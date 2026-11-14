//! Measurement: the part of the tool that stands in for hearing. `analyze`
//! turns audio into numbers, a list of concrete problems and a plain-language
//! description that an agent without ears can act on.

pub mod compare;
pub mod lint;
pub mod loudness;
pub mod spectral;
pub mod spectrogram;

use crate::audio::Sig;
use crate::dsp::resample::true_peak;
use crate::util::{db_to_lin, lin_to_db, round_to};
use serde_json::{json, Map, Value};
use spectral::{average_spectrum, band_shares, detect_key, detect_pitch, detect_rhythm, spectral_shape, stft, BANDS};

#[derive(Clone, Debug)]
pub struct Issue {
    /// "error" (must fix), "warning" (probably wrong) or "info".
    pub severity: &'static str,
    pub code: &'static str,
    pub message: String,
    pub fix: String,
}

impl Issue {
    pub fn to_json(&self) -> Value {
        json!({ "severity": self.severity, "code": self.code, "message": self.message, "fix": self.fix })
    }
}

#[derive(Clone, Debug)]
pub struct StereoInfo {
    pub correlation: f64,
    pub width: f64,
    pub balance_db: f64,
    pub mono_loss_db: f64,
}

#[derive(Clone, Debug)]
pub struct LoopSeam {
    /// Size of the discontinuity (step or kink) across the loop point in dBFS.
    pub jump_db: f64,
    /// The discontinuity relative to the normal curvature of the material.
    pub jump_ratio: f64,
    /// Level difference between the first and last 50 ms.
    pub level_diff_db: f64,
    pub audible: bool,
}

#[derive(Clone, Debug)]
pub struct Analysis {
    pub duration_s: f64,
    pub sample_rate: u32,
    pub channels: usize,
    pub peak_db: f64,
    pub true_peak_db: f64,
    pub rms_db: f64,
    pub lufs: f64,
    pub lufs_ungated: bool,
    pub momentary_max_lufs: f64,
    pub loudness_range: f64,
    pub crest_db: f64,
    pub dc_offset: f64,
    pub clipped_samples: usize,
    pub silence_start_s: f64,
    pub silence_end_s: f64,
    pub attack_ms: f64,
    pub decay_s: f64,
    pub effective_duration_s: f64,
    pub start_level_db: f64,
    pub end_level_db: f64,
    pub envelope_db: Vec<f64>,
    pub centroid_hz: f64,
    pub rolloff_hz: f64,
    pub flatness: f64,
    pub band_shares: Vec<f64>,
    /// Energy share in dB per third-octave band (see `third_octave_centers`).
    pub third_octave_db: Vec<f64>,
    /// Coarse time-by-band energy map in dB (rows = bands, low to high).
    pub band_envelope_db: Vec<Vec<f64>>,
    pub pitch: Option<spectral::Pitch>,
    pub onsets: Vec<f64>,
    pub tempo_bpm: Option<f64>,
    pub tempo_confidence: f64,
    pub key: Option<spectral::KeyEstimate>,
    pub stereo: Option<StereoInfo>,
    pub loop_seam: Option<LoopSeam>,
    pub silent: bool,
    pub issues: Vec<Issue>,
    pub description: String,
}

pub struct AnalyzeOpts {
    /// Treat the file as a loop and inspect the seam.
    pub is_loop: bool,
}

const ENV_POINTS: usize = 24;

/// Center frequencies of the third-octave bands used by `third_octave_db`.
pub fn third_octave_centers() -> Vec<f64> {
    (0..30).map(|k| 25.0 * 2f64.powf(k as f64 / 3.0)).collect()
}

fn third_octave(power: &[f64], bin_hz: f64) -> Vec<f64> {
    let total: f64 = power.iter().enumerate().filter(|(i, _)| *i as f64 * bin_hz >= 20.0).map(|(_, p)| *p).sum::<f64>().max(1e-30);
    third_octave_centers()
        .iter()
        .map(|fc| {
            let (lo, hi) = (fc / 2f64.powf(1.0 / 6.0), fc * 2f64.powf(1.0 / 6.0));
            let (a, b) = ((lo / bin_hz).ceil() as usize, ((hi / bin_hz) as usize).min(power.len() - 1));
            let e: f64 = if b >= a { power[a..=b].iter().sum() } else { power[a.min(power.len() - 1)] * (hi - lo) / bin_hz };
            round_to((10.0 * (e / total + 1e-12).log10()).max(-120.0), 1)
        })
        .collect()
}

fn rms(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
    }
}

pub fn analyze(sig: &Sig, sr: u32, opts: &AnalyzeOpts) -> Analysis {
    let srf = sr as f64;
    let n = sig.len();
    let mono = sig.to_mono();
    let duration_s = n as f64 / srf;
    let peak = sig.peak();
    let silent = peak < db_to_lin(-90.0);
    let tp = true_peak(sig);
    let all_rms = (sig.chans().iter().map(|c| c.iter().map(|v| v * v).sum::<f64>()).sum::<f64>() / (n.max(1) * sig.channels()) as f64).sqrt();
    let loud = loudness::measure(sig, sr);
    let dc = sig.chans().iter().map(|c| c.iter().sum::<f64>() / n.max(1) as f64).fold(0.0f64, |m, v| if v.abs() > m.abs() { v } else { m });
    let mut clipped = 0usize;
    for c in sig.chans() {
        let mut run = 0;
        for v in c {
            if v.abs() >= 0.999 {
                run += 1;
                if run >= 3 {
                    clipped += 1;
                }
            } else {
                run = 0;
            }
        }
    }

    // amplitude envelope: peak-of-channels RMS in 5 ms windows
    let win = ((0.005 * srf) as usize).max(1);
    let env: Vec<f64> = (0..n.div_ceil(win).max(1))
        .map(|k| {
            let (a, b) = (k * win, ((k + 1) * win).min(n));
            sig.chans().iter().map(|c| rms(&c[a..b])).fold(0.0, f64::max)
        })
        .collect();
    let env_peak = env.iter().cloned().fold(0.0, f64::max);
    let env_peak_idx = env.iter().position(|v| *v >= env_peak).unwrap_or(0);
    let th_sil = (env_peak * db_to_lin(-60.0)).max(db_to_lin(-80.0));
    let first = env.iter().position(|v| *v > th_sil);
    let last = env.iter().rposition(|v| *v > th_sil);
    let wsec = win as f64 / srf;
    let (silence_start_s, silence_end_s, effective_duration_s) = match (first, last) {
        (Some(f), Some(l)) => (f as f64 * wsec, (duration_s - (l + 1) as f64 * wsec).max(0.0), ((l + 1 - f) as f64 * wsec).min(duration_s)),
        _ => (duration_s, duration_s, 0.0),
    };
    // attack: from first audible window to 90% of the envelope peak
    let attack_ms = match first {
        Some(f) => {
            let target = env_peak * 0.9;
            let reach = env[f..].iter().position(|v| *v >= target).unwrap_or(0);
            // resolve sharp attacks below the 5 ms window size using raw samples
            if reach <= 1 {
                let a = f * win;
                let b = ((f + 2) * win).min(n);
                let seg_peak = mono[a..b].iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let start = mono[a..b].iter().position(|v| v.abs() > seg_peak * 0.05).unwrap_or(0);
                let hit = mono[a..b].iter().position(|v| v.abs() >= seg_peak * 0.9).unwrap_or(0);
                (hit.saturating_sub(start)) as f64 / srf * 1000.0
            } else {
                reach as f64 * wsec * 1000.0
            }
        }
        None => 0.0,
    };
    // decay: time from the envelope peak until it has fallen 20 dB
    let decay_s = {
        let th = env_peak * db_to_lin(-20.0);
        match env[env_peak_idx..].iter().position(|v| *v < th) {
            Some(k) => k as f64 * wsec,
            None => (env.len() - env_peak_idx) as f64 * wsec,
        }
    };
    let edge = ((0.005 * srf) as usize).clamp(1, n.max(1));
    let start_level_db = lin_to_db(sig.chans().iter().map(|c| c.first().copied().unwrap_or(0.0).abs()).fold(0.0, f64::max));
    let end_level_db = if n == 0 { -240.0 } else { lin_to_db(sig.chans().iter().flat_map(|c| c[n - edge..].iter()).fold(0.0f64, |m, v| m.max(v.abs()))) };
    let envelope_db: Vec<f64> = (0..ENV_POINTS)
        .map(|k| {
            let a = k * env.len() / ENV_POINTS;
            let b = (((k + 1) * env.len()) / ENV_POINTS).max(a + 1).min(env.len());
            if a >= env.len() {
                return -120.0;
            }
            round_to(lin_to_db(env[a..b].iter().cloned().fold(0.0, f64::max)).max(-120.0), 1)
        })
        .collect();

    // spectrum
    let st = stft(&mono, 4096, 1024, srf);
    let avg = average_spectrum(&st);
    let shape = spectral_shape(&avg, st.bin_hz());
    let shares = band_shares(&avg, st.bin_hz());
    let third_octave_db = third_octave(&avg, st.bin_hz());
    let cols = ENV_POINTS.min(st.frames.len()).max(1);
    let band_envelope_db: Vec<Vec<f64>> = BANDS
        .iter()
        .map(|b| {
            let lo = (b.1 / st.bin_hz()).ceil() as usize;
            let hi = ((b.2 / st.bin_hz()) as usize).min(st.size / 2);
            (0..cols)
                .map(|k| {
                    let a = k * st.frames.len() / cols;
                    let e = (((k + 1) * st.frames.len()) / cols).max(a + 1).min(st.frames.len());
                    let mut p = 0.0;
                    for f in &st.frames[a..e] {
                        if hi >= lo {
                            p += f[lo..=hi].iter().sum::<f64>();
                        }
                    }
                    // scaled so a full-scale sine reads about 0 dB
                    round_to((10.0 * (p / (e - a) as f64 * 4.0 + 1e-12).log10()).max(-120.0), 0)
                })
                .collect()
        })
        .collect();

    let pitch = if silent { None } else { detect_pitch(&mono, srf) };
    let rhythm = detect_rhythm(&mono, srf);
    let key = if duration_s >= 4.0 && !silent { detect_key(&mono, srf) } else { None };

    let stereo = sig.r.as_ref().map(|r| {
        let (mut ll, mut rr, mut lr, mut mm, mut ss) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (a, b) in sig.l.iter().zip(r) {
            ll += a * a;
            rr += b * b;
            lr += a * b;
            mm += (0.5 * (a + b)).powi(2);
            ss += (0.5 * (a - b)).powi(2);
        }
        let correlation = if ll > 0.0 && rr > 0.0 { lr / (ll * rr).sqrt() } else { 1.0 };
        let width = if mm + ss > 0.0 { ss / (mm + ss) } else { 0.0 };
        let balance_db = if ll > 0.0 && rr > 0.0 { 10.0 * (ll / rr).log10() } else { 0.0 };
        let mono_loss_db = if mm > 0.0 && ll + rr > 0.0 { 10.0 * (mm / (0.5 * (ll + rr))).log10() } else { 0.0 };
        StereoInfo { correlation, width, balance_db, mono_loss_db }
    });

    let loop_seam = if opts.is_loop && n > 64 {
        // A seamless loop is smooth across the wrap: compare the curvature at
        // the seam with the curvature typical of the material near both ends.
        let mut kink = 0.0f64;
        let mut local = 0.0f64;
        for c in sig.chans() {
            kink = kink.max((c[1] - 2.0 * c[0] + c[n - 1]).abs()).max((c[0] - 2.0 * c[n - 1] + c[n - 2]).abs());
            let k = 256.min(n / 4);
            let mut curv: Vec<f64> = (1..k).map(|i| (c[i + 1] - 2.0 * c[i] + c[i - 1]).abs()).chain((n - k..n - 1).map(|i| (c[i + 1] - 2.0 * c[i] + c[i - 1]).abs())).collect();
            curv.sort_by(|a, b| a.partial_cmp(b).unwrap());
            local = local.max(curv[(curv.len() as f64 * 0.95) as usize]);
        }
        let w = ((0.05 * srf) as usize).min(n / 2);
        let level_diff_db = lin_to_db(rms(&mono[..w]).max(1e-9)) - lin_to_db(rms(&mono[n - w..]).max(1e-9));
        let jump_ratio = if local > 1e-9 { kink / local } else if kink > 1e-6 { 99.0 } else { 0.0 };
        // A level step across the seam only matters for steady material (an
        // ambience that fades); in rhythmic material bars start and end differently.
        let inner = &envelope_db[2..ENV_POINTS - 2];
        let steady = inner.iter().cloned().fold(f64::MIN, f64::max) - inner.iter().cloned().fold(f64::MAX, f64::min) < 8.0;
        let audible = (kink > db_to_lin(-60.0) && jump_ratio > 3.0) || (steady && level_diff_db.abs() > 3.0);
        Some(LoopSeam { jump_db: lin_to_db(kink), jump_ratio, level_diff_db, audible })
    } else {
        None
    };

    let mut a = Analysis {
        duration_s,
        sample_rate: sr,
        channels: sig.channels(),
        peak_db: lin_to_db(peak),
        true_peak_db: lin_to_db(tp),
        rms_db: lin_to_db(all_rms),
        lufs: loud.integrated,
        lufs_ungated: loud.ungated,
        momentary_max_lufs: loud.momentary_max,
        loudness_range: loud.range,
        crest_db: if all_rms > 0.0 { lin_to_db(peak) - lin_to_db(all_rms) } else { 0.0 },
        dc_offset: dc,
        clipped_samples: clipped,
        silence_start_s,
        silence_end_s,
        attack_ms,
        decay_s,
        effective_duration_s,
        start_level_db,
        end_level_db,
        envelope_db,
        centroid_hz: shape.centroid_hz,
        rolloff_hz: shape.rolloff_hz,
        flatness: shape.flatness,
        band_shares: shares,
        third_octave_db,
        band_envelope_db,
        pitch,
        onsets: rhythm.onsets,
        tempo_bpm: rhythm.tempo_bpm,
        tempo_confidence: rhythm.tempo_confidence,
        key,
        stereo,
        loop_seam,
        silent,
        issues: Vec::new(),
        description: String::new(),
    };
    a.issues = find_issues(&a, opts);
    a.description = describe(&a, opts);
    a
}

/// Problems that are wrong for any asset, whatever its purpose.
fn find_issues(a: &Analysis, opts: &AnalyzeOpts) -> Vec<Issue> {
    let mut v = Vec::new();
    let mut add = |severity, code, message: String, fix: &str| v.push(Issue { severity, code, message, fix: fix.to_string() });
    if a.silent {
        add("error", "silent", "the file contains no audible sound".into(), "check that `output` points at the right node, that envelopes rise above zero and that gains are not extremely low");
        return v;
    }
    if a.clipped_samples > 0 {
        add("error", "clipping", format!("{} samples are clipped at full scale", a.clipped_samples), "lower the level before the output or add a limiter node as the last step");
    } else if a.true_peak_db > -0.1 {
        add("warning", "true_peak_high", format!("true peak is {:+.2} dBFS; lossy encoding or resampling in a game engine may clip", a.true_peak_db), "normalize to -1 dBFS peak (the default) or add `{type: limiter, ceiling_db: -1}`");
    }
    if a.dc_offset.abs() > 0.01 {
        add("warning", "dc_offset", format!("DC offset of {:.3} wastes headroom and can click at start and end", a.dc_offset), "add a `dcblock` node before the output, or a highpass at 20 Hz");
    }
    if a.silence_start_s > 0.02 {
        add("warning", "leading_silence", format!("{:.0} ms of silence before the sound starts; it will feel late when triggered", a.silence_start_s * 1000.0), "remove delays at the start: check envelope points, `shift` nodes and reverb predelay");
    }
    if !opts.is_loop && a.silence_end_s > 0.5 && a.silence_end_s > 0.3 * a.duration_s {
        add("info", "trailing_silence", format!("{:.2} s of silence at the end wastes memory", a.silence_end_s), "shorten `duration` to just past the end of the tail");
    }
    if !opts.is_loop && a.end_level_db > -50.0 {
        add("warning", "cut_off", format!("the sound is still at {:.0} dBFS when the file ends, so the tail is cut off", a.end_level_db), "increase `duration`, or shorten envelopes and reverb decay so the sound fades out before the end");
    }
    if !opts.is_loop && a.start_level_db > -20.0 {
        add("info", "abrupt_start", format!("the first sample is already at {:.0} dBFS, which adds a click to the attack", a.start_level_db), "fine for impacts; otherwise start oscillators at phase 0 or give the envelope a 1-3 ms attack");
    }
    if a.peak_db < -26.0 {
        add("warning", "too_quiet", format!("peak level is only {:.1} dBFS", a.peak_db), "enable normalization (`normalize: {peak_db: -1}`) or raise the output gain");
    }
    if let Some(s) = &a.stereo {
        if s.correlation < -0.2 {
            add("warning", "phase_cancellation", format!("left and right are out of phase (correlation {:.2}); the sound will thin out or vanish on mono speakers", s.correlation), "reduce `width` amounts above 1, or reduce stereo spread in chorus and unison");
        } else if s.mono_loss_db < -4.0 {
            add("info", "mono_loss", format!("loses {:.1} dB when summed to mono", -s.mono_loss_db), "reduce stereo width if the sound must work on phone speakers");
        }
        if s.balance_db.abs() > 3.0 {
            add("info", "off_center", format!("the {} channel is {:.1} dB louder", if s.balance_db > 0.0 { "left" } else { "right" }, s.balance_db.abs()), "fine if intentional; otherwise check `pan` positions");
        }
    }
    let low = a.band_shares[0];
    if low > 0.7 {
        add("warning", "mostly_sub_bass", format!("{:.0}% of the energy is below 60 Hz, which small speakers cannot reproduce", low * 100.0), "add harmonics (distort node) or a layer in the 100-2000 Hz range so the sound survives on small speakers");
    }
    if let Some(l) = &a.loop_seam {
        if l.audible {
            let msg = if !(l.jump_db > -60.0 && l.jump_ratio > 3.0) {
                format!("the loop does not join smoothly: level differs by {:.1} dB between end and start", l.level_diff_db)
            } else {
                format!("the loop point clicks: a {:.0} dBFS discontinuity where the end wraps to the start", l.jump_db)
            };
            add("error", "loop_seam", msg, "render with `loop: true` in the patch (or --loop), which crossfades the end into the start; avoid one-shot envelopes in looping sounds");
        }
    }
    v
}

fn describe(a: &Analysis, opts: &AnalyzeOpts) -> String {
    if a.silent {
        return "Silent.".into();
    }
    let mut parts: Vec<String> = Vec::new();
    let len = if a.duration_s < 0.15 {
        "Very short"
    } else if a.duration_s < 0.7 {
        "Short"
    } else if a.duration_s < 3.0 {
        "Medium-length"
    } else {
        "Long"
    };
    let bright = if a.centroid_hz < 250.0 {
        "very dark, boomy"
    } else if a.centroid_hz < 800.0 {
        "dark, warm"
    } else if a.centroid_hz < 2500.0 {
        "balanced"
    } else if a.centroid_hz < 6000.0 {
        "bright"
    } else {
        "very bright, thin"
    };
    let texture = match &a.pitch {
        Some(p) if p.tonal_fraction > 0.6 => "tonal",
        Some(p) if p.tonal_fraction > 0.25 => "partly tonal",
        _ if a.flatness > 0.25 => "noisy",
        _ => "inharmonic",
    };
    parts.push(format!("{len} ({:.2} s), {bright}, {texture} sound", a.duration_s));
    let attack = if a.attack_ms <= 5.0 {
        "a sharp, percussive attack"
    } else if a.attack_ms <= 40.0 {
        "a fast attack"
    } else if a.attack_ms <= 300.0 {
        "a soft attack"
    } else {
        "a slow swell"
    };
    let sustained = a.decay_s > 0.6 * a.duration_s;
    let tail = if opts.is_loop || (sustained && a.end_level_db > -50.0) {
        "a sustained level throughout".to_string()
    } else if sustained {
        "a sustained body".to_string()
    } else if a.decay_s < 0.08 {
        "an immediate decay".to_string()
    } else if a.decay_s < 0.5 {
        format!("a quick decay ({:.2} s to fall 20 dB)", a.decay_s)
    } else {
        format!("a long tail ({:.1} s to fall 20 dB)", a.decay_s)
    };
    parts.push(format!("with {attack} and {tail}"));
    if let Some(p) = &a.pitch {
        if p.tonal_fraction > 0.25 {
            let ratio = p.end_hz / p.start_hz;
            if ratio > 1.25 {
                parts.push(format!("Pitch rises from about {:.0} Hz to {:.0} Hz", p.start_hz, p.end_hz));
            } else if ratio < 0.8 {
                parts.push(format!("Pitch falls from about {:.0} Hz to {:.0} Hz", p.start_hz, p.end_hz));
            } else {
                parts.push(format!("Pitch is around {} ({:.0} Hz)", p.note, p.hz));
            }
        }
    }
    let (imax, share) = a.band_shares.iter().enumerate().fold((0, 0.0), |(bi, bv), (i, v)| if *v > bv { (i, *v) } else { (bi, bv) });
    let band_label = ["sub-bass (20-60 Hz)", "bass (60-250 Hz)", "low mids (250-500 Hz)", "mids (500 Hz-2 kHz)", "upper mids (2-6 kHz)", "highs (6-12 kHz)", "air (12-20 kHz)"][imax];
    parts.push(format!("Most energy ({:.0}%) is in the {band_label}", share * 100.0));
    match &a.stereo {
        None => parts.push("Mono".into()),
        Some(s) if s.width < 0.02 => parts.push("Stereo file but effectively mono".into()),
        Some(s) if s.width < 0.15 => parts.push("Narrow stereo image".into()),
        Some(s) if s.width < 0.4 => parts.push("Wide stereo image".into()),
        Some(_) => parts.push("Very wide stereo image".into()),
    }
    if let Some(bpm) = a.tempo_bpm {
        if a.is_musical() {
            parts.push(format!("Rhythmic, about {bpm:.0} BPM with {} onsets", a.onsets.len()));
        }
    }
    if let Some(k) = &a.key {
        if k.confidence > 0.3 && a.is_musical() {
            parts.push(format!("Sounds like {}", k.name));
        }
    }
    let dynamics = if a.crest_db > 20.0 {
        "very spiky dynamics"
    } else if a.crest_db > 12.0 {
        "punchy dynamics"
    } else if a.crest_db > 6.0 {
        "moderate dynamics"
    } else {
        "dense, compressed dynamics"
    };
    parts.push(format!("{} LUFS with {dynamics}", round_to(a.lufs, 1)));
    let mut s = parts.join(". ").replace(" sound. with", " sound with");
    s.push('.');
    s
}

fn r(x: f64, d: i32) -> Value {
    if x.is_finite() {
        json!(round_to(x, d))
    } else {
        Value::Null
    }
}

impl Analysis {
    /// True when the file looks like music (long, with a steady pulse) rather
    /// than a sound effect; tempo and key are only meaningful then.
    pub fn is_musical(&self) -> bool {
        self.duration_s >= 6.0 && self.onsets.len() >= 12 && self.tempo_bpm.is_some() && self.tempo_confidence > 0.3
    }

    /// Compact summary: the numbers an agent needs after every render.
    pub fn summary_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("duration_s".into(), r(self.duration_s, 3));
        m.insert("peak_db".into(), r(self.peak_db, 1));
        m.insert("true_peak_db".into(), r(self.true_peak_db, 1));
        m.insert("lufs".into(), r(self.lufs, 1));
        m.insert("centroid_hz".into(), r(self.centroid_hz, 0));
        m.insert("attack_ms".into(), r(self.attack_ms, 1));
        m.insert("decay_s".into(), r(self.decay_s, 3));
        if let Some(p) = &self.pitch {
            if p.tonal_fraction > 0.25 {
                m.insert("pitch".into(), json!(format!("{} ({:.0} Hz)", p.note, p.hz)));
            }
        }
        if self.is_musical() {
            if let Some(t) = self.tempo_bpm {
                m.insert("tempo_bpm".into(), r(t, 1));
            }
            if let Some(k) = &self.key {
                m.insert("key".into(), json!(k.name));
            }
        }
        m.insert("description".into(), json!(self.description));
        m.insert("issues".into(), Value::Array(self.issues.iter().map(|i| i.to_json()).collect()));
        Value::Object(m)
    }

    pub fn to_json(&self, detail: bool) -> Value {
        let mut bands = Map::new();
        for (b, s) in BANDS.iter().zip(&self.band_shares) {
            bands.insert(b.0.into(), json!({ "range_hz": [b.1, b.2], "percent": round_to(s * 100.0, 1), "db": r(10.0 * (s + 1e-12).log10(), 1) }));
        }
        let mut out = json!({
            "description": self.description,
            "issues": self.issues.iter().map(|i| i.to_json()).collect::<Vec<_>>(),
            "file": {
                "duration_s": r(self.duration_s, 4),
                "sample_rate": self.sample_rate,
                "channels": self.channels,
            },
            "level": {
                "peak_db": r(self.peak_db, 2),
                "true_peak_db": r(self.true_peak_db, 2),
                "rms_db": r(self.rms_db, 2),
                "lufs": r(self.lufs, 2),
                "lufs_is_ungated": self.lufs_ungated,
                "momentary_max_lufs": r(self.momentary_max_lufs, 2),
                "loudness_range_lu": r(self.loudness_range, 1),
                "crest_factor_db": r(self.crest_db, 1),
                "dc_offset": r(self.dc_offset, 5),
                "clipped_samples": self.clipped_samples,
            },
            "time": {
                "silence_start_s": r(self.silence_start_s, 3),
                "silence_end_s": r(self.silence_end_s, 3),
                "attack_ms": r(self.attack_ms, 1),
                "decay_s": r(self.decay_s, 3),
                "effective_duration_s": r(self.effective_duration_s, 3),
                "start_level_db": r(self.start_level_db.max(-120.0), 1),
                "end_level_db": r(self.end_level_db.max(-120.0), 1),
                "envelope_db": self.envelope_db,
            },
            "spectrum": {
                "centroid_hz": r(self.centroid_hz, 0),
                "rolloff_85_hz": r(self.rolloff_hz, 0),
                "flatness": r(self.flatness, 3),
                "bands": bands,
            },
        });
        let o = out.as_object_mut().unwrap();
        o.insert(
            "pitch".into(),
            match &self.pitch {
                Some(p) => json!({
                    "hz": r(p.hz, 1), "note": p.note, "cents_off": r(p.cents_off, 0), "clarity": r(p.clarity, 2),
                    "tonal_fraction": r(p.tonal_fraction, 2), "start_hz": r(p.start_hz, 1), "end_hz": r(p.end_hz, 1),
                }),
                None => Value::Null,
            },
        );
        o.insert(
            "rhythm".into(),
            json!({
                "onset_count": self.onsets.len(),
                "onsets_s": self.onsets.iter().take(if detail { 512 } else { 32 }).map(|t| round_to(*t, 3)).collect::<Vec<_>>(),
                "tempo_bpm": self.tempo_bpm.map(|t| round_to(t, 1)),
                "tempo_confidence": r(self.tempo_confidence, 2),
            }),
        );
        o.insert("key".into(), match &self.key {
            Some(k) => json!({ "name": k.name, "confidence": r(k.confidence, 2) }),
            None => Value::Null,
        });
        o.insert(
            "stereo".into(),
            match &self.stereo {
                Some(s) => json!({ "correlation": r(s.correlation, 2), "width": r(s.width, 2), "balance_db": r(s.balance_db, 1), "mono_loss_db": r(s.mono_loss_db, 1) }),
                None => Value::Null,
            },
        );
        if let Some(l) = &self.loop_seam {
            o.insert("loop_seam".into(), json!({ "jump_db": r(l.jump_db.max(-120.0), 1), "jump_ratio": r(l.jump_ratio, 1), "level_diff_db": r(l.level_diff_db, 1), "audible": l.audible }));
        }
        if detail {
            o.insert("third_octave".into(), json!({ "centers_hz": third_octave_centers().iter().map(|f| round_to(*f, 0)).collect::<Vec<_>>(), "db": self.third_octave_db }));
            let mut be = Map::new();
            for (b, row) in BANDS.iter().zip(&self.band_envelope_db) {
                be.insert(b.0.into(), json!(row));
            }
            o.insert("band_envelope_db".into(), json!({ "note": "energy per band over time, left to right = start to end of file; a full-scale sine reads about 0 dB", "bands": be }));
        }
        out
    }

    /// A text picture of the sound: level per band over time.
    pub fn ascii_map(&self) -> String {
        let ramp: Vec<char> = " .:-=+*#%@".chars().collect();
        let mut s = String::new();
        for (b, row) in BANDS.iter().zip(&self.band_envelope_db).rev() {
            s.push_str(&format!("{:>8} |", b.0));
            for v in row {
                let x = ((*v + 70.0) / 70.0).clamp(0.0, 1.0);
                s.push(ramp[(x * (ramp.len() - 1) as f64).round() as usize]);
            }
            s.push_str("|\n");
        }
        s.push_str(&format!("{:>8} +{}+\n", "", "-".repeat(self.band_envelope_db.first().map_or(0, |r| r.len()))));
        s.push_str(&format!("{:>8}  0s{:>width$.2}s\n", "", self.duration_s, width = self.band_envelope_db.first().map_or(4, |r| r.len().saturating_sub(3))));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn tone(f: f64, secs: f64, amp: f64) -> Vec<f64> {
        (0..(48000.0 * secs) as usize).map(|i| (TAU * f * i as f64 / 48000.0).sin() * amp).collect()
    }

    #[test]
    fn measures_a_decaying_tone() {
        let mut x = tone(440.0, 1.0, 0.8);
        for (i, v) in x.iter_mut().enumerate() {
            *v *= (-(i as f64) / 4800.0).exp();
        }
        let a = analyze(&Sig::mono(x), 48000, &AnalyzeOpts { is_loop: false });
        assert!(!a.silent);
        assert!((a.peak_db + 1.9).abs() < 0.6, "peak {}", a.peak_db);
        assert!(a.attack_ms < 5.0, "attack {}", a.attack_ms);
        assert!((a.decay_s - 0.23).abs() < 0.05, "decay {}", a.decay_s);
        assert_eq!(a.pitch.as_ref().unwrap().note, "A4");
        assert!(a.description.contains("percussive"));
        assert!(a.issues.iter().all(|i| i.code != "cut_off"), "{:?}", a.issues);
    }

    #[test]
    fn flags_problems() {
        let a = analyze(&Sig::mono(vec![0.0; 4800]), 48000, &AnalyzeOpts { is_loop: false });
        assert!(a.issues.iter().any(|i| i.code == "silent"));
        let clipped: Vec<f64> = tone(100.0, 0.5, 3.0).iter().map(|v| v.clamp(-1.0, 1.0)).collect();
        let a = analyze(&Sig::mono(clipped), 48000, &AnalyzeOpts { is_loop: false });
        assert!(a.issues.iter().any(|i| i.code == "clipping"));
        assert!(a.issues.iter().any(|i| i.code == "cut_off"));
        let x = tone(300.0, 0.5, 0.5);
        let inv: Vec<f64> = x.iter().map(|v| -v).collect();
        let a = analyze(&Sig::stereo(x, inv), 48000, &AnalyzeOpts { is_loop: false });
        assert!(a.issues.iter().any(|i| i.code == "phase_cancellation"));
    }

    #[test]
    fn loop_seam_detection() {
        // 440 Hz over exactly 1 s loops perfectly; 440.5 Hz does not
        let good = analyze(&Sig::mono(tone(440.0, 1.0, 0.5)), 48000, &AnalyzeOpts { is_loop: true });
        assert!(!good.loop_seam.unwrap().audible);
        let bad = analyze(&Sig::mono(tone(440.5, 1.0, 0.5)), 48000, &AnalyzeOpts { is_loop: true });
        assert!(bad.loop_seam.unwrap().audible);
    }
}
