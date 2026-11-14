//! Compare a candidate sound to a reference and say, in words and numbers,
//! how to move the candidate closer.

use super::{third_octave_centers, Analysis};
use crate::util::round_to;
use serde_json::{json, Value};

fn score(distance: f64, scale: f64) -> f64 {
    (-distance / scale).exp()
}

pub fn compare(cand: &Analysis, refr: &Analysis) -> Value {
    let mut hints: Vec<String> = Vec::new();

    // spectrum: third-octave profile distance, ignoring bands that are empty in both
    let (mut d_spec, mut cnt) = (0.0, 0.0);
    for (a, b) in cand.third_octave_db.iter().zip(&refr.third_octave_db) {
        let (a, b) = (a.max(-60.0), b.max(-60.0));
        if a > -60.0 || b > -60.0 {
            d_spec += (a - b).abs();
            cnt += 1.0;
        }
    }
    let d_spec = if cnt > 0.0 { d_spec / cnt } else { 0.0 };
    let s_spec = score(d_spec, 8.0);

    // envelope shape on a normalized time axis, relative to each sound's peak
    let norm = |e: &Vec<f64>| -> Vec<f64> {
        let p = e.iter().cloned().fold(f64::MIN, f64::max);
        e.iter().map(|v| (v - p).max(-60.0)).collect()
    };
    let (ec, er) = (norm(&cand.envelope_db), norm(&refr.envelope_db));
    let d_env = ec.iter().zip(&er).map(|(a, b)| (a - b).abs()).sum::<f64>() / ec.len().max(1) as f64;
    let s_env = score(d_env, 10.0);

    let dur_ratio = if refr.duration_s > 0.0 { cand.duration_s / refr.duration_s } else { 1.0 };
    let s_dur = score(dur_ratio.max(1e-6).log2().abs(), 0.6);
    let d_loud = cand.lufs - refr.lufs;
    let s_loud = score(d_loud.abs(), 8.0);

    let s_pitch = match (&cand.pitch, &refr.pitch) {
        (Some(a), Some(b)) if a.tonal_fraction > 0.25 && b.tonal_fraction > 0.25 => {
            let semis = 12.0 * (a.hz / b.hz).log2();
            if semis.abs() > 0.5 {
                hints.push(format!("pitch is {:.1} semitones {} than the reference ({} vs {}): {} `freq` by that amount", semis.abs(), if semis > 0.0 { "higher" } else { "lower" }, a.note, b.note, if semis > 0.0 { "lower" } else { "raise" }));
            }
            Some(score(semis.abs(), 4.0))
        }
        (Some(a), None) if a.tonal_fraction > 0.5 => {
            hints.push("the candidate has a clear pitch but the reference is noise-like: replace oscillators with filtered noise, or add inharmonic content".into());
            Some(0.4)
        }
        (None, Some(b)) if b.tonal_fraction > 0.5 => {
            hints.push(format!("the reference has a clear pitch ({}) but the candidate does not: add an oscillator or resonant filter at {:.0} Hz", b.note, b.hz));
            Some(0.4)
        }
        _ => None,
    };

    if dur_ratio > 1.25 {
        hints.push(format!("{:.2} s is longer than the reference ({:.2} s): shorten `duration` and envelope times", cand.duration_s, refr.duration_s));
    } else if dur_ratio < 0.8 {
        hints.push(format!("{:.2} s is shorter than the reference ({:.2} s): lengthen `duration` and envelope times", cand.duration_s, refr.duration_s));
    }
    let cent_oct = (cand.centroid_hz.max(1.0) / refr.centroid_hz.max(1.0)).log2();
    if cent_oct > 0.35 {
        hints.push(format!("brighter than the reference (centroid {:.0} Hz vs {:.0} Hz): lower filter cutoffs or cut highs with a highshelf", cand.centroid_hz, refr.centroid_hz));
    } else if cent_oct < -0.35 {
        hints.push(format!("darker than the reference (centroid {:.0} Hz vs {:.0} Hz): raise filter cutoffs, add noise or upper partials", cand.centroid_hz, refr.centroid_hz));
    }
    // largest broad-band differences
    let names = ["sub (20-60 Hz)", "bass (60-250 Hz)", "low mids (250-500 Hz)", "mids (500 Hz-2 kHz)", "upper mids (2-6 kHz)", "highs (6-12 kHz)", "air (12-20 kHz)"];
    let mut band_diffs: Vec<(usize, f64)> = (0..7)
        .map(|i| {
            let a = 10.0 * (cand.band_shares[i] + 1e-6).log10();
            let b = 10.0 * (refr.band_shares[i] + 1e-6).log10();
            (i, a.max(-40.0) - b.max(-40.0))
        })
        .collect();
    band_diffs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    for (i, d) in band_diffs.iter().take(2) {
        if d.abs() > 4.0 {
            hints.push(format!("{:.0} dB {} energy in the {} than the reference", d.abs(), if *d > 0.0 { "more" } else { "less" }, names[*i]));
        }
    }
    if cand.attack_ms > refr.attack_ms * 2.0 + 5.0 {
        hints.push(format!("attack is slower ({:.0} ms vs {:.0} ms): shorten the envelope attack", cand.attack_ms, refr.attack_ms));
    } else if refr.attack_ms > cand.attack_ms * 2.0 + 5.0 {
        hints.push(format!("attack is faster ({:.0} ms vs {:.0} ms): lengthen the envelope attack", cand.attack_ms, refr.attack_ms));
    }
    if cand.decay_s > refr.decay_s * 1.5 + 0.03 {
        hints.push(format!("decays more slowly ({:.2} s vs {:.2} s to fall 20 dB): shorten decay/release or reverb", cand.decay_s, refr.decay_s));
    } else if refr.decay_s > cand.decay_s * 1.5 + 0.03 {
        hints.push(format!("decays faster ({:.2} s vs {:.2} s to fall 20 dB): lengthen decay/release or add reverb", cand.decay_s, refr.decay_s));
    }
    if d_loud.abs() > 2.0 {
        hints.push(format!("{:.1} LU {} than the reference", d_loud.abs(), if d_loud > 0.0 { "louder" } else { "quieter" }));
    }
    let (wc, wr) = (cand.stereo.as_ref().map_or(0.0, |s| s.width), refr.stereo.as_ref().map_or(0.0, |s| s.width));
    if wr - wc > 0.1 {
        hints.push("narrower stereo image than the reference: add stereo noise, unison spread, chorus or reverb".into());
    } else if wc - wr > 0.1 {
        hints.push("wider stereo image than the reference: reduce spread, chorus or reverb width".into());
    }
    if (cand.flatness - refr.flatness).abs() > 0.15 {
        hints.push(if cand.flatness > refr.flatness { "noisier than the reference: reduce noise layers or narrow them with bandpass filters".into() } else { "more tonal than the reference: add noise layers or detune/inharmonic partials".into() });
    }

    let mut weighted = vec![(s_spec, 0.4), (s_env, 0.25), (s_dur, 0.15), (s_loud, 0.05)];
    if let Some(sp) = s_pitch {
        weighted.push((sp, 0.15));
    }
    let total_w: f64 = weighted.iter().map(|w| w.1).sum();
    let similarity = weighted.iter().map(|w| w.0 * w.1).sum::<f64>() / total_w * 100.0;
    if hints.is_empty() {
        hints.push("no significant differences found in duration, spectrum, envelope, pitch, loudness or width".into());
    }
    let centers = third_octave_centers();
    let spectrum_diff: Vec<Value> = centers
        .iter()
        .zip(cand.third_octave_db.iter().zip(&refr.third_octave_db))
        .filter(|(_, (a, b))| a.max(-60.0) > -60.0 || b.max(-60.0) > -60.0)
        .map(|(f, (a, b))| json!({ "hz": round_to(*f, 0), "diff_db": round_to(a.max(-60.0) - b.max(-60.0), 1) }))
        .collect();
    json!({
        "similarity": round_to(similarity, 1),
        "similarity_note": "0-100; above 85 is very close, below 50 is a different kind of sound. Based on spectrum, envelope, duration, pitch and loudness.",
        "scores": {
            "spectrum": round_to(s_spec * 100.0, 1),
            "envelope": round_to(s_env * 100.0, 1),
            "duration": round_to(s_dur * 100.0, 1),
            "loudness": round_to(s_loud * 100.0, 1),
            "pitch": s_pitch.map(|s| round_to(s * 100.0, 1)),
        },
        "how_to_get_closer": hints,
        "candidate": cand.summary_json(),
        "reference": refr.summary_json(),
        "spectrum_diff_db": spectrum_diff,
    })
}
