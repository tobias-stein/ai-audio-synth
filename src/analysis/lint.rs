//! Lint profiles: what "correct" looks like for a given kind of game audio
//! asset, checked against the measurements, with a concrete fix for each miss.

use super::{Analysis, Issue};
use crate::error::{Error, Result};
use crate::util::hint_from;
use serde_json::{json, Value};

pub struct Profile {
    pub name: &'static str,
    pub description: &'static str,
    pub duration: (f64, f64),
    /// Acceptable integrated loudness range in LUFS.
    pub lufs: Option<(f64, f64)>,
    pub max_true_peak_db: f64,
    pub max_attack_ms: Option<f64>,
    pub min_attack_ms: Option<f64>,
    pub is_loop: bool,
    /// Maximum share of energy below 250 Hz.
    pub max_low_share: Option<f64>,
    pub min_centroid_hz: Option<f64>,
    pub max_centroid_hz: Option<f64>,
    pub min_crest_db: Option<f64>,
    pub max_crest_db: Option<f64>,
    pub want_stereo: bool,
    pub max_silence_start_ms: f64,
}

const fn base(name: &'static str, description: &'static str, duration: (f64, f64)) -> Profile {
    Profile {
        name,
        description,
        duration,
        lufs: None,
        max_true_peak_db: -0.9,
        max_attack_ms: None,
        min_attack_ms: None,
        is_loop: false,
        max_low_share: None,
        min_centroid_hz: None,
        max_centroid_hz: None,
        min_crest_db: None,
        max_crest_db: None,
        want_stereo: false,
        max_silence_start_ms: 10.0,
    }
}

pub fn profiles() -> Vec<Profile> {
    vec![
        Profile { max_attack_ms: Some(20.0), max_low_share: Some(0.4), min_centroid_hz: Some(500.0), max_silence_start_ms: 5.0, ..base("ui", "Interface feedback: clicks, confirms, errors, notifications. Short, immediate, light on bass so it never masks gameplay audio.", (0.02, 1.0)) },
        Profile { ..base("oneshot", "Any one-shot sound effect. Checks only universal correctness: starts immediately, ends in silence, no clipping.", (0.03, 12.0)) },
        Profile { max_attack_ms: Some(12.0), min_crest_db: Some(9.0), ..base("impact", "Hits, explosions, slams, body falls. Needs an immediate, punchy transient and a tail that dies away.", (0.08, 8.0)) },
        Profile { max_attack_ms: Some(10.0), min_crest_db: Some(8.0), min_centroid_hz: Some(400.0), ..base("weapon", "Gunshots, lasers, blasters, melee swings with impact. Immediate attack; must cut through a busy mix.", (0.05, 4.0)) },
        Profile { min_attack_ms: Some(25.0), max_silence_start_ms: 40.0, ..base("whoosh", "Whooshes, swishes, risers and passes. Should swell rather than start abruptly.", (0.15, 10.0)) },
        Profile { is_loop: true, lufs: Some((-40.0, -18.0)), max_crest_db: Some(22.0), want_stereo: true, max_silence_start_ms: 1e9, ..base("ambience-loop", "Looping background beds: wind, rain, room tone, drones. Must loop seamlessly, stay steady, and sit well below foreground sounds.", (4.0, 600.0)) },
        Profile { is_loop: true, max_crest_db: Some(20.0), max_silence_start_ms: 1e9, ..base("sfx-loop", "Looping effects attached to objects: engines, fires, force fields, alarms. Must loop seamlessly.", (0.3, 120.0)) },
        Profile { is_loop: true, lufs: Some((-24.0, -12.0)), want_stereo: true, max_silence_start_ms: 1e9, ..base("music-loop", "Looping music cue or layer. Must loop seamlessly at a musical boundary.", (2.0, 900.0)) },
        Profile { lufs: Some((-20.0, -11.0)), want_stereo: true, max_silence_start_ms: 500.0, ..base("music", "A complete, mastered music track.", (8.0, 900.0)) },
        Profile { lufs: Some((-24.0, -10.0)), max_silence_start_ms: 30.0, ..base("stinger", "Short musical phrase: victory, defeat, discovery, level-up.", (0.4, 12.0)) },
    ]
}

pub fn profile_names() -> Vec<&'static str> {
    profiles().iter().map(|p| p.name).collect()
}

pub fn find_profile(name: &str) -> Result<Profile> {
    profiles().into_iter().find(|p| p.name == name).ok_or_else(|| Error::new("unknown_name", format!("unknown lint profile '{name}'")).hint(hint_from(name, &profile_names())))
}

pub struct LintResult {
    pub profile: &'static str,
    pub issues: Vec<Issue>,
}

impl LintResult {
    pub fn errors(&self) -> usize {
        self.issues.iter().filter(|i| i.severity == "error").count()
    }
    pub fn warnings(&self) -> usize {
        self.issues.iter().filter(|i| i.severity == "warning").count()
    }
    pub fn passed(&self, strict: bool) -> bool {
        self.errors() == 0 && (!strict || self.warnings() == 0)
    }
    pub fn to_json(&self, strict: bool) -> Value {
        json!({
            "profile": self.profile,
            "passed": self.passed(strict),
            "errors": self.errors(),
            "warnings": self.warnings(),
            "issues": self.issues.iter().map(|i| i.to_json()).collect::<Vec<_>>(),
        })
    }
}

/// Check an analysis (made with `is_loop` matching the profile) against a profile.
pub fn lint(a: &Analysis, p: &Profile) -> LintResult {
    let mut issues: Vec<Issue> = a.issues.clone();
    let mut add = |severity, code, message: String, fix: String| issues.push(Issue { severity, code, message, fix });
    if a.silent {
        return LintResult { profile: p.name, issues };
    }
    if a.duration_s < p.duration.0 {
        add("error", "too_short", format!("{:.2} s is shorter than the {:.2} s minimum for '{}'", a.duration_s, p.duration.0, p.name), "increase `duration` and lengthen envelopes to match".into());
    }
    if a.duration_s > p.duration.1 {
        add("error", "too_long", format!("{:.2} s is longer than the {:.1} s maximum for '{}'", a.duration_s, p.duration.1, p.name), "shorten `duration`, envelopes and reverb decay".into());
    }
    if a.true_peak_db > p.max_true_peak_db + 0.05 && a.clipped_samples == 0 && a.true_peak_db <= -0.1 {
        add("warning", "true_peak_high", format!("true peak {:+.2} dBFS is above the {:.1} dBFS ceiling", a.true_peak_db, p.max_true_peak_db), "normalize to -1 dBFS peak or end the chain with `{type: limiter, ceiling_db: -1}`".into());
    }
    if let Some((lo, hi)) = p.lufs {
        if a.lufs < lo {
            add("warning", "too_quiet_for_profile", format!("{:.1} LUFS is below the {lo}..{hi} LUFS range for '{}'", a.lufs, p.name), format!("raise the level: `normalize: {{lufs: {}}}`, or for scores set `master: {{lufs: {}}}`", (lo + hi) / 2.0, (lo + hi) / 2.0));
        } else if a.lufs > hi {
            add("warning", "too_loud_for_profile", format!("{:.1} LUFS is above the {lo}..{hi} LUFS range for '{}'", a.lufs, p.name), format!("lower the level: `normalize: {{lufs: {}}}`, or for scores set `master: {{lufs: {}}}`", (lo + hi) / 2.0, (lo + hi) / 2.0));
        }
    }
    if let Some(max) = p.max_attack_ms {
        if a.attack_ms > max {
            add("warning", "slow_attack", format!("attack takes {:.0} ms; '{}' sounds need {max:.0} ms or less to feel immediate", a.attack_ms, p.name), "shorten the envelope attack to 1-5 ms and add a click or noise-burst layer at t=0; a `transient` node with attack_db: 6 also helps".into());
        }
    }
    if let Some(min) = p.min_attack_ms {
        if a.attack_ms < min {
            add("warning", "abrupt_attack", format!("attack is only {:.0} ms; '{}' sounds should swell in over at least {min:.0} ms", a.attack_ms, p.name), "lengthen the first envelope segment so the level rises gradually".into());
        }
    }
    if a.silence_start_s * 1000.0 > p.max_silence_start_ms && a.silence_start_s <= 0.02 {
        add("warning", "leading_silence", format!("{:.0} ms of silence before the sound starts; '{}' sounds must respond instantly", a.silence_start_s * 1000.0, p.name), "remove delays at the start: envelope points should begin at t=0".into());
    }
    if let Some(max) = p.max_low_share {
        let low = a.band_shares[0] + a.band_shares[1];
        if low > max {
            add("warning", "too_much_bass", format!("{:.0}% of the energy is below 250 Hz; '{}' sounds should stay under {:.0}% so they do not mask other audio", low * 100.0, p.name, max * 100.0), "add a highpass around 200-300 Hz before the output, or raise the pitch".into());
        }
    }
    if let Some(min) = p.min_centroid_hz {
        if a.centroid_hz < min {
            add("warning", "too_dark", format!("spectral centroid {:.0} Hz is below {min:.0} Hz; the sound may be inaudible in a busy mix", a.centroid_hz), "raise filter cutoffs, add a brighter layer (noise burst, higher partials) or a highshelf boost".into());
        }
    }
    if let Some(max) = p.max_centroid_hz {
        if a.centroid_hz > max {
            add("warning", "too_bright", format!("spectral centroid {:.0} Hz is above {max:.0} Hz", a.centroid_hz), "lower filter cutoffs or add a lowpass before the output".into());
        }
    }
    if let Some(min) = p.min_crest_db {
        if a.crest_db < min {
            add("warning", "weak_transient", format!("crest factor {:.1} dB is below {min:.0} dB: the hit lacks punch relative to its tail", a.crest_db), "make the attack layer louder and shorter, reduce sustain or reverb mix, or add `{type: transient, attack_db: 6}`".into());
        }
    }
    if let Some(max) = p.max_crest_db {
        if a.crest_db > max {
            add("warning", "uneven_level", format!("crest factor {:.1} dB is above {max:.0} dB: isolated peaks stick out of what should be a steady bed", a.crest_db), "reduce the level of sporadic events or add a compressor before the output".into());
        }
    }
    if p.want_stereo && a.stereo.as_ref().map_or(true, |s| s.width < 0.01) {
        add("info", "mono_content", format!("the sound is mono; '{}' assets usually benefit from stereo width", p.name), "use `stereo: true` on noise, unison on oscillators, or add chorus/reverb".into());
    }
    if p.is_loop && a.loop_seam.is_none() {
        add("warning", "loop_not_checked", "the profile expects a loop but the seam was not analyzed".into(), "run lint with --loop".into());
    }
    // one entry per code: profile-specific findings replace generic ones
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::new();
    for i in issues.into_iter().rev() {
        if seen.insert(i.code) {
            unique.push(i);
        }
    }
    unique.reverse();
    unique.sort_by_key(|i| match i.severity {
        "error" => 0,
        "warning" => 1,
        _ => 2,
    });
    LintResult { profile: p.name, issues: unique }
}
