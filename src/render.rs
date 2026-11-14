//! Turning a patch into a finished asset: looping, normalization, safety
//! fades and warnings about problems the caller cannot hear.

use crate::analysis::loudness;
use crate::audio::Sig;
use crate::dsp::resample::true_peak;
use crate::error::{Error, Result};
use crate::graph::{render_graph, render_graph_probe, NodeLevel, Normalize, Patch, RenderOpts};
use crate::util::{db_to_lin, lin_to_db, round_to};

pub struct Asset {
    pub sig: Sig,
    /// Sample range to mark as a loop in the output file.
    pub loop_points: Option<(u32, u32)>,
    pub warnings: Vec<String>,
    /// Per-node levels when probing was requested.
    pub levels: Vec<NodeLevel>,
}

#[derive(Clone, Debug)]
pub struct AssetOpts {
    pub sr: u32,
    pub oversample: usize,
    pub seed: u64,
    pub params: Vec<(String, f64)>,
    pub duration: Option<f64>,
    /// Force looping on (None = follow the patch).
    pub force_loop: bool,
    pub normalize: Option<Normalize>,
    /// Record the level of every node (for `--probe`).
    pub probe: bool,
}

pub fn quality_to_oversample(q: &str) -> Result<usize> {
    match q {
        "draft" => Ok(1),
        "standard" => Ok(2),
        "high" => Ok(4),
        _ => Err(Error::usage(format!("unknown quality '{q}'")).hint("use draft (fast, 1x), standard (2x oversampling) or high (4x)")),
    }
}

/// Fold the extra `xfade` samples at the end of `raw` into its start so that
/// the result repeats without a seam. The crossfade law adapts to how similar
/// the two overlapping parts are: linear for identical (periodic) material,
/// equal-power for unrelated (noise-like) material, so level stays constant.
pub fn make_seamless(raw: &Sig, loop_len: usize, xfade: usize) -> Sig {
    let xfade = xfade.min(loop_len).min(raw.len().saturating_sub(loop_len));
    let (mut hh, mut tt, mut ht) = (0.0, 0.0, 0.0);
    for c in raw.chans() {
        for i in 0..xfade {
            let (h, t) = (c[i], c[loop_len + i]);
            hh += h * h;
            tt += t * t;
            ht += h * t;
        }
    }
    let rho = if hh > 0.0 && tt > 0.0 { (ht / (hh * tt).sqrt()).clamp(0.0, 1.0) } else { 0.0 };
    raw.map_channels(|_, x| {
        let mut out = x[..loop_len].to_vec();
        for i in 0..xfade {
            let p = (i as f64 + 0.5) / xfade as f64;
            let g = 1.0 / (p * p + (1.0 - p) * (1.0 - p) + 2.0 * rho * p * (1.0 - p)).sqrt();
            out[i] = x[i] * p * g + x[loop_len + i] * (1.0 - p) * g;
        }
        out
    })
}

pub fn apply_normalize(sig: &mut Sig, sr: u32, mode: &Normalize, warnings: &mut Vec<String>) {
    let tp = true_peak(sig);
    if tp < 1e-9 {
        warnings.push("the output is silent: check that envelopes are above zero and that `output` points at the right node".into());
        return;
    }
    match mode {
        Normalize::None => {
            if tp > 1.0 {
                warnings.push(format!("output clips: true peak is {:+.1} dBFS with normalization off; lower the level or enable normalize", lin_to_db(tp)));
            }
        }
        Normalize::Peak(db) => sig.scale(db_to_lin(*db) / tp),
        Normalize::Lufs(target) => {
            let l = loudness::measure(sig, sr).integrated;
            let mut g = db_to_lin(target - l);
            let ceiling = db_to_lin(-1.0);
            if tp * g > ceiling {
                let reached = l + lin_to_db(ceiling / tp);
                warnings.push(format!("cannot reach {target} LUFS without clipping; normalized to {} LUFS at -1 dBTP instead (the sound is too peaky: add a compressor or limiter)", round_to(reached, 1)));
                g = ceiling / tp;
            }
            sig.scale(g);
        }
    }
}

pub fn render_patch(patch: &Patch, o: &AssetOpts) -> Result<Asset> {
    let mut warnings = Vec::new();
    let dur = o.duration.or(patch.duration_with(&o.params)?).ok_or_else(|| Error::invalid("patch has no 'duration'").at("duration").hint("add e.g. `duration: 1.5` (seconds) or pass --duration"))?;
    let looped = o.force_loop || patch.loop_crossfade.is_some();
    let xfade = if looped {
        let x = patch.loop_crossfade.filter(|x| *x >= 0.0).unwrap_or((dur * 0.25).min(1.0));
        // a duration override keeps the patch's crossfade in proportion
        let x = if o.duration.is_some() { x.min(dur * 0.4) } else { x };
        if x > dur * 0.5 {
            return Err(Error::invalid(format!("loop crossfade {x}s is more than half the duration {dur}s")).at("loop.crossfade").hint("shorten the crossfade or lengthen the duration"));
        }
        x
    } else {
        0.0
    };
    let mut ro = RenderOpts::new(o.sr, o.oversample, o.seed);
    ro.params = o.params.clone();
    ro.duration = Some(dur + xfade);
    ro.gate = patch.gate;
    let mut levels = Vec::new();
    let raw = render_graph_probe(patch, &ro, if o.probe { Some(&mut levels) } else { None })?;
    let n = (dur * o.sr as f64).round() as usize;
    let mut sig = if looped {
        make_seamless(&raw, n.min(raw.len()), (xfade * o.sr as f64).round() as usize)
    } else {
        let mut s = raw;
        s.resize(n);
        s
    };
    let mode = o.normalize.clone().unwrap_or(patch.normalize.clone());
    apply_normalize(&mut sig, o.sr, &mode, &mut warnings);
    if !looped {
        // if the sound is still audible at the very end it was cut off
        let w = ((0.003 * o.sr as f64) as usize).min(sig.len());
        let tail_peak = sig.chans().iter().flat_map(|c| c[c.len() - w..].iter()).fold(0.0f64, |m, v| m.max(v.abs()));
        if tail_peak > db_to_lin(-50.0) {
            warnings.push(format!(
                "the sound is still at {:.0} dBFS when the file ends, so it is cut off (a short fade was applied): increase `duration`, or shorten envelopes and reverb decay",
                lin_to_db(tail_peak)
            ));
            let f = ((0.008 * o.sr as f64) as usize).min(sig.len());
            sig.fade(0, f);
        } else if tail_peak > 0.0 {
            sig.fade(0, w);
        }
    }
    let loop_points = if looped { Some((0, sig.len().saturating_sub(1) as u32)) } else { None };
    Ok(Asset { sig, loop_points, warnings, levels })
}

/// Render one note of an instrument patch for the sequencer: gate plus tail,
/// trailing silence trimmed, no normalization.
pub fn render_note(patch: &Patch, freq: Option<f64>, vel: f64, gate: f64, seed: u64, sr: u32, oversample: usize, params: &[(String, f64)]) -> Result<Sig> {
    let mut ro = RenderOpts::new(sr, oversample, seed);
    ro.freq = freq;
    ro.vel = vel;
    ro.gate = Some(gate);
    ro.params = params.to_vec();
    let mut sig = render_graph(patch, &ro)?;
    let keep = sig.last_audible(-84.0).map(|i| i + 1).unwrap_or(0);
    let min_len = ((0.005 * sr as f64) as usize).min(sig.len());
    sig.resize(keep.max(min_len));
    let f = ((0.004 * sr as f64) as usize).min(sig.len());
    sig.fade(0, f);
    // velocity scales loudness on a perceptual curve; patches may also read `vel` for tone
    sig.scale(vel.clamp(0.0, 1.5).powf(1.5));
    Ok(sig)
}
