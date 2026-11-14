//! Physical models: a plucked string (extended Karplus-Strong) and a modal
//! resonator bank for struck metal, wood, glass and membranes.

use super::filter::one_pole_coef;
use crate::util::Rng;
use std::f64::consts::TAU;

pub struct PluckArgs {
    pub freq: f64,
    /// T60 of the fundamental in seconds.
    pub decay: f64,
    /// 0 = dull, 1 = bright.
    pub brightness: f64,
    /// Pluck position along the string (0..0.5); shapes the harmonic content.
    pub position: f64,
}

pub fn pluck(a: &PluckArgs, sr: f64, n: usize, seed: u64) -> Vec<f64> {
    let freq = a.freq.clamp(20.0, sr * 0.2);
    let period = sr / freq;
    let b = 0.5 * (1.0 - a.brightness.clamp(0.0, 1.0)); // loop filter blend
    let loop_delay = period - b; // the averaging filter delays by `b` samples
    let di = loop_delay.floor() as usize;
    let fr = loop_delay - di as f64;
    let len = di + 2;
    let mut buf = vec![0.0f64; len];
    // excitation: one period of noise, softened and comb-filtered by pick position
    let mut rng = Rng::new(seed);
    let p = period.ceil() as usize;
    let mut exc: Vec<f64> = (0..p).map(|_| rng.bipolar()).collect();
    let lp = one_pole_coef(800.0 + 15000.0 * a.brightness.clamp(0.0, 1.0), sr);
    let mut s = 0.0;
    for v in exc.iter_mut() {
        s += lp * (*v - s);
        *v = s;
    }
    let mean = exc.iter().sum::<f64>() / p as f64;
    for v in exc.iter_mut() {
        *v -= mean;
    }
    let pos = (a.position.clamp(0.02, 0.5) * period) as usize;
    let shaped: Vec<f64> = (0..p + pos).map(|i| (if i < p { exc[i] } else { 0.0 }) - if i >= pos { exc[i - pos] } else { 0.0 }).collect();
    let peak = shaped.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-9);

    let g = 10f64.powf(-3.0 / (freq * a.decay.max(0.01)));
    let mut w = 0usize;
    let mut prev = 0.0;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // read `loop_delay` samples back with linear interpolation
        let r0 = (w + len - di) % len;
        let r1 = (w + len - di - 1) % len;
        let d = buf[r0] * (1.0 - fr) + buf[r1] * fr;
        let filtered = (1.0 - b) * d + b * prev;
        prev = d;
        let x = if i < shaped.len() { shaped[i] / peak } else { 0.0 };
        let y = x + filtered * g;
        buf[w] = y;
        w = (w + 1) % len;
        out.push(y);
    }
    out
}

/// Named mode tables: (frequency ratio, relative gain).
pub fn modal_preset(name: &str) -> Option<Vec<(f64, f64)>> {
    let ratios: &[f64] = match name {
        "bell" => &[0.5, 1.0, 1.183, 1.506, 2.0, 2.514, 2.662, 3.011, 4.166, 5.433, 6.796, 8.215],
        "bar" => &[1.0, 2.756, 5.404, 8.933, 13.344, 18.64],
        "marimba" => &[1.0, 3.984, 9.2, 16.2],
        "plate" => &[1.0, 2.5, 4.0, 5.0, 6.5, 8.5, 9.0, 10.0, 12.5, 13.0, 14.5, 16.0, 17.0, 18.5, 20.0],
        "membrane" => &[1.0, 1.593, 2.136, 2.296, 2.653, 2.918, 3.156, 3.501, 3.6, 3.652, 4.06, 4.154],
        "glass" => &[1.0, 2.32, 4.25, 6.63, 9.38, 12.22],
        "wood" => &[1.0, 2.572, 4.644, 6.984, 9.723, 12.0],
        "bowl" => &[1.0, 2.71, 5.12, 8.0, 11.3, 14.9],
        "tube" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "metal" => &[1.0, 1.47, 2.09, 2.56, 3.39, 4.08, 4.65, 5.27, 6.11, 7.24, 8.36, 9.87, 11.4, 13.1],
        _ => return None,
    };
    Some(ratios.iter().enumerate().map(|(i, &r)| (r, 1.0 / (1.0 + 0.35 * i as f64))).collect())
}

pub const MODAL_PRESETS: [&str; 10] = ["bell", "bar", "marimba", "plate", "membrane", "glass", "wood", "bowl", "tube", "metal"];

pub struct ModalArgs {
    pub freq: f64,
    /// (ratio, gain, decay multiplier)
    pub modes: Vec<(f64, f64, f64)>,
    /// T60 of the lowest mode in seconds.
    pub decay: f64,
    /// How much faster high modes die away (0 = all equal).
    pub damping: f64,
    /// Internal rate divided by output rate. Excitation impulses are scaled by
    /// this factor so that level does not depend on the oversampling setting.
    pub rate_ratio: f64,
}

/// Excite a bank of two-pole resonators with `exc`.
pub fn modal(exc: &[f64], a: &ModalArgs, sr: f64, nyquist: f64) -> Vec<f64> {
    let n = exc.len();
    let mut out = vec![0.0; n];
    let lowest = a.modes.iter().map(|m| m.0).fold(f64::MAX, f64::min).max(1e-6);
    for &(ratio, gain, dmul) in &a.modes {
        let f = a.freq * ratio;
        if f >= nyquist || f < 10.0 || gain == 0.0 {
            continue;
        }
        let t60 = (a.decay * dmul / (ratio / lowest).powf(a.damping)).max(0.005);
        let r = (-6.907755 / (t60 * sr)).exp();
        let w = TAU * f / sr;
        let (a1, a2) = (2.0 * r * w.cos(), -r * r);
        let g = gain * w.sin() / a.rate_ratio.max(1.0);
        let (mut y1, mut y2) = (0.0, 0.0);
        for i in 0..n {
            let y = a1 * y1 + a2 * y2 + g * exc[i];
            y2 = y1;
            y1 = y;
            out[i] += y;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dominant_freq(x: &[f64], sr: f64) -> f64 {
        crate::analysis::spectral::detect_pitch(x, sr).map_or(0.0, |p| p.hz)
    }

    #[test]
    fn pluck_is_in_tune_and_decays() {
        let sr = 96000.0;
        let x = pluck(&PluckArgs { freq: 220.0, decay: 1.0, brightness: 0.2, position: 0.25 }, sr, 96000, 1);
        let f = dominant_freq(&x[20000..60000], sr);
        assert!((f - 220.0).abs() < 4.0, "pitch {f}");
        let e = |s: &[f64]| s.iter().map(|v| v * v).sum::<f64>();
        assert!(e(&x[80000..]) < 0.05 * e(&x[..16000]));
    }

    #[test]
    fn modal_rings_at_the_requested_pitch() {
        let sr = 96000.0;
        let mut exc = vec![0.0; 96000];
        exc[0] = 1.0;
        let y = modal(&exc, &ModalArgs { freq: 440.0, modes: vec![(1.0, 1.0, 1.0)], decay: 2.0, damping: 0.0, rate_ratio: 1.0 }, sr, 24000.0);
        let f = dominant_freq(&y[..48000], sr);
        assert!((f - 440.0).abs() < 3.0, "pitch {f}");
        let peak = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(peak > 0.8 && peak < 1.2, "peak {peak}");
    }
}
