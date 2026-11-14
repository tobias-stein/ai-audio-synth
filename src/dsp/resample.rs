//! Integer-ratio resampling with Kaiser-windowed sinc filters, used for the
//! engine's oversampling and for true-peak measurement.

use crate::audio::Sig;
use crate::util::bessel_i0;
use std::f64::consts::PI;
use std::sync::OnceLock;

/// Linear-phase lowpass. `cutoff` is in cycles per sample (0..0.5).
pub fn kaiser_lowpass(taps: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let m = (taps - 1) as f64;
    let i0b = bessel_i0(beta);
    let mut h: Vec<f64> = (0..taps)
        .map(|i| {
            let x = i as f64 - m / 2.0;
            let sinc = if x.abs() < 1e-12 { 2.0 * cutoff } else { (2.0 * PI * cutoff * x).sin() / (PI * x) };
            let r = 2.0 * i as f64 / m - 1.0;
            let w = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b;
            sinc * w
        })
        .collect();
    let sum: f64 = h.iter().sum();
    for v in h.iter_mut() {
        *v /= sum;
    }
    h
}

/// One side of a half-band lowpass: coefficients for offsets 1, 3, 5, ...
/// from the center tap (which is exactly 0.5; even offsets are zero).
fn halfband_side() -> &'static [f64] {
    static TAPS: OnceLock<Vec<f64>> = OnceLock::new();
    TAPS.get_or_init(|| {
        const SIDE: usize = 24;
        let full = kaiser_lowpass(4 * SIDE - 1, 0.25, 10.0);
        let c = 2 * SIDE - 1;
        let mut side: Vec<f64> = (0..SIDE).map(|i| full[c + 2 * i + 1]).collect();
        // the two sides together must carry the other half of the DC gain
        let sum: f64 = side.iter().sum::<f64>() * 2.0;
        for v in side.iter_mut() {
            *v *= 0.5 / sum;
        }
        side
    })
}

/// Zero-phase decimation by two (about 100 dB stopband from 0.28 of the input rate).
fn decimate2(x: &[f64]) -> Vec<f64> {
    let h = halfband_side();
    let side = h.len();
    let n_out = x.len().div_ceil(2);
    // odd-indexed samples, zero-padded so the filter never branches at the edges
    let mut odd = vec![0.0; n_out + 2 * side];
    for (j, v) in x.iter().skip(1).step_by(2).enumerate() {
        odd[side + j] = *v;
    }
    (0..n_out)
        .map(|k| {
            let fwd = &odd[side + k..side + k + side];
            let back = &odd[k..side + k];
            let mut acc = 0.0;
            for i in 0..side {
                acc += h[i] * (fwd[i] + back[side - 1 - i]);
            }
            0.5 * x[2 * k] + acc
        })
        .collect()
}

/// Zero-phase interpolation by two; input samples pass through unchanged.
fn interpolate2(x: &[f64]) -> Vec<f64> {
    let h = halfband_side();
    let side = h.len();
    let n = x.len();
    let mut padded = vec![0.0; n + 2 * side];
    padded[side..side + n].copy_from_slice(x);
    let mut out = Vec::with_capacity(2 * n);
    for k in 0..n {
        out.push(x[k]);
        let fwd = &padded[side + k + 1..side + k + 1 + side];
        let back = &padded[k + 1..side + k + 1];
        let mut acc = 0.0;
        for i in 0..side {
            acc += h[i] * (fwd[i] + back[side - 1 - i]);
        }
        out.push(2.0 * acc);
    }
    out
}

/// Zero-phase decimation by a power of two.
pub fn decimate(x: &[f64], m: usize) -> Vec<f64> {
    assert!(m.is_power_of_two(), "resampling factors must be powers of two");
    let mut v = x.to_vec();
    let mut f = m;
    while f > 1 {
        v = decimate2(&v);
        f /= 2;
    }
    v
}

/// Zero-phase interpolation by a power of two.
pub fn interpolate(x: &[f64], m: usize) -> Vec<f64> {
    assert!(m.is_power_of_two(), "resampling factors must be powers of two");
    let mut v = x.to_vec();
    let mut f = m;
    while f > 1 {
        v = interpolate2(&v);
        f /= 2;
    }
    v
}

pub fn decimate_sig(s: &Sig, m: usize) -> Sig {
    s.map_channels(|_, c| decimate(c, m))
}

/// Per-sample peak magnitude including inter-sample peaks (4x oversampled).
pub fn inter_sample_peaks(x: &[f64]) -> Vec<f64> {
    let up = interpolate(x, 4);
    (0..x.len())
        .map(|i| {
            let mut p = x[i].abs();
            for k in 0..4 {
                p = p.max(up[i * 4 + k].abs());
            }
            // peaks just before this sample belong to the previous interval but
            // counting them here too makes limiting safer
            if i > 0 {
                p = p.max(up[i * 4 - 1].abs()).max(up[i * 4 - 2].abs());
            }
            p
        })
        .collect()
}

/// True peak of a signal in linear units (ITU-R BS.1770 style, 4x oversampling).
pub fn true_peak(s: &Sig) -> f64 {
    let mut p = 0.0f64;
    for c in s.chans() {
        if c.is_empty() {
            continue;
        }
        for v in interpolate(c, 4) {
            p = p.max(v.abs());
        }
        for v in c {
            p = p.max(v.abs());
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn decimate_keeps_passband_and_kills_stopband() {
        let sr = 96000.0;
        let n = 48000;
        let tone = |f: f64| -> Vec<f64> { (0..n).map(|i| (TAU * f * i as f64 / sr).sin()).collect() };
        let rms = |v: &[f64]| (v.iter().map(|s| s * s).sum::<f64>() / v.len() as f64).sqrt();
        let pass = decimate(&tone(1000.0), 2);
        assert_eq!(pass.len(), n / 2);
        assert!((rms(&pass[1000..n / 2 - 1000]) - 0.7071).abs() < 0.001);
        let stop = decimate(&tone(30000.0), 2);
        assert!(rms(&stop[1000..n / 2 - 1000]) < 1e-4);
    }

    #[test]
    fn interpolate_then_decimate_is_identity() {
        let n = 4000;
        let x: Vec<f64> = (0..n).map(|i| (TAU * 0.01 * i as f64).sin() * 0.5).collect();
        let y = decimate(&interpolate(&x, 4), 4);
        for i in 200..n - 200 {
            assert!((x[i] - y[i]).abs() < 1e-3, "sample {i}: {} vs {}", x[i], y[i]);
        }
    }

    #[test]
    fn true_peak_exceeds_sample_peak() {
        // a sine at fs/4 with 45 degree phase has samples at 0.707 of its real peak
        let x: Vec<f64> = (0..4000).map(|i| (TAU * 0.25 * i as f64 + PI / 4.0).sin()).collect();
        let s = Sig::mono(x);
        assert!(s.peak() < 0.72);
        let tp = true_peak(&s);
        assert!(tp > 0.98 && tp < 1.02, "true peak {tp}");
    }
}
