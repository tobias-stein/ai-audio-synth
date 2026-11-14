//! Integer-ratio resampling with Kaiser-windowed sinc filters, used for the
//! engine's oversampling and for true-peak measurement.

use crate::audio::Sig;
use crate::util::bessel_i0;
use std::f64::consts::PI;

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

/// Zero-phase decimation by an integer factor (about 100 dB stopband).
pub fn decimate(x: &[f64], m: usize) -> Vec<f64> {
    if m <= 1 {
        return x.to_vec();
    }
    let taps = 64 * m - 1;
    let h = kaiser_lowpass(taps, 0.5 / m as f64, 10.0);
    let c = (taps / 2) as isize;
    let n_out = x.len().div_ceil(m);
    let n = x.len() as isize;
    (0..n_out)
        .map(|k| {
            let center = (k * m) as isize;
            let lo = (center - c).max(0);
            let hi = (center + c).min(n - 1);
            let mut acc = 0.0;
            for j in lo..=hi {
                acc += x[j as usize] * h[(j - center + c) as usize];
            }
            acc
        })
        .collect()
}

/// Zero-phase interpolation by an integer factor.
pub fn interpolate(x: &[f64], m: usize) -> Vec<f64> {
    if m <= 1 {
        return x.to_vec();
    }
    let taps = 32 * m - 1;
    let h = kaiser_lowpass(taps, 0.5 / m as f64, 9.0);
    let c = (taps / 2) as isize;
    let n = x.len() as isize;
    let mf = m as f64;
    (0..x.len() * m)
        .map(|i| {
            let i = i as isize;
            // input samples sit at multiples of m in the upsampled grid
            let first = (i - c).max(0).div_euclid(m as isize) + if (i - c).max(0).rem_euclid(m as isize) != 0 { 1 } else { 0 };
            let last = ((i + c) / m as isize).min(n - 1);
            let mut acc = 0.0;
            let mut k = first;
            while k <= last {
                acc += x[k as usize] * h[(k * m as isize - i + c) as usize];
                k += 1;
            }
            acc * mf
        })
        .collect()
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
