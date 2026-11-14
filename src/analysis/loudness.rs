//! Loudness measurement following ITU-R BS.1770-4 / EBU R 128.

use crate::audio::Sig;
use crate::dsp::filter::Biquad;
use std::f64::consts::PI;

fn k_weight(x: &[f64], sr: f64) -> Vec<f64> {
    // Stage 1: high shelf (+4 dB above ~1.7 kHz); stage 2: highpass at 38 Hz.
    // Coefficients are derived for the actual sample rate.
    let mut shelf = {
        let (f0, g, q) = (1681.974450955533, 3.999843853973347, 0.7071752369554196);
        let k = (PI * f0 / sr).tan();
        let vh = 10f64.powf(g / 20.0);
        let vb = vh.powf(0.4996667741545416);
        let a0 = 1.0 + k / q + k * k;
        Biquad::from_coeffs((vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0, 1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0)
    };
    let mut hp = {
        let (f0, q) = (38.13547087602444, 0.5003270373238773);
        let k = (PI * f0 / sr).tan();
        let a0 = 1.0 + k / q + k * k;
        Biquad::from_coeffs(1.0, -2.0, 1.0, 1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0)
    };
    x.iter().map(|&v| hp.tick(shelf.tick(v))).collect()
}

#[derive(Clone, Debug)]
pub struct Loudness {
    /// Integrated loudness in LUFS (gated when the signal is long enough).
    pub integrated: f64,
    /// True when the file was too short for gating (< 0.4 s) and the value is
    /// the ungated loudness of the whole file.
    pub ungated: bool,
    /// Loudest 400 ms window in LUFS.
    pub momentary_max: f64,
    /// Loudness range in LU (0 for short files).
    pub range: f64,
}

fn to_lufs(ms: f64) -> f64 {
    if ms <= 1e-12 {
        -120.0
    } else {
        -0.691 + 10.0 * ms.log10()
    }
}

pub fn measure(sig: &Sig, sr: u32) -> Loudness {
    let srf = sr as f64;
    let weighted: Vec<Vec<f64>> = sig.chans().iter().map(|c| k_weight(c, srf)).collect();
    let n = sig.len();
    let block = (0.4 * srf) as usize;
    let hop = (0.1 * srf) as usize;
    let whole = |from: usize, to: usize| -> f64 {
        if to <= from {
            return 0.0;
        }
        weighted.iter().map(|c| c[from..to].iter().map(|v| v * v).sum::<f64>() / (to - from) as f64).sum()
    };
    if n < block || hop == 0 {
        let l = to_lufs(whole(0, n));
        return Loudness { integrated: l, ungated: true, momentary_max: l, range: 0.0 };
    }
    // prefix sums of squared samples make block energies cheap
    let prefix: Vec<Vec<f64>> = weighted
        .iter()
        .map(|c| {
            let mut p = Vec::with_capacity(n + 1);
            let mut acc = 0.0;
            p.push(0.0);
            for v in c {
                acc += v * v;
                p.push(acc);
            }
            p
        })
        .collect();
    let mut blocks = Vec::new();
    let mut start = 0;
    while start + block <= n {
        let ms: f64 = prefix.iter().map(|p| (p[start + block] - p[start]) / block as f64).sum();
        blocks.push(ms);
        start += hop;
    }
    let momentary_max = blocks.iter().cloned().fold(0.0, f64::max);
    let abs_gated: Vec<f64> = blocks.iter().cloned().filter(|&ms| to_lufs(ms) > -70.0).collect();
    if abs_gated.is_empty() {
        return Loudness { integrated: -120.0, ungated: false, momentary_max: to_lufs(momentary_max), range: 0.0 };
    }
    let mean = abs_gated.iter().sum::<f64>() / abs_gated.len() as f64;
    let rel = to_lufs(mean) - 10.0;
    let gated: Vec<f64> = abs_gated.iter().cloned().filter(|&ms| to_lufs(ms) > rel).collect();
    let integrated = if gated.is_empty() { to_lufs(mean) } else { to_lufs(gated.iter().sum::<f64>() / gated.len() as f64) };
    // loudness range: spread of block loudness between the 10th and 95th percentiles
    let rel_lra = to_lufs(mean) - 20.0;
    let mut lv: Vec<f64> = abs_gated.iter().map(|&ms| to_lufs(ms)).filter(|&l| l > rel_lra).collect();
    lv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let range = if lv.len() >= 4 {
        let q = |p: f64| lv[((lv.len() - 1) as f64 * p).round() as usize];
        q(0.95) - q(0.10)
    } else {
        0.0
    };
    Loudness { integrated, ungated: false, momentary_max: to_lufs(momentary_max), range }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn reference_sine_reads_minus_3_lufs() {
        // BS.1770: a 0 dBFS 997 Hz sine in one channel reads -3.01 LUFS.
        let sr = 48000;
        let x: Vec<f64> = (0..sr * 5).map(|i| (TAU * 997.0 * i as f64 / sr as f64).sin()).collect();
        let l = measure(&Sig::mono(x), sr);
        assert!((l.integrated + 3.01).abs() < 0.1, "got {}", l.integrated);
        assert!(!l.ungated);
    }

    #[test]
    fn stereo_sine_is_3_db_louder_and_rate_independent() {
        for sr in [44100u32, 48000, 96000] {
            let x: Vec<f64> = (0..sr * 3).map(|i| (TAU * 997.0 * i as f64 / sr as f64).sin() * 0.1).collect();
            let l = measure(&Sig::stereo(x.clone(), x), sr);
            assert!((l.integrated + 20.0).abs() < 0.15, "sr {sr}: got {}", l.integrated);
        }
    }

    #[test]
    fn short_and_silent_inputs() {
        let l = measure(&Sig::mono(vec![0.5; 1000]), 48000);
        assert!(l.ungated);
        let s = measure(&Sig::mono(vec![0.0; 96000]), 48000);
        assert!(s.integrated < -100.0);
    }
}
