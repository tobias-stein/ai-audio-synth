//! Filters: a modulatable state-variable filter (Simper/Cytomic topology),
//! static biquads, one-pole smoothers, comb and formant filters.

use super::P;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FilterMode {
    Lowpass,
    Highpass,
    Bandpass,
    Notch,
    Peak,
    LowShelf,
    HighShelf,
    Allpass,
}

/// Trapezoidal state-variable filter. Stable under fast cutoff modulation.
#[derive(Clone, Debug)]
pub struct Svf {
    ic1: f64,
    ic2: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    m0: f64,
    m1: f64,
    m2: f64,
}

impl Svf {
    pub fn new() -> Self {
        Svf { ic1: 0.0, ic2: 0.0, a1: 0.0, a2: 0.0, a3: 0.0, m0: 0.0, m1: 0.0, m2: 0.0 }
    }

    pub fn set(&mut self, mode: FilterMode, fc: f64, q: f64, gain_db: f64, sr: f64) {
        let fc = fc.clamp(5.0, sr * 0.49);
        let q = q.max(0.05);
        let a = 10f64.powf(gain_db / 40.0);
        let mut g = (PI * fc / sr).tan();
        let mut k = 1.0 / q;
        match mode {
            FilterMode::Lowpass => {
                self.m0 = 0.0;
                self.m1 = 0.0;
                self.m2 = 1.0;
            }
            FilterMode::Highpass => {
                self.m0 = 1.0;
                self.m1 = -k;
                self.m2 = -1.0;
            }
            FilterMode::Bandpass => {
                // constant 0 dB peak gain
                self.m0 = 0.0;
                self.m1 = k;
                self.m2 = 0.0;
            }
            FilterMode::Notch => {
                self.m0 = 1.0;
                self.m1 = -k;
                self.m2 = 0.0;
            }
            FilterMode::Allpass => {
                self.m0 = 1.0;
                self.m1 = -2.0 * k;
                self.m2 = 0.0;
            }
            FilterMode::Peak => {
                k = 1.0 / (q * a);
                self.m0 = 1.0;
                self.m1 = k * (a * a - 1.0);
                self.m2 = 0.0;
            }
            FilterMode::LowShelf => {
                g /= a.sqrt();
                self.m0 = 1.0;
                self.m1 = k * (a - 1.0);
                self.m2 = a * a - 1.0;
            }
            FilterMode::HighShelf => {
                g *= a.sqrt();
                self.m0 = a * a;
                self.m1 = k * (1.0 - a) * a;
                self.m2 = 1.0 - a * a;
            }
        }
        self.a1 = 1.0 / (1.0 + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
    }

    #[inline(always)]
    pub fn tick(&mut self, v0: f64) -> f64 {
        let v3 = v0 - self.ic2;
        let v1 = self.a1 * self.ic1 + self.a2 * v3;
        let v2 = self.ic2 + self.a2 * self.ic1 + self.a3 * v3;
        self.ic1 = 2.0 * v1 - self.ic1;
        self.ic2 = 2.0 * v2 - self.ic2;
        self.m0 * v0 + self.m1 * v1 + self.m2 * v2
    }
}

impl Default for Svf {
    fn default() -> Self {
        Self::new()
    }
}

/// Run an SVF over a buffer with per-sample modulatable parameters.
/// `stages` cascades identical sections (2 gives a 24 dB/octave slope).
pub fn svf_process(x: &[f64], mode: FilterMode, cutoff: &P, q: &P, gain_db: &P, stages: usize, sr: f64) -> Vec<f64> {
    let mut out = x.to_vec();
    let all_const = cutoff.is_const() && q.is_const() && gain_db.is_const();
    for _ in 0..stages.max(1) {
        let mut f = Svf::new();
        if all_const {
            f.set(mode, cutoff.at(0), q.at(0), gain_db.at(0), sr);
            for s in out.iter_mut() {
                *s = f.tick(*s);
            }
        } else {
            let (mut lc, mut lq, mut lg) = (f64::NAN, f64::NAN, f64::NAN);
            for (i, s) in out.iter_mut().enumerate() {
                let (c, qq, g) = (cutoff.at(i), q.at(i), gain_db.at(i));
                if c != lc || qq != lq || g != lg {
                    f.set(mode, c, qq, g, sr);
                    lc = c;
                    lq = qq;
                    lg = g;
                }
                *s = f.tick(*s);
            }
        }
    }
    out
}

/// Direct-form-II-transposed biquad for fixed filters.
#[derive(Clone, Debug)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    pub fn from_coeffs(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Biquad { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0, z1: 0.0, z2: 0.0 }
    }
    pub fn highpass(fc: f64, q: f64, sr: f64) -> Self {
        let w = 2.0 * PI * fc / sr;
        let (s, c) = w.sin_cos();
        let alpha = s / (2.0 * q);
        Self::from_coeffs((1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }
    pub fn lowpass(fc: f64, q: f64, sr: f64) -> Self {
        let w = 2.0 * PI * fc / sr;
        let (s, c) = w.sin_cos();
        let alpha = s / (2.0 * q);
        Self::from_coeffs((1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }
    pub fn bandpass(fc: f64, q: f64, sr: f64) -> Self {
        let w = 2.0 * PI * fc / sr;
        let (s, c) = w.sin_cos();
        let alpha = s / (2.0 * q);
        Self::from_coeffs(alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }
    pub fn high_shelf(fc: f64, gain_db: f64, q: f64, sr: f64) -> Self {
        let a = 10f64.powf(gain_db / 40.0);
        let w = 2.0 * PI * fc / sr;
        let (s, c) = w.sin_cos();
        let alpha = s / (2.0 * q);
        let sa = 2.0 * a.sqrt() * alpha;
        Self::from_coeffs(
            a * ((a + 1.0) + (a - 1.0) * c + sa),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
            a * ((a + 1.0) + (a - 1.0) * c - sa),
            (a + 1.0) - (a - 1.0) * c + sa,
            2.0 * ((a - 1.0) - (a + 1.0) * c),
            (a + 1.0) - (a - 1.0) * c - sa,
        )
    }
    #[inline(always)]
    pub fn tick(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
    pub fn process(&mut self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|v| self.tick(*v)).collect()
    }
}

/// One-pole lowpass coefficient for a given cutoff.
pub fn one_pole_coef(fc: f64, sr: f64) -> f64 {
    1.0 - (-2.0 * PI * fc.max(0.0) / sr).exp()
}

/// Removes DC with a 10 Hz one-pole highpass.
pub fn dc_block(x: &[f64], sr: f64) -> Vec<f64> {
    let r = 1.0 - 2.0 * PI * 10.0 / sr;
    let (mut x1, mut y1) = (0.0, 0.0);
    x.iter()
        .map(|&v| {
            let y = v - x1 + r * y1;
            x1 = v;
            y1 = y;
            y
        })
        .collect()
}

/// Feedback comb filter tuned to `freq`, with a damping lowpass in the loop.
pub fn comb(x: &[f64], freq: &P, feedback: &P, damp_hz: f64, sr: f64) -> Vec<f64> {
    let max_len = (sr / 10.0) as usize + 4;
    let mut buf = vec![0.0; max_len];
    let mut w = 0usize;
    let lp = one_pole_coef(damp_hz, sr);
    let mut state = 0.0;
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let d = (sr / freq.at(i).clamp(10.5, sr * 0.45)).clamp(1.0, (max_len - 3) as f64);
            let rp = w as f64 - d;
            let rp = if rp < 0.0 { rp + max_len as f64 } else { rp };
            let i0 = rp.floor() as usize;
            let fr = rp - i0 as f64;
            let a = buf[i0 % max_len];
            let b = buf[(i0 + 1) % max_len];
            let delayed = a + (b - a) * fr;
            state += lp * (delayed - state);
            let y = v + state * feedback.at(i).clamp(-0.999, 0.999);
            buf[w] = y;
            w = (w + 1) % max_len;
            y
        })
        .collect()
}

/// Vowel formant table (F1..F3 in Hz) for a, e, i, o, u.
const VOWELS: [[f64; 3]; 5] = [[730.0, 1090.0, 2440.0], [530.0, 1840.0, 2480.0], [270.0, 2290.0, 3010.0], [570.0, 840.0, 2410.0], [300.0, 870.0, 2240.0]];

/// Three parallel resonators morphing between vowels. `vowel` runs 0..4 (a e i o u).
pub fn formant(x: &[f64], vowel: &P, shift: f64, q: f64, sr: f64) -> Vec<f64> {
    let mut filters = [Svf::new(), Svf::new(), Svf::new()];
    let gains = [1.0, 0.6, 0.35];
    let mut last = f64::NAN;
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let p = vowel.at(i).clamp(0.0, 4.0);
            if i % 16 == 0 && p != last {
                last = p;
                let i0 = (p.floor() as usize).min(3);
                let fr = p - i0 as f64;
                for (k, f) in filters.iter_mut().enumerate() {
                    let fc = (VOWELS[i0][k] + (VOWELS[i0 + 1][k] - VOWELS[i0][k]) * fr) * shift;
                    f.set(FilterMode::Bandpass, fc, q, 0.0, sr);
                }
            }
            let mut y = 0.0;
            for (k, f) in filters.iter_mut().enumerate() {
                y += f.tick(v) * gains[k];
            }
            y
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn gain_at(mode: FilterMode, fc: f64, q: f64, gain_db: f64, f: f64) -> f64 {
        let sr = 96000.0;
        let n = 96000;
        let x: Vec<f64> = (0..n).map(|i| (TAU * f * i as f64 / sr).sin()).collect();
        let y = svf_process(&x, mode, &P::C(fc), &P::C(q), &P::C(gain_db), 1, sr);
        let rms = |v: &[f64]| (v.iter().map(|s| s * s).sum::<f64>() / v.len() as f64).sqrt();
        20.0 * (rms(&y[n / 2..]) / rms(&x[n / 2..])).log10()
    }

    #[test]
    fn lowpass_response() {
        assert!(gain_at(FilterMode::Lowpass, 1000.0, 0.707, 0.0, 100.0).abs() < 0.2);
        assert!((gain_at(FilterMode::Lowpass, 1000.0, 0.707, 0.0, 1000.0) + 3.0).abs() < 0.3);
        assert!(gain_at(FilterMode::Lowpass, 1000.0, 0.707, 0.0, 8000.0) < -30.0);
    }

    #[test]
    fn highpass_and_bandpass_response() {
        assert!(gain_at(FilterMode::Highpass, 1000.0, 0.707, 0.0, 100.0) < -35.0);
        assert!(gain_at(FilterMode::Highpass, 1000.0, 0.707, 0.0, 10000.0).abs() < 0.2);
        assert!(gain_at(FilterMode::Bandpass, 1000.0, 4.0, 0.0, 1000.0).abs() < 0.2);
        assert!(gain_at(FilterMode::Bandpass, 1000.0, 4.0, 0.0, 100.0) < -20.0);
    }

    #[test]
    fn shelves_and_peak() {
        assert!((gain_at(FilterMode::Peak, 1000.0, 1.0, 6.0, 1000.0) - 6.0).abs() < 0.2);
        assert!(gain_at(FilterMode::Peak, 1000.0, 1.0, 6.0, 50.0).abs() < 0.2);
        assert!((gain_at(FilterMode::LowShelf, 500.0, 0.707, 6.0, 30.0) - 6.0).abs() < 0.3);
        assert!(gain_at(FilterMode::LowShelf, 500.0, 0.707, 6.0, 10000.0).abs() < 0.2);
        assert!((gain_at(FilterMode::HighShelf, 2000.0, 0.707, -6.0, 15000.0) + 6.0).abs() < 0.3);
        assert!(gain_at(FilterMode::HighShelf, 2000.0, 0.707, -6.0, 100.0).abs() < 0.2);
    }
}
