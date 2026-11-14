//! Band-limited oscillators and noise sources.

use super::P;
use crate::audio::Sig;
use crate::util::Rng;
use std::f64::consts::{PI, TAU};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Wave {
    Sine,
    Saw,
    Square,
    Pulse,
    Triangle,
}

impl Wave {
    pub const NAMES: [&'static str; 5] = ["sine", "saw", "square", "pulse", "triangle"];
    pub fn parse(s: &str) -> Option<Wave> {
        Some(match s {
            "sine" | "sin" => Wave::Sine,
            "saw" | "sawtooth" => Wave::Saw,
            "square" => Wave::Square,
            "pulse" => Wave::Pulse,
            "triangle" | "tri" => Wave::Triangle,
            _ => return None,
        })
    }
}

/// Two-sample polynomial band-limited step residual.
#[inline(always)]
fn polyblep(t: f64, dt: f64) -> f64 {
    if t < dt {
        let x = t / dt;
        x + x - x * x - 1.0
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

/// Two-sample polynomial band-limited ramp residual (for slope discontinuities).
#[inline(always)]
fn polyblamp(t: f64, dt: f64) -> f64 {
    if t < dt {
        let x = 1.0 - t / dt;
        x * x * x / 3.0
    } else if t > 1.0 - dt {
        let x = 1.0 + (t - 1.0) / dt;
        x * x * x / 3.0
    } else {
        0.0
    }
}

#[inline(always)]
fn wave_sample(wave: Wave, t: f64, dt: f64, pw: f64) -> f64 {
    match wave {
        Wave::Sine => (TAU * t).sin(),
        Wave::Saw => 2.0 * t - 1.0 - polyblep(t, dt),
        Wave::Square | Wave::Pulse => {
            let pw = if wave == Wave::Square { 0.5 } else { pw.clamp(0.02, 0.98) };
            let mut v = if t < pw { 1.0 } else { -1.0 };
            v += polyblep(t, dt);
            let t2 = t + 1.0 - pw;
            v -= polyblep(t2 - t2.floor(), dt);
            // remove the DC offset a non-50% pulse carries
            v - (2.0 * pw - 1.0)
        }
        Wave::Triangle => {
            let mut v = 4.0 * (t - 0.5).abs() - 1.0;
            v -= 4.0 * dt * polyblamp(t, dt);
            let t2 = t + 0.5;
            v += 4.0 * dt * polyblamp(t2 - t2.floor(), dt);
            v
        }
    }
}

pub struct OscArgs<'a> {
    pub wave: Wave,
    pub freq: &'a P,
    pub pw: &'a P,
    /// Phase modulation input in radians.
    pub pm: Option<&'a P>,
    pub unison: usize,
    pub detune_cents: f64,
    pub spread: f64,
    /// Start phase in cycles; `None` picks a seeded random phase per voice.
    pub phase: Option<f64>,
    pub sr: f64,
    pub n: usize,
    pub seed: u64,
}

pub fn osc(a: &OscArgs) -> Sig {
    let unison = a.unison.max(1);
    let stereo = unison > 1 && a.spread > 0.0;
    let mut l = vec![0.0; a.n];
    let mut r = if stereo { vec![0.0; a.n] } else { Vec::new() };
    let mut rng = Rng::new(a.seed);
    let norm = 1.0 / (unison as f64).sqrt();
    for v in 0..unison {
        // voices spread evenly across the detune range and the stereo field
        let pos = if unison == 1 { 0.0 } else { v as f64 / (unison - 1) as f64 * 2.0 - 1.0 };
        let ratio = 2f64.powf(pos * a.detune_cents * 0.5 / 1200.0);
        let mut ph = match a.phase {
            Some(p) if unison == 1 => p.rem_euclid(1.0),
            Some(p) => (p + rng.uniform()).rem_euclid(1.0),
            None => rng.uniform(),
        };
        let pan = pos * a.spread;
        let (gl, gr) = if stereo {
            let ang = (pan + 1.0) * 0.25 * PI;
            (ang.cos() * norm * std::f64::consts::SQRT_2, ang.sin() * norm * std::f64::consts::SQRT_2)
        } else {
            (norm, norm)
        };
        let mut last_pm = a.pm.map(|p| p.at(0) / TAU).unwrap_or(0.0);
        for i in 0..a.n {
            let inc = (a.freq.at(i) * ratio / a.sr).clamp(-0.5, 0.5);
            let (t, dt) = match a.pm {
                Some(pm) => {
                    let m = pm.at(i) / TAU;
                    let d = (inc + (m - last_pm)).abs().clamp(1e-9, 0.5);
                    last_pm = m;
                    ((ph + m).rem_euclid(1.0), d)
                }
                None => (ph, inc.abs().max(1e-9)),
            };
            let s = wave_sample(a.wave, t, dt, a.pw.at(i));
            l[i] += s * gl;
            if stereo {
                r[i] += s * gr;
            }
            ph += inc;
            ph -= ph.floor();
        }
    }
    if stereo {
        Sig::stereo(l, r)
    } else {
        Sig::mono(l)
    }
}

/// Sum of sine partials. Each partial is (frequency ratio, amplitude, T60 decay in
/// seconds or 0 for none). Partials at or above `nyquist` are skipped.
pub fn additive(freq: &P, partials: &[(f64, f64, f64)], sr: f64, n: usize, nyquist: f64) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for &(ratio, amp, decay) in partials {
        if amp == 0.0 {
            continue;
        }
        let mut ph = 0.0f64;
        let k = if decay > 0.0 { (-6.907755 / (decay * sr)).exp() } else { 1.0 };
        let mut env = amp;
        for (i, o) in out.iter_mut().enumerate() {
            let f = freq.at(i) * ratio;
            if f < nyquist {
                *o += (TAU * ph).sin() * env;
            }
            ph += f / sr;
            ph -= ph.floor();
            env *= k;
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoiseColor {
    White,
    Pink,
    Brown,
}

impl NoiseColor {
    pub const NAMES: [&'static str; 3] = ["white", "pink", "brown"];
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "white" => NoiseColor::White,
            "pink" => NoiseColor::Pink,
            "brown" | "red" => NoiseColor::Brown,
            _ => return None,
        })
    }
}

pub fn noise(color: NoiseColor, sr: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    match color {
        NoiseColor::White => (0..n).map(|_| rng.bipolar()).collect(),
        NoiseColor::Pink => {
            // Paul Kellet's filter bank, a close approximation of -3 dB/octave.
            let mut b = [0.0f64; 7];
            (0..n)
                .map(|_| {
                    let w = rng.bipolar();
                    b[0] = 0.99886 * b[0] + w * 0.0555179;
                    b[1] = 0.99332 * b[1] + w * 0.0750759;
                    b[2] = 0.96900 * b[2] + w * 0.1538520;
                    b[3] = 0.86650 * b[3] + w * 0.3104856;
                    b[4] = 0.55000 * b[4] + w * 0.5329522;
                    b[5] = -0.7616 * b[5] - w * 0.0168980;
                    let out = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362;
                    b[6] = w * 0.115926;
                    out * 0.16
                })
                .collect()
        }
        NoiseColor::Brown => {
            // Leaky integrator with a 20 Hz corner, scaled to about 0.3 RMS.
            let a = 1.0 - TAU * 20.0 / sr;
            let k = 0.3 * ((1.0 - a * a) * 3.0).sqrt();
            let mut y = 0.0;
            (0..n)
                .map(|_| {
                    y = a * y + k * rng.bipolar();
                    y
                })
                .collect()
        }
    }
}

/// Random impulses at an average rate of `density` per second.
pub fn dust(density: &P, bipolar: bool, sr: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let p = density.at(i).max(0.0) / sr;
            if rng.uniform() < p {
                if bipolar {
                    rng.bipolar()
                } else {
                    rng.uniform().max(0.05)
                }
            } else {
                0.0
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LfoWave {
    Sine,
    Triangle,
    Saw,
    Square,
    /// Sample-and-hold random steps.
    Steps,
    /// Smoothly interpolated random values.
    Smooth,
}

impl LfoWave {
    pub const NAMES: [&'static str; 6] = ["sine", "triangle", "saw", "square", "steps", "smooth"];
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "sine" | "sin" => LfoWave::Sine,
            "triangle" | "tri" => LfoWave::Triangle,
            "saw" => LfoWave::Saw,
            "square" => LfoWave::Square,
            "steps" | "sh" | "random" => LfoWave::Steps,
            "smooth" => LfoWave::Smooth,
            _ => return None,
        })
    }
}

/// Low-frequency oscillator, output in [-1, 1].
pub fn lfo(wave: LfoWave, rate: &P, phase: f64, sr: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut ph = phase.rem_euclid(1.0);
    let mut cur = rng.bipolar();
    let mut next = rng.bipolar();
    (0..n)
        .map(|i| {
            let v = match wave {
                LfoWave::Sine => (TAU * ph).sin(),
                LfoWave::Triangle => 1.0 - 4.0 * (ph - 0.5).abs(),
                LfoWave::Saw => 2.0 * ph - 1.0,
                LfoWave::Square => {
                    if ph < 0.5 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                LfoWave::Steps => cur,
                LfoWave::Smooth => {
                    let s = ph * ph * (3.0 - 2.0 * ph);
                    cur + (next - cur) * s
                }
            };
            ph += rate.at(i).max(0.0) / sr;
            if ph >= 1.0 {
                ph -= ph.floor();
                cur = next;
                next = rng.bipolar();
            }
            v
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(x: &[f64]) -> f64 {
        (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
    }

    #[test]
    fn waves_are_bounded_and_centered() {
        let f = P::C(440.0);
        let pw = P::C(0.3);
        for wave in [Wave::Sine, Wave::Saw, Wave::Square, Wave::Pulse, Wave::Triangle] {
            let s = osc(&OscArgs { wave, freq: &f, pw: &pw, pm: None, unison: 1, detune_cents: 0.0, spread: 0.0, phase: Some(0.0), sr: 96000.0, n: 96000, seed: 1 });
            let peak = s.peak();
            assert!(peak <= 1.6 && peak > 0.9, "{wave:?} peak {peak}");
            let mean = s.l.iter().sum::<f64>() / s.l.len() as f64;
            assert!(mean.abs() < 0.01, "{wave:?} mean {mean}");
        }
    }

    #[test]
    fn unison_goes_stereo() {
        let f = P::C(220.0);
        let pw = P::C(0.5);
        let s = osc(&OscArgs { wave: Wave::Saw, freq: &f, pw: &pw, pm: None, unison: 7, detune_cents: 30.0, spread: 0.8, phase: None, sr: 96000.0, n: 9600, seed: 3 });
        assert!(s.is_stereo());
    }

    #[test]
    fn noise_levels() {
        for c in [NoiseColor::White, NoiseColor::Pink, NoiseColor::Brown] {
            let x = noise(c, 96000.0, 96000, 5);
            let r = rms(&x);
            assert!(r > 0.1 && r < 0.7, "{c:?} rms {r}");
        }
    }
}
