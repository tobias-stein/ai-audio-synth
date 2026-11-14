//! Time-based and non-linear effects.

use super::filter::{dc_block, one_pole_coef};
use super::resample::{decimate, interpolate};
use super::P;
use crate::audio::Sig;
use crate::util::db_to_lin;
use std::f64::consts::{PI, TAU};

/// Circular delay line with cubic interpolation.
pub struct DelayLine {
    buf: Vec<f64>,
    w: usize,
}

impl DelayLine {
    pub fn new(max_samples: usize) -> Self {
        DelayLine { buf: vec![0.0; max_samples + 8], w: 0 }
    }
    #[inline(always)]
    pub fn push(&mut self, x: f64) {
        self.buf[self.w] = x;
        self.w += 1;
        if self.w == self.buf.len() {
            self.w = 0;
        }
    }
    /// Value pushed `d` samples ago (d >= 1), integer delay.
    #[inline(always)]
    pub fn tap(&self, d: usize) -> f64 {
        let len = self.buf.len();
        self.buf[(self.w + len - d.clamp(1, len - 1)) % len]
    }
    /// Fractional delay read with Hermite interpolation.
    #[inline(always)]
    pub fn read(&self, d: f64) -> f64 {
        let len = self.buf.len();
        let d = d.clamp(2.0, (len - 4) as f64);
        let di = d.floor();
        let fr = d - di;
        let base = self.w + 2 * len - di as usize;
        let xm1 = self.buf[(base + 1) % len];
        let x0 = self.buf[base % len];
        let x1 = self.buf[(base - 1) % len];
        let x2 = self.buf[(base - 2) % len];
        let c0 = x0;
        let c1 = 0.5 * (x1 - xm1);
        let c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2;
        let c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1);
        ((c3 * fr + c2) * fr + c1) * fr + c0
    }
}

pub struct DelayArgs<'a> {
    pub time: &'a P,
    pub feedback: &'a P,
    pub mix: &'a P,
    pub damp_hz: f64,
    pub pingpong: bool,
}

pub fn delay(input: &Sig, a: &DelayArgs, sr: f64) -> Sig {
    let n = input.len();
    let max = ((a.time.max().clamp(0.0005, 10.0)) * sr) as usize + 8;
    let stereo = input.is_stereo() || a.pingpong;
    let mut dl = DelayLine::new(max);
    let mut dr = DelayLine::new(max);
    let lp = one_pole_coef(a.damp_hz, sr);
    let (mut sl, mut sr_state) = (0.0, 0.0);
    let mut l = vec![0.0; n];
    let mut r = vec![0.0; if stereo { n } else { 0 }];
    let in_r = input.right();
    for i in 0..n {
        let d = a.time.at(i).clamp(0.0005, 10.0) * sr;
        let fb = a.feedback.at(i).clamp(0.0, 0.98);
        let mix = a.mix.at(i).clamp(0.0, 1.0);
        let wl = dl.read(d);
        sl += lp * (wl - sl);
        if stereo {
            let wr = dr.read(d);
            sr_state += lp * (wr - sr_state);
            if a.pingpong {
                let mono = 0.5 * (input.l[i] + in_r[i]);
                dl.push(mono + sr_state * fb);
                dr.push(sl * fb);
            } else {
                dl.push(input.l[i] + sl * fb);
                dr.push(in_r[i] + sr_state * fb);
            }
            l[i] = input.l[i] * (1.0 - mix) + wl * mix;
            r[i] = in_r[i] * (1.0 - mix) + wr * mix;
        } else {
            dl.push(input.l[i] + sl * fb);
            l[i] = input.l[i] * (1.0 - mix) + wl * mix;
        }
    }
    if stereo {
        Sig::stereo(l, r)
    } else {
        Sig::mono(l)
    }
}

/// Sine oscillator advanced by rotation, far cheaper than calling `sin` per sample.
struct Rotor {
    s: f64,
    c: f64,
    ds: f64,
    dc: f64,
}

impl Rotor {
    fn new(freq: f64, phase_cycles: f64, sr: f64) -> Self {
        let (s, c) = (TAU * phase_cycles).sin_cos();
        let (ds, dc) = (TAU * freq / sr).sin_cos();
        Rotor { s, c, ds, dc }
    }
    #[inline(always)]
    fn next(&mut self) -> f64 {
        let v = self.s;
        let s = self.s * self.dc + self.c * self.ds;
        let c = self.c * self.dc - self.s * self.ds;
        self.s = s;
        self.c = c;
        v
    }
}

/// Multi-voice chorus; always returns stereo.
pub fn chorus(input: &Sig, rate: f64, depth_ms: f64, mix: f64, voices: usize, sr: f64) -> Sig {
    let n = input.len();
    let voices = voices.clamp(1, 6);
    let base = 0.012 * sr;
    let depth = depth_ms * 0.001 * sr;
    let max = (base + depth) as usize + 16;
    let ins = [&input.l[..], input.right()];
    let mut outs: [Vec<f64>; 2] = [vec![0.0; n], vec![0.0; n]];
    let norm = 1.0 / (voices as f64).sqrt();
    for c in 0..2 {
        let mut line = DelayLine::new(max);
        let mut lfos: Vec<Rotor> = (0..voices).map(|v| Rotor::new(rate * (1.0 + 0.07 * v as f64), v as f64 / voices as f64 + c as f64 * 0.25, sr)).collect();
        for i in 0..n {
            let mut wet = 0.0;
            for lfo in lfos.iter_mut() {
                wet += line.read(base + depth * (0.5 + 0.5 * lfo.next()));
            }
            line.push(ins[c][i]);
            outs[c][i] = ins[c][i] * (1.0 - mix) + wet * norm * mix;
        }
    }
    let [l, r] = outs;
    Sig::stereo(l, r)
}

pub fn flanger(input: &Sig, rate: f64, depth_ms: f64, feedback: f64, mix: f64, sr: f64) -> Sig {
    let fb = feedback.clamp(-0.95, 0.95);
    input.map_channels(|c, x| {
        let max = ((1.0 + depth_ms) * 0.001 * sr) as usize + 16;
        let mut line = DelayLine::new(max);
        let mut out = Vec::with_capacity(x.len());
        for (i, &v) in x.iter().enumerate() {
            let ph = i as f64 / sr * rate + c as f64 * 0.25;
            let d = (0.0005 + depth_ms * 0.001 * (0.5 + 0.5 * (TAU * ph).sin())) * sr;
            let wet = line.read(d);
            line.push(v + wet * fb);
            out.push(v * (1.0 - mix * 0.5) + wet * mix);
        }
        out
    })
}

pub fn phaser(input: &Sig, rate: f64, f_lo: f64, f_hi: f64, feedback: f64, stages: usize, mix: f64, sr: f64) -> Sig {
    let stages = stages.clamp(2, 12);
    let fb = feedback.clamp(-0.95, 0.95);
    input.map_channels(|c, x| {
        let mut x1 = vec![0.0; stages];
        let mut y1 = vec![0.0; stages];
        let mut last = 0.0;
        let mut out = Vec::with_capacity(x.len());
        let mut coef = 0.0;
        for (i, &v) in x.iter().enumerate() {
            if i % 8 == 0 {
                let ph = i as f64 / sr * rate + c as f64 * 0.25;
                let s = 0.5 + 0.5 * (TAU * ph).sin();
                let fc = f_lo * (f_hi / f_lo).powf(s);
                let t = (PI * fc / sr).tan();
                coef = (t - 1.0) / (t + 1.0);
            }
            let mut s = v + last * fb;
            for k in 0..stages {
                let y = coef * s + x1[k] - coef * y1[k];
                x1[k] = s;
                y1[k] = y;
                s = y;
            }
            last = s;
            out.push(v * (1.0 - mix * 0.5) + s * mix);
        }
        out
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Tanh,
    Soft,
    Hard,
    Fold,
    Tube,
}

impl Shape {
    pub const NAMES: [&'static str; 5] = ["tanh", "soft", "hard", "fold", "tube"];
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "tanh" => Shape::Tanh,
            "soft" => Shape::Soft,
            "hard" | "clip" => Shape::Hard,
            "fold" => Shape::Fold,
            "tube" => Shape::Tube,
            _ => return None,
        })
    }
    #[inline(always)]
    fn apply(&self, x: f64) -> f64 {
        match self {
            Shape::Tanh => x.tanh(),
            Shape::Soft => x / (1.0 + x.abs()),
            Shape::Hard => x.clamp(-1.0, 1.0),
            Shape::Fold => {
                // triangle wavefolder with period 4
                let t = (x + 1.0).rem_euclid(4.0);
                if t < 2.0 {
                    t - 1.0
                } else {
                    3.0 - t
                }
            }
            Shape::Tube => {
                if x >= 0.0 {
                    x.tanh()
                } else {
                    (0.7 * x).tanh() / 0.7 * 0.9
                }
            }
        }
    }
}

fn ln_cosh(x: f64) -> f64 {
    let a = x.abs();
    a + (-2.0 * a).exp().ln_1p() - std::f64::consts::LN_2
}

impl Shape {
    /// Antiderivative of the shaping function, used for antialiasing.
    #[inline(always)]
    fn integral(&self, x: f64) -> f64 {
        match self {
            Shape::Tanh => ln_cosh(x),
            Shape::Soft => x.abs() - x.abs().ln_1p(),
            Shape::Hard => {
                if x.abs() <= 1.0 {
                    0.5 * x * x
                } else {
                    x.abs() - 0.5
                }
            }
            Shape::Fold => {
                let t = (x + 1.0).rem_euclid(4.0);
                if t < 2.0 {
                    0.5 * t * t - t
                } else {
                    -0.5 * t * t + 3.0 * t - 4.0
                }
            }
            Shape::Tube => {
                if x >= 0.0 {
                    ln_cosh(x)
                } else {
                    ln_cosh(0.7 * x) * 0.9 / 0.49
                }
            }
        }
    }

    /// Antialiased shaping of the step from `prev` to `x` (first-order
    /// antiderivative method): the average of the shaper over the interval
    /// between two samples instead of its value at one point.
    #[inline(always)]
    fn apply_aa(&self, x: f64, prev: f64) -> f64 {
        let d = x - prev;
        if d.abs() < 1e-5 {
            self.apply(0.5 * (x + prev))
        } else {
            (self.integral(x) - self.integral(prev)) / d
        }
    }
}

/// Waveshaping distortion. Harmonics are kept from aliasing by internal
/// oversampling combined with antiderivative antialiasing.
pub fn distort(input: &Sig, shape: Shape, drive_db: &P, mix: f64, oversample: usize, sr: f64) -> Sig {
    let os = oversample.max(1);
    input.map_channels(|_, x| {
        let up = interpolate(x, os);
        let mut prev = 0.0;
        let mut shape_step = |v: f64| {
            let y = shape.apply_aa(v, prev);
            prev = v;
            y
        };
        let shaped: Vec<f64> = match drive_db {
            P::C(d) => {
                let g = db_to_lin(*d);
                up.iter().map(|&v| shape_step(v * g)).collect()
            }
            P::B(_) => up
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let g = db_to_lin(drive_db.at((i / os).min(x.len().saturating_sub(1))));
                    shape_step(v * g)
                })
                .collect(),
        };
        let mut wet = decimate(&shaped, os);
        // waveshaping an asymmetric signal creates a DC offset; remove it
        wet = dc_block(&wet, sr);
        wet.iter().zip(x).map(|(w, d)| d * (1.0 - mix) + w * mix).collect()
    })
}

/// Bit-depth and sample-rate reduction (deliberately aliased, for retro sounds).
pub fn bitcrush(input: &Sig, bits: &P, rate: &P, sr: f64) -> Sig {
    input.map_channels(|_, x| {
        let mut held = 0.0;
        let mut ph = 1.0;
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                ph += rate.at(i).clamp(20.0, sr) / sr;
                if ph >= 1.0 {
                    ph -= ph.floor();
                    let levels = 2f64.powf(bits.at(i).clamp(1.0, 24.0) - 1.0);
                    held = (v * levels).round() / levels;
                }
                held
            })
            .collect()
    })
}

/// Equal-power pan (-1 left .. 1 right). Stereo input is balanced instead.
pub fn pan(input: &Sig, pos: &P) -> Sig {
    let n = input.len();
    let mut l = vec![0.0; n];
    let mut r = vec![0.0; n];
    let in_r = input.right();
    let stereo = input.is_stereo();
    for i in 0..n {
        let p = pos.at(i).clamp(-1.0, 1.0);
        let ang = (p + 1.0) * 0.25 * PI;
        let (gl, gr) = (ang.cos(), ang.sin());
        if stereo {
            l[i] = input.l[i] * gl * std::f64::consts::SQRT_2;
            r[i] = in_r[i] * gr * std::f64::consts::SQRT_2;
        } else {
            l[i] = input.l[i] * gl;
            r[i] = input.l[i] * gr;
        }
    }
    Sig::stereo(l, r)
}

/// Mid/side width: 0 = mono, 1 = unchanged, 2 = extra wide.
pub fn width(input: &Sig, amount: f64) -> Sig {
    let in_r = input.right();
    let n = input.len();
    let mut l = vec![0.0; n];
    let mut r = vec![0.0; n];
    for i in 0..n {
        let m = 0.5 * (input.l[i] + in_r[i]);
        let s = 0.5 * (input.l[i] - in_r[i]) * amount;
        l[i] = m + s;
        r[i] = m - s;
    }
    Sig::stereo(l, r)
}

/// Variable-speed playback of the input (tape-style pitch and time change).
pub fn speed(input: &Sig, rate: &P) -> Sig {
    input.map_channels(|_, x| {
        let n = x.len();
        let mut pos = 0.0f64;
        let get = |i: isize| -> f64 {
            if i < 0 || i as usize >= n {
                0.0
            } else {
                x[i as usize]
            }
        };
        (0..n)
            .map(|i| {
                let i0 = pos.floor() as isize;
                let fr = pos - i0 as f64;
                let (xm1, x0, x1, x2) = (get(i0 - 1), get(i0), get(i0 + 1), get(i0 + 2));
                let c1 = 0.5 * (x1 - xm1);
                let c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2;
                let c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1);
                pos += rate.at(i).max(0.0);
                ((c3 * fr + c2) * fr + c1) * fr + x0
            })
            .collect()
    })
}

/// Envelope-difference transient shaper: boost or cut attacks and sustains.
pub fn transient(input: &Sig, attack_db: f64, sustain_db: f64, sr: f64) -> Sig {
    let n = input.len();
    let in_r = input.right();
    let coef = |ms: f64| (-1.0 / (ms * 0.001 * sr)).exp();
    let (a_fast, a_slow, r_short, r_long) = (coef(0.5), coef(20.0), coef(40.0), coef(400.0));
    let follow = |env: &mut f64, x: f64, a: f64, r: f64| {
        let c = if x > *env { a } else { r };
        *env = c * *env + (1.0 - c) * x;
    };
    let (mut fast, mut slow, mut long) = (0.0, 0.0, 0.0);
    let mut gains = vec![1.0; n];
    for i in 0..n {
        let x = input.l[i].abs().max(in_r[i].abs());
        follow(&mut fast, x, a_fast, r_short);
        follow(&mut slow, x, a_slow, r_short);
        follow(&mut long, x, a_fast, r_long);
        let att = ((fast - slow) / (fast + 1e-9)).clamp(0.0, 1.0);
        let sus = ((long - fast) / (long + 1e-9)).clamp(0.0, 1.0);
        gains[i] = db_to_lin(attack_db * att + sustain_db * sus);
    }
    input.map_channels(|_, x| x.iter().zip(&gains).map(|(v, g)| v * g).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_line_reads_back() {
        let mut d = DelayLine::new(100);
        for i in 0..50 {
            d.push(i as f64);
        }
        assert_eq!(d.tap(1), 49.0);
        assert_eq!(d.tap(10), 40.0);
        assert!((d.read(10.0) - 40.0).abs() < 1e-9);
        assert!((d.read(10.5) - 39.5).abs() < 1e-9);
    }

    #[test]
    fn delay_echoes_at_the_right_time() {
        let sr = 48000.0;
        let mut x = vec![0.0; 48000];
        x[0] = 1.0;
        let out = delay(&Sig::mono(x), &DelayArgs { time: &P::C(0.25), feedback: &P::C(0.0), mix: &P::C(1.0), damp_hz: 20000.0, pingpong: false }, sr);
        let (imax, _) = out.l.iter().enumerate().fold((0, 0.0), |(bi, bv), (i, v)| if v.abs() > bv { (i, v.abs()) } else { (bi, bv) });
        assert!((imax as i64 - 12000).abs() <= 2, "echo at {imax}");
    }

    #[test]
    fn antialiased_shaper_matches_plain_shaper_for_slow_signals() {
        for sh in [Shape::Tanh, Shape::Soft, Shape::Hard, Shape::Fold, Shape::Tube] {
            let mut prev = -3.0f64;
            for i in 1..6000 {
                let x = -3.0 + i as f64 * 0.001;
                let aa = sh.apply_aa(x, prev);
                let plain = sh.apply(0.5 * (x + prev));
                assert!((aa - plain).abs() < 2e-3, "{sh:?} at {x}: {aa} vs {plain}");
                prev = x;
            }
        }
    }

    #[test]
    fn distortion_is_bounded() {
        let x: Vec<f64> = (0..4800).map(|i| (i as f64 * 0.05).sin() * 4.0).collect();
        for sh in [Shape::Tanh, Shape::Soft, Shape::Hard, Shape::Fold] {
            let y = distort(&Sig::mono(x.clone()), sh, &P::C(12.0), 1.0, 4, 48000.0);
            assert!(y.peak() < 1.3, "{sh:?} peak {}", y.peak());
        }
    }
}
