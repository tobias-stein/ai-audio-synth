//! Algorithmic reverb: input diffusion into an 8-line feedback delay network
//! with a Hadamard mixing matrix, frequency-dependent decay and slow delay
//! modulation to avoid metallic ringing.

use super::filter::one_pole_coef;
use super::fx::DelayLine;
use crate::audio::Sig;
use crate::util::Rng;
use std::f64::consts::TAU;

pub struct ReverbArgs {
    /// Time for the tail to fall by 60 dB, in seconds.
    pub decay: f64,
    /// Room size multiplier (0.3 small room .. 2.5 huge hall).
    pub size: f64,
    /// High-frequency damping corner in Hz.
    pub damping_hz: f64,
    pub predelay_ms: f64,
    /// 0 = dry only, 1 = wet only.
    pub mix: f64,
    pub width: f64,
    pub diffusion: f64,
    pub lowcut_hz: f64,
    pub modulation: f64,
}

const LINE_MS: [f64; 8] = [31.7, 37.9, 43.3, 47.9, 55.1, 61.3, 68.9, 76.1];
const DIFF_MS: [f64; 4] = [4.77, 3.59, 12.73, 9.31];

fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    let mut i = 2;
    while i * i <= n {
        if n % i == 0 {
            return false;
        }
        i += 1;
    }
    true
}

fn next_prime(mut n: usize) -> usize {
    while !is_prime(n) {
        n += 1;
    }
    n
}

struct Allpass {
    line: DelayLine,
    len: usize,
    g: f64,
}

impl Allpass {
    fn new(len: usize, g: f64) -> Self {
        Allpass { line: DelayLine::new(len + 2), len, g }
    }
    #[inline(always)]
    fn tick(&mut self, x: f64) -> f64 {
        let d = self.line.tap(self.len);
        let v = x + d * self.g;
        self.line.push(v);
        d - v * self.g
    }
}

#[inline(always)]
fn hadamard8(v: &mut [f64; 8]) {
    let mut h = 1;
    while h < 8 {
        let mut i = 0;
        while i < 8 {
            for j in i..i + h {
                let a = v[j];
                let b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
    let s = 1.0 / 8f64.sqrt();
    for x in v.iter_mut() {
        *x *= s;
    }
}

pub fn reverb(input: &Sig, a: &ReverbArgs, sr: f64, seed: u64) -> Sig {
    let n = input.len();
    let in_r = input.right();
    let size = a.size.clamp(0.1, 4.0);
    let decay = a.decay.max(0.05);
    let mut rng = Rng::new(seed);

    let lens: Vec<usize> = LINE_MS.iter().map(|ms| next_prime((ms * 0.001 * sr * size) as usize)).collect();
    let max_mod = (0.0008 * sr * a.modulation.max(0.0)) + 4.0;
    let mut lines: Vec<DelayLine> = lens.iter().map(|&l| DelayLine::new(l + max_mod as usize + 8)).collect();
    // per-line feedback gain for the requested T60
    let gains: Vec<f64> = lens.iter().map(|&l| 10f64.powf(-3.0 * l as f64 / (decay * sr))).collect();
    let g_avg = gains.iter().sum::<f64>() / 8.0;
    let lp = one_pole_coef(a.damping_hz.clamp(200.0, sr * 0.45), sr);
    let mut damp = [0.0f64; 8];
    let mod_rate: Vec<f64> = (0..8).map(|_| rng.range(0.07, 0.6)).collect();
    let mod_phase: Vec<f64> = (0..8).map(|_| rng.uniform()).collect();
    let mod_depth = 0.0004 * sr * a.modulation.max(0.0);

    let dg = a.diffusion.clamp(0.0, 0.9);
    let mut diff_l: Vec<Allpass> = DIFF_MS.iter().map(|ms| Allpass::new(next_prime((ms * 0.001 * sr * size.sqrt()) as usize), dg)).collect();
    let mut diff_r: Vec<Allpass> = DIFF_MS.iter().map(|ms| Allpass::new(next_prime((ms * 1.07 * 0.001 * sr * size.sqrt()) as usize), dg)).collect();

    let pre = (a.predelay_ms.clamp(0.0, 500.0) * 0.001 * sr) as usize;
    let mut pre_l = DelayLine::new(pre + 2);
    let mut pre_r = DelayLine::new(pre + 2);

    // Keeps perceived wet level roughly stable across decay times.
    let norm = (1.0 - g_avg * g_avg).max(1e-6).powf(0.25) * 0.8;
    let hp = one_pole_coef(a.lowcut_hz.clamp(5.0, 2000.0), sr);
    let (mut hp_l, mut hp_r) = (0.0, 0.0);
    let mix = a.mix.clamp(0.0, 1.0);
    let width = a.width.clamp(0.0, 2.0);

    let mut out_l = vec![0.0; n];
    let mut out_r = vec![0.0; n];
    let mut o = [0.0f64; 8];
    for i in 0..n {
        let (mut xl, mut xr) = (input.l[i], in_r[i]);
        if pre > 0 {
            let (pl, pr) = (pre_l.tap(pre), pre_r.tap(pre));
            pre_l.push(xl);
            pre_r.push(xr);
            xl = pl;
            xr = pr;
        }
        for ap in diff_l.iter_mut() {
            xl = ap.tick(xl);
        }
        for ap in diff_r.iter_mut() {
            xr = ap.tick(xr);
        }
        let t = i as f64 / sr;
        for k in 0..8 {
            let d = lens[k] as f64 + mod_depth * (1.0 + (TAU * (mod_rate[k] * t + mod_phase[k])).sin());
            let v = lines[k].read(d);
            o[k] = v;
        }
        let mut f = [0.0f64; 8];
        for k in 0..8 {
            damp[k] += lp * (o[k] - damp[k]);
            f[k] = damp[k] * gains[k];
        }
        hadamard8(&mut f);
        for k in 0..8 {
            let inj = if k % 2 == 0 { xl } else { xr };
            lines[k].push(inj * 0.5 + f[k]);
        }
        let wl = (o[0] - o[2] + o[4] - o[6]) * norm;
        let wr = (o[1] - o[3] + o[5] - o[7]) * norm;
        hp_l += hp * (wl - hp_l);
        hp_r += hp * (wr - hp_r);
        let (wl, wr) = (wl - hp_l, wr - hp_r);
        let m = 0.5 * (wl + wr);
        let s = 0.5 * (wl - wr) * width;
        out_l[i] = input.l[i] * (1.0 - mix) + (m + s) * mix;
        out_r[i] = in_r[i] * (1.0 - mix) + (m - s) * mix;
    }
    Sig::stereo(out_l, out_r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(decay: f64) -> ReverbArgs {
        ReverbArgs { decay, size: 1.0, damping_hz: 18000.0, predelay_ms: 0.0, mix: 1.0, width: 1.0, diffusion: 0.7, lowcut_hz: 20.0, modulation: 0.0 }
    }

    fn energy_db(s: &Sig, from: usize, to: usize) -> f64 {
        let e: f64 = s.l[from..to].iter().chain(s.right()[from..to].iter()).map(|v| v * v).sum::<f64>() / (2 * (to - from)) as f64;
        10.0 * e.max(1e-30).log10()
    }

    #[test]
    fn decay_time_matches_request() {
        let sr = 48000.0;
        let mut x = vec![0.0; (sr * 3.0) as usize];
        x[0] = 1.0;
        let out = reverb(&Sig::mono(x), &args(2.0), sr, 1);
        assert!(!out.has_non_finite());
        // level should fall about 30 dB per second for a 2 s T60
        let w = 4800;
        let e1 = energy_db(&out, (0.5 * sr) as usize, (0.5 * sr) as usize + w);
        let e2 = energy_db(&out, (1.5 * sr) as usize, (1.5 * sr) as usize + w);
        let drop = e1 - e2;
        assert!((drop - 30.0).abs() < 5.0, "dropped {drop} dB over one second");
    }

    #[test]
    fn stable_with_long_decay() {
        let sr = 48000.0;
        let x: Vec<f64> = (0..(sr * 2.0) as usize).map(|i| ((i * 7919) % 1000) as f64 / 500.0 - 1.0).collect();
        let out = reverb(&Sig::mono(x), &args(20.0), sr, 1);
        assert!(!out.has_non_finite());
        assert!(out.peak() < 8.0, "peak {}", out.peak());
    }
}
