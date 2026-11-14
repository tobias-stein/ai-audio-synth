//! Dynamics: compressor and a look-ahead true-peak limiter.

use super::resample::{inter_sample_peaks, true_peak};
use crate::audio::Sig;
use crate::util::{db_to_lin, lin_to_db};
use std::collections::VecDeque;

pub struct CompArgs {
    pub threshold_db: f64,
    pub ratio: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    pub knee_db: f64,
    pub makeup_db: f64,
}

/// Feed-forward compressor with soft knee. Channels are linked. If `key` is
/// given it drives the detector instead of the input (sidechain ducking).
/// Returns the processed signal and the maximum gain reduction in dB.
pub fn compressor(input: &Sig, key: Option<&Sig>, a: &CompArgs, sr: f64) -> (Sig, f64) {
    let n = input.len();
    let det = key.unwrap_or(input);
    let det_r = det.right();
    let att = (-1.0 / (a.attack_ms.max(0.01) * 0.001 * sr)).exp();
    let rel = (-1.0 / (a.release_ms.max(1.0) * 0.001 * sr)).exp();
    let ratio = a.ratio.max(1.0);
    let knee = a.knee_db.max(0.0);
    let slope = 1.0 - 1.0 / ratio;
    let mut gr = 0.0f64; // smoothed gain reduction in dB (<= 0)
    let mut max_gr = 0.0f64;
    let makeup = a.makeup_db;
    let mut gains = vec![1.0; n];
    for i in 0..n {
        let level = if i < det.len() { det.l[i].abs().max(det_r[i].abs()) } else { 0.0 };
        let over = lin_to_db(level) - a.threshold_db;
        let target = if knee > 0.0 && over.abs() <= knee / 2.0 {
            -slope * (over + knee / 2.0).powi(2) / (2.0 * knee)
        } else if over > 0.0 {
            -slope * over
        } else {
            0.0
        };
        let c = if target < gr { att } else { rel };
        gr = c * gr + (1.0 - c) * target;
        max_gr = max_gr.min(gr);
        gains[i] = db_to_lin(gr + makeup);
    }
    (input.map_channels(|_, x| x.iter().zip(&gains).map(|(v, g)| v * g).collect()), -max_gr)
}

/// Brickwall limiter. Being offline it can look ahead: the gain curve is the
/// sliding minimum of the required gain, smoothed so it never overshoots.
/// With `true_peak` the detector sees inter-sample peaks. Returns the limited
/// signal and the maximum gain reduction in dB.
pub fn limiter(input: &Sig, ceiling_db: f64, release_ms: f64, lookahead_ms: f64, use_true_peak: bool, sr: f64) -> (Sig, f64) {
    let n = input.len();
    if n == 0 {
        return (input.clone(), 0.0);
    }
    let ceiling = db_to_lin(ceiling_db);
    let look = ((lookahead_ms.max(0.1) * 0.001 * sr) as usize).max(1);
    let mut out = input.clone();
    let mut total_gain = vec![1.0f64; n];
    // Gain modulation can nudge inter-sample peaks back over the ceiling, so
    // repeat the pass on the result until it holds (normally one or two passes).
    for _ in 0..4 {
        let gain = limiter_gain(&out, ceiling, look, release_ms, use_true_peak, sr);
        if gain.iter().all(|g| *g >= 1.0) {
            break;
        }
        out = out.map_channels(|_, x| x.iter().zip(&gain).map(|(v, g)| v * g).collect());
        for (t, g) in total_gain.iter_mut().zip(&gain) {
            *t *= g;
        }
        if !use_true_peak {
            break;
        }
    }
    if use_true_peak {
        let tp = true_peak(&out);
        if tp > ceiling {
            out.scale(ceiling / tp);
        }
    }
    let max_gr = -lin_to_db(total_gain.iter().cloned().fold(1.0, f64::min));
    (out, max_gr)
}

fn limiter_gain(input: &Sig, ceiling: f64, look: usize, release_ms: f64, use_true_peak: bool, sr: f64) -> Vec<f64> {
    let n = input.len();
    // required gain per sample
    let mut need = vec![1.0f64; n];
    for c in input.chans() {
        let peaks: Vec<f64> = if use_true_peak { inter_sample_peaks(c) } else { c.iter().map(|v| v.abs()).collect() };
        for i in 0..n {
            // a hair of margin so that rounding never leaves a residue
            if peaks[i] > ceiling * 0.9995 {
                need[i] = need[i].min(ceiling * 0.999 / peaks[i]);
            }
        }
    }
    if need.iter().all(|g| *g >= 1.0) {
        return need;
    }
    // sliding minimum over [i, i + look]
    let mut gmin = vec![1.0f64; n];
    let mut dq: VecDeque<usize> = VecDeque::new();
    let mut j = 0usize;
    for i in 0..n {
        while j < n && j <= i + look {
            while let Some(&b) = dq.back() {
                if need[b] >= need[j] {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(j);
            j += 1;
        }
        while let Some(&f) = dq.front() {
            if f < i {
                dq.pop_front();
            } else {
                break;
            }
        }
        gmin[i] = need[*dq.front().unwrap()];
    }
    // exponential release toward unity
    let rel = 1.0 - (-1.0 / (release_ms.max(1.0) * 0.001 * sr)).exp();
    let mut gs = vec![1.0f64; n];
    let mut prev = 1.0f64;
    for i in 0..n {
        let recovered = prev + (1.0 - prev) * rel;
        prev = gmin[i].min(recovered);
        gs[i] = prev;
    }
    // moving average over the look-ahead window turns steps into ramps
    let mut gain = vec![1.0f64; n];
    let mut acc = 0.0;
    for i in 0..n {
        acc += gs[i];
        if i >= look {
            acc -= gs[i - look];
        }
        gain[i] = (acc / (i + 1).min(look) as f64).min(1.0);
    }
    gain
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn limiter_holds_the_ceiling() {
        let sr = 48000.0;
        let x: Vec<f64> = (0..48000).map(|i| (TAU * 997.0 * i as f64 / sr).sin() * if i > 10000 && i < 20000 { 2.5 } else { 0.4 }).collect();
        let (out, gr) = limiter(&Sig::mono(x.clone()), -1.0, 100.0, 5.0, true, sr);
        assert!(true_peak(&out) <= db_to_lin(-1.0) + 1e-6, "true peak {}", lin_to_db(true_peak(&out)));
        assert!(gr > 8.0);
        // quiet parts far from the loud burst are untouched
        assert!((out.l[2000] - x[2000]).abs() < 1e-9);
    }

    #[test]
    fn compressor_reduces_loud_signal() {
        let sr = 48000.0;
        let x: Vec<f64> = (0..48000).map(|i| (TAU * 220.0 * i as f64 / sr).sin() * 0.9).collect();
        let a = CompArgs { threshold_db: -20.0, ratio: 4.0, attack_ms: 5.0, release_ms: 100.0, knee_db: 0.0, makeup_db: 0.0 };
        let (out, gr) = compressor(&Sig::mono(x), None, &a, sr);
        // input peaks at -0.9 dB: 19.1 dB over, 4:1 leaves about 14.3 dB reduction
        assert!(gr > 12.0 && gr < 15.0, "gr {gr}");
        let settled = out.l[24000..].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(settled < 0.2, "settled peak {settled}");
    }
}
