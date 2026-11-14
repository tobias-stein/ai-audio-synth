//! Frequency-domain measurements: average spectrum, bands, pitch, onsets,
//! tempo and key.

use crate::util::{hann, hz_to_midi, midi_to_name, pitch_class_name};
use rustfft::{num_complex::Complex, FftPlanner};

pub struct Stft {
    /// Power spectra, one per frame, `size / 2 + 1` bins each.
    pub frames: Vec<Vec<f64>>,
    pub size: usize,
    pub hop: usize,
    pub sr: f64,
}

impl Stft {
    pub fn bin_hz(&self) -> f64 {
        self.sr / self.size as f64
    }
    pub fn frame_time(&self, i: usize) -> f64 {
        (i * self.hop) as f64 / self.sr
    }
}

/// Short-time Fourier transform with a Hann window. Short inputs are
/// zero-padded to one frame.
pub fn stft(x: &[f64], size: usize, hop: usize, sr: f64) -> Stft {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(size);
    let win = hann(size);
    // scaled so that the bins of one frame sum to the mean square of the signal
    let wnorm: f64 = win.iter().map(|w| w * w).sum::<f64>() * size as f64 / 2.0;
    let mut frames = Vec::new();
    let mut buf = vec![Complex::new(0.0, 0.0); size];
    let mut start = 0usize;
    loop {
        for i in 0..size {
            let v = if start + i < x.len() { x[start + i] } else { 0.0 };
            buf[i] = Complex::new(v * win[i], 0.0);
        }
        fft.process(&mut buf);
        frames.push(buf[..size / 2 + 1].iter().map(|c| c.norm_sqr() / wnorm).collect());
        start += hop;
        if start + size > x.len() {
            break;
        }
    }
    Stft { frames, size, hop, sr }
}

/// Average power spectrum over frames that are not silent.
pub fn average_spectrum(s: &Stft) -> Vec<f64> {
    let energies: Vec<f64> = s.frames.iter().map(|f| f.iter().sum::<f64>()).collect();
    let max = energies.iter().cloned().fold(0.0, f64::max);
    let bins = s.size / 2 + 1;
    let mut avg = vec![0.0; bins];
    let mut count = 0;
    for (f, e) in s.frames.iter().zip(&energies) {
        if *e >= max * 1e-6 && *e > 0.0 {
            for (a, v) in avg.iter_mut().zip(f) {
                *a += v;
            }
            count += 1;
        }
    }
    if count > 0 {
        for a in avg.iter_mut() {
            *a /= count as f64;
        }
    }
    avg
}

pub const BANDS: [(&str, f64, f64); 7] = [
    ("sub", 20.0, 60.0),
    ("bass", 60.0, 250.0),
    ("low_mid", 250.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("high_mid", 2000.0, 6000.0),
    ("high", 6000.0, 12000.0),
    ("air", 12000.0, 20000.0),
];

/// Share of total energy in each named band (fractions summing to about 1).
pub fn band_shares(power: &[f64], bin_hz: f64) -> Vec<f64> {
    let mut out = vec![0.0; BANDS.len()];
    let mut total = 0.0;
    for (i, p) in power.iter().enumerate() {
        let f = i as f64 * bin_hz;
        if f < 20.0 || f > 20000.0 {
            continue;
        }
        total += p;
        for (k, b) in BANDS.iter().enumerate() {
            if f >= b.1 && f < b.2 {
                out[k] += p;
            }
        }
    }
    if total > 0.0 {
        for v in out.iter_mut() {
            *v /= total;
        }
    }
    out
}

pub struct SpectralShape {
    pub centroid_hz: f64,
    pub rolloff_hz: f64,
    pub flatness: f64,
}

pub fn spectral_shape(power: &[f64], bin_hz: f64) -> SpectralShape {
    let (mut num, mut den, mut total) = (0.0, 0.0, 0.0);
    for (i, p) in power.iter().enumerate() {
        let f = i as f64 * bin_hz;
        if f < 20.0 {
            continue;
        }
        let m = p.sqrt();
        num += f * m;
        den += m;
        total += p;
    }
    let centroid_hz = if den > 0.0 { num / den } else { 0.0 };
    let mut acc = 0.0;
    let mut rolloff_hz = 0.0;
    for (i, p) in power.iter().enumerate() {
        let f = i as f64 * bin_hz;
        if f < 20.0 {
            continue;
        }
        acc += p;
        if acc >= 0.85 * total {
            rolloff_hz = f;
            break;
        }
    }
    // flatness between 50 Hz and 16 kHz: 1 = noise-like, 0 = tonal
    let lo = (50.0 / bin_hz).ceil() as usize;
    let hi = ((16000.0 / bin_hz) as usize).min(power.len().saturating_sub(1));
    let mut flatness = 0.0;
    if hi > lo {
        let n = (hi - lo + 1) as f64;
        let mean = power[lo..=hi].iter().sum::<f64>() / n;
        if mean > 0.0 {
            let log_mean = power[lo..=hi].iter().map(|p| (p + 1e-30).ln()).sum::<f64>() / n;
            flatness = (log_mean.exp() / mean).clamp(0.0, 1.0);
        }
    }
    SpectralShape { centroid_hz, rolloff_hz, flatness }
}

#[derive(Clone, Debug)]
pub struct Pitch {
    pub hz: f64,
    pub note: String,
    pub cents_off: f64,
    /// 0..1: how clearly periodic the sound is.
    pub clarity: f64,
    /// Fraction of non-silent frames that had a clear pitch.
    pub tonal_fraction: f64,
    pub start_hz: f64,
    pub end_hz: f64,
}

/// Pitch tracking by normalized autocorrelation (McLeod-style peak picking).
pub fn detect_pitch(x: &[f64], sr: f64) -> Option<Pitch> {
    let size = 4096usize;
    let hop = 1024usize;
    if x.len() < size / 4 {
        return None;
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(size * 2);
    let ifft = planner.plan_fft_inverse(size * 2);
    let min_lag = (sr / 4000.0).ceil() as usize;
    let max_lag = ((sr / 40.0) as usize).min(size / 2);
    let peak_energy = x.iter().map(|v| v * v).fold(0.0, f64::max);
    let mut track: Vec<(f64, f64)> = Vec::new(); // (hz, clarity)
    let mut frames = 0usize;
    let mut start = 0usize;
    while start < x.len() {
        let frame: Vec<f64> = (0..size).map(|i| if start + i < x.len() { x[start + i] } else { 0.0 }).collect();
        let energy: f64 = frame.iter().map(|v| v * v).sum::<f64>() / size as f64;
        start += hop;
        if energy < peak_energy * 1e-5 || energy < 1e-10 {
            if start + size / 2 > x.len() {
                break;
            }
            continue;
        }
        frames += 1;
        let mut buf: Vec<Complex<f64>> = frame.iter().map(|v| Complex::new(*v, 0.0)).chain(std::iter::repeat(Complex::new(0.0, 0.0)).take(size)).collect();
        fft.process(&mut buf);
        for c in buf.iter_mut() {
            *c = Complex::new(c.norm_sqr(), 0.0);
        }
        ifft.process(&mut buf);
        let acf: Vec<f64> = buf[..size].iter().map(|c| c.re / (2.0 * size as f64)).collect();
        // normalized square difference function
        let mut m = 2.0 * acf[0];
        let mut nsdf = vec![0.0; max_lag + 1];
        nsdf[0] = 1.0;
        for lag in 1..=max_lag {
            m -= frame[lag - 1] * frame[lag - 1] + frame[size - lag] * frame[size - lag];
            nsdf[lag] = if m > 1e-12 { 2.0 * acf[lag] / m } else { 0.0 };
        }
        // peaks after the first zero crossing
        let mut lag = 1;
        while lag < max_lag && nsdf[lag] > 0.0 {
            lag += 1;
        }
        let mut peaks: Vec<(usize, f64)> = Vec::new();
        while lag < max_lag {
            if nsdf[lag] > 0.0 && nsdf[lag] >= nsdf[lag - 1] && nsdf[lag] > nsdf[lag + 1] && lag >= min_lag {
                peaks.push((lag, nsdf[lag]));
            }
            lag += 1;
        }
        if let Some(best) = peaks.iter().map(|p| p.1).fold(None, |m: Option<f64>, v| Some(m.map_or(v, |x| x.max(v)))) {
            if let Some(&(l, v)) = peaks.iter().find(|p| p.1 >= 0.9 * best) {
                // parabolic interpolation around the peak
                let (a, b, c) = (nsdf[l - 1], nsdf[l], nsdf[l + 1]);
                let denom = a - 2.0 * b + c;
                let shift = if denom.abs() > 1e-12 { 0.5 * (a - c) / denom } else { 0.0 };
                track.push((sr / (l as f64 + shift), v.clamp(0.0, 1.0)));
            }
        }
        if start + size / 2 > x.len() {
            break;
        }
    }
    if frames == 0 {
        return None;
    }
    let clear: Vec<(f64, f64)> = track.iter().cloned().filter(|p| p.1 >= 0.8).collect();
    if clear.is_empty() {
        return None;
    }
    let mut hz: Vec<f64> = clear.iter().map(|p| p.0).collect();
    hz.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = hz[hz.len() / 2];
    let clarity = clear.iter().map(|p| p.1).sum::<f64>() / clear.len() as f64;
    let midi = hz_to_midi(median);
    let nearest = midi.round();
    Some(Pitch {
        hz: median,
        note: midi_to_name(nearest as i32),
        cents_off: (midi - nearest) * 100.0,
        clarity,
        tonal_fraction: clear.len() as f64 / frames as f64,
        start_hz: clear.first().unwrap().0,
        end_hz: clear.last().unwrap().0,
    })
}

pub struct Rhythm {
    pub onsets: Vec<f64>,
    pub tempo_bpm: Option<f64>,
    pub tempo_confidence: f64,
}

/// Onset detection by spectral flux, and tempo from the autocorrelation of
/// the onset-strength curve.
pub fn detect_rhythm(x: &[f64], sr: f64) -> Rhythm {
    let size = 1024;
    let hop = 256;
    let s = stft(x, size, hop, sr);
    let nf = s.frames.len();
    let mut flux = vec![0.0; nf];
    let logmag = |p: f64| (1.0 + 2200.0 * p.sqrt()).ln();
    for t in 1..nf {
        let mut acc = 0.0;
        for (a, b) in s.frames[t].iter().zip(&s.frames[t - 1]) {
            let d = logmag(*a) - logmag(*b);
            if d > 0.0 {
                acc += d;
            }
        }
        flux[t] = acc;
    }
    let frame_rate = sr / hop as f64;
    let mean = flux.iter().sum::<f64>() / nf.max(1) as f64;
    let w = (0.1 * frame_rate) as usize;
    let min_gap = (0.05 * frame_rate).ceil() as usize;
    let mut onsets = Vec::new();
    let mut last: Option<usize> = None;
    for t in 1..nf.saturating_sub(1) {
        let lo = t.saturating_sub(w);
        let hi = (t + w + 1).min(nf);
        let local = flux[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
        let is_peak = flux[t] > flux[t - 1] && flux[t] >= flux[t + 1];
        if is_peak && flux[t] > local * 1.5 + mean * 0.5 && last.map_or(true, |l| t - l >= min_gap) {
            onsets.push(s.frame_time(t));
            last = Some(t);
        }
    }
    let mut tempo_bpm = None;
    let mut tempo_confidence = 0.0;
    if x.len() as f64 / sr >= 3.0 && onsets.len() >= 6 {
        let f: Vec<f64> = flux.iter().map(|v| (v - mean).max(0.0)).collect();
        let acf0: f64 = f.iter().map(|v| v * v).sum();
        if acf0 > 0.0 {
            let mut best = (0.0, 0.0);
            let mut bpm = 50.0;
            while bpm <= 220.0 {
                let lag = 60.0 / bpm * frame_rate;
                let l = lag.round() as usize;
                if l < f.len() {
                    // include the double period so that true beats win over syncopation
                    let mut acc = 0.0;
                    for (mult, wgt) in [(1usize, 1.0), (2, 0.5)] {
                        let ll = l * mult;
                        if ll < f.len() {
                            let n = f.len() - ll;
                            acc += wgt * (0..n).map(|i| f[i] * f[i + ll]).sum::<f64>() / n as f64 * f.len() as f64;
                        }
                    }
                    // gentle preference for moderate tempi
                    let prior = (-0.5 * ((bpm / 115.0).log2() / 1.0).powi(2)).exp();
                    let score = acc / acf0 * prior;
                    if score > best.1 {
                        best = (bpm, score);
                    }
                }
                bpm += 0.5;
            }
            if best.1 > 0.05 {
                tempo_bpm = Some(best.0);
                tempo_confidence = (best.1 / 1.5).clamp(0.0, 1.0);
            }
        }
    }
    Rhythm { onsets, tempo_bpm, tempo_confidence }
}

#[derive(Clone, Debug)]
pub struct KeyEstimate {
    pub name: String,
    pub confidence: f64,
}

/// Musical key from an averaged chroma vector (Krumhansl-Schmuckler profiles).
pub fn detect_key(x: &[f64], sr: f64) -> Option<KeyEstimate> {
    let size = 8192;
    let s = stft(x, size, size / 2, sr);
    let bin_hz = s.bin_hz();
    let mut chroma = [0.0f64; 12];
    for f in &s.frames {
        for (i, p) in f.iter().enumerate() {
            let hz = i as f64 * bin_hz;
            if !(65.0..=2100.0).contains(&hz) {
                continue;
            }
            let pc = (hz_to_midi(hz).round() as i32).rem_euclid(12) as usize;
            chroma[pc] += p.sqrt();
        }
    }
    let total: f64 = chroma.iter().sum();
    if total <= 0.0 {
        return None;
    }
    const MAJOR: [f64; 12] = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
    const MINOR: [f64; 12] = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
    let corr = |profile: &[f64; 12], tonic: usize| -> f64 {
        let xs: Vec<f64> = (0..12).map(|i| chroma[(i + tonic) % 12]).collect();
        let mx = xs.iter().sum::<f64>() / 12.0;
        let my = profile.iter().sum::<f64>() / 12.0;
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for i in 0..12 {
            sxy += (xs[i] - mx) * (profile[i] - my);
            sxx += (xs[i] - mx).powi(2);
            syy += (profile[i] - my).powi(2);
        }
        if sxx > 0.0 && syy > 0.0 {
            sxy / (sxx * syy).sqrt()
        } else {
            0.0
        }
    };
    let mut scores: Vec<(f64, String)> = Vec::new();
    for tonic in 0..12 {
        scores.push((corr(&MAJOR, tonic), format!("{} major", pitch_class_name(tonic as i32))));
        scores.push((corr(&MINOR, tonic), format!("{} minor", pitch_class_name(tonic as i32))));
    }
    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let (best, name) = scores[0].clone();
    if best < 0.4 {
        return None;
    }
    Some(KeyEstimate { name, confidence: ((best - scores[1].0) * 4.0 + best * 0.5).clamp(0.0, 1.0) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn pitch_of_a_sine_and_of_noise() {
        let sr = 48000.0;
        let x: Vec<f64> = (0..48000).map(|i| (TAU * 261.63 * i as f64 / sr).sin() * 0.5).collect();
        let p = detect_pitch(&x, sr).unwrap();
        assert!((p.hz - 261.63).abs() < 1.0, "hz {}", p.hz);
        assert_eq!(p.note, "C4");
        assert!(p.clarity > 0.9);
        let mut rng = crate::util::Rng::new(1);
        let n: Vec<f64> = (0..48000).map(|_| rng.bipolar()).collect();
        let pn = detect_pitch(&n, sr);
        assert!(pn.map_or(true, |p| p.tonal_fraction < 0.2));
    }

    #[test]
    fn bands_and_centroid() {
        let sr = 48000.0;
        let x: Vec<f64> = (0..48000).map(|i| (TAU * 100.0 * i as f64 / sr).sin()).collect();
        let s = stft(&x, 4096, 2048, sr);
        let avg = average_spectrum(&s);
        let shares = band_shares(&avg, s.bin_hz());
        assert!(shares[1] > 0.95, "bass share {}", shares[1]);
        let shape = spectral_shape(&avg, s.bin_hz());
        assert!((shape.centroid_hz - 100.0).abs() < 30.0, "centroid {}", shape.centroid_hz);
        assert!(shape.flatness < 0.01);
    }

    #[test]
    fn tempo_of_a_click_track() {
        let sr = 48000.0;
        let mut x = vec![0.0; 48000 * 8];
        let beat = (sr * 0.5) as usize; // 120 bpm
        let mut rng = crate::util::Rng::new(2);
        for b in 0..16 {
            for i in 0..2000 {
                x[b * beat + i] += rng.bipolar() * (-(i as f64) / 300.0).exp();
            }
        }
        let r = detect_rhythm(&x, sr);
        assert!(r.onsets.len() >= 14 && r.onsets.len() <= 18, "onsets {}", r.onsets.len());
        let bpm = r.tempo_bpm.unwrap();
        assert!((bpm - 120.0).abs() < 2.0, "bpm {bpm}");
    }

    #[test]
    fn key_of_a_c_major_scale() {
        let sr = 48000.0;
        let notes = [60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60, 67, 60];
        let mut x = Vec::new();
        for m in notes {
            let f = crate::util::midi_to_hz(m as f64);
            x.extend((0..24000).map(|i| (TAU * f * i as f64 / sr).sin() * 0.3));
        }
        let k = detect_key(&x, sr).unwrap();
        assert_eq!(k.name, "C major");
    }
}
