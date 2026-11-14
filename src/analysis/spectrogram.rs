//! Spectrogram image, for agents that can look at pictures.

use super::spectral::stft;
use crate::audio::Sig;
use crate::error::{Error, Result};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

fn colormap(x: f64) -> [u8; 3] {
    // black -> indigo -> magenta -> orange -> pale yellow
    const STOPS: [(f64, [f64; 3]); 5] = [(0.0, [0.0, 0.0, 0.0]), (0.3, [50.0, 15.0, 110.0]), (0.55, [180.0, 40.0, 110.0]), (0.8, [250.0, 140.0, 40.0]), (1.0, [255.0, 250.0, 200.0])];
    let x = x.clamp(0.0, 1.0);
    for w in STOPS.windows(2) {
        if x <= w[1].0 {
            let t = (x - w[0].0) / (w[1].0 - w[0].0);
            let mut c = [0u8; 3];
            for k in 0..3 {
                c[k] = (w[0].1[k] + (w[1].1[k] - w[0].1[k]) * t).round() as u8;
            }
            return c;
        }
    }
    [255, 250, 200]
}

/// Write a log-frequency spectrogram (20 Hz at the bottom to 20 kHz at the
/// top, time left to right, 80 dB range). Returns (width, height).
pub fn write_png(sig: &Sig, sr: u32, path: &Path, width: usize, height: usize) -> Result<(usize, usize)> {
    let mono = sig.to_mono();
    let size = 4096;
    let hop = (mono.len() / width.max(1)).clamp(64, 2048);
    let s = stft(&mono, size, hop, sr as f64);
    let width = width.min(s.frames.len()).max(1);
    let bin_hz = s.bin_hz();
    let f_hi = 20000f64.min(sr as f64 / 2.0);
    let mut peak = 1e-12f64;
    for f in &s.frames {
        for p in f {
            peak = peak.max(*p);
        }
    }
    let mut img = vec![0u8; width * height * 3];
    for xpix in 0..width {
        let fa = xpix * s.frames.len() / width;
        let fb = (((xpix + 1) * s.frames.len()) / width).max(fa + 1).min(s.frames.len());
        for ypix in 0..height {
            let frac_lo = (height - 1 - ypix) as f64 / height as f64;
            let frac_hi = (height - ypix) as f64 / height as f64;
            let lo = 20.0 * (f_hi / 20.0).powf(frac_lo);
            let hi = 20.0 * (f_hi / 20.0).powf(frac_hi);
            let (a, b) = ((lo / bin_hz).floor() as usize, ((hi / bin_hz).ceil() as usize).min(size / 2));
            let mut p = 0.0f64;
            for f in &s.frames[fa..fb] {
                for v in &f[a..=b.max(a)] {
                    p = p.max(*v);
                }
            }
            let db = 10.0 * (p / peak + 1e-12).log10();
            let c = colormap((db + 80.0) / 80.0);
            let o = (ypix * width + xpix) * 3;
            img[o..o + 3].copy_from_slice(&c);
        }
    }
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| Error::io(format!("cannot create directory {}: {e}", dir.display())))?;
        }
    }
    let file = File::create(path).map_err(|e| Error::io(format!("cannot write {}: {e}", path.display())))?;
    let mut enc = png::Encoder::new(BufWriter::new(file), width as u32, height as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| Error::io(format!("cannot write PNG: {e}")))?;
    writer.write_image_data(&img).map_err(|e| Error::io(format!("cannot write PNG: {e}")))?;
    Ok((width, height))
}
