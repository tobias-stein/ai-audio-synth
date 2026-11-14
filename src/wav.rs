//! WAV reading and writing (PCM 16/24-bit and 32-bit float), including the
//! `smpl` chunk so loop points survive import into game audio middleware.

use crate::audio::Sig;
use crate::error::{Error, Result};
use crate::util::Rng;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SampleFormat {
    Pcm16,
    Pcm24,
    Float32,
}

impl SampleFormat {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "16" | "pcm16" | "wav16" => Ok(SampleFormat::Pcm16),
            "24" | "pcm24" | "wav24" => Ok(SampleFormat::Pcm24),
            "32f" | "float" | "float32" | "f32" => Ok(SampleFormat::Float32),
            _ => Err(Error::usage(format!("unknown sample format '{s}'")).hint("use one of: 16, 24, 32f")),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            SampleFormat::Pcm16 => "pcm16",
            SampleFormat::Pcm24 => "pcm24",
            SampleFormat::Float32 => "float32",
        }
    }
    fn bits(&self) -> u16 {
        match self {
            SampleFormat::Pcm16 => 16,
            SampleFormat::Pcm24 => 24,
            SampleFormat::Float32 => 32,
        }
    }
}

/// Write `sig` to `path`. If `loop_points` is set, a `smpl` chunk marks the
/// sample range as a forward loop. PCM output is TPDF-dithered with a fixed
/// seed so files are reproducible.
pub fn write_wav(path: &Path, sig: &Sig, sample_rate: u32, fmt: SampleFormat, loop_points: Option<(u32, u32)>) -> Result<()> {
    let ch = sig.channels() as u16;
    let n = sig.len();
    let bytes_per = (fmt.bits() / 8) as usize;
    let data_len = n * ch as usize * bytes_per;
    let mut data = Vec::with_capacity(data_len);
    let mut rng = Rng::new(0x5EED_D17E);
    let chans = sig.chans();
    for i in 0..n {
        for c in &chans {
            let x = c[i];
            match fmt {
                SampleFormat::Float32 => data.extend_from_slice(&(x as f32).to_le_bytes()),
                SampleFormat::Pcm16 => {
                    let d = rng.uniform() - rng.uniform();
                    let v = (x * 32767.0 + d).round().clamp(-32768.0, 32767.0) as i16;
                    data.extend_from_slice(&v.to_le_bytes());
                }
                SampleFormat::Pcm24 => {
                    let d = rng.uniform() - rng.uniform();
                    let v = (x * 8_388_607.0 + d).round().clamp(-8_388_608.0, 8_388_607.0) as i32;
                    data.extend_from_slice(&v.to_le_bytes()[..3]);
                }
            }
        }
    }
    let mut out = Vec::with_capacity(data_len + 128);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&[0, 0, 0, 0]); // patched below
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    let tag: u16 = if fmt == SampleFormat::Float32 { 3 } else { 1 };
    out.extend_from_slice(&tag.to_le_bytes());
    out.extend_from_slice(&ch.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    let block = ch * fmt.bits() / 8;
    out.extend_from_slice(&(sample_rate * block as u32).to_le_bytes());
    out.extend_from_slice(&block.to_le_bytes());
    out.extend_from_slice(&fmt.bits().to_le_bytes());
    if fmt == SampleFormat::Float32 {
        out.extend_from_slice(b"fact");
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&(n as u32).to_le_bytes());
    }
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    out.extend_from_slice(&data);
    if data_len % 2 == 1 {
        out.push(0);
    }
    if let Some((start, end)) = loop_points {
        out.extend_from_slice(b"smpl");
        out.extend_from_slice(&60u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // manufacturer
        out.extend_from_slice(&0u32.to_le_bytes()); // product
        out.extend_from_slice(&(1_000_000_000u32 / sample_rate).to_le_bytes()); // sample period ns
        out.extend_from_slice(&60u32.to_le_bytes()); // MIDI unity note
        out.extend_from_slice(&0u32.to_le_bytes()); // pitch fraction
        out.extend_from_slice(&0u32.to_le_bytes()); // SMPTE format
        out.extend_from_slice(&0u32.to_le_bytes()); // SMPTE offset
        out.extend_from_slice(&1u32.to_le_bytes()); // number of loops
        out.extend_from_slice(&0u32.to_le_bytes()); // sampler data
        out.extend_from_slice(&0u32.to_le_bytes()); // cue point id
        out.extend_from_slice(&0u32.to_le_bytes()); // type: forward
        out.extend_from_slice(&start.to_le_bytes());
        out.extend_from_slice(&end.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // fraction
        out.extend_from_slice(&0u32.to_le_bytes()); // play count: infinite
    }
    let riff_len = (out.len() - 8) as u32;
    out[4..8].copy_from_slice(&riff_len.to_le_bytes());
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(dir).map_err(|e| Error::io(format!("cannot create directory {}: {e}", dir.display())))?;
        }
    }
    fs::write(path, out).map_err(|e| Error::io(format!("cannot write {}: {e}", path.display())))
}

pub struct WavData {
    pub sig: Sig,
    pub sample_rate: u32,
    pub bits: u16,
    pub float: bool,
    pub loop_points: Option<(u32, u32)>,
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}
fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

pub fn read_wav(path: &Path) -> Result<WavData> {
    let b = fs::read(path).map_err(|e| Error::io(format!("cannot read {}: {e}", path.display())))?;
    let bad = |m: &str| Error::new("unsupported_format", format!("{}: {m}", path.display())).hint("aas reads uncompressed WAV files (PCM 8/16/24/32-bit or float 32/64-bit)");
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }
    let mut pos = 12;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // tag, channels, rate, bits
    let mut data: Option<&[u8]> = None;
    let mut loop_points = None;
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let len = u32_at(&b, pos + 4) as usize;
        let start = pos + 8;
        let end = (start + len).min(b.len());
        let body = &b[start..end];
        match id {
            b"fmt " if body.len() >= 16 => {
                let mut tag = u16_at(body, 0);
                if tag == 0xFFFE && body.len() >= 26 {
                    tag = u16_at(body, 24); // WAVE_FORMAT_EXTENSIBLE sub-format
                }
                fmt = Some((tag, u16_at(body, 2), u32_at(body, 4), u16_at(body, 14)));
            }
            b"data" => data = Some(body),
            b"smpl" if body.len() >= 60 && u32_at(body, 28) >= 1 => {
                loop_points = Some((u32_at(body, 44), u32_at(body, 48)));
            }
            _ => {}
        }
        pos = start + len + (len & 1);
    }
    let (tag, ch, rate, bits) = fmt.ok_or_else(|| bad("missing fmt chunk"))?;
    let data = data.ok_or_else(|| bad("missing data chunk"))?;
    if ch == 0 || rate == 0 {
        return Err(bad("invalid channel count or sample rate"));
    }
    let bytes = (bits / 8) as usize;
    let frame = bytes * ch as usize;
    if frame == 0 {
        return Err(bad("invalid bit depth"));
    }
    let n = data.len() / frame;
    let decode = |s: &[u8]| -> Option<f64> {
        Some(match (tag, bits) {
            (1, 8) => (s[0] as f64 - 128.0) / 128.0,
            (1, 16) => i16::from_le_bytes([s[0], s[1]]) as f64 / 32768.0,
            (1, 24) => (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f64 / 8_388_608.0,
            (1, 32) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64 / 2_147_483_648.0,
            (3, 32) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64,
            (3, 64) => f64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]),
            _ => return None,
        })
    };
    if decode(&[0u8; 8]).is_none() {
        return Err(bad(&format!("unsupported encoding (format tag {tag}, {bits}-bit)")));
    }
    let mut l = Vec::with_capacity(n);
    let mut r = if ch >= 2 { Some(Vec::with_capacity(n)) } else { None };
    for i in 0..n {
        let f = &data[i * frame..(i + 1) * frame];
        l.push(decode(&f[0..bytes]).unwrap());
        if let Some(r) = r.as_mut() {
            r.push(decode(&f[bytes..2 * bytes]).unwrap());
        }
    }
    Ok(WavData { sig: Sig { l, r }, sample_rate: rate, bits, float: tag == 3, loop_points })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let n = 1000;
        let l: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).sin() * 0.5).collect();
        let r: Vec<f64> = (0..n).map(|i| (i as f64 * 0.07).cos() * 0.25).collect();
        let sig = Sig::stereo(l.clone(), r);
        let dir = std::env::temp_dir();
        for (fmt, tol) in [(SampleFormat::Pcm16, 1e-4), (SampleFormat::Pcm24, 1e-6), (SampleFormat::Float32, 1e-7)] {
            let p = dir.join(format!("aas_rt_{}.wav", fmt.name()));
            write_wav(&p, &sig, 48000, fmt, Some((10, 900))).unwrap();
            let w = read_wav(&p).unwrap();
            assert_eq!(w.sample_rate, 48000);
            assert_eq!(w.sig.len(), n);
            assert!(w.sig.is_stereo());
            assert_eq!(w.loop_points, Some((10, 900)));
            for i in 0..n {
                assert!((w.sig.l[i] - l[i]).abs() < tol, "{fmt:?} sample {i}");
            }
            let _ = std::fs::remove_file(p);
        }
    }
}
