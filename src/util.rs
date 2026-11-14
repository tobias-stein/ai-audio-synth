//! Small shared helpers: deterministic RNG, unit conversions, name matching.

use std::f64::consts::PI;

/// xoshiro256++ seeded through splitmix64. Implemented here (rather than using a
/// crate) so that output is bit-identical across versions and platforms.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }
    pub fn next_u64(&mut self) -> u64 {
        let r = self.s[0].wrapping_add(self.s[3]).rotate_left(23).wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in [-1, 1).
    pub fn bipolar(&mut self) -> f64 {
        self.uniform() * 2.0 - 1.0
    }
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
    /// Approximately normal (sum of uniforms), mean 0, std 1.
    pub fn gauss(&mut self) -> f64 {
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.uniform();
        }
        s - 6.0
    }
}

/// Stable 64-bit FNV-1a hash, used to derive per-node seeds from names.
pub fn hash_str(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

pub fn db_to_lin(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

pub fn lin_to_db(x: f64) -> f64 {
    if x <= 1e-12 {
        -240.0
    } else {
        20.0 * x.log10()
    }
}

pub fn midi_to_hz(m: f64) -> f64 {
    440.0 * 2f64.powf((m - 69.0) / 12.0)
}

pub fn hz_to_midi(f: f64) -> f64 {
    69.0 + 12.0 * (f / 440.0).log2()
}

const NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

pub fn midi_to_name(m: i32) -> String {
    format!("{}{}", NAMES[m.rem_euclid(12) as usize], m.div_euclid(12) - 1)
}

pub fn pitch_class_name(pc: i32) -> &'static str {
    NAMES[pc.rem_euclid(12) as usize]
}

/// Parse a pitch-class name like "C", "F#", "Bb". Returns (pitch class, chars consumed).
pub fn parse_pitch_class(s: &str) -> Option<(i32, usize)> {
    let mut chars = s.chars();
    let base = match chars.next()?.to_ascii_uppercase() {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };
    let mut pc = base;
    let mut used = 1;
    for c in chars {
        match c {
            '#' | '♯' => pc += 1,
            'b' | '♭' => pc -= 1,
            _ => break,
        }
        used += c.len_utf8();
    }
    Some((pc, used))
}

/// Parse a note name like "C4", "F#2", "Bb-1" into a MIDI note number (C4 = 60).
pub fn parse_note(s: &str) -> Option<i32> {
    let s = s.trim();
    let (pc, used) = parse_pitch_class(s)?;
    let oct: i32 = s[used..].parse().ok()?;
    let m = (oct + 1) * 12 + pc;
    if (0..=135).contains(&m) {
        Some(m)
    } else {
        None
    }
}

pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Closest candidate to `name`, if any is reasonably close.
pub fn did_you_mean<'a, I: IntoIterator<Item = &'a str>>(name: &str, candidates: I) -> Option<String> {
    let lname = name.to_lowercase();
    let mut best: Option<(usize, &str)> = None;
    for c in candidates {
        let lc = c.to_lowercase();
        let mut d = levenshtein(&lname, &lc);
        if lc.contains(&lname) || lname.contains(&lc) {
            d = d.min(1);
        }
        if best.map_or(true, |(bd, _)| d < bd) {
            best = Some((d, c));
        }
    }
    let (d, c) = best?;
    if d <= (name.len() / 2).max(2) {
        Some(c.to_string())
    } else {
        None
    }
}

pub fn hint_from(name: &str, candidates: &[&str]) -> String {
    match did_you_mean(name, candidates.iter().copied()) {
        Some(s) => format!("did you mean '{}'? Valid: {}", s, candidates.join(", ")),
        None => format!("valid: {}", candidates.join(", ")),
    }
}

pub fn round_to(x: f64, digits: i32) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let p = 10f64.powi(digits);
    (x * p).round() / p
}

pub fn hann(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()).collect()
}

/// Modified Bessel function of the first kind, order zero (for Kaiser windows).
pub fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut k = 1.0;
    loop {
        term *= (x / (2.0 * k)) * (x / (2.0 * k));
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
        k += 1.0;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn notes() {
        assert_eq!(parse_note("C4"), Some(60));
        assert_eq!(parse_note("A4"), Some(69));
        assert_eq!(parse_note("Bb2"), Some(46));
        assert_eq!(parse_note("F#-1"), Some(6));
        assert_eq!(midi_to_name(61), "C#4");
        assert!((midi_to_hz(69.0) - 440.0).abs() < 1e-9);
    }
    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let u = a.uniform();
        assert!((0.0..1.0).contains(&u));
    }
    #[test]
    fn suggestions() {
        assert_eq!(did_you_mean("lowpas", ["lowpass", "highpass"]), Some("lowpass".into()));
        assert_eq!(did_you_mean("zzzzzzzz", ["lowpass", "highpass"]), None);
    }
}
