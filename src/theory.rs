//! Music theory helpers: keys and scales, chord symbols, voicings and the
//! compact note and drum notations used in scores.

use crate::error::{Error, Result};
use crate::util::{hint_from, midi_to_name, parse_note, parse_pitch_class, pitch_class_name};

pub const SCALES: &[(&str, &[i32])] = &[
    ("major", &[0, 2, 4, 5, 7, 9, 11]),
    ("minor", &[0, 2, 3, 5, 7, 8, 10]),
    ("harmonic_minor", &[0, 2, 3, 5, 7, 8, 11]),
    ("melodic_minor", &[0, 2, 3, 5, 7, 9, 11]),
    ("dorian", &[0, 2, 3, 5, 7, 9, 10]),
    ("phrygian", &[0, 1, 3, 5, 7, 8, 10]),
    ("lydian", &[0, 2, 4, 6, 7, 9, 11]),
    ("mixolydian", &[0, 2, 4, 5, 7, 9, 10]),
    ("locrian", &[0, 1, 3, 5, 6, 8, 10]),
    ("pentatonic_major", &[0, 2, 4, 7, 9]),
    ("pentatonic_minor", &[0, 3, 5, 7, 10]),
    ("blues", &[0, 3, 5, 6, 7, 10]),
    ("whole_tone", &[0, 2, 4, 6, 8, 10]),
    ("chromatic", &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
];

#[derive(Clone, Debug)]
pub struct Key {
    pub tonic: i32,
    pub scale_name: &'static str,
    pub intervals: &'static [i32],
}

impl Key {
    /// Parse "C minor", "F# major", "D dorian", "Am" or "Bb".
    pub fn parse(s: &str) -> Result<Key> {
        let t = s.trim();
        let bad = || Error::invalid(format!("cannot read key '{s}'")).hint(format!("write a tonic and a scale, e.g. \"C minor\" or \"F# dorian\". Scales: {}", SCALES.iter().map(|x| x.0).collect::<Vec<_>>().join(", ")));
        let (pc, used) = parse_pitch_class(t).ok_or_else(bad)?;
        let rest = t[used..].trim().to_lowercase().replace([' ', '-'], "_");
        let name = match rest.as_str() {
            "" | "maj" => "major",
            "m" | "min" => "minor",
            other => other,
        };
        let (scale_name, intervals) = SCALES.iter().find(|x| x.0 == name).map(|x| (x.0, x.1)).ok_or_else(|| {
            let names: Vec<&str> = SCALES.iter().map(|x| x.0).collect();
            Error::invalid(format!("unknown scale '{name}' in key '{s}'")).hint(hint_from(name, &names))
        })?;
        Ok(Key { tonic: pc.rem_euclid(12), scale_name, intervals })
    }
    pub fn name(&self) -> String {
        format!("{} {}", pitch_class_name(self.tonic), self.scale_name)
    }
    pub fn contains(&self, midi: i32) -> bool {
        self.intervals.contains(&(midi - self.tonic).rem_euclid(12))
    }
    /// Nearest note in the scale (ties resolve downward).
    pub fn snap(&self, midi: i32) -> i32 {
        for d in 0..7 {
            if self.contains(midi - d) {
                return midi - d;
            }
            if self.contains(midi + d) {
                return midi + d;
            }
        }
        midi
    }
    pub fn note_names(&self) -> Vec<String> {
        self.intervals.iter().map(|i| pitch_class_name(self.tonic + i).to_string()).collect()
    }
    /// Diatonic triads/sevenths built on each scale degree, as chord symbols.
    pub fn diatonic_chords(&self) -> Vec<String> {
        if self.intervals.len() != 7 {
            return Vec::new();
        }
        (0..7)
            .map(|d| {
                let at = |k: usize| self.intervals[(d + k) % 7] + if d + k >= 7 { 12 } else { 0 };
                let (third, fifth) = (at(2) - at(0), at(4) - at(0));
                let q = match (third, fifth) {
                    (4, 7) => "",
                    (3, 7) => "m",
                    (3, 6) => "dim",
                    (4, 8) => "aug",
                    _ => "?",
                };
                format!("{}{}", pitch_class_name(self.tonic + self.intervals[d]), q)
            })
            .collect()
    }
}

const CHORDS: &[(&str, &[i32])] = &[
    ("", &[0, 4, 7]),
    ("maj", &[0, 4, 7]),
    ("m", &[0, 3, 7]),
    ("min", &[0, 3, 7]),
    ("dim", &[0, 3, 6]),
    ("aug", &[0, 4, 8]),
    ("5", &[0, 7]),
    ("sus2", &[0, 2, 7]),
    ("sus4", &[0, 5, 7]),
    ("sus", &[0, 5, 7]),
    ("6", &[0, 4, 7, 9]),
    ("m6", &[0, 3, 7, 9]),
    ("7", &[0, 4, 7, 10]),
    ("maj7", &[0, 4, 7, 11]),
    ("M7", &[0, 4, 7, 11]),
    ("m7", &[0, 3, 7, 10]),
    ("min7", &[0, 3, 7, 10]),
    ("mMaj7", &[0, 3, 7, 11]),
    ("dim7", &[0, 3, 6, 9]),
    ("m7b5", &[0, 3, 6, 10]),
    ("7sus4", &[0, 5, 7, 10]),
    ("add9", &[0, 4, 7, 14]),
    ("madd9", &[0, 3, 7, 14]),
    ("9", &[0, 4, 7, 10, 14]),
    ("maj9", &[0, 4, 7, 11, 14]),
    ("m9", &[0, 3, 7, 10, 14]),
    ("11", &[0, 7, 10, 14, 17]),
    ("m11", &[0, 3, 7, 10, 14, 17]),
    ("13", &[0, 4, 7, 10, 14, 21]),
];

#[derive(Clone, Debug)]
pub struct Chord {
    pub root: i32,
    pub intervals: Vec<i32>,
    pub bass: Option<i32>,
    pub symbol: String,
}

impl Chord {
    /// Parse a chord symbol such as "Cm7", "F#maj7", "Bbsus4" or "C/E".
    pub fn parse(sym: &str) -> Result<Chord> {
        let bad = |why: &str| Error::invalid(format!("cannot read chord '{sym}': {why}")).hint(format!("write a root and a quality, e.g. C, Am, F#m7, Bbmaj7, Gsus4, C/E. Qualities: {}", CHORDS.iter().map(|c| if c.0.is_empty() { "(major)" } else { c.0 }).collect::<Vec<_>>().join(" ")));
        let (main, bass) = match sym.split_once('/') {
            Some((m, b)) => (m, Some(parse_pitch_class(b).filter(|x| x.1 == b.len()).ok_or_else(|| bad("the bass note after '/' is not a note name"))?.0)),
            None => (sym, None),
        };
        let (root, used) = parse_pitch_class(main).ok_or_else(|| bad("it does not start with a note name"))?;
        let quality = &main[used..];
        let intervals = CHORDS.iter().find(|c| c.0 == quality).ok_or_else(|| bad(&format!("unknown quality '{quality}'")))?.1.to_vec();
        Ok(Chord { root: root.rem_euclid(12), intervals, bass: bass.map(|b| b.rem_euclid(12)), symbol: sym.to_string() })
    }

    /// MIDI notes in close position with the root in `octave` (C4 = octave 4).
    pub fn close(&self, octave: i32) -> Vec<i32> {
        let base = (octave + 1) * 12 + self.root;
        let mut v: Vec<i32> = self.intervals.iter().map(|i| base + i).collect();
        if let Some(b) = self.bass {
            let mut bn = (octave + 1) * 12 + b;
            while bn >= base {
                bn -= 12;
            }
            v.insert(0, bn);
        }
        v
    }

    /// Open voicing: root an octave down, third raised an octave.
    pub fn open(&self, octave: i32) -> Vec<i32> {
        let mut v = self.close(octave);
        let has_bass = self.bass.is_some();
        let root_idx = if has_bass { 1 } else { 0 };
        if v.len() > root_idx + 2 {
            v[root_idx + 1] += 12;
            if !has_bass {
                v[root_idx] -= 12;
            }
        }
        v.sort();
        v
    }

    /// Inversion of the close voicing whose average pitch is nearest to `center`.
    pub fn near(&self, center: f64, octave: i32) -> Vec<i32> {
        let mut best: Option<(f64, Vec<i32>)> = None;
        for oct in octave - 1..=octave + 1 {
            let base: Vec<i32> = self.intervals.iter().map(|i| (oct + 1) * 12 + self.root + i).collect();
            for inv in 0..base.len() {
                let mut v = base.clone();
                for x in v.iter_mut().take(inv) {
                    *x += 12;
                }
                v.sort();
                let avg = v.iter().sum::<i32>() as f64 / v.len() as f64;
                let d = (avg - center).abs();
                if best.as_ref().map_or(true, |b| d < b.0) {
                    best = Some((d, v));
                }
            }
        }
        best.unwrap().1
    }

    pub fn note_names(&self, octave: i32) -> Vec<String> {
        self.close(octave).iter().map(|m| midi_to_name(*m)).collect()
    }
}

/// One step of a note pattern, in beats.
#[derive(Clone, Debug, PartialEq)]
pub struct NoteEvent {
    pub at: f64,
    pub dur: f64,
    pub notes: Vec<i32>,
    pub vel: f64,
}

pub fn parse_duration(s: &str) -> Option<f64> {
    if let Some((a, b)) = s.split_once('/') {
        let (a, b) = (a.parse::<f64>().ok()?, b.parse::<f64>().ok()?);
        if b > 0.0 && a > 0.0 {
            return Some(a / b);
        }
        return None;
    }
    s.parse::<f64>().ok().filter(|d| *d > 0.0)
}

/// Split a token into head, optional duration and optional velocity.
fn split_token(tok: &str) -> (&str, Option<&str>, Option<&str>) {
    let (rest, vel) = match tok.split_once('@') {
        Some((r, v)) => (r, Some(v)),
        None => (tok, None),
    };
    match rest.split_once(':') {
        Some((h, d)) => (h, Some(d), vel),
        None => (rest, None, vel),
    }
}

pub const NOTE_SYNTAX_HINT: &str = "tokens are separated by spaces: NOTE[:BEATS][@VELOCITY]. Examples: `C4` `Eb3:0.5` `F#5:2@0.6` `C4+E4+G4:4` (chord) `_:1` (rest) `~:1` (extend the previous note). `|` marks bar lines.";

/// Parse the note mini-notation. Returns the events and the total length in
/// beats. If bar lines are present, every bar must be `beats_per_bar` long.
pub fn parse_notes(src: &str, default_dur: f64, default_vel: f64, beats_per_bar: f64) -> Result<(Vec<NoteEvent>, f64)> {
    let mut events: Vec<NoteEvent> = Vec::new();
    let mut pos = 0.0f64;
    let mut bar_start = 0.0f64;
    let mut bar_no = 1;
    let has_bars = src.contains('|');
    let spaced = src.replace('|', " | ");
    let tokens: Vec<&str> = spaced.split_whitespace().collect();
    let check_bar = |pos: f64, bar_start: f64, bar_no: i32| -> Result<()> {
        let len = pos - bar_start;
        if (len - beats_per_bar).abs() > 1e-6 {
            return Err(Error::invalid(format!("bar {bar_no} is {len} beats long but the time signature needs {beats_per_bar}"))
                .hint(format!("{} the durations in bar {bar_no} so they add up to {beats_per_bar} beats", if len > beats_per_bar { "shorten" } else { "lengthen" })));
        }
        Ok(())
    };
    for (ti, tok) in tokens.iter().enumerate() {
        if *tok == "|" {
            if pos > bar_start {
                check_bar(pos, bar_start, bar_no)?;
                bar_no += 1;
            }
            bar_start = pos;
            continue;
        }
        let (head, dur_s, vel_s) = split_token(tok);
        let dur = match dur_s {
            Some(d) => parse_duration(d).ok_or_else(|| Error::invalid(format!("bad duration '{d}' in '{tok}'")).hint("durations are in beats: 1, 0.5, 1.5 or fractions like 1/3"))?,
            None => default_dur,
        };
        let vel = match vel_s {
            Some(v) => v.parse::<f64>().ok().filter(|v| (0.0..=1.0).contains(v)).ok_or_else(|| Error::invalid(format!("bad velocity '{v}' in '{tok}'")).hint("velocity is a number from 0 to 1, e.g. C4:1@0.7"))?,
            None => default_vel,
        };
        match head {
            "_" | "r" | "." => {}
            "~" => match events.last_mut() {
                Some(e) if (e.at + e.dur - pos).abs() < 1e-9 => e.dur += dur,
                _ => return Err(Error::invalid(format!("'~' at token {} has no note directly before it to extend", ti + 1)).hint("put ~ right after a note, or use a rest `_` instead")),
            },
            _ => {
                let mut notes = Vec::new();
                for part in head.trim_matches(|c| c == '[' || c == ']').split(['+', ',']) {
                    let m = parse_note(part).ok_or_else(|| Error::invalid(format!("cannot read note '{part}' in '{tok}'")).hint(NOTE_SYNTAX_HINT))?;
                    notes.push(m);
                }
                events.push(NoteEvent { at: pos, dur, notes, vel });
            }
        }
        pos += dur;
    }
    if has_bars && pos > bar_start {
        check_bar(pos, bar_start, bar_no)?;
    }
    Ok((events, pos))
}

/// Parse a drum step string: `x` hit, `X` accent, `o` ghost note, `.` rest.
/// Returns (step index, velocity) pairs and the number of steps.
pub fn parse_steps(src: &str) -> Result<(Vec<(usize, f64)>, usize)> {
    let mut hits = Vec::new();
    let mut i = 0usize;
    for c in src.chars() {
        match c {
            'x' => hits.push((i, 0.8)),
            'X' => hits.push((i, 1.0)),
            'o' => hits.push((i, 0.45)),
            '.' | '-' | '_' => {}
            ' ' | '|' => continue,
            other => return Err(Error::invalid(format!("unexpected character '{other}' in drum steps \"{src}\"")).hint("use x (hit), X (accent), o (ghost note) and . (rest); spaces and | are ignored")),
        }
        i += 1;
    }
    Ok((hits, i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys() {
        let k = Key::parse("C minor").unwrap();
        assert!(k.contains(63) && !k.contains(64));
        assert_eq!(k.snap(64), 63);
        assert_eq!(Key::parse("F# dorian").unwrap().name(), "F# dorian");
        assert_eq!(Key::parse("Am").unwrap().scale_name, "minor");
        assert!(Key::parse("C minro").unwrap_err().hint.unwrap().contains("minor"));
        assert_eq!(Key::parse("C major").unwrap().diatonic_chords(), vec!["C", "Dm", "Em", "F", "G", "Am", "Bdim"]);
    }

    #[test]
    fn chords() {
        assert_eq!(Chord::parse("Cm7").unwrap().close(4), vec![60, 63, 67, 70]);
        assert_eq!(Chord::parse("F#").unwrap().close(3), vec![54, 58, 61]);
        assert_eq!(Chord::parse("C/E").unwrap().close(4), vec![52, 60, 64, 67]);
        assert!(Chord::parse("Cxyz").is_err());
        let near = Chord::parse("G").unwrap().near(64.0, 4);
        assert!(near.iter().all(|n| (55..=74).contains(n)), "{near:?}");
    }

    #[test]
    fn note_notation() {
        let (ev, len) = parse_notes("C4 Eb4:0.5 _:0.5 G4:2@0.5 ~:1 C4+E4+G4:1", 1.0, 0.8, 4.0).unwrap();
        assert_eq!(len, 6.0);
        assert_eq!(ev.len(), 4);
        assert_eq!(ev[0], NoteEvent { at: 0.0, dur: 1.0, notes: vec![60], vel: 0.8 });
        assert_eq!(ev[2].dur, 3.0);
        assert_eq!(ev[2].vel, 0.5);
        assert_eq!(ev[3].notes, vec![60, 64, 67]);
        assert!(parse_notes("C4 H4", 1.0, 0.8, 4.0).is_err());
    }

    #[test]
    fn bars_are_checked() {
        assert!(parse_notes("C4:2 D4:2 | E4:4", 1.0, 0.8, 4.0).is_ok());
        let e = parse_notes("C4:2 D4:1 | E4:4", 1.0, 0.8, 4.0).unwrap_err();
        assert!(e.message.contains("bar 1"), "{}", e.message);
    }

    #[test]
    fn drum_steps() {
        let (hits, n) = parse_steps("x..o X...|x...").unwrap();
        assert_eq!(n, 12);
        assert_eq!(hits, vec![(0, 0.8), (3, 0.45), (4, 1.0), (8, 0.8)]);
        assert!(parse_steps("x.q.").is_err());
    }
}
