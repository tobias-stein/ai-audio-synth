//! Scores: music described as instruments, patterns, sections and an
//! arrangement, rendered through a mixer with sends, ducking and mastering.

use crate::analysis::loudness;
use crate::audio::Sig;
use crate::doc::{self, DocKind};
use crate::dsp::dynamics::limiter;
use crate::dsp::fx::pan as pan_fx;
use crate::dsp::resample::true_peak;
use crate::dsp::P;
use crate::error::{Error, Result};
use crate::graph::{as_obj, check_keys, get_f64, get_str, render_graph, type_name, Patch, RenderOpts};
use crate::render::render_note;
use crate::theory::{parse_duration, parse_notes, parse_steps, Chord, Key, NoteEvent, NOTE_SYNTAX_HINT};
use crate::util::{db_to_lin, hash_str, hint_from, lin_to_db, midi_to_hz, midi_to_name, round_to, Rng};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

pub struct KitPiece {
    pub name: String,
    pub patch: Rc<Patch>,
    pub params: Vec<(String, f64)>,
    pub gain: f64,
    pub pan: f64,
}

pub enum Voice {
    Patch { patch: Rc<Patch>, params: Vec<(String, f64)> },
    Kit(Vec<KitPiece>),
}

pub struct Duck {
    /// Instrument name, or `instrument.piece` for one drum of a kit.
    pub source: String,
    pub amount_db: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
}

pub struct Instrument {
    pub name: String,
    pub voice: Voice,
    pub gain: f64,
    pub pan: f64,
    pub transpose: i32,
    pub fx: Option<Patch>,
    pub sends: Vec<(String, f64)>,
    pub duck: Option<Duck>,
    pub mute: bool,
}

pub struct Bus {
    pub name: String,
    pub fx: Option<Patch>,
    pub gain: f64,
}

pub struct Master {
    pub fx: Option<Patch>,
    pub lufs: Option<f64>,
    pub ceiling_db: f64,
    pub fade_in: f64,
    pub fade_out: f64,
}

pub enum PatternKind {
    Notes(Vec<NoteEvent>),
    /// (piece name, beat, velocity)
    Drums(Vec<(String, f64, f64)>),
}

pub struct Pattern {
    pub name: String,
    pub kind: PatternKind,
    pub length: f64,
}

pub struct Clip {
    pub pattern: String,
    pub transpose: i32,
    pub vel: f64,
}

pub struct Section {
    pub name: String,
    pub bars: f64,
    /// instrument name -> clips played one after another, repeating to fill the section
    pub tracks: Vec<(String, Vec<Clip>)>,
}

pub struct Score {
    pub title: String,
    pub tempo: f64,
    pub beats_per_bar: f64,
    pub time_signature: String,
    pub key: Option<Key>,
    pub scale_lock: bool,
    pub swing: f64,
    pub swing_unit: f64,
    pub humanize_ms: f64,
    pub humanize_vel: f64,
    pub seed: u64,
    pub tail: f64,
    pub instruments: Vec<Instrument>,
    pub buses: Vec<Bus>,
    pub master: Master,
    pub patterns: Vec<Pattern>,
    pub sections: Vec<Section>,
    pub arrangement: Vec<String>,
    pub warnings: Vec<String>,
}

const SCORE_KEYS: &[&str] = &["kind", "title", "description", "tempo", "time_signature", "key", "scale_lock", "swing", "swing_unit", "humanize", "seed", "tail", "instruments", "buses", "master", "patterns", "sections", "arrangement"];
const INSTRUMENT_KEYS: &[&str] = &["patch", "kit", "params", "gain_db", "pan", "transpose", "fx", "sends", "duck", "mute"];
const PATTERN_KEYS: &[&str] = &["notes", "events", "drums", "chords", "step", "length", "transpose", "octave", "vel", "voicing", "style", "rate", "gate"];

fn load_patch(v: &Value, base_dir: &Path) -> Result<Patch> {
    match v {
        Value::String(s) => {
            let d = doc::load_from(s, base_dir)?;
            if d.kind != DocKind::Patch {
                return Err(Error::invalid(format!("'{s}' is not a patch document")));
            }
            Patch::from_value(&d.value).map_err(|e| Error { message: format!("in {}: {}", d.origin, e.message), ..e })
        }
        Value::Object(_) => Patch::from_value(v),
        other => Err(Error::invalid(format!("a patch must be a path, a builtin:<id> reference or an inline patch mapping, found {}", type_name(other))).hint("e.g. patch: builtin:instruments/bass_sub")),
    }
}

fn parse_params(v: Option<&Value>) -> Result<Vec<(String, f64)>> {
    let mut out = Vec::new();
    if let Some(v) = v {
        let o = as_obj(v, "'params'")?;
        for k in o.keys() {
            let val = get_f64(o, k)?.ok_or_else(|| Error::invalid("parameter value must be a number").at(k.clone()))?;
            out.push((k.clone(), val));
        }
    }
    Ok(out)
}

/// Check parameter overrides against the patch they target.
fn check_params(patch: &Patch, params: &[(String, f64)]) -> Result<()> {
    for (k, _) in params {
        if k != "freq" && k != "vel" && !patch.params.iter().any(|p| &p.name == k) {
            let names: Vec<&str> = patch.params.iter().map(|p| p.name.as_str()).collect();
            return Err(Error::new("unknown_name", format!("the patch has no parameter '{k}'")).at(k.clone()).hint(if names.is_empty() { "this patch exposes no parameters".to_string() } else { hint_from(k, &names) }));
        }
    }
    Ok(())
}

/// Build an effect-chain patch from a list of node mappings; each node's
/// input is the previous node (or the incoming audio for the first).
pub fn parse_fx_chain(v: &Value) -> Result<Option<Patch>> {
    let arr = match v {
        Value::Null => return Ok(None),
        Value::Array(a) => a,
        other => return Err(Error::invalid(format!("'fx' must be a list of effect nodes, found {}", type_name(other))).hint("e.g. fx: [{type: lowpass, cutoff: 2000}, {type: reverb, decay: 2}]")),
    };
    if arr.is_empty() {
        return Ok(None);
    }
    let mut nodes = Map::new();
    for (i, item) in arr.iter().enumerate() {
        let mut o = as_obj(item, "an effect").map_err(|e| e.at(i.to_string()))?.clone();
        if !o.contains_key("in") && o.get("type").and_then(|t| t.as_str()) != Some("mix") {
            o.insert("in".into(), json!(if i == 0 { "input".to_string() } else { format!("fx{}", i - 1) }));
        }
        nodes.insert(format!("fx{i}"), Value::Object(o));
    }
    let doc = json!({ "nodes": nodes, "output": format!("fx{}", arr.len() - 1), "normalize": "none" });
    let patch = Patch::from_value(&doc).map_err(|mut e| {
        // "nodes.fx2.cutoff" -> "2.cutoff"
        if let Some(p) = e.path.take() {
            e.path = Some(p.trim_start_matches("nodes.fx").to_string());
        }
        e
    })?;
    Ok(Some(patch))
}

fn db(o: &Map<String, Value>, key: &str, default: f64) -> Result<f64> {
    let v = get_f64(o, key)?.unwrap_or(default);
    if !(-120.0..=24.0).contains(&v) {
        return Err(Error::invalid(format!("{key} {v} is out of range")).at(key).hint("use a gain between -120 and +24 dB"));
    }
    Ok(v)
}

fn pan_of(o: &Map<String, Value>) -> Result<f64> {
    let v = get_f64(o, "pan")?.unwrap_or(0.0);
    if !(-1.0..=1.0).contains(&v) {
        return Err(Error::invalid(format!("pan {v} is out of range")).at("pan").hint("pan runs from -1 (left) to 1 (right)"));
    }
    Ok(v)
}

fn parse_instrument(name: &str, v: &Value, base_dir: &Path) -> Result<Instrument> {
    let o = as_obj(v, "an instrument")?;
    check_keys(o, INSTRUMENT_KEYS)?;
    let voice = match (o.get("patch"), o.get("kit")) {
        (Some(p), None) => {
            let patch = load_patch(p, base_dir).map_err(|e| e.within("patch"))?;
            let params = parse_params(o.get("params")).map_err(|e| e.within("params"))?;
            check_params(&patch, &params).map_err(|e| e.within("params"))?;
            Voice::Patch { patch: Rc::new(patch), params }
        }
        (None, Some(k)) => {
            let ko = as_obj(k, "'kit'").map_err(|e| e.at("kit"))?;
            if ko.is_empty() {
                return Err(Error::invalid("kit is empty").at("kit"));
            }
            let mut pieces = Vec::new();
            for (pname, pv) in ko {
                let path = format!("kit.{pname}");
                let piece = (|| -> Result<KitPiece> {
                    match pv {
                        Value::Object(po) if po.contains_key("patch") => {
                            check_keys(po, &["patch", "params", "gain_db", "pan"])?;
                            let patch = load_patch(&po["patch"], base_dir).map_err(|e| e.within("patch"))?;
                            let params = parse_params(po.get("params")).map_err(|e| e.within("params"))?;
                            check_params(&patch, &params).map_err(|e| e.within("params"))?;
                            Ok(KitPiece { name: pname.clone(), patch: Rc::new(patch), params, gain: db_to_lin(db(po, "gain_db", 0.0)?), pan: pan_of(po)? })
                        }
                        other => Ok(KitPiece { name: pname.clone(), patch: Rc::new(load_patch(other, base_dir)?), params: Vec::new(), gain: 1.0, pan: 0.0 }),
                    }
                })()
                .map_err(|e| e.within(&path))?;
                pieces.push(piece);
            }
            Voice::Kit(pieces)
        }
        _ => return Err(Error::invalid("an instrument needs exactly one of 'patch' or 'kit'").hint("e.g. {patch: builtin:instruments/bass_sub} or {kit: {kick: builtin:drums/kick, snare: builtin:drums/snare}}")),
    };
    let mut sends = Vec::new();
    if let Some(sv) = o.get("sends") {
        let so = as_obj(sv, "'sends'").map_err(|e| e.at("sends"))?;
        for k in so.keys() {
            sends.push((k.clone(), db_to_lin(db(so, k, 0.0).map_err(|e| e.within("sends"))?)));
        }
    }
    let duck = match o.get("duck") {
        None | Some(Value::Null) => None,
        Some(dv) => {
            let d = as_obj(dv, "'duck'").map_err(|e| e.at("duck"))?;
            check_keys(d, &["source", "amount_db", "attack_ms", "release_ms"]).map_err(|e| e.within("duck"))?;
            Some(Duck {
                source: get_str(d, "source").map_err(|e| e.within("duck"))?.ok_or_else(|| Error::invalid("duck needs a 'source'").at("duck.source").hint("e.g. source: drums.kick"))?.to_string(),
                amount_db: get_f64(d, "amount_db").map_err(|e| e.within("duck"))?.unwrap_or(6.0).abs(),
                attack_ms: get_f64(d, "attack_ms").map_err(|e| e.within("duck"))?.unwrap_or(5.0).max(0.1),
                release_ms: get_f64(d, "release_ms").map_err(|e| e.within("duck"))?.unwrap_or(180.0).max(1.0),
            })
        }
    };
    Ok(Instrument {
        name: name.to_string(),
        voice,
        gain: db_to_lin(db(o, "gain_db", 0.0)?),
        pan: pan_of(o)?,
        transpose: get_f64(o, "transpose")?.unwrap_or(0.0) as i32,
        fx: match o.get("fx") {
            Some(f) => parse_fx_chain(f).map_err(|e| e.within("fx"))?,
            None => None,
        },
        sends,
        duck,
        mute: o.get("mute").and_then(|m| m.as_bool()).unwrap_or(false),
    })
}

fn parse_pattern(name: &str, v: &Value, beats_per_bar: f64) -> Result<Pattern> {
    // shorthand: a bare string is a note pattern
    let shorthand;
    let o = match v {
        Value::String(s) => {
            let mut m = Map::new();
            m.insert("notes".into(), Value::String(s.clone()));
            shorthand = m;
            &shorthand
        }
        other => as_obj(other, "a pattern")?,
    };
    check_keys(o, PATTERN_KEYS)?;
    let step_default = if o.contains_key("drums") { 0.25 } else { 1.0 };
    let step = get_f64(o, "step")?.unwrap_or(step_default);
    if !(step > 0.0 && step <= 64.0) {
        return Err(Error::invalid("step must be above 0 beats").at("step"));
    }
    let vel = get_f64(o, "vel")?.unwrap_or(0.8);
    if !(0.0..=1.0).contains(&vel) {
        return Err(Error::invalid("vel must be between 0 and 1").at("vel"));
    }
    let shift = (get_f64(o, "transpose")?.unwrap_or(0.0) + 12.0 * get_f64(o, "octave")?.filter(|_| !o.contains_key("chords")).unwrap_or(0.0)) as i32;
    let gate = get_f64(o, "gate")?.unwrap_or(1.0).clamp(0.05, 4.0);
    let kinds = ["notes", "events", "drums", "chords"].iter().filter(|k| o.contains_key(**k)).count();
    if kinds != 1 {
        return Err(Error::invalid("a pattern needs exactly one of 'notes', 'events', 'drums' or 'chords'").hint("e.g. {notes: \"C3:1 Eb3:1 G3:2\"} or {drums: {kick: \"x...x...\"}} or {chords: \"Cm:4 Ab:4\"}"));
    }
    let (kind, natural_len) = if let Some(nv) = o.get("notes") {
        let text = match nv {
            Value::String(s) => s.clone(),
            Value::Array(a) => a.iter().map(|x| x.as_str().map(String::from).ok_or_else(|| Error::invalid("each entry of 'notes' must be a string").at("notes"))).collect::<Result<Vec<_>>>()?.join(" | "),
            other => return Err(Error::invalid(format!("'notes' must be a string, found {}", type_name(other))).at("notes").hint(NOTE_SYNTAX_HINT)),
        };
        let (mut ev, len) = parse_notes(&text, step, vel, beats_per_bar).map_err(|e| e.at("notes"))?;
        for e in ev.iter_mut() {
            e.dur *= gate;
            for n in e.notes.iter_mut() {
                *n += shift;
            }
        }
        (PatternKind::Notes(ev), len)
    } else if let Some(ev) = o.get("events") {
        let arr = ev.as_array().ok_or_else(|| Error::invalid("'events' must be a list").at("events").hint("e.g. events: [{at: 0, note: C4, dur: 1, vel: 0.8}]"))?;
        let mut out = Vec::new();
        let mut end = 0.0f64;
        for (i, item) in arr.iter().enumerate() {
            let path = format!("events.{i}");
            let eo = as_obj(item, "an event").map_err(|e| e.at(path.clone()))?;
            check_keys(eo, &["at", "note", "dur", "vel"]).map_err(|e| e.within(&path))?;
            let at = get_f64(eo, "at").map_err(|e| e.within(&path))?.ok_or_else(|| Error::invalid("event needs 'at' (beats from the pattern start)").at(path.clone()))?;
            let dur = get_f64(eo, "dur").map_err(|e| e.within(&path))?.unwrap_or(step);
            if at < 0.0 || dur <= 0.0 {
                return Err(Error::invalid("'at' must be 0 or more and 'dur' above 0").at(path));
            }
            let names: Vec<String> = match eo.get("note") {
                Some(Value::String(s)) => vec![s.clone()],
                Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
                Some(Value::Number(n)) => vec![midi_to_name(n.as_f64().unwrap_or(60.0) as i32)],
                _ => return Err(Error::invalid("event needs 'note': a note name such as C4, or a list of names").at(path)),
            };
            let mut notes = Vec::new();
            for nm in &names {
                notes.push(crate::util::parse_note(nm).ok_or_else(|| Error::invalid(format!("cannot read note '{nm}'")).at(path.clone()).hint("note names look like C4, F#3, Bb2"))? + shift);
            }
            let v = get_f64(eo, "vel").map_err(|e| e.within(&path))?.unwrap_or(vel).clamp(0.0, 1.0);
            end = end.max(at + dur);
            out.push(NoteEvent { at, dur: dur * gate, notes, vel: v });
        }
        out.sort_by(|a, b| a.at.partial_cmp(&b.at).unwrap());
        (PatternKind::Notes(out), (end / beats_per_bar).ceil().max(1.0) * beats_per_bar)
    } else if let Some(dv) = o.get("drums") {
        let dobj = as_obj(dv, "'drums'").map_err(|e| e.at("drums"))?;
        let mut hits = Vec::new();
        let mut steps = 0usize;
        for (piece, sv) in dobj {
            let s = sv.as_str().ok_or_else(|| Error::invalid("drum steps must be a string such as \"x...x...\"").at(format!("drums.{piece}")))?;
            let (h, n) = parse_steps(s).map_err(|e| e.at(format!("drums.{piece}")))?;
            if steps != 0 && n != steps {
                return Err(Error::invalid(format!("'{piece}' has {n} steps but earlier rows have {steps}")).at(format!("drums.{piece}")).hint("all rows of a drum pattern must have the same number of steps"));
            }
            steps = n;
            for (i, v) in h {
                hits.push((piece.clone(), i as f64 * step, v * vel / 0.8));
            }
        }
        (PatternKind::Drums(hits), steps as f64 * step)
    } else {
        let text = get_str(o, "chords")?.ok_or_else(|| Error::invalid("'chords' must be a string such as \"Cm:4 Ab:4 Eb:4 Bb:4\"").at("chords"))?;
        let octave = get_f64(o, "octave")?.unwrap_or(3.0) as i32;
        let voicing = get_str(o, "voicing")?.unwrap_or("smooth");
        let style = get_str(o, "style")?.unwrap_or("block");
        const VOICINGS: [&str; 3] = ["smooth", "close", "open"];
        const STYLES: [&str; 6] = ["block", "arp_up", "arp_down", "arp_updown", "strum", "root"];
        if !VOICINGS.contains(&voicing) {
            return Err(Error::invalid(format!("unknown voicing '{voicing}'")).at("voicing").hint(hint_from(voicing, &VOICINGS)));
        }
        if !STYLES.contains(&style) {
            return Err(Error::invalid(format!("unknown style '{style}'")).at("style").hint(hint_from(style, &STYLES)));
        }
        let rate = get_f64(o, "rate")?.unwrap_or(0.5);
        if rate <= 0.0 {
            return Err(Error::invalid("rate must be above 0 beats").at("rate"));
        }
        let mut ev = Vec::new();
        let mut pos = 0.0;
        let mut center = ((octave + 1) * 12 + 7) as f64;
        for tok in text.replace('|', " ").split_whitespace() {
            let (rest, v) = match tok.split_once('@') {
                Some((r, v)) => (r, v.parse::<f64>().ok().filter(|v| (0.0..=1.0).contains(v)).ok_or_else(|| Error::invalid(format!("bad velocity in '{tok}'")).at("chords"))?),
                None => (tok, vel),
            };
            let (sym, dur) = match rest.rsplit_once(':') {
                Some((s, d)) => (s, parse_duration(d).ok_or_else(|| Error::invalid(format!("bad duration in '{tok}'")).at("chords").hint("chord tokens look like Cm7:4 (symbol, colon, beats)"))?),
                None => (rest, beats_per_bar),
            };
            if sym != "_" {
                let chord = Chord::parse(sym).map_err(|e| e.at("chords"))?;
                let mut notes = match voicing {
                    "close" => chord.close(octave),
                    "open" => chord.open(octave),
                    _ => chord.near(center, octave),
                };
                if voicing == "smooth" {
                    center = 0.5 * center + 0.5 * (notes.iter().sum::<i32>() as f64 / notes.len() as f64);
                }
                for n in notes.iter_mut() {
                    *n += shift;
                }
                match style {
                    "block" => ev.push(NoteEvent { at: pos, dur: dur * gate, notes, vel: v }),
                    "root" => ev.push(NoteEvent { at: pos, dur: dur * gate, notes: vec![(octave + 1) * 12 + chord.bass.unwrap_or(chord.root) + shift], vel: v }),
                    "strum" => {
                        for (i, n) in notes.iter().enumerate() {
                            let off = i as f64 * 0.03;
                            ev.push(NoteEvent { at: pos + off, dur: (dur - off).max(0.05) * gate, notes: vec![*n], vel: v });
                        }
                    }
                    _ => {
                        let seq: Vec<i32> = match style {
                            "arp_down" => notes.iter().rev().cloned().collect(),
                            "arp_updown" if notes.len() > 2 => notes.iter().cloned().chain(notes[1..notes.len() - 1].iter().rev().cloned()).collect(),
                            _ => notes.clone(),
                        };
                        let mut t = 0.0;
                        let mut i = 0;
                        while t < dur - 1e-9 {
                            ev.push(NoteEvent { at: pos + t, dur: rate.min(dur - t) * gate, notes: vec![seq[i % seq.len()]], vel: v });
                            t += rate;
                            i += 1;
                        }
                    }
                }
            }
            pos += dur;
        }
        (PatternKind::Notes(ev), pos)
    };
    let length = get_f64(o, "length")?.unwrap_or(natural_len);
    if !(length > 0.0) {
        return Err(Error::invalid("pattern is empty: it has no length").hint("add notes, or set `length` in beats"));
    }
    Ok(Pattern { name: name.to_string(), kind, length })
}

fn parse_clip(v: &Value) -> Result<Vec<Clip>> {
    match v {
        Value::String(s) => Ok(vec![Clip { pattern: s.clone(), transpose: 0, vel: 1.0 }]),
        Value::Array(a) => {
            let mut out = Vec::new();
            for (i, item) in a.iter().enumerate() {
                out.extend(parse_clip(item).map_err(|e| e.within(&i.to_string()))?);
            }
            if out.is_empty() {
                return Err(Error::invalid("pattern list is empty"));
            }
            Ok(out)
        }
        Value::Object(o) => {
            check_keys(o, &["pattern", "patterns", "transpose", "vel", "gain_db"])?;
            let transpose = get_f64(o, "transpose")?.unwrap_or(0.0) as i32;
            let vel = get_f64(o, "vel")?.unwrap_or(1.0) * db_to_lin(get_f64(o, "gain_db")?.unwrap_or(0.0)).powf(1.0 / 1.5);
            let inner = o.get("pattern").or_else(|| o.get("patterns")).ok_or_else(|| Error::invalid("needs 'pattern'").hint("e.g. {pattern: bass_a, transpose: 12}"))?;
            let mut clips = parse_clip(inner)?;
            for c in clips.iter_mut() {
                c.transpose += transpose;
                c.vel *= vel;
            }
            Ok(clips)
        }
        other => Err(Error::invalid(format!("expected a pattern name, a list of names or a mapping, found {}", type_name(other)))),
    }
}

impl Score {
    pub fn from_value(v: &Value, base_dir: &Path) -> Result<Score> {
        let o = as_obj(v, "a score")?;
        check_keys(o, SCORE_KEYS)?;
        let mut warnings = Vec::new();
        let tempo = get_f64(o, "tempo")?.unwrap_or(120.0);
        if !(20.0..=400.0).contains(&tempo) {
            return Err(Error::invalid(format!("tempo {tempo} is out of range")).at("tempo").hint("tempo is in beats per minute, between 20 and 400"));
        }
        let ts = match o.get("time_signature") {
            None => "4/4".to_string(),
            Some(Value::String(s)) => s.clone(),
            Some(other) => return Err(Error::invalid(format!("time_signature must be a string such as \"4/4\", found {}", type_name(other))).at("time_signature").hint("quote it: time_signature: \"3/4\"")),
        };
        let beats_per_bar = ts
            .split_once('/')
            .and_then(|(a, b)| Some((a.trim().parse::<f64>().ok()?, b.trim().parse::<f64>().ok()?)))
            .filter(|(a, b)| *a >= 1.0 && *a <= 32.0 && [1.0, 2.0, 4.0, 8.0, 16.0].contains(b))
            .map(|(a, b)| a * 4.0 / b)
            .ok_or_else(|| Error::invalid(format!("cannot read time signature '{ts}'")).at("time_signature").hint("e.g. \"4/4\", \"3/4\", \"6/8\", \"7/8\""))?;
        let key = match get_str(o, "key")? {
            Some(k) => Some(Key::parse(k).map_err(|e| e.at("key"))?),
            None => None,
        };
        let (mut humanize_ms, mut humanize_vel) = (0.0, 0.0);
        if let Some(h) = o.get("humanize") {
            let ho = as_obj(h, "'humanize'").map_err(|e| e.at("humanize"))?;
            check_keys(ho, &["timing_ms", "velocity"]).map_err(|e| e.within("humanize"))?;
            humanize_ms = get_f64(ho, "timing_ms").map_err(|e| e.within("humanize"))?.unwrap_or(0.0).clamp(0.0, 50.0);
            humanize_vel = get_f64(ho, "velocity").map_err(|e| e.within("humanize"))?.unwrap_or(0.0).clamp(0.0, 0.5);
        }

        let inst_v = o.get("instruments").ok_or_else(|| Error::invalid("score is missing 'instruments'").hint("run `aas library show examples/minimal_score` for a starting point"))?;
        let iobj = as_obj(inst_v, "'instruments'").map_err(|e| e.at("instruments"))?;
        let mut instruments = Vec::new();
        for (name, iv) in iobj {
            instruments.push(parse_instrument(name, iv, base_dir).map_err(|e| e.within(&format!("instruments.{name}")))?);
        }
        if instruments.is_empty() {
            return Err(Error::invalid("score has no instruments").at("instruments"));
        }

        let mut buses = Vec::new();
        if let Some(bv) = o.get("buses") {
            for (name, b) in as_obj(bv, "'buses'").map_err(|e| e.at("buses"))? {
                let path = format!("buses.{name}");
                let bo = as_obj(b, "a bus").map_err(|e| e.at(path.clone()))?;
                check_keys(bo, &["fx", "gain_db"]).map_err(|e| e.within(&path))?;
                buses.push(Bus {
                    name: name.clone(),
                    fx: match bo.get("fx") {
                        Some(f) => parse_fx_chain(f).map_err(|e| e.within(&format!("{path}.fx")))?,
                        None => None,
                    },
                    gain: db_to_lin(db(bo, "gain_db", 0.0).map_err(|e| e.within(&path))?),
                });
            }
        }
        let bus_names: Vec<&str> = buses.iter().map(|b| b.name.as_str()).collect();
        for i in &instruments {
            for (s, _) in &i.sends {
                if !bus_names.contains(&s.as_str()) {
                    return Err(Error::new("unknown_name", format!("instrument '{}' sends to bus '{s}', which is not defined", i.name))
                        .at(format!("instruments.{}.sends.{s}", i.name))
                        .hint(if bus_names.is_empty() { "add a `buses:` section, e.g. buses: {hall: {fx: [{type: reverb, decay: 3, mix: 1}]}}".to_string() } else { hint_from(s, &bus_names) }));
                }
            }
            if let Some(d) = &i.duck {
                let (src_inst, piece) = match d.source.split_once('.') {
                    Some((a, b)) => (a, Some(b)),
                    None => (d.source.as_str(), None),
                };
                let names: Vec<&str> = instruments.iter().map(|x| x.name.as_str()).collect();
                let src = instruments.iter().find(|x| x.name == src_inst).ok_or_else(|| Error::new("unknown_name", format!("duck source '{}' is not an instrument", d.source)).at(format!("instruments.{}.duck.source", i.name)).hint(hint_from(src_inst, &names)))?;
                if src.name == i.name {
                    return Err(Error::invalid("an instrument cannot duck itself").at(format!("instruments.{}.duck.source", i.name)));
                }
                if let Some(p) = piece {
                    let ok = matches!(&src.voice, Voice::Kit(k) if k.iter().any(|x| x.name == p));
                    if !ok {
                        return Err(Error::new("unknown_name", format!("duck source '{}': instrument '{src_inst}' has no kit piece '{p}'", d.source)).at(format!("instruments.{}.duck.source", i.name)));
                    }
                }
            }
        }

        let master = {
            let empty = Map::new();
            let mo = match o.get("master") {
                Some(m) => as_obj(m, "'master'").map_err(|e| e.at("master"))?,
                None => &empty,
            };
            check_keys(mo, &["fx", "lufs", "ceiling_db", "fade_in", "fade_out"]).map_err(|e| e.within("master"))?;
            let lufs = match mo.get("lufs") {
                Some(Value::String(s)) if s == "none" => None,
                Some(Value::Bool(false)) => None,
                _ => Some(get_f64(mo, "lufs").map_err(|e| e.within("master"))?.unwrap_or(-16.0)),
            };
            if let Some(l) = lufs {
                if !(-60.0..=-5.0).contains(&l) {
                    return Err(Error::invalid(format!("master lufs {l} is out of range")).at("master.lufs").hint("typical targets: -16 for game music, -14 for streaming, -23 for broadcast"));
                }
            }
            Master {
                fx: match mo.get("fx") {
                    Some(f) => parse_fx_chain(f).map_err(|e| e.within("master.fx"))?,
                    None => None,
                },
                lufs,
                ceiling_db: get_f64(mo, "ceiling_db").map_err(|e| e.within("master"))?.unwrap_or(-1.0).clamp(-20.0, 0.0),
                fade_in: get_f64(mo, "fade_in").map_err(|e| e.within("master"))?.unwrap_or(0.0).max(0.0),
                fade_out: get_f64(mo, "fade_out").map_err(|e| e.within("master"))?.unwrap_or(0.0).max(0.0),
            }
        };

        let mut patterns = Vec::new();
        if let Some(pv) = o.get("patterns") {
            for (name, p) in as_obj(pv, "'patterns'").map_err(|e| e.at("patterns"))? {
                patterns.push(parse_pattern(name, p, beats_per_bar).map_err(|e| e.within(&format!("patterns.{name}")))?);
            }
        }
        let scale_lock = o.get("scale_lock").and_then(|b| b.as_bool()).unwrap_or(false);
        if let Some(k) = &key {
            for p in patterns.iter_mut() {
                if let PatternKind::Notes(ev) = &mut p.kind {
                    let mut off: Vec<String> = Vec::new();
                    for e in ev.iter_mut() {
                        for n in e.notes.iter_mut() {
                            if !k.contains(*n) {
                                if scale_lock {
                                    *n = k.snap(*n);
                                } else if off.len() < 6 && !off.contains(&midi_to_name(*n)) {
                                    off.push(midi_to_name(*n));
                                }
                            }
                        }
                    }
                    if !off.is_empty() {
                        warnings.push(format!("pattern '{}' uses notes outside {}: {} (fine if intentional; set `scale_lock: true` to snap them into the key)", p.name, k.name(), off.join(", ")));
                    }
                }
            }
        }

        let sec_v = o.get("sections").ok_or_else(|| Error::invalid("score is missing 'sections'").hint("e.g. sections: {main: {bars: 4, tracks: {bass: bass_a}}}"))?;
        let pat_names: Vec<&str> = patterns.iter().map(|p| p.name.as_str()).collect();
        let inst_names: Vec<&str> = instruments.iter().map(|i| i.name.as_str()).collect();
        let mut sections = Vec::new();
        for (name, sv) in as_obj(sec_v, "'sections'").map_err(|e| e.at("sections"))? {
            let path = format!("sections.{name}");
            let so = as_obj(sv, "a section").map_err(|e| e.at(path.clone()))?;
            check_keys(so, &["bars", "tracks"]).map_err(|e| e.within(&path))?;
            let bars = get_f64(so, "bars").map_err(|e| e.within(&path))?.ok_or_else(|| Error::invalid("section needs 'bars'").at(path.clone()))?;
            if !(bars > 0.0 && bars <= 2000.0) {
                return Err(Error::invalid("bars must be above 0").at(format!("{path}.bars")));
            }
            let mut tracks = Vec::new();
            if let Some(tv) = so.get("tracks") {
                for (inst, cv) in as_obj(tv, "'tracks'").map_err(|e| e.at(format!("{path}.tracks")))? {
                    let tpath = format!("{path}.tracks.{inst}");
                    let Some(instrument) = instruments.iter().find(|i| &i.name == inst) else {
                        return Err(Error::new("unknown_name", format!("section '{name}' uses instrument '{inst}', which is not defined")).at(tpath).hint(hint_from(inst, &inst_names)));
                    };
                    let clips = parse_clip(cv).map_err(|e| e.within(&tpath))?;
                    for c in &clips {
                        let Some(p) = patterns.iter().find(|p| p.name == c.pattern) else {
                            return Err(Error::new("unknown_name", format!("pattern '{}' is not defined", c.pattern)).at(tpath).hint(hint_from(&c.pattern, &pat_names)));
                        };
                        match (&p.kind, &instrument.voice) {
                            (PatternKind::Drums(hits), Voice::Kit(pieces)) => {
                                let piece_names: Vec<&str> = pieces.iter().map(|x| x.name.as_str()).collect();
                                for (piece, _, _) in hits {
                                    if !piece_names.contains(&piece.as_str()) {
                                        return Err(Error::new("unknown_name", format!("pattern '{}' plays drum '{piece}' but kit '{inst}' has no such piece", p.name)).at(format!("patterns.{}.drums.{piece}", p.name)).hint(hint_from(piece, &piece_names)));
                                    }
                                }
                            }
                            (PatternKind::Notes(_), Voice::Patch { .. }) => {}
                            (PatternKind::Drums(_), Voice::Patch { .. }) => {
                                return Err(Error::invalid(format!("pattern '{}' is a drum pattern but instrument '{inst}' is not a kit", p.name)).at(tpath).hint("give the instrument a `kit:`, or use a `notes:` pattern"));
                            }
                            (PatternKind::Notes(_), Voice::Kit(_)) => {
                                return Err(Error::invalid(format!("pattern '{}' is a note pattern but instrument '{inst}' is a drum kit", p.name)).at(tpath).hint("use a `drums:` pattern for kits, e.g. {drums: {kick: \"x...x...\"}}"));
                            }
                        }
                    }
                    tracks.push((inst.clone(), clips));
                }
            }
            sections.push(Section { name: name.clone(), bars, tracks });
        }
        if sections.is_empty() {
            return Err(Error::invalid("score has no sections").at("sections"));
        }
        let sec_names: Vec<&str> = sections.iter().map(|s| s.name.as_str()).collect();
        let mut arrangement = Vec::new();
        match o.get("arrangement") {
            None => arrangement = sections.iter().map(|s| s.name.clone()).collect(),
            Some(Value::Array(a)) => {
                for (i, item) in a.iter().enumerate() {
                    let (name, repeat) = match item {
                        Value::String(s) => (s.clone(), 1),
                        Value::Object(io) => {
                            check_keys(io, &["section", "repeat"]).map_err(|e| e.within(&format!("arrangement.{i}")))?;
                            (get_str(io, "section")?.ok_or_else(|| Error::invalid("needs 'section'").at(format!("arrangement.{i}")))?.to_string(), get_f64(io, "repeat")?.unwrap_or(1.0).max(1.0) as usize)
                        }
                        other => return Err(Error::invalid(format!("arrangement entries are section names, found {}", type_name(other))).at(format!("arrangement.{i}"))),
                    };
                    if !sec_names.contains(&name.as_str()) {
                        return Err(Error::new("unknown_name", format!("arrangement uses section '{name}', which is not defined")).at(format!("arrangement.{i}")).hint(hint_from(&name, &sec_names)));
                    }
                    for _ in 0..repeat {
                        arrangement.push(name.clone());
                    }
                }
            }
            Some(other) => return Err(Error::invalid(format!("'arrangement' must be a list of section names, found {}", type_name(other))).at("arrangement")),
        }
        if arrangement.is_empty() {
            return Err(Error::invalid("arrangement is empty").at("arrangement"));
        }
        for i in &instruments {
            if !sections.iter().any(|s| arrangement.contains(&s.name) && s.tracks.iter().any(|t| t.0 == i.name)) {
                warnings.push(format!("instrument '{}' is never played by any section in the arrangement", i.name));
            }
        }
        for p in &patterns {
            if !sections.iter().any(|s| s.tracks.iter().any(|t| t.1.iter().any(|c| c.pattern == p.name))) {
                warnings.push(format!("pattern '{}' is defined but never used", p.name));
            }
        }
        for s in &sections {
            let len = s.bars * beats_per_bar;
            for (inst, clips) in &s.tracks {
                let total: f64 = clips.iter().map(|c| patterns.iter().find(|p| p.name == c.pattern).unwrap().length).sum();
                let reps = len / total;
                if (reps - reps.round()).abs() > 1e-6 {
                    warnings.push(format!("section '{}', instrument '{inst}': the patterns span {total} beats, which does not divide the section's {len} beats evenly, so the last repeat is cut short", s.name));
                }
            }
        }
        Ok(Score {
            title: get_str(o, "title")?.unwrap_or("").to_string(),
            tempo,
            beats_per_bar,
            time_signature: ts,
            key,
            scale_lock,
            swing: get_f64(o, "swing")?.unwrap_or(0.0).clamp(0.0, 1.0),
            swing_unit: get_f64(o, "swing_unit")?.unwrap_or(0.5).clamp(0.0625, 4.0),
            humanize_ms,
            humanize_vel,
            seed: get_f64(o, "seed")?.unwrap_or(1.0) as u64,
            tail: get_f64(o, "tail")?.unwrap_or(4.0).clamp(0.0, 60.0),
            instruments,
            buses,
            master,
            patterns,
            sections,
            arrangement,
            warnings,
        })
    }

    pub fn total_beats(&self, arrangement: &[String]) -> f64 {
        arrangement.iter().map(|n| self.sections.iter().find(|s| &s.name == n).map_or(0.0, |s| s.bars * self.beats_per_bar)).sum()
    }

    pub fn beat_seconds(&self) -> f64 {
        60.0 / self.tempo
    }
}

/// A note or drum hit placed on the timeline, in seconds.
struct Hit {
    at: f64,
    dur: f64,
    midi: i32,
    piece: Option<usize>,
    vel: f64,
}

#[derive(Clone, Debug)]
pub struct ScoreOpts {
    pub sr: u32,
    pub oversample: usize,
    pub seed: Option<u64>,
    pub looped: bool,
    /// Render only these sections, in this order, instead of the arrangement.
    pub arrangement: Option<Vec<String>>,
    /// Use this master gain instead of measuring loudness (keeps separately
    /// rendered sections at consistent levels).
    pub fixed_gain: Option<f64>,
}

pub struct RenderedScore {
    pub master: Sig,
    /// Musical length in samples (the loop length when looping).
    pub musical_samples: usize,
    pub master_gain: f64,
    pub limiter_gr_db: f64,
    pub warnings: Vec<String>,
    pub timeline: Vec<Value>,
    pub note_count: usize,
}

fn collect_hits(score: &Score, inst: &Instrument, arrangement: &[String], seed: u64) -> Vec<Hit> {
    let spb = score.beat_seconds();
    let mut rng = Rng::new(seed ^ hash_str(&inst.name));
    let mut hits = Vec::new();
    let mut section_start = 0.0f64;
    let swing_time = |beat: f64| -> f64 {
        if score.swing <= 0.0 {
            return beat;
        }
        let u = score.swing_unit;
        let p = beat.rem_euclid(2.0 * u);
        if p >= u - 1e-9 {
            beat + score.swing * u / 3.0 * ((2.0 * u - p) / u).clamp(0.0, 1.0)
        } else {
            beat
        }
    };
    for sec_name in arrangement {
        let Some(sec) = score.sections.iter().find(|s| &s.name == sec_name) else { continue };
        let sec_len = sec.bars * score.beats_per_bar;
        if let Some((_, clips)) = sec.tracks.iter().find(|t| t.0 == inst.name) {
            let mut pos = 0.0f64;
            let mut ci = 0usize;
            while pos < sec_len - 1e-9 {
                let clip = &clips[ci % clips.len()];
                let pat = score.patterns.iter().find(|p| p.name == clip.pattern).unwrap();
                let mut place = |at_beats: f64, dur_beats: f64, midi: i32, piece: Option<usize>, vel: f64| {
                    if at_beats >= sec_len - 1e-9 {
                        return;
                    }
                    let jitter = if score.humanize_ms > 0.0 { rng.gauss() * 0.5 * score.humanize_ms * 0.001 } else { 0.0 };
                    let vj = if score.humanize_vel > 0.0 { rng.gauss() * 0.5 * score.humanize_vel } else { 0.0 };
                    let at = (swing_time(section_start + at_beats) * spb + jitter).max(0.0);
                    hits.push(Hit { at, dur: (dur_beats * spb).max(0.01), midi, piece, vel: (vel * clip.vel + vj).clamp(0.02, 1.2) });
                };
                match (&pat.kind, &inst.voice) {
                    (PatternKind::Notes(ev), _) => {
                        for e in ev {
                            if e.at >= pat.length - 1e-9 {
                                continue;
                            }
                            for n in &e.notes {
                                place(pos + e.at, e.dur, n + clip.transpose + inst.transpose, None, e.vel);
                            }
                        }
                    }
                    (PatternKind::Drums(dh), Voice::Kit(pieces)) => {
                        for (piece, at, vel) in dh {
                            if let Some(pi) = pieces.iter().position(|p| &p.name == piece) {
                                place(pos + at, 0.25, 60, Some(pi), *vel);
                            }
                        }
                    }
                    _ => {}
                }
                pos += pat.length;
                ci += 1;
            }
        }
        section_start += sec_len;
    }
    hits.sort_by(|a, b| a.at.partial_cmp(&b.at).unwrap());
    hits
}

fn apply_fx(fx: &Option<Patch>, sig: Sig, sr: u32, seed: u64, what: &str) -> Result<Sig> {
    match fx {
        None => Ok(sig),
        Some(p) => {
            let mut ro = RenderOpts::new(sr, 1, seed);
            ro.input = Some(Rc::new(sig));
            render_graph(p, &ro).map_err(|e| e.within(what))
        }
    }
}

/// Envelope of a signal, stored at 1/32 rate, used as a ducking key.
fn key_envelope(sig: &Sig, sr: u32, attack_ms: f64, release_ms: f64) -> Vec<f64> {
    let hop = 32usize;
    let step_s = hop as f64 / sr as f64;
    let a = (-step_s / (attack_ms * 0.001)).exp();
    let r = (-step_s / (release_ms * 0.001)).exp();
    let right = sig.right();
    let mut env = 0.0f64;
    let mut out = Vec::with_capacity(sig.len() / hop + 1);
    let mut i = 0;
    while i < sig.len() {
        let end = (i + hop).min(sig.len());
        let mut pk = 0.0f64;
        for k in i..end {
            pk = pk.max(sig.l[k].abs()).max(right[k].abs());
        }
        let c = if pk > env { a } else { r };
        env = c * env + (1.0 - c) * pk;
        out.push(env);
        i = end;
    }
    let peak = out.iter().cloned().fold(0.0, f64::max);
    if peak > 0.0 {
        for v in out.iter_mut() {
            *v = (*v / peak).min(1.0);
        }
    }
    out
}

fn wrap_tail(sig: &Sig, loop_len: usize) -> Sig {
    sig.map_channels(|_, x| {
        let mut out = x[..loop_len.min(x.len())].to_vec();
        out.resize(loop_len, 0.0);
        for (i, v) in x.iter().enumerate().skip(loop_len) {
            out[i % loop_len] += v;
        }
        out
    })
}

/// Render the score. `stem_sink`, when given, receives every instrument and
/// bus signal (post-fader, pre-master, already scaled by the master gain).
pub fn render_score(score: &Score, o: &ScoreOpts, mut stem_sink: Option<&mut dyn FnMut(&str, &Sig) -> Result<()>>) -> Result<RenderedScore> {
    let sr = o.sr;
    let srf = sr as f64;
    let seed = o.seed.unwrap_or(score.seed);
    let arrangement: Vec<String> = o.arrangement.clone().unwrap_or_else(|| score.arrangement.clone());
    let musical_s = score.total_beats(&arrangement) * score.beat_seconds();
    let musical_samples = (musical_s * srf).round() as usize;
    let total = musical_samples + (score.tail * srf) as usize;
    if total as f64 / srf > 900.0 {
        return Err(Error::invalid(format!("the arrangement is {:.0} s long; the limit is 15 minutes", total as f64 / srf)).at("arrangement"));
    }
    let mut warnings = Vec::new();
    let mut note_count = 0usize;

    // which signals are needed as ducking keys
    let duck_sources: Vec<&str> = score.instruments.iter().filter_map(|i| i.duck.as_ref().map(|d| d.source.as_str())).collect();
    let mut keys: HashMap<String, Sig> = HashMap::new();

    // instruments that others duck against must be rendered first
    let mut order: Vec<usize> = (0..score.instruments.len()).collect();
    order.sort_by_key(|&i| {
        let name = &score.instruments[i].name;
        let is_source = duck_sources.iter().any(|s| s.split('.').next() == Some(name.as_str()));
        (if is_source { 0 } else { 1 }, i)
    });

    let mut mix = Sig::silence(total, true);
    let mut bus_in: Vec<Sig> = score.buses.iter().map(|_| Sig::silence(total, true)).collect();
    let mut pending_stems: Vec<(String, Sig)> = Vec::new();
    let keep_stems = stem_sink.is_some();

    for &ii in &order {
        let inst = &score.instruments[ii];
        if inst.mute {
            continue;
        }
        let hits = collect_hits(score, inst, &arrangement, seed);
        if hits.is_empty() {
            continue;
        }
        note_count += hits.len();
        let mut track = Sig::silence(total, true);
        let mut piece_tracks: HashMap<usize, Sig> = HashMap::new();
        // Identical notes are rendered a few times with different seeds and then
        // reused in rotation: natural variation without re-rendering every hit.
        let mut cache: HashMap<(Option<usize>, i32, u32, u64), Vec<Rc<Sig>>> = HashMap::new();
        for h in hits.iter() {
            let offset = (h.at * srf).round() as usize;
            if offset >= total {
                continue;
            }
            let vel = (h.vel * 24.0).round().max(1.0) / 24.0;
            let gate_ms = (h.dur * 1000.0).round() as u64;
            let key = (h.piece, h.midi, (vel * 24.0).round() as u32, gate_ms);
            let variants = if h.piece.is_some() { 4 } else { 2 };
            let entry = cache.entry(key).or_default();
            let (uses, have) = (entry.len(), entry.iter().filter(|s| !s.is_empty()).count());
            let _ = uses;
            let sig: Rc<Sig> = if have < variants {
                let note_seed = seed ^ hash_str(&inst.name) ^ (h.midi as u64).wrapping_mul(0x9E37_79B9) ^ (gate_ms << 20) ^ ((key.2 as u64) << 40) ^ (h.piece.unwrap_or(0) as u64).wrapping_mul(0xC2B2_AE35) ^ ((have as u64 + 1) << 56);
                let s = match (&inst.voice, h.piece) {
                    (Voice::Patch { patch, params }, _) => {
                        if !(0..=127).contains(&h.midi) {
                            continue;
                        }
                        render_note(patch, Some(midi_to_hz(h.midi as f64)), vel, gate_ms as f64 / 1000.0, note_seed, sr, o.oversample, params).map_err(|e| e.within(&format!("instruments.{}.patch", inst.name)))?
                    }
                    (Voice::Kit(pieces), Some(pi)) => {
                        let piece = &pieces[pi];
                        let s = render_note(&piece.patch, None, vel, gate_ms as f64 / 1000.0, note_seed, sr, o.oversample, &piece.params).map_err(|e| e.within(&format!("instruments.{}.kit.{}", inst.name, piece.name)))?;
                        if piece.pan != 0.0 {
                            pan_fx(&s, &P::C(piece.pan))
                        } else {
                            s
                        }
                    }
                    _ => continue,
                };
                let rc = Rc::new(s);
                entry.push(rc.clone());
                rc
            } else {
                // rotate through the rendered variants
                let rc = entry.remove(0);
                entry.push(rc.clone());
                rc
            };
            match (&inst.voice, h.piece) {
                (Voice::Kit(pieces), Some(pi)) => {
                    let piece = &pieces[pi];
                    mix_clipped(&mut track, &sig, offset, piece.gain, total);
                    let key_name = format!("{}.{}", inst.name, piece.name);
                    if duck_sources.contains(&key_name.as_str()) {
                        let pt = piece_tracks.entry(pi).or_insert_with(|| Sig::silence(total, false));
                        mix_clipped(pt, &Sig::mono(sig.to_mono()), offset, piece.gain, total);
                    }
                }
                _ => mix_clipped(&mut track, &sig, offset, 1.0, total),
            }
        }
        if let Voice::Kit(pieces) = &inst.voice {
            for (pi, s) in piece_tracks {
                keys.insert(format!("{}.{}", inst.name, pieces[pi].name), s);
            }
        }
        if duck_sources.contains(&inst.name.as_str()) {
            keys.insert(inst.name.clone(), track.clone());
        }
        let mut track = apply_fx(&inst.fx, track, sr, seed ^ hash_str(&inst.name), &format!("instruments.{}.fx", inst.name))?;
        if let Some(d) = &inst.duck {
            if let Some(k) = keys.get(&d.source) {
                let env = key_envelope(k, sr, d.attack_ms, d.release_ms);
                let n = track.len();
                let gain_at = |i: usize| -> f64 {
                    let p = i as f64 / 32.0;
                    let i0 = (p.floor() as usize).min(env.len().saturating_sub(1));
                    let i1 = (i0 + 1).min(env.len().saturating_sub(1));
                    let e = env[i0] + (env[i1] - env[i0]) * (p - p.floor());
                    db_to_lin(-d.amount_db * e)
                };
                for i in 0..n {
                    let g = gain_at(i);
                    track.l[i] *= g;
                    if let Some(r) = track.r.as_mut() {
                        r[i] *= g;
                    }
                }
            }
        }
        let mut track = if inst.pan != 0.0 { pan_fx(&track, &P::C(inst.pan)) } else { track.into_stereo() };
        track.scale(inst.gain);
        track.resize(total);
        for (bname, g) in &inst.sends {
            if let Some(bi) = score.buses.iter().position(|b| &b.name == bname) {
                bus_in[bi].mix_in(&track, 0, *g);
            }
        }
        mix.mix_in(&track, 0, 1.0);
        if keep_stems {
            pending_stems.push((inst.name.clone(), track));
        }
    }
    for (bi, bus) in score.buses.iter().enumerate() {
        let input = std::mem::take(&mut bus_in[bi]);
        if input.peak() == 0.0 {
            continue;
        }
        let mut out = apply_fx(&bus.fx, input, sr, seed ^ hash_str(&bus.name), &format!("buses.{}.fx", bus.name))?.into_stereo();
        out.scale(bus.gain);
        out.resize(total);
        mix.mix_in(&out, 0, 1.0);
        if keep_stems {
            pending_stems.push((format!("bus_{}", bus.name), out));
        }
    }
    if mix.peak() < 1e-9 {
        return Err(Error::new("render_error", "the score rendered to silence").hint("check that sections reference patterns with notes, that instruments are not muted and that gains are not extremely low"));
    }

    let mut master = apply_fx(&score.master.fx, mix, sr, seed ^ 0x4D41_5354, "master.fx")?.into_stereo();
    master.resize(total);
    if o.looped {
        master = wrap_tail(&master, musical_samples);
    } else {
        // keep the tail but drop trailing silence
        let keep = master.last_audible(-72.0).map(|i| i + (0.05 * srf) as usize).unwrap_or(musical_samples).max(musical_samples).min(total);
        master.resize(keep);
        let f = (0.02 * srf) as usize;
        master.fade(0, f.min(master.len()));
        if score.master.fade_in > 0.0 || score.master.fade_out > 0.0 {
            master.fade((score.master.fade_in * srf) as usize, (score.master.fade_out * srf) as usize);
        }
    }

    let ceiling = db_to_lin(score.master.ceiling_db);
    let gain = match (o.fixed_gain, score.master.lufs) {
        (Some(g), _) => g,
        (None, Some(target)) => {
            let l = loudness::measure(&master, sr).integrated;
            db_to_lin(target - l)
        }
        (None, None) => 1.0,
    };
    master.scale(gain);
    let mut gr = 0.0;
    if true_peak(&master) > ceiling {
        if o.looped {
            // limit two passes of the loop and keep the second, so the gain
            // curve is continuous across the loop point
            let doubled = master.map_channels(|_, x| x.iter().chain(x.iter()).cloned().collect());
            let (lim, g) = limiter(&doubled, score.master.ceiling_db, 120.0, 5.0, true, srf);
            master = lim.map_channels(|_, x| x[musical_samples..].to_vec());
            gr = g;
        } else {
            let (lim, g) = limiter(&master, score.master.ceiling_db, 120.0, 5.0, true, srf);
            master = lim;
            gr = g;
        }
        if gr > 6.0 {
            warnings.push(format!(
                "the limiter is reducing peaks by {:.1} dB to reach {} LUFS, which will audibly squash the mix: lower `master.lufs`, reduce the loudest instrument, or add a compressor to the master fx",
                gr,
                score.master.lufs.map_or("the target".to_string(), |l| l.to_string())
            ));
        }
    }

    if let Some(sink) = stem_sink.as_mut() {
        for (name, mut s) in pending_stems {
            if o.looped {
                s = wrap_tail(&s, musical_samples);
            } else {
                s.resize(master.len());
            }
            s.scale(gain);
            sink(&name, &s)?;
        }
    }

    // timeline for manifests
    let mut timeline = Vec::new();
    let mut beat = 0.0;
    for name in &arrangement {
        let sec = score.sections.iter().find(|s| &s.name == name).unwrap();
        timeline.push(json!({
            "section": name,
            "start_s": round_to(beat * score.beat_seconds(), 4),
            "start_bar": round_to(beat / score.beats_per_bar + 1.0, 3),
            "bars": sec.bars,
        }));
        beat += sec.bars * score.beats_per_bar;
    }
    Ok(RenderedScore { master, musical_samples, master_gain: gain, limiter_gr_db: gr, warnings, timeline, note_count })
}

fn mix_clipped(dst: &mut Sig, src: &Sig, offset: usize, gain: f64, total: usize) {
    if offset >= total {
        return;
    }
    let n = src.len().min(total - offset);
    if src.is_stereo() && !dst.is_stereo() {
        dst.r = Some(dst.l.clone());
    }
    for i in 0..n {
        dst.l[offset + i] += src.l[i] * gain;
    }
    if let Some(r) = dst.r.as_mut() {
        let sr = src.right();
        for i in 0..n {
            r[offset + i] += sr[i] * gain;
        }
    }
}

impl Score {
    /// Facts about the score for `validate` output.
    pub fn summary(&self) -> Value {
        let beats = self.total_beats(&self.arrangement);
        json!({
            "title": self.title,
            "tempo_bpm": self.tempo,
            "time_signature": self.time_signature,
            "key": self.key.as_ref().map(|k| k.name()),
            "scale_lock": self.scale_lock,
            "bars": round_to(beats / self.beats_per_bar, 3),
            "duration_s": round_to(beats * self.beat_seconds(), 3),
            "instruments": self.instruments.iter().map(|i| i.name.clone()).collect::<Vec<_>>(),
            "buses": self.buses.iter().map(|b| b.name.clone()).collect::<Vec<_>>(),
            "sections": self.sections.iter().map(|s| json!({"name": s.name, "bars": s.bars})).collect::<Vec<_>>(),
            "arrangement": self.arrangement,
            "master_lufs_target": self.master.lufs,
            "master_gain_note": format!("master gain is chosen to hit the LUFS target, then a true-peak limiter holds {} dBFS", lin_to_db(db_to_lin(self.master.ceiling_db)).round()),
        })
    }
}
