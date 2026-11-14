//! The node library: a data-driven registry (used for validation and for the
//! `aas nodes` documentation) plus the implementation of every node type.

use crate::audio::Sig;
use crate::dsp::dynamics::{compressor, limiter, CompArgs};
use crate::dsp::filter::{comb, dc_block, formant, svf_process, FilterMode};
use crate::dsp::fx::{bitcrush, chorus, delay, distort, flanger, pan, phaser, speed, transient, width, DelayArgs, Shape};
use crate::dsp::osc::{additive, dust, lfo, noise, osc, LfoWave, NoiseColor, OscArgs, Wave};
use crate::dsp::physical::{modal, modal_preset, pluck, ModalArgs, PluckArgs, MODAL_PRESETS};
use crate::dsp::resample::interpolate;
use crate::dsp::reverb::{reverb, ReverbArgs};
use crate::dsp::P;
use crate::error::{Error, Result};
use crate::expr::Val;
use crate::graph::{type_name, Args};
use crate::util::{db_to_lin, hint_from};
use serde_json::Value;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// Audio input: a node name or expression; may be stereo.
    Sig,
    /// Modulatable parameter: a number, or an expression evaluated per sample.
    Par,
    /// Constant number (expressions allowed, evaluated once).
    Num,
    Int,
    Bool,
    Enum(&'static [&'static str]),
    /// List of audio inputs.
    SigList,
    /// Node-specific structured data (lists of points, partials, modes).
    Raw,
}

impl Kind {
    pub fn label(&self) -> String {
        match self {
            Kind::Sig => "signal".into(),
            Kind::Par => "number or expression (modulatable per sample)".into(),
            Kind::Num => "number".into(),
            Kind::Int => "integer".into(),
            Kind::Bool => "true/false".into(),
            Kind::Enum(o) => format!("one of: {}", o.join(", ")),
            Kind::SigList => "list of signals".into(),
            Kind::Raw => "structured".into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParamSpec {
    pub name: &'static str,
    pub kind: Kind,
    pub default: Option<&'static str>,
    pub required: bool,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub doc: &'static str,
}

#[derive(Clone, Debug)]
pub struct NodeSpec {
    pub name: &'static str,
    pub category: &'static str,
    pub doc: &'static str,
    pub params: Vec<ParamSpec>,
    pub example: &'static str,
}

#[derive(Clone, Debug)]
pub enum NodeData {
    /// (ratio, amplitude, decay seconds)
    Partials(Vec<(f64, f64, f64)>),
    /// (ratio, gain, decay multiplier)
    Modes(Vec<(f64, f64, f64)>),
    /// (time, value)
    Points(Vec<(f64, f64)>),
}

fn input() -> ParamSpec {
    ParamSpec { name: "in", kind: Kind::Sig, default: None, required: true, min: None, max: None, doc: "input signal: a node name or an expression such as \"osc1 * env\"" }
}
fn sig_opt(name: &'static str, doc: &'static str) -> ParamSpec {
    ParamSpec { name, kind: Kind::Sig, default: None, required: false, min: None, max: None, doc }
}
fn par(name: &'static str, default: &'static str, min: f64, max: f64, doc: &'static str) -> ParamSpec {
    ParamSpec { name, kind: Kind::Par, default: Some(default), required: false, min: Some(min), max: Some(max), doc }
}
fn par_req(name: &'static str, min: f64, max: f64, doc: &'static str) -> ParamSpec {
    ParamSpec { name, kind: Kind::Par, default: None, required: true, min: Some(min), max: Some(max), doc }
}
fn par_opt(name: &'static str, doc: &'static str) -> ParamSpec {
    ParamSpec { name, kind: Kind::Par, default: None, required: false, min: None, max: None, doc }
}
fn num(name: &'static str, default: &'static str, min: f64, max: f64, doc: &'static str) -> ParamSpec {
    ParamSpec { name, kind: Kind::Num, default: Some(default), required: false, min: Some(min), max: Some(max), doc }
}
fn num_req(name: &'static str, min: f64, max: f64, doc: &'static str) -> ParamSpec {
    ParamSpec { name, kind: Kind::Num, default: None, required: true, min: Some(min), max: Some(max), doc }
}
fn int(name: &'static str, default: &'static str, min: f64, max: f64, doc: &'static str) -> ParamSpec {
    ParamSpec { name, kind: Kind::Int, default: Some(default), required: false, min: Some(min), max: Some(max), doc }
}
fn flag(name: &'static str, default: bool, doc: &'static str) -> ParamSpec {
    ParamSpec { name, kind: Kind::Bool, default: Some(if default { "true" } else { "false" }), required: false, min: None, max: None, doc }
}
fn choice(name: &'static str, options: &'static [&'static str], default: &'static str, doc: &'static str) -> ParamSpec {
    ParamSpec { name, kind: Kind::Enum(options), default: Some(default), required: false, min: None, max: None, doc }
}
fn amp() -> ParamSpec {
    par("amp", "1", -100.0, 100.0, "output level multiplier; use an envelope here to shape loudness, e.g. amp: env")
}
fn freq() -> ParamSpec {
    par("freq", "freq", 0.0, 20000.0, "frequency in Hz; defaults to the note frequency `freq`")
}

const F: f64 = 1e9;

fn build_specs() -> Vec<NodeSpec> {
    let s = |name, category, doc, params, example| NodeSpec { name, category, doc, params, example };
    vec![
        // ---- sources
        s("osc", "source", "Band-limited oscillator. With unison > 1 it becomes a detuned stereo stack (supersaw). `pm` gives FM-style phase modulation.", vec![
            choice("wave", &Wave::NAMES, "sine", "waveform"),
            freq(), amp(),
            par("pw", "0.5", 0.02, 0.98, "pulse width for wave: pulse (0.5 = square)"),
            par_opt("pm", "phase modulation input in radians, e.g. \"mod * 3\" for FM with index 3"),
            int("unison", "1", 1.0, 16.0, "number of detuned voices"),
            num("detune", "20", 0.0, 200.0, "total unison detune spread in cents"),
            num("spread", "0.7", 0.0, 1.0, "unison stereo spread"),
            num("phase", "0", 0.0, 1.0, "start phase in cycles (unison voices add a random offset)"),
        ], "{type: osc, wave: saw, freq: \"freq * 0.5\", unison: 5, detune: 25}"),
        s("noise", "source", "White, pink or brown noise.", vec![
            choice("color", &NoiseColor::NAMES, "white", "spectral tilt: white is bright, pink is balanced, brown is rumbly"),
            amp(),
            flag("stereo", false, "independent noise per channel (wide) instead of mono"),
        ], "{type: noise, color: pink, amp: env}"),
        s("dust", "source", "Random impulses, for crackle, rain, fire, geiger and vinyl textures.", vec![
            par("density", "20", 0.0, 50000.0, "average impulses per second"),
            amp(),
            flag("bipolar", true, "impulses of both polarities"),
        ], "{type: dust, density: \"5 + 200 * env\"}"),
        s("impulse", "source", "A single-sample click; the usual exciter for modal and comb resonators.", vec![
            num("at", "0", 0.0, 600.0, "time of the impulse in seconds"),
            num("amp", "1", -100.0, 100.0, "impulse height"),
        ], "{type: impulse}"),
        s("additive", "source", "Sum of sine partials. `partials` is a list of harmonic amplitudes [1, 0.5, 0.33] or of [ratio, amplitude, decay_seconds] triples for inharmonic, individually decaying partials.", vec![
            freq(), amp(),
            ParamSpec { name: "partials", kind: Kind::Raw, default: Some("default"), required: false, min: None, max: None, doc: "list of amplitudes for harmonics 1,2,3... or list of [ratio, amplitude, decay_seconds]" },
        ], "{type: additive, partials: [1, 0.5, 0.33, 0.25]}"),
        s("pluck", "source", "Plucked string (Karplus-Strong). Good for guitar, harp, koto and bass plucks.", vec![
            num("freq", "freq", 20.0, 8000.0, "string pitch in Hz (constant)"),
            num("decay", "2", 0.01, 30.0, "time for the note to die away, in seconds"),
            num("brightness", "0.5", 0.0, 1.0, "0 = dull and soft, 1 = bright and twangy"),
            num("position", "0.2", 0.02, 0.5, "pluck position along the string; near 0 is thin, 0.5 is hollow"),
            amp(),
        ], "{type: pluck, decay: 3, brightness: 0.6}"),
        s("modal", "source", "Bank of resonators modelling a struck object. Excite it with an impulse or a short noise burst via `in`; with no input it is struck once at t=0.", vec![
            sig_opt("in", "excitation signal (default: a single impulse at t=0)"),
            num("freq", "freq", 10.0, 20000.0, "frequency of the lowest mode in Hz"),
            ParamSpec { name: "modes", kind: Kind::Raw, default: Some("bar"), required: false, min: None, max: None, doc: "preset name (bell, bar, marimba, plate, membrane, glass, wood, bowl, tube, metal) or list of [ratio, gain, decay_multiplier]" },
            num("decay", "1.5", 0.005, 60.0, "ring time of the lowest mode in seconds"),
            num("damping", "0.6", 0.0, 3.0, "how much faster high modes fade: 0 = all ring equally (glassy), 2 = dull thud"),
            amp(),
        ], "{type: modal, modes: bell, freq: 440, decay: 4}"),
        // ---- modulators
        s("adsr", "modulator", "Attack-decay-sustain-release envelope, 0..1, driven by the note gate.", vec![
            num("attack", "0.01", 0.0, 60.0, "seconds to reach full level"),
            num("decay", "0.1", 0.0, 60.0, "seconds to fall to the sustain level"),
            num("sustain", "0.7", 0.0, 1.0, "level held while the gate is on"),
            num("release", "0.2", 0.0, 60.0, "seconds to fade out after the gate ends"),
            par("gate", "gate", -F, F, "gate signal (> 0.5 = held)"),
        ], "{type: adsr, attack: 0.005, decay: 0.2, sustain: 0.5, release: 0.3}"),
        s("env", "modulator", "Breakpoint envelope through a list of [time, value] points. Holds the last value afterwards.", vec![
            ParamSpec { name: "points", kind: Kind::Raw, default: None, required: true, min: None, max: None, doc: "list of [time_seconds, value] pairs in time order" },
            choice("curve", &["lin", "exp", "smooth", "hold"], "lin", "interpolation between points: exp gives natural-sounding decays"),
            num("time_scale", "1", 0.0, 600.0, "multiplies all point times; use \"dur\" with times in 0..1"),
        ], "{type: env, points: [[0, 0], [0.01, 1], [0.4, 0]], curve: exp}"),
        s("lfo", "modulator", "Low-frequency oscillator, -1..1 (or 0..1 with unipolar).", vec![
            choice("wave", &LfoWave::NAMES, "sine", "shape; steps = random sample-and-hold, smooth = wandering random"),
            par("rate", "1", 0.0, 20000.0, "cycles per second"),
            par("amp", "1", -F, F, "output scale"),
            num("phase", "0", 0.0, 1.0, "start phase in cycles"),
            flag("unipolar", false, "output 0..1 instead of -1..1"),
        ], "{type: lfo, wave: smooth, rate: 0.3}"),
        s("sweep", "modulator", "One-shot ramp from one value to another; the standard way to make pitch drops and filter sweeps.", vec![
            num_req("from", -F, F, "start value"),
            num_req("to", -F, F, "end value"),
            num("time", "dur", 0.0, 600.0, "seconds the ramp takes (default: the whole sound)"),
            choice("curve", &["lin", "exp"], "exp", "exp sounds even for frequencies; needs both values above 0"),
            num("delay", "0", 0.0, 600.0, "seconds to hold the start value before ramping"),
        ], "{type: sweep, from: 900, to: 60, time: 0.25}"),
        // ---- filters
        s("lowpass", "filter", "Removes highs above the cutoff. The main tool for making things darker, softer or more distant.", filter_params(), "{type: lowpass, in: osc1, cutoff: \"200 + 5000 * env\", q: 2}"),
        s("highpass", "filter", "Removes lows below the cutoff. Use it to clear rumble and make sounds thinner or smaller.", filter_params(), "{type: highpass, in: src, cutoff: 300}"),
        s("bandpass", "filter", "Keeps a band around the cutoff. Narrow (high q) bands make whistles and resonances.", filter_params(), "{type: bandpass, in: src, cutoff: 1200, q: 6}"),
        s("notch", "filter", "Removes a band around the cutoff.", filter_params(), "{type: notch, in: src, cutoff: 1000, q: 4}"),
        s("peak", "filter", "Bell EQ: boosts or cuts around a frequency.", eq_params(), "{type: peak, in: src, freq: 3000, gain_db: 4, q: 1}"),
        s("lowshelf", "filter", "Shelf EQ: boosts or cuts everything below a frequency.", eq_params(), "{type: lowshelf, in: src, freq: 120, gain_db: 3}"),
        s("highshelf", "filter", "Shelf EQ: boosts or cuts everything above a frequency.", eq_params(), "{type: highshelf, in: src, freq: 6000, gain_db: -4}"),
        s("comb", "filter", "Feedback comb filter: a short tuned resonance that adds a metallic or tubular pitch.", vec![
            input(),
            par("freq", "freq", 10.5, 20000.0, "resonant pitch in Hz"),
            par("feedback", "0.7", -0.999, 0.999, "resonance amount; close to 1 rings long"),
            num("damp", "8000", 100.0, 40000.0, "lowpass inside the loop in Hz; lower is duller"),
        ], "{type: comb, in: src, freq: 220, feedback: 0.9}"),
        s("formant", "filter", "Vowel filter. Makes any harmonically rich input sound voice-like; morph `vowel` over time for talking or creature sounds.", vec![
            input(),
            par("vowel", "0", 0.0, 4.0, "0=a 1=e 2=i 3=o 4=u, fractional values morph"),
            num("shift", "1", 0.25, 4.0, "formant frequency multiplier: below 1 = bigger creature, above 1 = smaller"),
            num("q", "8", 0.5, 50.0, "resonance sharpness"),
        ], "{type: formant, in: src, vowel: \"2 + 2 * lfo1\"}"),
        s("dcblock", "filter", "Removes DC offset (a constant shift that wastes headroom and causes clicks).", vec![input()], "{type: dcblock, in: src}"),
        // ---- shaping
        s("distort", "shaper", "Waveshaping distortion, oversampled to avoid aliasing. Output stays within +/-1 for tanh, soft and hard.", vec![
            input(),
            par("drive_db", "12", -24.0, 60.0, "gain into the shaper; more = dirtier"),
            choice("shape", &Shape::NAMES, "tanh", "tanh = warm, soft = gentle, hard = harsh clip, fold = buzzy metallic, tube = asymmetric warmth"),
            num("mix", "1", 0.0, 1.0, "dry/wet"),
            num("level_db", "0", -60.0, 24.0, "output trim"),
        ], "{type: distort, in: bass, drive_db: 18, shape: tube}"),
        s("bitcrush", "shaper", "Bit-depth and sample-rate reduction for retro and lo-fi grit (aliases on purpose).", vec![
            input(),
            par("bits", "8", 1.0, 24.0, "bit depth"),
            par("rate", "8000", 20.0, 192000.0, "sample-and-hold rate in Hz"),
        ], "{type: bitcrush, in: src, bits: 6, rate: 11025}"),
        s("gain", "utility", "Level change in decibels.", vec![input(), par("db", "0", -120.0, 60.0, "gain in dB")], "{type: gain, in: src, db: -6}"),
        s("mix", "utility", "Sums several signals, optionally with per-input gains in dB.", vec![
            ParamSpec { name: "inputs", kind: Kind::SigList, default: None, required: true, min: None, max: None, doc: "list of signals to sum" },
            ParamSpec { name: "gains_db", kind: Kind::SigList, default: None, required: false, min: None, max: None, doc: "optional list of gains in dB, one per input" },
        ], "{type: mix, inputs: [body, click, tail], gains_db: [0, -6, -9]}"),
        s("expr", "utility", "Any expression as a node, for custom math between signals.", vec![
            ParamSpec { name: "expr", kind: Kind::Sig, default: None, required: true, min: None, max: None, doc: "expression, e.g. \"a * b\" (ring modulation) or \"tanh(a * 3)\"" },
        ], "{type: expr, expr: \"osc1 * osc2\"}"),
        s("fade", "utility", "Fades the start and/or end of a signal.", vec![
            input(),
            num("fade_in", "0", 0.0, 600.0, "seconds"),
            num("fade_out", "0", 0.0, 600.0, "seconds, measured back from the end of the sound"),
        ], "{type: fade, in: src, fade_in: 0.5, fade_out: 1}"),
        s("shift", "utility", "Delays a signal in time without feedback, to offset one layer against another.", vec![
            input(),
            num_req("time", 0.0, 600.0, "seconds"),
        ], "{type: shift, in: tail, time: 0.08}"),
        s("reverse", "utility", "Plays the signal backwards (reverse cymbals, sucking effects).", vec![input()], "{type: reverse, in: src}"),
        // ---- dynamics
        s("compressor", "dynamics", "Reduces level above a threshold. With `sidechain` it ducks the input when the sidechain is loud.", vec![
            input(),
            num("threshold_db", "-18", -80.0, 0.0, "level where compression starts"),
            num("ratio", "4", 1.0, 100.0, "how hard to compress (4 = 4:1)"),
            num("attack_ms", "10", 0.01, 500.0, "how fast it clamps down"),
            num("release_ms", "120", 1.0, 5000.0, "how fast it lets go"),
            num("knee_db", "6", 0.0, 40.0, "softness of the threshold"),
            num("makeup_db", "0", -24.0, 40.0, "gain after compression"),
            sig_opt("sidechain", "signal that drives the detector instead of the input"),
        ], "{type: compressor, in: drums, threshold_db: -16, ratio: 3}"),
        s("limiter", "dynamics", "Look-ahead true-peak brickwall limiter. Put it last to guarantee no clipping.", vec![
            input(),
            num("ceiling_db", "-1", -40.0, 0.0, "maximum true-peak level"),
            num("release_ms", "80", 1.0, 2000.0, "recovery time"),
        ], "{type: limiter, in: src, ceiling_db: -1}"),
        s("transient", "dynamics", "Transient shaper: makes attacks punchier or softer and tails longer or shorter, independent of level.", vec![
            input(),
            num("attack_db", "0", -24.0, 24.0, "boost (+) or soften (-) attacks"),
            num("sustain_db", "0", -24.0, 24.0, "lengthen (+) or shorten (-) tails"),
        ], "{type: transient, in: hit, attack_db: 6, sustain_db: -3}"),
        // ---- space
        s("delay", "space", "Echo with feedback and damping; ping-pong bounces between left and right.", vec![
            input(),
            par("time", "0.25", 0.0005, 10.0, "delay time in seconds"),
            par("feedback", "0.35", 0.0, 0.98, "how much each echo feeds the next"),
            par("mix", "0.3", 0.0, 1.0, "dry/wet"),
            num("damp", "6000", 100.0, 40000.0, "lowpass on the echoes in Hz"),
            flag("pingpong", false, "alternate echoes left and right"),
        ], "{type: delay, in: lead, time: 0.375, feedback: 0.4, pingpong: true}"),
        s("chorus", "space", "Thickens and widens a sound with modulated copies. Output is stereo.", vec![
            input(),
            num("rate", "0.8", 0.01, 20.0, "modulation speed in Hz"),
            num("depth_ms", "4", 0.0, 30.0, "modulation depth"),
            num("mix", "0.5", 0.0, 1.0, "dry/wet"),
            int("voices", "3", 1.0, 6.0, "number of copies"),
        ], "{type: chorus, in: pad}"),
        s("flanger", "space", "Short modulated delay with feedback: jet-plane sweeps.", vec![
            input(),
            num("rate", "0.3", 0.01, 20.0, "sweep speed in Hz"),
            num("depth_ms", "3", 0.0, 20.0, "sweep depth"),
            num("feedback", "0.5", -0.95, 0.95, "intensity"),
            num("mix", "0.5", 0.0, 1.0, "dry/wet"),
        ], "{type: flanger, in: src, rate: 0.2, feedback: 0.7}"),
        s("phaser", "space", "Sweeping notches: swirling, sci-fi movement.", vec![
            input(),
            num("rate", "0.4", 0.01, 20.0, "sweep speed in Hz"),
            num("min_freq", "300", 20.0, 20000.0, "bottom of the sweep in Hz"),
            num("max_freq", "3000", 20.0, 20000.0, "top of the sweep in Hz"),
            num("feedback", "0.5", -0.95, 0.95, "intensity"),
            int("stages", "6", 2.0, 12.0, "number of all-pass stages"),
            num("mix", "0.7", 0.0, 1.0, "dry/wet"),
        ], "{type: phaser, in: pad, rate: 0.2}"),
        s("reverb", "space", "Algorithmic room/hall reverb. Output is stereo. Remember to make the patch `duration` long enough for the tail.", vec![
            input(),
            num("decay", "2", 0.05, 60.0, "time for the tail to fade by 60 dB, in seconds"),
            num("size", "1", 0.1, 4.0, "room size: 0.3 = small room, 1 = hall, 2.5 = cathedral"),
            num("damping", "6000", 200.0, 40000.0, "high-frequency absorption corner in Hz; lower = darker, warmer"),
            num("predelay_ms", "10", 0.0, 500.0, "gap before the reverb starts; larger = bigger space, clearer source"),
            num("mix", "0.3", 0.0, 1.0, "dry/wet (1 = reverb only)"),
            num("width", "1", 0.0, 2.0, "stereo width of the tail"),
            num("diffusion", "0.7", 0.0, 0.9, "smoothness of the onset"),
            num("lowcut", "100", 5.0, 2000.0, "removes lows from the tail, in Hz, to keep it from getting muddy"),
            num("modulation", "1", 0.0, 4.0, "slow internal movement that prevents metallic ringing"),
        ], "{type: reverb, in: dry, decay: 3.5, size: 1.5, mix: 0.35}"),
        s("pan", "space", "Places a signal in the stereo field. Output is stereo.", vec![
            input(),
            par("pos", "0", -1.0, 1.0, "-1 = left, 0 = center, 1 = right; modulate for movement"),
        ], "{type: pan, in: src, pos: \"sin(t * 2)\"}"),
        s("width", "space", "Stereo width control (mid/side).", vec![
            input(),
            num("amount", "1", 0.0, 3.0, "0 = mono, 1 = unchanged, 2 = extra wide"),
        ], "{type: width, in: src, amount: 1.5}"),
        s("speed", "utility", "Variable-speed playback of its input: tape stops, pitch dives and rises of complex sounds.", vec![
            input(),
            par("rate", "1", 0.0, 16.0, "playback speed: 1 = normal, 2 = octave up and twice as fast"),
        ], "{type: speed, in: src, rate: \"1 - 0.8 * t / dur\"}"),
    ]
}

fn filter_params() -> Vec<ParamSpec> {
    vec![
        input(),
        par_req("cutoff", 5.0, 96000.0, "cutoff frequency in Hz"),
        par("q", "0.707", 0.05, 100.0, "resonance: 0.707 = flat, 2-10 = pronounced peak"),
        choice("slope", &["12", "24", "36", "48"], "12", "steepness in dB per octave"),
    ]
}

fn eq_params() -> Vec<ParamSpec> {
    vec![
        input(),
        par_req("freq", 5.0, 96000.0, "center or corner frequency in Hz"),
        par_req("gain_db", -48.0, 48.0, "boost (+) or cut (-) in dB"),
        par("q", "0.707", 0.05, 100.0, "bandwidth: higher is narrower"),
    ]
}

pub fn specs() -> &'static [NodeSpec] {
    static SPECS: OnceLock<Vec<NodeSpec>> = OnceLock::new();
    SPECS.get_or_init(build_specs)
}

pub fn spec_for(name: &str) -> Option<&'static NodeSpec> {
    specs().iter().find(|s| s.name == name)
}

pub fn spec_names() -> Vec<&'static str> {
    specs().iter().map(|s| s.name).collect()
}

fn num_list(v: &Value, what: &str) -> Result<Vec<f64>> {
    let arr = v.as_array().ok_or_else(|| Error::invalid(format!("{what} must be a list of numbers, found {}", type_name(v))))?;
    arr.iter().map(|x| x.as_f64().ok_or_else(|| Error::invalid(format!("{what} must contain only numbers, found {}", type_name(x))))).collect()
}

pub fn parse_node_data(node_type: &str, param: &str, v: &Value) -> Result<NodeData> {
    match (node_type, param) {
        ("additive", "partials") => {
            if v.as_str() == Some("default") {
                return Ok(NodeData::Partials((1..=8).map(|h| (h as f64, 1.0 / h as f64, 0.0)).collect()));
            }
            let arr = v.as_array().ok_or_else(|| Error::invalid(format!("must be a list, found {}", type_name(v))).hint("e.g. partials: [1, 0.5, 0.33] or partials: [[1, 1, 2.0], [2.76, 0.5, 1.2]]"))?;
            if arr.is_empty() || arr.len() > 256 {
                return Err(Error::invalid("needs between 1 and 256 partials"));
            }
            let mut out = Vec::new();
            for (i, item) in arr.iter().enumerate() {
                match item {
                    Value::Number(n) => out.push(((i + 1) as f64, n.as_f64().unwrap_or(0.0), 0.0)),
                    Value::Array(_) => {
                        let l = num_list(item, "a partial").map_err(|e| e.within(&i.to_string()))?;
                        if l.len() < 2 || l.len() > 3 || l[0] <= 0.0 {
                            return Err(Error::invalid("a partial is [ratio, amplitude] or [ratio, amplitude, decay_seconds] with ratio > 0").at(i.to_string()));
                        }
                        out.push((l[0], l[1], l.get(2).copied().unwrap_or(0.0).max(0.0)));
                    }
                    other => return Err(Error::invalid(format!("a partial must be a number or a list, found {}", type_name(other))).at(i.to_string())),
                }
            }
            Ok(NodeData::Partials(out))
        }
        ("modal", "modes") => match v {
            Value::String(name) => {
                let m = modal_preset(name).ok_or_else(|| Error::invalid(format!("unknown mode preset '{name}'")).hint(hint_from(name, &MODAL_PRESETS)))?;
                Ok(NodeData::Modes(m.into_iter().map(|(r, g)| (r, g, 1.0)).collect()))
            }
            Value::Array(arr) => {
                if arr.is_empty() || arr.len() > 128 {
                    return Err(Error::invalid("needs between 1 and 128 modes"));
                }
                let mut out = Vec::new();
                for (i, item) in arr.iter().enumerate() {
                    match item {
                        Value::Number(n) => out.push((n.as_f64().unwrap_or(1.0), 1.0 / (1.0 + 0.35 * i as f64), 1.0)),
                        _ => {
                            let l = num_list(item, "a mode").map_err(|e| e.within(&i.to_string()))?;
                            if l.is_empty() || l.len() > 3 || l[0] <= 0.0 {
                                return Err(Error::invalid("a mode is [ratio], [ratio, gain] or [ratio, gain, decay_multiplier] with ratio > 0").at(i.to_string()));
                            }
                            out.push((l[0], l.get(1).copied().unwrap_or(1.0), l.get(2).copied().unwrap_or(1.0).max(0.001)));
                        }
                    }
                }
                Ok(NodeData::Modes(out))
            }
            other => Err(Error::invalid(format!("must be a preset name or a list of modes, found {}", type_name(other))).hint(format!("presets: {}", MODAL_PRESETS.join(", ")))),
        },
        ("env", "points") => {
            let arr = v.as_array().ok_or_else(|| Error::invalid(format!("must be a list of [time, value] pairs, found {}", type_name(v))).hint("e.g. points: [[0, 0], [0.01, 1], [0.5, 0]]"))?;
            if arr.is_empty() {
                return Err(Error::invalid("needs at least one [time, value] point"));
            }
            let mut out: Vec<(f64, f64)> = Vec::new();
            for (i, item) in arr.iter().enumerate() {
                let l = num_list(item, "a point").map_err(|e| e.within(&i.to_string()))?;
                if l.len() != 2 {
                    return Err(Error::invalid("a point is [time, value]").at(i.to_string()));
                }
                if l[0] < 0.0 || out.last().map_or(false, |p| l[0] < p.0) {
                    return Err(Error::invalid(format!("point times must be non-negative and increasing; {} comes after {}", l[0], out.last().map_or(0.0, |p| p.0))).at(i.to_string()));
                }
                out.push((l[0], l[1]));
            }
            Ok(NodeData::Points(out))
        }
        _ => Err(Error::invalid(format!("node '{node_type}' has no structured parameter '{param}'"))),
    }
}

fn apply_amp(mut s: Sig, amp: &P) -> Sig {
    match amp {
        P::C(a) if *a == 1.0 => s,
        P::C(a) => {
            s.scale(*a);
            s
        }
        P::B(b) => {
            for (i, x) in s.l.iter_mut().enumerate() {
                *x *= b[i];
            }
            if let Some(r) = s.r.as_mut() {
                for (i, x) in r.iter_mut().enumerate() {
                    *x *= b[i];
                }
            }
            s
        }
    }
}

fn val_at(v: &Val, i: usize, right: bool) -> f64 {
    match v {
        Val::C(c) => *c,
        Val::S(s) => {
            if right {
                s.right()[i]
            } else {
                s.l[i]
            }
        }
    }
}

pub fn eval_node(a: &Args) -> Result<Sig> {
    let c = a.ctx;
    let (sr, n) = (c.sr, c.n);
    Ok(match a.node.spec.name {
        "osc" => {
            let freq = a.p("freq");
            let pw = a.p("pw");
            let pm = a.opt_p("pm");
            let s = osc(&OscArgs {
                wave: Wave::parse(a.str("wave")).unwrap_or(Wave::Sine),
                freq: &freq,
                pw: &pw,
                pm: pm.as_ref(),
                unison: a.int("unison"),
                detune_cents: a.num("detune"),
                spread: a.num("spread"),
                phase: Some(a.num("phase")),
                max_freq: c.nyquist(),
                sr,
                n,
                seed: a.seed(),
            });
            apply_amp(s, &a.p("amp"))
        }
        "noise" => {
            let color = NoiseColor::parse(a.str("color")).unwrap_or(NoiseColor::White);
            // generated at the output rate and band-limited up, so level and
            // spectrum do not depend on the oversampling setting
            let gen = |seed: u64| -> Vec<f64> {
                let mut v = interpolate(&noise(color, c.base_sr, n.div_ceil(c.os), seed), c.os);
                v.truncate(n);
                v
            };
            let l = gen(a.seed());
            let s = if a.flag("stereo") { Sig::stereo(l, gen(a.seed() ^ 0xDEAD_BEEF)) } else { Sig::mono(l) };
            apply_amp(s, &a.p("amp"))
        }
        "dust" => {
            let mut v = dust(&a.p("density"), a.flag("bipolar"), sr, n, a.seed());
            for x in v.iter_mut() {
                *x *= c.os as f64; // keep impulse energy through downsampling
            }
            apply_amp(Sig::mono(v), &a.p("amp"))
        }
        "impulse" => {
            let mut v = vec![0.0; n];
            let i = (a.num("at") * sr).round() as usize;
            if i < n {
                // scaled so the impulse keeps its energy through downsampling
                v[i] = a.num("amp") * c.os as f64;
            }
            Sig::mono(v)
        }
        "additive" => {
            let Some(NodeData::Partials(p)) = &a.node.data else { unreachable!() };
            apply_amp(Sig::mono(additive(&a.p("freq"), p, sr, n, c.nyquist())), &a.p("amp"))
        }
        "pluck" => {
            let v = pluck(&PluckArgs { freq: a.num("freq"), decay: a.num("decay"), brightness: a.num("brightness"), position: a.num("position") }, sr, n, a.seed());
            apply_amp(Sig::mono(v), &a.p("amp"))
        }
        "modal" => {
            let Some(NodeData::Modes(m)) = &a.node.data else { unreachable!() };
            let exc: Vec<f64> = if a.has("in") {
                a.sig("in").to_mono()
            } else {
                let mut v = vec![0.0; n];
                v[0] = c.os as f64;
                v
            };
            let args = ModalArgs { freq: a.num("freq"), modes: m.clone(), decay: a.num("decay"), damping: a.num("damping"), rate_ratio: c.os as f64 };
            let total_gain: f64 = m.iter().map(|x| x.1.abs()).sum::<f64>().max(1e-9);
            let mut out = modal(&exc, &args, sr, c.nyquist());
            // keep level independent of how many modes a preset has
            let k = 1.0 / total_gain.sqrt();
            for x in out.iter_mut() {
                *x *= k;
            }
            apply_amp(Sig::mono(out), &a.p("amp"))
        }
        "adsr" => {
            let (att, dec, sus, rel) = (a.num("attack"), a.num("decay"), a.num("sustain"), a.num("release"));
            let gate = a.p("gate");
            let att_inc = if att <= 0.0 { 1.0 } else { 1.0 / (att * sr) };
            let dec_k = if dec <= 0.0 { 0.0 } else { (-4.6 / (dec * sr)).exp() };
            let rel_k = if rel <= 0.0 { 0.0 } else { (-6.9 / (rel * sr)).exp() };
            let mut level = 0.0f64;
            let mut attacking = true;
            let mut was_on = false;
            Sig::mono(
                (0..n)
                    .map(|i| {
                        let on = gate.at(i) > 0.5;
                        if on && !was_on {
                            attacking = true;
                        }
                        was_on = on;
                        if on {
                            if attacking {
                                level += att_inc;
                                if level >= 1.0 {
                                    level = 1.0;
                                    attacking = false;
                                }
                            } else {
                                level = sus + (level - sus) * dec_k;
                            }
                        } else {
                            level *= rel_k;
                            if level < 1e-6 {
                                level = 0.0;
                            }
                        }
                        level
                    })
                    .collect(),
            )
        }
        "env" => {
            let Some(NodeData::Points(pts)) = &a.node.data else { unreachable!() };
            let ts = a.num("time_scale");
            let curve = a.str("curve");
            let mut seg = 0usize;
            Sig::mono(
                (0..n)
                    .map(|i| {
                        let t = i as f64 / sr;
                        while seg + 1 < pts.len() && t >= pts[seg + 1].0 * ts {
                            seg += 1;
                        }
                        if t < pts[0].0 * ts {
                            return pts[0].1;
                        }
                        if seg + 1 >= pts.len() {
                            return pts[pts.len() - 1].1;
                        }
                        let (t0, v0) = (pts[seg].0 * ts, pts[seg].1);
                        let (t1, v1) = (pts[seg + 1].0 * ts, pts[seg + 1].1);
                        let x = if t1 > t0 { (t - t0) / (t1 - t0) } else { 1.0 };
                        match curve {
                            "hold" => v0,
                            "smooth" => v0 + (v1 - v0) * (x * x * (3.0 - 2.0 * x)),
                            "exp" => {
                                // exponential approach: fast at first, then settling (a -60 dB curve)
                                let k = 1.0 - (-6.9 * x).exp();
                                v0 + (v1 - v0) * k / (1.0 - (-6.9f64).exp())
                            }
                            _ => v0 + (v1 - v0) * x,
                        }
                    })
                    .collect(),
            )
        }
        "lfo" => {
            let wave = LfoWave::parse(a.str("wave")).unwrap_or(LfoWave::Sine);
            let mut v = lfo(wave, &a.p("rate"), a.num("phase"), sr, n, a.seed());
            if a.flag("unipolar") {
                for x in v.iter_mut() {
                    *x = 0.5 + 0.5 * *x;
                }
            }
            apply_amp(Sig::mono(v), &a.p("amp"))
        }
        "sweep" => {
            let (from, to, time, del) = (a.num("from"), a.num("to"), a.num("time").max(1e-6), a.num("delay"));
            let exp = a.str("curve") == "exp" && from > 0.0 && to > 0.0;
            Sig::mono(
                (0..n)
                    .map(|i| {
                        let x = ((i as f64 / sr - del) / time).clamp(0.0, 1.0);
                        if exp {
                            from * (to / from).powf(x)
                        } else {
                            from + (to - from) * x
                        }
                    })
                    .collect(),
            )
        }
        ty @ ("lowpass" | "highpass" | "bandpass" | "notch") => {
            let mode = match ty {
                "lowpass" => FilterMode::Lowpass,
                "highpass" => FilterMode::Highpass,
                "bandpass" => FilterMode::Bandpass,
                _ => FilterMode::Notch,
            };
            let (cut, q) = (a.p("cutoff"), a.p("q"));
            let stages = a.str("slope").parse::<usize>().unwrap_or(12) / 12;
            a.sig("in").map_channels(|_, x| svf_process(x, mode, &cut, &q, &P::C(0.0), stages, sr))
        }
        ty @ ("peak" | "lowshelf" | "highshelf") => {
            let mode = match ty {
                "peak" => FilterMode::Peak,
                "lowshelf" => FilterMode::LowShelf,
                _ => FilterMode::HighShelf,
            };
            let (f, g, q) = (a.p("freq"), a.p("gain_db"), a.p("q"));
            a.sig("in").map_channels(|_, x| svf_process(x, mode, &f, &q, &g, 1, sr))
        }
        "comb" => {
            let (f, fb, damp) = (a.p("freq"), a.p("feedback"), a.num("damp"));
            a.sig("in").map_channels(|_, x| comb(x, &f, &fb, damp, sr))
        }
        "formant" => {
            let (v, shift, q) = (a.p("vowel"), a.num("shift"), a.num("q"));
            a.sig("in").map_channels(|_, x| formant(x, &v, shift, q, sr))
        }
        "dcblock" => a.sig("in").map_channels(|_, x| dc_block(x, sr)),
        "distort" => {
            let shape = Shape::parse(a.str("shape")).unwrap_or(Shape::Tanh);
            // extra oversampling on top of the engine's: 4x in total for draft
            // and standard quality, 8x for high
            let os = if c.os == 1 { 4 } else { 2 };
            let mut s = distort(&a.sig("in"), shape, &a.p("drive_db"), a.num("mix"), os, sr);
            s.scale(db_to_lin(a.num("level_db")));
            s
        }
        "bitcrush" => bitcrush(&a.sig("in"), &a.p("bits"), &a.p("rate"), sr),
        "gain" => {
            let db = a.p("db");
            let g = match &db {
                P::C(v) => P::C(db_to_lin(*v)),
                P::B(b) => P::B(std::rc::Rc::new(b.iter().map(|v| db_to_lin(*v)).collect())),
            };
            apply_amp((*a.sig("in")).clone(), &g)
        }
        "mix" => {
            let inputs = a.lists.get("inputs").map(|v| v.as_slice()).unwrap_or(&[]);
            let gains = a.lists.get("gains_db");
            if let Some(g) = gains {
                if g.len() != inputs.len() {
                    return Err(Error::invalid(format!("gains_db has {} entries but inputs has {}", g.len(), inputs.len())).at("gains_db").hint("give one gain per input"));
                }
            }
            let stereo = inputs.iter().any(|v| matches!(v, Val::S(s) if s.is_stereo()));
            let mut out = Sig::silence(n, stereo);
            for (k, v) in inputs.iter().enumerate() {
                let g = gains.map(|g| db_to_lin(val_at(&g[k], 0, false))).unwrap_or(1.0);
                for i in 0..n {
                    out.l[i] += val_at(v, i, false) * g;
                }
                if let Some(r) = out.r.as_mut() {
                    for (i, x) in r.iter_mut().enumerate() {
                        *x += val_at(v, i, true) * g;
                    }
                }
            }
            out
        }
        "expr" => (*a.sig("expr")).clone(),
        "fade" => {
            let mut s = (*a.sig("in")).clone();
            let fi = (a.num("fade_in") * sr) as usize;
            let fo = (a.num("fade_out") * sr) as usize;
            s.fade(fi, fo);
            s
        }
        "shift" => {
            let d = (a.num("time") * sr).round() as usize;
            a.sig("in").map_channels(|_, x| {
                let mut v = vec![0.0; n];
                if d < n {
                    v[d..].copy_from_slice(&x[..n - d]);
                }
                v
            })
        }
        "reverse" => a.sig("in").map_channels(|_, x| x.iter().rev().cloned().collect()),
        "compressor" => {
            let args = CompArgs { threshold_db: a.num("threshold_db"), ratio: a.num("ratio"), attack_ms: a.num("attack_ms"), release_ms: a.num("release_ms"), knee_db: a.num("knee_db"), makeup_db: a.num("makeup_db") };
            let key = if a.has("sidechain") { Some(a.sig("sidechain")) } else { None };
            compressor(&a.sig("in"), key.as_deref(), &args, sr).0
        }
        "limiter" => limiter(&a.sig("in"), a.num("ceiling_db"), a.num("release_ms"), 5.0, c.os == 1, sr).0,
        "transient" => transient(&a.sig("in"), a.num("attack_db"), a.num("sustain_db"), sr),
        "delay" => {
            let (time, fb, mix) = (a.p("time"), a.p("feedback"), a.p("mix"));
            delay(&a.sig("in"), &DelayArgs { time: &time, feedback: &fb, mix: &mix, damp_hz: a.num("damp"), pingpong: a.flag("pingpong") }, sr)
        }
        "chorus" => chorus(&a.sig("in"), a.num("rate"), a.num("depth_ms"), a.num("mix"), a.int("voices"), sr),
        "flanger" => flanger(&a.sig("in"), a.num("rate"), a.num("depth_ms"), a.num("feedback"), a.num("mix"), sr),
        "phaser" => {
            let (lo, hi) = (a.num("min_freq"), a.num("max_freq"));
            if hi <= lo {
                return Err(Error::invalid("max_freq must be above min_freq").at("max_freq"));
            }
            phaser(&a.sig("in"), a.num("rate"), lo, hi.min(sr * 0.45), a.num("feedback"), a.int("stages"), a.num("mix"), sr)
        }
        "reverb" => reverb(
            &a.sig("in"),
            &ReverbArgs {
                decay: a.num("decay"),
                size: a.num("size"),
                damping_hz: a.num("damping"),
                predelay_ms: a.num("predelay_ms"),
                mix: a.num("mix"),
                width: a.num("width"),
                diffusion: a.num("diffusion"),
                lowcut_hz: a.num("lowcut"),
                modulation: a.num("modulation"),
            },
            sr,
            a.seed(),
        ),
        "pan" => pan(&a.sig("in"), &a.p("pos")),
        "width" => width(&a.sig("in"), a.num("amount")),
        "speed" => speed(&a.sig("in"), &a.p("rate")),
        other => return Err(Error::new("internal_error", format!("node type '{other}' has no implementation"))),
    })
}
