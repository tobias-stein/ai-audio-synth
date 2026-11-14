//! Patch documents: parsing, validation and evaluation of the node graph.

use crate::audio::Sig;
use crate::dsp::resample::decimate_sig;
use crate::dsp::P;
use crate::error::{Error, Result};
use crate::expr::{Expr, Val, CONSTANTS};
use crate::nodes::{eval_node, parse_node_data, spec_for, spec_names, Kind, NodeData, NodeSpec};
use crate::util::{did_you_mean, hash_str, hint_from, hz_to_midi, Rng};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

/// Variables every expression can read, with documentation.
pub const BUILTIN_VARS: &[(&str, &str)] = &[
    ("t", "time in seconds since the start of the sound (a signal)"),
    ("freq", "note frequency in Hz (440 unless a score note or --set freq=... overrides it)"),
    ("vel", "note velocity 0..1 (1 unless a score note overrides it)"),
    ("gate", "1 while the note is held, 0 after release (a signal); drives adsr"),
    ("dur", "total length of the render in seconds"),
    ("gate_dur", "how long the gate is held, in seconds"),
    ("note", "MIDI note number matching freq (69 = A4)"),
    ("sr", "output sample rate in Hz"),
    ("input", "the incoming audio when the document is used as an effect chain"),
];

#[derive(Clone, Debug)]
pub struct PatchParam {
    pub name: String,
    pub default: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub doc: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Normalize {
    None,
    Peak(f64),
    Lufs(f64),
}

#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub spec: &'static NodeSpec,
    pub exprs: BTreeMap<&'static str, Expr>,
    pub strs: BTreeMap<&'static str, String>,
    pub flags: BTreeMap<&'static str, bool>,
    pub lists: BTreeMap<&'static str, Vec<Expr>>,
    pub data: Option<NodeData>,
    pub deps: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Patch {
    pub name: String,
    pub description: String,
    pub category: String,
    pub tags: Vec<String>,
    pub profile: Option<String>,
    pub duration: Option<f64>,
    pub gate: Option<f64>,
    pub tail: f64,
    pub seed: Option<u64>,
    pub params: Vec<PatchParam>,
    pub nodes: Vec<Node>,
    pub order: Vec<usize>,
    pub output: Expr,
    pub loop_crossfade: Option<f64>,
    pub normalize: Normalize,
}

const PATCH_KEYS: &[&str] = &["kind", "name", "description", "category", "tags", "profile", "duration", "gate", "tail", "seed", "params", "nodes", "output", "loop", "normalize"];

pub fn as_obj<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>> {
    v.as_object().ok_or_else(|| Error::invalid(format!("{what} must be a mapping of keys to values, found {}", type_name(v))))
}

pub fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "a mapping",
    }
}

pub fn check_keys(obj: &Map<String, Value>, allowed: &[&str]) -> Result<()> {
    for k in obj.keys() {
        if !allowed.contains(&k.as_str()) {
            return Err(Error::invalid(format!("unknown key '{k}'")).at(k.clone()).hint(hint_from(k, allowed)));
        }
    }
    Ok(())
}

pub fn get_f64(obj: &Map<String, Value>, key: &str) -> Result<Option<f64>> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => Ok(n.as_f64()),
        Some(Value::String(s)) => match Expr::parse(s).ok().and_then(|e| e.eval(&|_| None, &mut Rng::new(0)).ok()).and_then(|v| v.as_const()) {
            Some(v) => Ok(Some(v)),
            None => Err(Error::invalid(format!("'{key}' must be a number, found \"{s}\"")).at(key)),
        },
        Some(v) => Err(Error::invalid(format!("'{key}' must be a number, found {}", type_name(v))).at(key)),
    }
}

pub fn get_str<'a>(obj: &'a Map<String, Value>, key: &str) -> Result<Option<&'a str>> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(v) => Err(Error::invalid(format!("'{key}' must be a string, found {}", type_name(v))).at(key)),
    }
}

fn value_to_expr(v: &Value) -> Result<Expr> {
    match v {
        Value::Number(n) => Ok(Expr::Num(n.as_f64().unwrap_or(0.0))),
        Value::String(s) => Expr::parse(s),
        Value::Bool(b) => Ok(Expr::Num(if *b { 1.0 } else { 0.0 })),
        other => Err(Error::invalid(format!("expected a number or an expression string, found {}", type_name(other)))),
    }
}

fn parse_node(name: &str, v: &Value) -> Result<Node> {
    let obj = as_obj(v, "a node")?;
    let ty = get_str(obj, "type")?.ok_or_else(|| Error::invalid("node is missing 'type'").hint(format!("add e.g. `type: osc`. Node types: {}", spec_names().join(", "))))?;
    let spec = spec_for(ty).ok_or_else(|| Error::new("unknown_name", format!("unknown node type '{ty}'")).at("type").hint(hint_from(ty, &spec_names())))?;
    let mut node = Node { name: name.to_string(), spec, exprs: BTreeMap::new(), strs: BTreeMap::new(), flags: BTreeMap::new(), lists: BTreeMap::new(), data: None, deps: Vec::new() };
    let param_names: Vec<&str> = spec.params.iter().map(|p| p.name).collect();
    for (k, val) in obj {
        if k == "type" {
            continue;
        }
        let Some(ps) = spec.params.iter().find(|p| p.name == k) else {
            return Err(Error::invalid(format!("node type '{ty}' has no parameter '{k}'")).at(k.clone()).hint(hint_from(k, &param_names)));
        };
        let res: Result<()> = (|| {
            match ps.kind {
                Kind::Sig | Kind::Par | Kind::Num | Kind::Int => {
                    let e = value_to_expr(val)?;
                    if let Some(lit) = e.is_literal() {
                        check_range(ps.min, ps.max, lit)?;
                        if ps.kind == Kind::Int && lit.fract() != 0.0 {
                            return Err(Error::invalid(format!("must be a whole number, found {lit}")));
                        }
                    }
                    node.exprs.insert(ps.name, e);
                }
                Kind::Bool => match val {
                    Value::Bool(b) => {
                        node.flags.insert(ps.name, *b);
                    }
                    other => return Err(Error::invalid(format!("must be true or false, found {}", type_name(other)))),
                },
                Kind::Enum(options) => {
                    let s = match val {
                        Value::String(s) => s.clone(),
                        Value::Number(n) => n.to_string(),
                        other => return Err(Error::invalid(format!("must be one of: {}; found {}", options.join(", "), type_name(other)))),
                    };
                    if !options.contains(&s.as_str()) {
                        return Err(Error::invalid(format!("'{s}' is not a valid value")).hint(hint_from(&s, options)));
                    }
                    node.strs.insert(ps.name, s);
                }
                Kind::SigList => {
                    let arr = val.as_array().ok_or_else(|| Error::invalid(format!("must be a list of signals, found {}", type_name(val))))?;
                    if arr.is_empty() {
                        return Err(Error::invalid("list must not be empty"));
                    }
                    let mut v = Vec::new();
                    for (i, item) in arr.iter().enumerate() {
                        v.push(value_to_expr(item).map_err(|e| e.within(&i.to_string()))?);
                    }
                    node.lists.insert(ps.name, v);
                }
                Kind::Raw => {
                    node.data = Some(parse_node_data(spec.name, ps.name, val)?);
                }
            }
            Ok(())
        })();
        res.map_err(|e| e.within(k))?;
    }
    for ps in &spec.params {
        let present = node.exprs.contains_key(ps.name) || node.strs.contains_key(ps.name) || node.flags.contains_key(ps.name) || node.lists.contains_key(ps.name) || (ps.kind == Kind::Raw && node.data.is_some());
        if present {
            continue;
        }
        match (ps.default, ps.required) {
            (Some(d), _) => match ps.kind {
                Kind::Sig | Kind::Par | Kind::Num | Kind::Int => {
                    node.exprs.insert(ps.name, Expr::parse(d).expect("node default expression must parse"));
                }
                Kind::Bool => {
                    node.flags.insert(ps.name, d == "true");
                }
                Kind::Enum(_) => {
                    node.strs.insert(ps.name, d.to_string());
                }
                Kind::Raw => {
                    node.data = Some(parse_node_data(spec.name, ps.name, &Value::String(d.to_string()))?);
                }
                Kind::SigList => {}
            },
            (None, true) => {
                return Err(Error::invalid(format!("node type '{ty}' requires parameter '{}'", ps.name)).hint(format!("{}: {}", ps.name, ps.doc)));
            }
            (None, false) => {}
        }
    }
    let mut deps = Vec::new();
    for e in node.exprs.values() {
        e.idents(&mut deps);
    }
    for l in node.lists.values() {
        for e in l {
            e.idents(&mut deps);
        }
    }
    node.deps = deps;
    Ok(node)
}

fn check_range(min: Option<f64>, max: Option<f64>, v: f64) -> Result<()> {
    if let Some(lo) = min {
        if v < lo {
            return Err(Error::invalid(format!("value {v} is below the minimum {lo}")).hint(format!("use a value >= {lo}")));
        }
    }
    if let Some(hi) = max {
        if v > hi {
            return Err(Error::invalid(format!("value {v} is above the maximum {hi}")).hint(format!("use a value <= {hi}")));
        }
    }
    Ok(())
}

impl Patch {
    pub fn from_value(v: &Value) -> Result<Patch> {
        let obj = as_obj(v, "a patch")?;
        check_keys(obj, PATCH_KEYS)?;
        let mut params = Vec::new();
        if let Some(pv) = obj.get("params") {
            let pobj = as_obj(pv, "'params'").map_err(|e| e.at("params"))?;
            for (name, val) in pobj {
                let path = format!("params.{name}");
                if BUILTIN_VARS.iter().any(|b| b.0 == name && !matches!(b.0, "freq" | "vel")) || CONSTANTS.iter().any(|c| c.0 == name) {
                    return Err(Error::invalid(format!("'{name}' is a built-in name and cannot be a parameter")).at(path).hint("rename the parameter"));
                }
                let p = match val {
                    Value::Number(n) => PatchParam { name: name.clone(), default: n.as_f64().unwrap_or(0.0), min: None, max: None, doc: String::new() },
                    Value::Object(o) => {
                        check_keys(o, &["default", "min", "max", "doc"]).map_err(|e| e.within(&path))?;
                        let default = get_f64(o, "default").map_err(|e| e.within(&path))?.ok_or_else(|| Error::invalid("parameter needs a 'default'").at(path.clone()))?;
                        PatchParam {
                            name: name.clone(),
                            default,
                            min: get_f64(o, "min").map_err(|e| e.within(&path))?,
                            max: get_f64(o, "max").map_err(|e| e.within(&path))?,
                            doc: get_str(o, "doc").map_err(|e| e.within(&path))?.unwrap_or("").to_string(),
                        }
                    }
                    other => return Err(Error::invalid(format!("parameter must be a number or a mapping with 'default', found {}", type_name(other))).at(path)),
                };
                check_range(p.min, p.max, p.default).map_err(|e| e.at(path))?;
                params.push(p);
            }
        }

        let nodes_v = obj.get("nodes").ok_or_else(|| Error::invalid("patch is missing 'nodes'").hint("add a `nodes:` mapping; run `aas examples show patch-minimal` for a starting point"))?;
        let nobj = as_obj(nodes_v, "'nodes'").map_err(|e| e.at("nodes"))?;
        if nobj.is_empty() {
            return Err(Error::invalid("'nodes' is empty").at("nodes"));
        }
        let mut nodes = Vec::new();
        for (name, nv) in nobj {
            if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || name.chars().next().map_or(true, |c| c.is_ascii_digit()) {
                return Err(Error::invalid(format!("node name '{name}' is not valid")).at(format!("nodes.{name}")).hint("use letters, digits and underscores, starting with a letter"));
            }
            if BUILTIN_VARS.iter().any(|b| b.0 == name) || CONSTANTS.iter().any(|c| c.0 == name) || params.iter().any(|p| &p.name == name) {
                return Err(Error::invalid(format!("node name '{name}' clashes with a built-in variable or a parameter")).at(format!("nodes.{name}")).hint("rename the node"));
            }
            nodes.push(parse_node(name, nv).map_err(|e| e.within(&format!("nodes.{name}")))?);
        }

        let output = match obj.get("output") {
            Some(v) => value_to_expr(v).map_err(|e| e.at("output"))?,
            None => {
                if nodes.iter().any(|n| n.name == "out") {
                    Expr::Var("out".into())
                } else {
                    return Err(Error::invalid("patch has no 'output' and no node named 'out'").at("output").hint(format!("add `output: {}` (the last node) or name your final node 'out'", nodes.last().unwrap().name)));
                }
            }
        };

        // name resolution
        let mut known: Vec<&str> = BUILTIN_VARS.iter().map(|b| b.0).collect();
        known.extend(CONSTANTS.iter().map(|c| c.0));
        known.extend(params.iter().map(|p| p.name.as_str()));
        known.extend(nodes.iter().map(|n| n.name.as_str()));
        let unknown = |name: &str| -> Error {
            let mut e = Error::new("unknown_name", format!("'{name}' is not a node, a parameter or a built-in variable"));
            e.hint = Some(match did_you_mean(name, known.iter().copied()) {
                Some(s) => format!("did you mean '{s}'?"),
                None => format!("nodes: {}", nodes.iter().map(|n| n.name.as_str()).collect::<Vec<_>>().join(", ")),
            });
            e
        };
        for n in &nodes {
            for d in &n.deps {
                if !known.contains(&d.as_str()) {
                    return Err(unknown(d).at(format!("nodes.{}", n.name)));
                }
                if d == &n.name {
                    return Err(Error::invalid(format!("node '{}' refers to itself", n.name)).at(format!("nodes.{}", n.name)).hint("feedback loops are not supported between nodes; use the feedback parameter of delay, comb or flanger"));
                }
            }
        }
        let mut out_ids = Vec::new();
        output.idents(&mut out_ids);
        for d in &out_ids {
            if !known.contains(&d.as_str()) {
                return Err(unknown(d).at("output"));
            }
        }

        // topological order (stable: declaration order where possible)
        let index: HashMap<&str, usize> = nodes.iter().enumerate().map(|(i, n)| (n.name.as_str(), i)).collect();
        let mut state = vec![0u8; nodes.len()];
        let mut order = Vec::with_capacity(nodes.len());
        fn visit(i: usize, nodes: &[Node], index: &HashMap<&str, usize>, state: &mut [u8], order: &mut Vec<usize>) -> Result<()> {
            match state[i] {
                2 => return Ok(()),
                1 => {
                    return Err(Error::invalid(format!("nodes form a cycle through '{}'", nodes[i].name)).at(format!("nodes.{}", nodes[i].name)).hint("a node cannot depend on its own output; use the feedback parameter of delay, comb or flanger instead"));
                }
                _ => {}
            }
            state[i] = 1;
            for d in &nodes[i].deps {
                if let Some(&j) = index.get(d.as_str()) {
                    visit(j, nodes, index, state, order)?;
                }
            }
            state[i] = 2;
            order.push(i);
            Ok(())
        }
        for i in 0..nodes.len() {
            visit(i, &nodes, &index, &mut state, &mut order)?;
        }

        let duration = get_f64(obj, "duration")?;
        if let Some(d) = duration {
            if !(d > 0.0 && d <= 600.0) {
                return Err(Error::invalid(format!("duration {d} is out of range")).at("duration").hint("use a duration between 0 and 600 seconds"));
            }
        }
        let gate = get_f64(obj, "gate")?;
        let tail = get_f64(obj, "tail")?.unwrap_or(0.5);
        if !(0.0..=60.0).contains(&tail) {
            return Err(Error::invalid("tail must be between 0 and 60 seconds").at("tail"));
        }
        let loop_crossfade = match obj.get("loop") {
            None | Some(Value::Null) | Some(Value::Bool(false)) => None,
            Some(Value::Bool(true)) => Some(-1.0),
            Some(Value::Object(o)) => {
                check_keys(o, &["crossfade"]).map_err(|e| e.within("loop"))?;
                Some(get_f64(o, "crossfade").map_err(|e| e.within("loop"))?.unwrap_or(-1.0))
            }
            Some(other) => return Err(Error::invalid(format!("'loop' must be true, false or {{crossfade: seconds}}, found {}", type_name(other))).at("loop")),
        };
        let normalize = match obj.get("normalize") {
            None | Some(Value::Null) => Normalize::Peak(-1.0),
            Some(Value::Bool(false)) => Normalize::None,
            Some(Value::String(s)) if s == "none" => Normalize::None,
            Some(Value::Object(o)) => {
                check_keys(o, &["peak_db", "lufs"]).map_err(|e| e.within("normalize"))?;
                match (get_f64(o, "peak_db").map_err(|e| e.within("normalize"))?, get_f64(o, "lufs").map_err(|e| e.within("normalize"))?) {
                    (Some(p), None) if p <= 0.0 => Normalize::Peak(p),
                    (None, Some(l)) if l <= 0.0 => Normalize::Lufs(l),
                    _ => return Err(Error::invalid("normalize needs exactly one of peak_db or lufs, with a value of 0 or below").at("normalize").hint("e.g. `normalize: {peak_db: -1}` or `normalize: {lufs: -18}` or `normalize: none`")),
                }
            }
            Some(_) => return Err(Error::invalid("'normalize' must be none, {peak_db: x} or {lufs: x}").at("normalize")),
        };
        let tags = match obj.get("tags") {
            Some(Value::Array(a)) => a.iter().filter_map(|t| t.as_str().map(String::from)).collect(),
            _ => Vec::new(),
        };
        Ok(Patch {
            name: get_str(obj, "name")?.unwrap_or("").to_string(),
            description: get_str(obj, "description")?.unwrap_or("").to_string(),
            category: get_str(obj, "category")?.unwrap_or("").to_string(),
            tags,
            profile: get_str(obj, "profile")?.map(String::from),
            duration,
            gate,
            tail,
            seed: get_f64(obj, "seed")?.map(|s| s as u64),
            params,
            nodes,
            order,
            output,
            loop_crossfade,
            normalize,
        })
    }
}

/// Options for one evaluation of a patch graph.
#[derive(Clone, Debug)]
pub struct RenderOpts {
    pub sr: u32,
    pub oversample: usize,
    pub seed: u64,
    pub freq: Option<f64>,
    pub vel: f64,
    pub gate: Option<f64>,
    pub duration: Option<f64>,
    pub params: Vec<(String, f64)>,
    /// Incoming audio when the patch is used as an effect chain (at `sr`).
    pub input: Option<Rc<Sig>>,
}

impl RenderOpts {
    pub fn new(sr: u32, oversample: usize, seed: u64) -> Self {
        RenderOpts { sr, oversample, seed, freq: None, vel: 1.0, gate: None, duration: None, params: Vec::new(), input: None }
    }
}

/// Per-render context handed to node implementations.
pub struct Ctx {
    /// Internal (oversampled) sample rate.
    pub sr: f64,
    pub base_sr: f64,
    pub os: usize,
    pub n: usize,
    pub seed: u64,
    pub dur: f64,
}

impl Ctx {
    /// Highest frequency that survives the final downsampling.
    pub fn nyquist(&self) -> f64 {
        self.base_sr * 0.45
    }
}

/// Evaluated arguments of one node.
pub struct Args<'a> {
    pub node: &'a Node,
    pub vals: HashMap<&'static str, Val>,
    pub lists: HashMap<&'static str, Vec<Val>>,
    pub ctx: &'a Ctx,
}

impl<'a> Args<'a> {
    pub fn has(&self, name: &str) -> bool {
        self.vals.contains_key(name)
    }
    /// Audio input (mono or stereo). Constants become constant signals.
    pub fn sig(&self, name: &str) -> Rc<Sig> {
        match self.vals.get(name) {
            Some(Val::S(s)) => s.clone(),
            Some(Val::C(v)) => Rc::new(Sig::mono(vec![*v; self.ctx.n])),
            None => Rc::new(Sig::silence(self.ctx.n, false)),
        }
    }
    /// Modulatable parameter (mono).
    pub fn p(&self, name: &str) -> P {
        match self.vals.get(name) {
            Some(Val::C(v)) => P::C(*v),
            Some(Val::S(s)) => P::B(Rc::new(if s.is_stereo() { s.to_mono() } else { s.l.clone() })),
            None => P::C(0.0),
        }
    }
    pub fn opt_p(&self, name: &str) -> Option<P> {
        if self.has(name) {
            Some(self.p(name))
        } else {
            None
        }
    }
    /// Constant parameter. A signal is sampled at time zero.
    pub fn num(&self, name: &str) -> f64 {
        self.p(name).first()
    }
    pub fn int(&self, name: &str) -> usize {
        self.num(name).round().max(0.0) as usize
    }
    pub fn str(&self, name: &str) -> &str {
        self.node.strs.get(name).map(|s| s.as_str()).unwrap_or("")
    }
    pub fn flag(&self, name: &str) -> bool {
        self.node.flags.get(name).copied().unwrap_or(false)
    }
    pub fn seed(&self) -> u64 {
        hash_str(&self.node.name) ^ self.ctx.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }
}

fn node_path(node: &str, param: &str) -> String {
    format!("nodes.{node}.{param}")
}

/// Evaluate the graph and return audio at the output sample rate, before any
/// normalization, looping or fades.
pub fn render_graph(patch: &Patch, o: &RenderOpts) -> Result<Sig> {
    let os = o.oversample.max(1);
    let sr_int = o.sr as f64 * os as f64;
    let gate_dur_opt = o.gate.or(patch.gate);
    let dur = match (o.duration, patch.duration, &o.input) {
        (Some(d), _, _) => d,
        (None, _, Some(input)) => input.len() as f64 / o.sr as f64,
        (None, _, None) if o.gate.is_some() => o.gate.unwrap() + patch.tail,
        (None, Some(d), None) => d,
        (None, None, None) => {
            return Err(Error::invalid("patch has no 'duration'").at("duration").hint("add e.g. `duration: 1.5` (seconds)"));
        }
    };
    let n = ((dur * sr_int).round() as usize).max(1);
    if n > (sr_int as usize) * 900 {
        return Err(Error::invalid("render would be longer than 15 minutes").at("duration"));
    }
    let gate_dur = gate_dur_opt.unwrap_or(dur).min(dur);
    let ctx = Ctx { sr: sr_int, base_sr: o.sr as f64, os, n, seed: o.seed, dur };

    let mut env: HashMap<String, Val> = HashMap::new();
    let freq = o.freq.unwrap_or(440.0);
    env.insert("freq".into(), Val::C(freq));
    env.insert("vel".into(), Val::C(o.vel));
    for p in &patch.params {
        env.insert(p.name.clone(), Val::C(p.default));
    }
    if let Some(f) = o.freq {
        env.insert("freq".into(), Val::C(f));
    }
    for (k, v) in &o.params {
        let known_param = patch.params.iter().find(|p| &p.name == k);
        if known_param.is_none() && k != "freq" && k != "vel" {
            let names: Vec<&str> = patch.params.iter().map(|p| p.name.as_str()).chain(["freq", "vel"]).collect();
            return Err(Error::new("unknown_name", format!("patch has no parameter '{k}'")).at(format!("params.{k}")).hint(hint_from(k, &names)));
        }
        if let Some(p) = known_param {
            check_range(p.min, p.max, *v).map_err(|e| e.at(format!("params.{k}")))?;
        }
        env.insert(k.clone(), Val::C(*v));
    }
    let freq = env.get("freq").and_then(|v| v.as_const()).unwrap_or(freq);
    env.insert("note".into(), Val::C(hz_to_midi(freq.max(1.0))));
    env.insert("dur".into(), Val::C(dur));
    env.insert("gate_dur".into(), Val::C(gate_dur));
    env.insert("sr".into(), Val::C(o.sr as f64));
    env.insert("t".into(), Val::S(Rc::new(Sig::mono((0..n).map(|i| i as f64 / sr_int).collect()))));
    let gate_n = (gate_dur * sr_int).round() as usize;
    env.insert("gate".into(), Val::S(Rc::new(Sig::mono((0..n).map(|i| if i < gate_n { 1.0 } else { 0.0 }).collect()))));
    if let Some(input) = &o.input {
        let mut s = (**input).clone();
        s.resize(n);
        env.insert("input".into(), Val::S(Rc::new(s)));
    } else {
        env.insert("input".into(), Val::C(0.0));
    }

    let mut rng = Rng::new(o.seed ^ 0xA5A5_5A5A_1234_5678);
    for &idx in &patch.order {
        let node = &patch.nodes[idx];
        let mut vals: HashMap<&'static str, Val> = HashMap::new();
        for (name, e) in &node.exprs {
            let lookup = |n: &str| env.get(n).cloned();
            let v = e.eval(&lookup, &mut rng).map_err(|er| er.at(node_path(&node.name, name)))?;
            let ps = node.spec.params.iter().find(|p| p.name == *name).unwrap();
            if matches!(ps.kind, Kind::Num | Kind::Int) {
                let c = match &v {
                    Val::C(c) => *c,
                    Val::S(s) => s.l.first().copied().unwrap_or(0.0),
                };
                if !c.is_finite() {
                    return Err(Error::invalid(format!("'{name}' evaluated to a non-finite number")).at(node_path(&node.name, name)));
                }
                check_range(ps.min, ps.max, c).map_err(|er| er.at(node_path(&node.name, name)))?;
            }
            vals.insert(*name, v);
        }
        let mut lists: HashMap<&'static str, Vec<Val>> = HashMap::new();
        for (name, l) in &node.lists {
            let mut out = Vec::new();
            for e in l {
                let lookup = |n: &str| env.get(n).cloned();
                out.push(e.eval(&lookup, &mut rng).map_err(|er| er.at(node_path(&node.name, name)))?);
            }
            lists.insert(*name, out);
        }
        let args = Args { node, vals, lists, ctx: &ctx };
        let mut sig = eval_node(&args).map_err(|e| e.within(&format!("nodes.{}", node.name)))?;
        if sig.len() != n {
            sig.resize(n);
        }
        if sig.has_non_finite() {
            return Err(Error::new("render_error", format!("node '{}' ({}) produced non-finite samples", node.name, node.spec.name))
                .at(format!("nodes.{}", node.name))
                .hint("a parameter is probably out of its sensible range: check feedback values (< 1), filter cutoffs and q (> 0)"));
        }
        env.insert(node.name.clone(), Val::S(Rc::new(sig)));
    }
    let lookup = |n: &str| env.get(n).cloned();
    let out = match patch.output.eval(&lookup, &mut rng).map_err(|e| e.at("output"))? {
        Val::S(s) => (*s).clone(),
        Val::C(_) => return Err(Error::invalid("'output' is a constant, not a signal").at("output").hint("point output at a node, e.g. `output: out`")),
    };
    Ok(decimate_sig(&out, os))
}
