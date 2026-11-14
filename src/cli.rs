//! Command-line interface. Every command prints one JSON object to stdout
//! (`"ok": true` or `"ok": false` with a structured error) and sets an exit
//! code, so a calling agent never has to parse prose.

use crate::analysis::lint::{find_profile, lint, profiles};
use crate::analysis::{analyze, compare::compare, spectrogram, Analysis, AnalyzeOpts};
use crate::audio::Sig;
use crate::doc::{self, Doc, DocKind};
use crate::error::{Error, Result};
use crate::expr::FUNCTION_DOCS;
use crate::graph::{as_obj, check_keys, get_f64, get_str, Patch, BUILTIN_VARS};
use crate::library;
use crate::nodes::{specs, Kind};
use crate::render::{quality_to_oversample, render_patch, AssetOpts};
use crate::score::{render_score, Score, ScoreOpts};
use crate::theory::{Chord, Key, SCALES};
use crate::util::{hint_from, midi_to_hz, midi_to_name, parse_note, round_to};
use crate::wav::{read_wav, write_wav, SampleFormat};
use clap::{Args, Parser, Subcommand};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

const GUIDE: &str = include_str!("../docs/AGENT_GUIDE.md");

#[derive(Parser)]
#[command(name = "aas", version, about = "Procedural audio synthesis for AI agents: describe sound and music as text, render WAV files, and read back measurements instead of listening.", after_help = "Start with `aas guide`. All commands print JSON to stdout.")]
struct Cli {
    /// Print compact single-line JSON instead of indented JSON
    #[arg(long, global = true)]
    compact: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Render a patch (sound) or score (music) document to WAV and report measurements
    Render(RenderArgs),
    /// Render a built-in sound-effect recipe, e.g. `aas sfx explosion --set size=0.8`
    Sfx(SfxArgs),
    /// Check a document for errors without rendering
    Validate {
        /// Path to a .yaml/.json document, builtin:<id>, or - for stdin
        source: String,
    },
    /// Measure a WAV file: loudness, spectrum, envelope, pitch, rhythm, stereo, problems
    Analyze(AnalyzeArgs),
    /// Check a WAV file against a profile for its intended use (ui, impact, music-loop, ...)
    Lint(LintArgs),
    /// Compare a candidate WAV to a reference WAV and list how to get closer
    Compare {
        candidate: PathBuf,
        reference: PathBuf,
    },
    /// Plain-language description of a WAV file plus a text picture of its energy over time
    Describe {
        file: PathBuf,
        /// Treat the file as a loop
        #[arg(long = "loop")]
        is_loop: bool,
    },
    /// Write a spectrogram PNG (for agents that can view images)
    Spectrogram {
        file: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long, default_value_t = 1200)]
        width: usize,
        #[arg(long, default_value_t = 480)]
        height: usize,
    },
    /// Browse the built-in library of recipes, instruments, drums and example scores
    Library {
        #[command(subcommand)]
        cmd: LibraryCmd,
    },
    /// Reference for node types, expression functions and variables
    Nodes {
        /// Show full details for one node type
        name: Option<String>,
        /// List expression functions and built-in variables instead
        #[arg(long)]
        functions: bool,
    },
    /// Machine-readable reference for the document formats
    Schema {
        /// patch, score or pack
        kind: Option<String>,
    },
    /// List lint profiles and what they check
    Profiles,
    /// Music theory lookups
    Theory {
        #[command(subcommand)]
        cmd: TheoryCmd,
    },
    /// Render a batch of assets from a pack manifest and write manifest.json
    Pack(PackArgs),
    /// Print the guide for agents (Markdown)
    Guide,
}

#[derive(Args, Clone)]
struct RenderCommon {
    /// Output WAV path, or a directory ending in /
    #[arg(short, long)]
    output: Option<String>,
    /// Override a patch parameter: --set name=value (repeatable). Notes work for freq: --set freq=C3
    #[arg(long = "set", value_name = "NAME=VALUE")]
    set: Vec<String>,
    /// Random seed; the same seed always gives the same file
    #[arg(long)]
    seed: Option<u64>,
    /// Render N variations with consecutive seeds (for round-robin playback)
    #[arg(long, default_value_t = 1)]
    variations: usize,
    /// Output sample rate in Hz
    #[arg(long, default_value_t = 48000)]
    sr: u32,
    /// Sample format: 16, 24 or 32f
    #[arg(long, default_value = "24")]
    bits: String,
    /// draft (fast), standard (2x oversampling) or high (4x)
    #[arg(long, default_value = "standard")]
    quality: String,
    /// Make the output loop seamlessly
    #[arg(long = "loop")]
    looped: bool,
    /// Override the patch duration in seconds
    #[arg(long)]
    duration: Option<f64>,
    /// Scores: also write one file per instrument and bus
    #[arg(long)]
    stems: bool,
    /// Scores: also write each section as its own file, with its tail, for adaptive music
    #[arg(long)]
    sections: bool,
    /// Lint the result against this profile (default: the document's own `profile`)
    #[arg(long)]
    profile: Option<String>,
    /// Skip the measurement step (faster)
    #[arg(long)]
    no_analysis: bool,
    /// Patches: skip normalization and keep the level the graph produces
    #[arg(long)]
    raw: bool,
    /// Patches: report the level of every node, to see which layer is loud or silent
    #[arg(long)]
    probe: bool,
}

#[derive(Args)]
struct RenderArgs {
    /// Path to a .yaml/.json document, builtin:<id>, or - for stdin
    source: String,
    #[command(flatten)]
    common: RenderCommon,
}

#[derive(Args)]
struct SfxArgs {
    /// Recipe name, e.g. explosion (see `aas library list --category sfx`)
    recipe: String,
    #[command(flatten)]
    common: RenderCommon,
}

#[derive(Args)]
struct AnalyzeArgs {
    file: PathBuf,
    /// Treat the file as a loop and inspect the seam
    #[arg(long = "loop")]
    is_loop: bool,
    /// Include third-octave spectrum and the band-by-time energy map
    #[arg(long)]
    detail: bool,
}

#[derive(Args)]
struct LintArgs {
    file: PathBuf,
    /// Profile name (see `aas profiles`)
    #[arg(long)]
    profile: String,
    /// Fail on warnings as well as errors
    #[arg(long)]
    strict: bool,
}

#[derive(Subcommand)]
enum LibraryCmd {
    /// List built-in documents
    List {
        /// Only this category: sfx, instruments, drums or examples
        #[arg(long)]
        category: Option<String>,
    },
    /// Print the source of a built-in document (YAML) so it can be studied or modified
    Show { id: String },
    /// Copy a built-in document to a file for editing
    Copy { id: String, dest: PathBuf },
}

#[derive(Subcommand)]
enum TheoryCmd {
    /// Notes and diatonic chords of a key, e.g. `aas theory key "D minor"`
    Key { name: String },
    /// Notes of a chord symbol, e.g. `aas theory chord Fmaj7`
    Chord {
        symbol: String,
        #[arg(long, default_value_t = 3)]
        octave: i32,
    },
    /// MIDI number and frequency of a note, e.g. `aas theory note C4`
    Note { name: String },
    /// List scale names
    Scales,
}

#[derive(Args)]
struct PackArgs {
    /// Path to a pack manifest (.yaml/.json)
    manifest: String,
    /// Output directory (default: the manifest's `output_dir`, else ./out)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

pub fn run() -> i32 {
    let cli = match Cli::try_parse() {
        Ok(c) => c,
        Err(e) => {
            use clap::error::ErrorKind;
            if matches!(e.kind(), ErrorKind::DisplayHelp | ErrorKind::DisplayVersion | ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand) {
                let _ = e.print();
                return if matches!(e.kind(), ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand) { 2 } else { 0 };
            }
            let msg = e.render().to_string();
            let first = msg.lines().find(|l| !l.trim().is_empty()).unwrap_or("invalid command line").trim_start_matches("error: ").to_string();
            let err = Error::usage(first).hint("run `aas --help` or `aas <command> --help` for usage, or `aas guide` for the full guide");
            println!("{}", serde_json::to_string_pretty(&json!({ "ok": false, "error": err.to_json() })).unwrap());
            return 2;
        }
    };
    let compact = cli.compact;
    let print = |v: &Value| {
        if compact {
            println!("{}", serde_json::to_string(v).unwrap());
        } else {
            println!("{}", serde_json::to_string_pretty(v).unwrap());
        }
    };
    match dispatch(cli.cmd) {
        Ok(Output::Json(mut v, code)) => {
            if let Some(o) = v.as_object_mut() {
                let mut m = Map::new();
                m.insert("ok".into(), json!(code == 0));
                for (k, val) in std::mem::take(o) {
                    m.insert(k, val);
                }
                *o = m;
            }
            print(&v);
            code
        }
        Ok(Output::Text(t)) => {
            print!("{t}");
            if !t.ends_with('\n') {
                println!();
            }
            0
        }
        Err(e) => {
            print(&json!({ "ok": false, "error": e.to_json() }));
            e.exit_code()
        }
    }
}

enum Output {
    Json(Value, i32),
    Text(String),
}

fn ok(v: Value) -> Result<Output> {
    Ok(Output::Json(v, 0))
}

fn dispatch(cmd: Cmd) -> Result<Output> {
    match cmd {
        Cmd::Render(a) => cmd_render(&a.source, &a.common),
        Cmd::Sfx(a) => {
            let id = if a.recipe.contains('/') { a.recipe.clone() } else { format!("sfx/{}", a.recipe) };
            cmd_render(&format!("builtin:{id}"), &a.common)
        }
        Cmd::Validate { source } => cmd_validate(&source),
        Cmd::Analyze(a) => {
            let (an, _, loop_in_file) = analyze_file(&a.file, a.is_loop)?;
            let mut v = an.to_json(a.detail);
            let o = v.as_object_mut().unwrap();
            let mut out = Map::new();
            out.insert("file_path".into(), json!(a.file.display().to_string()));
            out.insert("analyzed_as_loop".into(), json!(a.is_loop || loop_in_file));
            out.extend(std::mem::take(o));
            ok(Value::Object(out))
        }
        Cmd::Lint(a) => {
            let p = find_profile(&a.profile)?;
            let (an, _, _) = analyze_file(&a.file, p.is_loop)?;
            let res = lint(&an, &p);
            let passed = res.passed(a.strict);
            let mut v = res.to_json(a.strict);
            v["file_path"] = json!(a.file.display().to_string());
            v["summary"] = an.summary_json();
            v["summary"].as_object_mut().unwrap().remove("issues");
            Ok(Output::Json(v, if passed { 0 } else { 5 }))
        }
        Cmd::Compare { candidate, reference } => {
            let (a, _, _) = analyze_file(&candidate, false)?;
            let (b, _, _) = analyze_file(&reference, false)?;
            ok(compare(&a, &b))
        }
        Cmd::Describe { file, is_loop } => {
            let (an, _, _) = analyze_file(&file, is_loop)?;
            ok(json!({
                "description": an.description,
                "energy_map": an.ascii_map().lines().collect::<Vec<_>>(),
                "energy_map_note": "rows are frequency bands (air at the top, sub at the bottom), columns are time from start to end; darker characters ( .:-=+*#%@) mean more energy",
                "issues": an.issues.iter().map(|i| i.to_json()).collect::<Vec<_>>(),
            }))
        }
        Cmd::Spectrogram { file, output, width, height } => {
            let w = read_wav(&file)?;
            let out = output.unwrap_or_else(|| file.with_extension("png"));
            let (pw, ph) = spectrogram::write_png(&w.sig, w.sample_rate, &out, width.clamp(64, 4000), height.clamp(64, 2000))?;
            ok(json!({
                "path": out.display().to_string(),
                "width": pw,
                "height": ph,
                "x_axis": format!("time, 0 to {:.3} s left to right", w.sig.len() as f64 / w.sample_rate as f64),
                "y_axis": format!("frequency, logarithmic, 20 Hz at the bottom to {:.0} Hz at the top", 20000f64.min(w.sample_rate as f64 / 2.0)),
                "color": "black = silent, purple = quiet, orange = loud, pale yellow = loudest; 80 dB range",
            }))
        }
        Cmd::Library { cmd } => cmd_library(cmd),
        Cmd::Nodes { name, functions } => cmd_nodes(name, functions),
        Cmd::Schema { kind } => cmd_schema(kind),
        Cmd::Profiles => ok(json!({
            "profiles": profiles().iter().map(|p| json!({
                "name": p.name,
                "description": p.description,
                "duration_s": [p.duration.0, p.duration.1],
                "loop": p.is_loop,
                "lufs_range": p.lufs.map(|l| vec![l.0, l.1]),
                "max_attack_ms": p.max_attack_ms,
                "min_attack_ms": p.min_attack_ms,
            })).collect::<Vec<_>>(),
            "usage": "aas lint file.wav --profile <name>, or set `profile: <name>` in a patch so every render is checked",
        })),
        Cmd::Theory { cmd } => cmd_theory(cmd),
        Cmd::Pack(a) => cmd_pack(&a),
        Cmd::Guide => Ok(Output::Text(GUIDE.to_string())),
    }
}

fn analyze_file(path: &Path, is_loop: bool) -> Result<(Analysis, Sig, bool)> {
    let w = read_wav(path)?;
    if w.sig.is_empty() {
        return Err(Error::new("unsupported_format", format!("{} contains no audio", path.display())));
    }
    let loop_in_file = w.loop_points.is_some();
    let an = analyze(&w.sig, w.sample_rate, &AnalyzeOpts { is_loop: is_loop || loop_in_file });
    Ok((an, w.sig, loop_in_file))
}

fn parse_sets(sets: &[String]) -> Result<Vec<(String, f64)>> {
    let mut out = Vec::new();
    for s in sets {
        let (k, v) = s.split_once('=').ok_or_else(|| Error::usage(format!("--set expects NAME=VALUE, got '{s}'")).hint("e.g. --set size=0.8 --set freq=C3"))?;
        let val = match v.trim().parse::<f64>() {
            Ok(x) => x,
            Err(_) => match parse_note(v.trim()) {
                Some(m) if k.trim() == "freq" => midi_to_hz(m as f64),
                _ => return Err(Error::usage(format!("--set {k}: '{v}' is not a number")).hint("values are numbers; for freq a note name such as C3 also works")),
            },
        };
        out.push((k.trim().to_string(), val));
    }
    Ok(out)
}

fn doc_stem(d: &Doc) -> String {
    let base = d.origin.trim_start_matches("builtin:");
    Path::new(base).file_stem().map(|s| s.to_string_lossy().to_string()).filter(|s| s != "stdin" && !s.is_empty()).unwrap_or_else(|| "output".into())
}

/// Resolve the output path for a render: explicit file, explicit directory or default name.
fn out_path(output: &Option<String>, stem: &str, suffix: &str) -> PathBuf {
    match output {
        Some(o) if o.ends_with('/') || o.ends_with('\\') || Path::new(o).is_dir() => Path::new(o).join(format!("{stem}{suffix}.wav")),
        Some(o) => {
            let p = Path::new(o);
            if suffix.is_empty() {
                p.to_path_buf()
            } else {
                let s = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| stem.to_string());
                p.with_file_name(format!("{s}{suffix}.wav"))
            }
        }
        None => PathBuf::from(format!("{stem}{suffix}.wav")),
    }
}

struct FileReport {
    json: Value,
    lint_failed: bool,
}

/// Write a WAV file and build its report entry (with measurements and lint).
fn write_and_report(path: &Path, sig: &Sig, sr: u32, fmt: SampleFormat, loop_points: Option<(u32, u32)>, analysis: bool, profile: Option<&str>, extra: Value) -> Result<FileReport> {
    if sig.has_non_finite() {
        return Err(Error::new("render_error", "the render produced non-finite samples").hint("a parameter is probably far out of range: check feedback, q and gain values"));
    }
    write_wav(path, sig, sr, fmt, loop_points)?;
    let mut m = Map::new();
    m.insert("path".into(), json!(path.display().to_string()));
    m.insert("duration_s".into(), json!(round_to(sig.len() as f64 / sr as f64, 4)));
    m.insert("sample_rate".into(), json!(sr));
    m.insert("channels".into(), json!(sig.channels()));
    m.insert("format".into(), json!(fmt.name()));
    if let Some((a, b)) = loop_points {
        m.insert("loop".into(), json!({ "start_sample": a, "end_sample": b, "note": "loop points are embedded in the WAV smpl chunk" }));
    }
    if let Some(o) = extra.as_object() {
        for (k, v) in o {
            m.insert(k.clone(), v.clone());
        }
    }
    let mut lint_failed = false;
    if analysis {
        let is_loop = loop_points.is_some();
        let an = analyze(sig, sr, &AnalyzeOpts { is_loop });
        m.insert("analysis".into(), an.summary_json());
        if let Some(pn) = profile {
            let p = find_profile(pn)?;
            let res = lint(&an, &p);
            lint_failed = !res.passed(false);
            m.insert("lint".into(), res.to_json(false));
        }
    }
    Ok(FileReport { json: Value::Object(m), lint_failed })
}

fn cmd_render(source: &str, c: &RenderCommon) -> Result<Output> {
    let started = Instant::now();
    let d = doc::load(source)?;
    let fmt = SampleFormat::parse(&c.bits)?;
    let os = quality_to_oversample(&c.quality)?;
    if !(8000..=192000).contains(&c.sr) {
        return Err(Error::usage(format!("sample rate {} is out of range", c.sr)).hint("use 44100, 48000 (game standard) or 96000"));
    }
    if c.variations == 0 || c.variations > 64 {
        return Err(Error::usage("--variations must be between 1 and 64"));
    }
    let stem = doc_stem(&d);
    let mut outputs = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut lint_failed = false;
    match d.kind {
        DocKind::Patch => {
            if c.stems || c.sections {
                return Err(Error::usage("--stems and --sections apply to scores, not patches"));
            }
            let patch = Patch::from_value(&d.value)?;
            let params = parse_sets(&c.set)?;
            let base_seed = c.seed.or(patch.seed).unwrap_or(1);
            let profile = c.profile.clone().or(patch.profile.clone());
            for v in 0..c.variations {
                let seed = base_seed + v as u64;
                let asset = render_patch(&patch, &AssetOpts { sr: c.sr, oversample: os, seed, params: params.clone(), duration: c.duration, force_loop: c.looped, normalize: if c.raw { Some(crate::graph::Normalize::None) } else { None }, probe: c.probe })?;
                let suffix = if c.variations > 1 { format!("_{:02}", v + 1) } else { String::new() };
                let path = out_path(&c.output, &stem, &suffix);
                let mut rep = write_and_report(&path, &asset.sig, c.sr, fmt, asset.loop_points, !c.no_analysis, profile.as_deref(), json!({ "seed": seed }))?;
                if !asset.warnings.is_empty() {
                    rep.json["warnings"] = json!(asset.warnings);
                }
                if c.probe {
                    rep.json["node_levels"] = json!({
                        "note": "level of each node's output before final normalization; -240 means silent. Compare layers feeding a mix to judge their balance.",
                        "nodes": asset.levels.iter().map(|l| json!({ "node": l.name, "type": l.node_type, "peak_db": round_to(l.peak_db, 1), "rms_db": round_to(l.rms_db, 1), "stereo": l.stereo })).collect::<Vec<_>>(),
                    });
                }
                lint_failed |= rep.lint_failed;
                outputs.push(rep.json);
            }
        }
        DocKind::Score => {
            if !c.set.is_empty() {
                return Err(Error::usage("--set applies to patches; for scores edit instrument `params` in the document"));
            }
            if c.variations > 1 {
                return Err(Error::usage("--variations applies to patches; for scores change --seed"));
            }
            let score = Score::from_value(&d.value, &d.base_dir)?;
            warnings.extend(score.warnings.iter().cloned());
            let opts = ScoreOpts { sr: c.sr, oversample: os, seed: c.seed, looped: c.looped, arrangement: None, fixed_gain: None };
            let path = out_path(&c.output, &stem, "");
            let stem_dir = path.with_file_name(format!("{}_stems", path.file_stem().unwrap().to_string_lossy()));
            let mut stem_reports: Vec<Value> = Vec::new();
            let loop_len = (score.total_beats(&score.arrangement) * score.beat_seconds() * c.sr as f64).round() as u32;
            let lp = if c.looped { Some((0u32, loop_len.saturating_sub(1))) } else { None };
            let rendered = {
                let mut sink = |name: &str, s: &Sig| -> Result<()> {
                    let p = stem_dir.join(format!("{name}.wav"));
                    let rep = write_and_report(&p, s, c.sr, fmt, lp, false, None, json!({ "stem": name }))?;
                    stem_reports.push(rep.json);
                    Ok(())
                };
                render_score(&score, &opts, if c.stems { Some(&mut sink) } else { None })?
            };
            warnings.extend(rendered.warnings.iter().cloned());
            let profile = c.profile.clone().or_else(|| Some(if c.looped { "music-loop".to_string() } else { "music".to_string() }).filter(|_| rendered.musical_samples as f64 / c.sr as f64 >= 8.0));
            let extra = json!({
                "role": "master",
                "tempo_bpm": score.tempo,
                "time_signature": score.time_signature,
                "key": score.key.as_ref().map(|k| k.name()),
                "bars": round_to(score.total_beats(&score.arrangement) / score.beats_per_bar, 3),
                "musical_length_s": round_to(rendered.musical_samples as f64 / c.sr as f64, 4),
                "musical_length_samples": rendered.musical_samples,
                "notes_rendered": rendered.note_count,
                "master_gain_db": round_to(crate::util::lin_to_db(rendered.master_gain), 2),
                "limiter_reduction_db": round_to(rendered.limiter_gr_db, 2),
                "timeline": rendered.timeline,
            });
            let rep = write_and_report(&path, &rendered.master, c.sr, fmt, lp, !c.no_analysis, profile.as_deref(), extra)?;
            lint_failed |= rep.lint_failed;
            outputs.push(rep.json);
            if c.stems {
                outputs.push(json!({ "role": "stems", "note": "post-fader, pre-master-limiter; summing all stems reproduces the mix before limiting", "files": stem_reports }));
            }
            if c.sections {
                let mut seen: Vec<String> = Vec::new();
                let mut files = Vec::new();
                for name in &score.arrangement {
                    if seen.contains(name) {
                        continue;
                    }
                    seen.push(name.clone());
                    let so = ScoreOpts { arrangement: Some(vec![name.clone()]), fixed_gain: Some(rendered.master_gain), looped: false, ..opts.clone() };
                    let r = render_score(&score, &so, None)?;
                    let p = path.with_file_name(format!("{}_{}.wav", path.file_stem().unwrap().to_string_lossy(), name));
                    let extra = json!({
                        "section": name,
                        "musical_length_s": round_to(r.musical_samples as f64 / c.sr as f64, 4),
                        "musical_length_samples": r.musical_samples,
                        "bars": score.sections.iter().find(|s| &s.name == name).map(|s| s.bars),
                    });
                    files.push(write_and_report(&p, &r.master, c.sr, fmt, None, false, None, extra)?.json);
                }
                outputs.push(json!({
                    "role": "sections",
                    "note": "each file holds one section plus its ring-out tail; in game middleware set the exit cue at musical_length_samples so the tail overlaps the next section",
                    "files": files,
                }));
            }
        }
        DocKind::Pack => return Err(Error::usage("this is a pack manifest: use `aas pack <manifest>`")),
    }
    let mut v = json!({
        "command": "render",
        "source": d.origin,
        "kind": if d.kind == DocKind::Patch { "patch" } else { "score" },
        "quality": c.quality,
        "outputs": outputs,
        "render_time_s": round_to(started.elapsed().as_secs_f64(), 2),
    });
    if !warnings.is_empty() {
        v["warnings"] = json!(warnings);
    }
    if lint_failed {
        v["next_step"] = json!("lint found errors: apply the `fix` of each issue in outputs[].lint.issues and render again");
    }
    ok(v)
}

fn cmd_validate(source: &str) -> Result<Output> {
    let d = doc::load(source)?;
    match d.kind {
        DocKind::Patch => {
            let p = Patch::from_value(&d.value)?;
            ok(json!({
                "command": "validate",
                "source": d.origin,
                "kind": "patch",
                "valid": true,
                "nodes": p.nodes.iter().map(|n| json!({"name": n.name, "type": n.spec.name})).collect::<Vec<_>>(),
                "params": p.params.iter().map(|x| json!({"name": x.name, "default": x.default, "min": x.min, "max": x.max, "doc": x.doc})).collect::<Vec<_>>(),
                "duration_s": p.duration_with(&[])?,
                "loop": p.loop_crossfade.is_some(),
                "profile": p.profile,
                "note": "valid means the document is well-formed; render it to check how it sounds",
            }))
        }
        DocKind::Score => {
            let s = Score::from_value(&d.value, &d.base_dir)?;
            let mut v = json!({ "command": "validate", "source": d.origin, "kind": "score", "valid": true, "summary": s.summary() });
            if !s.warnings.is_empty() {
                v["warnings"] = json!(s.warnings);
            }
            ok(v)
        }
        DocKind::Pack => {
            let assets = parse_pack(&d)?;
            ok(json!({ "command": "validate", "source": d.origin, "kind": "pack", "valid": true, "assets": assets.items.len() }))
        }
    }
}

fn entry_meta(e: &library::Entry) -> Value {
    let v = doc::parse_text(e.text, &e.id).ok();
    let get = |k: &str| v.as_ref().and_then(|v| v.get(k)).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let mut m = json!({ "id": e.id, "description": get("description") });
    let kind = v.as_ref().and_then(|v| doc::detect_kind(v, &e.id).ok());
    if kind == Some(DocKind::Patch) {
        if let Some(Ok(p)) = v.as_ref().map(Patch::from_value) {
            if !p.params.is_empty() {
                m["params"] = Value::Array(p.params.iter().map(|x| json!({ "name": x.name, "default": x.default, "min": x.min, "max": x.max, "doc": x.doc })).collect());
            }
            if let Ok(Some(d)) = p.duration_with(&[]) {
                m["duration_s"] = json!(round_to(d, 3));
            }
            if let Some(pr) = p.profile {
                m["profile"] = json!(pr);
            }
            if p.loop_crossfade.is_some() {
                m["loop"] = json!(true);
            }
        }
    } else if kind == Some(DocKind::Score) {
        m["kind"] = json!("score");
    }
    m
}

fn cmd_library(cmd: LibraryCmd) -> Result<Output> {
    match cmd {
        LibraryCmd::List { category } => {
            let all = library::entries();
            let cats: Vec<String> = {
                let mut c: Vec<String> = all.iter().filter_map(|e| e.id.split('/').next().map(String::from)).collect();
                c.dedup();
                c
            };
            if let Some(c) = &category {
                if !cats.contains(c) {
                    let refs: Vec<&str> = cats.iter().map(|s| s.as_str()).collect();
                    return Err(Error::new("unknown_name", format!("unknown category '{c}'")).hint(hint_from(c, &refs)));
                }
            }
            let items: Vec<Value> = all.iter().filter(|e| category.as_ref().map_or(true, |c| e.id.starts_with(&format!("{c}/")))).map(entry_meta).collect();
            ok(json!({
                "categories": cats,
                "count": items.len(),
                "items": items,
                "usage": {
                    "render_a_recipe": "aas sfx <name> --set <param>=<value> -o out.wav",
                    "use_in_a_score": "instruments: {bass: {patch: builtin:instruments/bass_sub}}",
                    "read_the_source": "aas library show <id>",
                },
            }))
        }
        LibraryCmd::Show { id } => Ok(Output::Text(library::get(id.trim_start_matches("builtin:"))?.to_string())),
        LibraryCmd::Copy { id, dest } => {
            let text = library::get(id.trim_start_matches("builtin:"))?;
            if let Some(dir) = dest.parent() {
                if !dir.as_os_str().is_empty() {
                    std::fs::create_dir_all(dir).map_err(|e| Error::io(format!("cannot create {}: {e}", dir.display())))?;
                }
            }
            std::fs::write(&dest, text).map_err(|e| Error::io(format!("cannot write {}: {e}", dest.display())))?;
            ok(json!({ "copied": id, "to": dest.display().to_string(), "next_step": format!("edit the file, then `aas render {}`", dest.display()) }))
        }
    }
}

fn cmd_nodes(name: Option<String>, functions: bool) -> Result<Output> {
    if functions {
        return ok(json!({
            "functions": FUNCTION_DOCS.iter().map(|f| json!({"signature": f.0, "doc": f.1})).collect::<Vec<_>>(),
            "variables": BUILTIN_VARS.iter().map(|f| json!({"name": f.0, "doc": f.1})).collect::<Vec<_>>(),
            "constants": ["pi", "tau", "e"],
            "operators": "+ - * / ^ and parentheses",
            "notes": [
                "Any numeric field accepts an expression in quotes, e.g. cutoff: \"200 + 4000 * env\".",
                "Node names are variables: referencing one reads that node's output signal.",
                "Patch `params` are variables too and can be overridden with --set.",
            ],
        }));
    }
    let param_json = |p: &crate::nodes::ParamSpec| {
        let mut m = json!({ "name": p.name, "type": p.kind.label(), "doc": p.doc, "required": p.required });
        if let Some(d) = p.default {
            m["default"] = json!(d);
        }
        if matches!(p.kind, Kind::Par | Kind::Num | Kind::Int) {
            if let (Some(lo), Some(hi)) = (p.min, p.max) {
                if lo > -1e8 && hi < 1e8 {
                    m["range"] = json!([lo, hi]);
                }
            }
        }
        m
    };
    match name {
        Some(n) => {
            let names: Vec<&str> = specs().iter().map(|s| s.name).collect();
            let s = specs().iter().find(|s| s.name == n).ok_or_else(|| Error::new("unknown_name", format!("unknown node type '{n}'")).hint(hint_from(&n, &names)))?;
            ok(json!({
                "node": s.name,
                "category": s.category,
                "doc": s.doc,
                "params": s.params.iter().map(param_json).collect::<Vec<_>>(),
                "example": s.example,
            }))
        }
        None => {
            let mut cats: Vec<&str> = Vec::new();
            for s in specs() {
                if !cats.contains(&s.category) {
                    cats.push(s.category);
                }
            }
            let mut by_cat = Map::new();
            for c in cats {
                by_cat.insert(c.to_string(), Value::Array(specs().iter().filter(|s| s.category == c).map(|s| json!({ "node": s.name, "doc": s.doc, "params": s.params.iter().map(|p| p.name).collect::<Vec<_>>(), "example": s.example })).collect()));
            }
            ok(json!({ "nodes": by_cat, "details": "aas nodes <name> shows every parameter with type, default and range", "expressions": "aas nodes --functions" }))
        }
    }
}

fn cmd_schema(kind: Option<String>) -> Result<Output> {
    let patch = json!({
        "kind": "\"patch\" (optional; inferred from `nodes`)",
        "description": "string: what the sound is",
        "duration": "number, seconds. Required for sounds; ignored for instruments in a score (note length + tail is used)",
        "gate": "number, seconds the gate stays on (default: the whole duration). adsr releases when the gate ends",
        "tail": "number, seconds kept after a note ends when used as an instrument (default 0.5)",
        "seed": "integer: default random seed",
        "params": "mapping name -> number, or name -> {default, min, max, doc}. Readable in expressions; overridable with --set or from a score",
        "nodes": "mapping name -> {type, ...parameters}. See `aas nodes`. Order does not matter",
        "output": "node name or expression (default: the node named `out`)",
        "loop": "true or {crossfade: seconds}: make the result loop seamlessly",
        "normalize": "{peak_db: -1} (default), {lufs: -18} or none",
        "profile": "lint profile name checked on every render (see `aas profiles`)",
        "category": "string", "tags": "list of strings", "name": "string",
    });
    let score = json!({
        "kind": "\"score\" (optional; inferred)",
        "title": "string", "description": "string",
        "tempo": "number, beats per minute (default 120)",
        "time_signature": "string such as \"4/4\" (default), \"3/4\", \"6/8\"",
        "key": "string such as \"C minor\" or \"F# dorian\"; enables out-of-key warnings",
        "scale_lock": "true: snap out-of-key notes to the key",
        "swing": "0..1: delay off-beats (1 = full triplet swing)", "swing_unit": "beats; 0.5 swings eighths (default), 0.25 sixteenths",
        "humanize": "{timing_ms, velocity}: random variation per note",
        "seed": "integer", "tail": "seconds of ring-out kept after the last bar (default 4)",
        "instruments": {
            "<name>": {
                "patch": "path, builtin:<id> or inline patch mapping (pitched instrument)",
                "kit": "mapping drum name -> patch reference or {patch, params, gain_db, pan} (drum kit)",
                "params": "mapping of patch parameter overrides",
                "gain_db": "number", "pan": "-1..1", "transpose": "semitones",
                "fx": "list of effect nodes applied in order, e.g. [{type: lowpass, cutoff: 3000}, {type: delay, time: 0.3}]",
                "sends": "mapping bus name -> send level in dB",
                "duck": "{source: <instrument or instrument.piece>, amount_db, attack_ms, release_ms}: sidechain ducking",
                "mute": "true/false",
            }
        },
        "buses": { "<name>": { "fx": "list of effect nodes (use mix: 1 on reverbs/delays here)", "gain_db": "number" } },
        "master": { "fx": "list of effect nodes", "lufs": "target loudness (default -16) or none", "ceiling_db": "true-peak ceiling (default -1)", "fade_in": "seconds", "fade_out": "seconds" },
        "patterns": {
            "<name>": {
                "notes": "string in note notation: \"C3:1 Eb3:0.5 _:0.5 G3:2\" (NOTE:BEATS@VELOCITY, _ rest, ~ tie, + chord, | bar line), or a list of strings (one per bar)",
                "drums": "mapping drum name -> step string: x hit, X accent, o ghost, . rest",
                "chords": "string of chord symbols: \"Cm:4 Ab:4 Eb:4 Bb:4\" (SYMBOL:BEATS)",
                "events": "list of {at, note, dur, vel} (beats)",
                "step": "beats per token without an explicit duration (notes: 1, drums: 0.25)",
                "length": "beats (default: derived from the content)",
                "octave": "notes: octave shift; chords: octave of the voicing (default 3)",
                "transpose": "semitones", "vel": "default velocity 0..1", "gate": "note length multiplier (0.5 = staccato)",
                "voicing": "chords: smooth (default, minimal movement), close, open",
                "style": "chords: block (default), arp_up, arp_down, arp_updown, strum, root", "rate": "chords: beats per arpeggio note",
            }
        },
        "sections": { "<name>": { "bars": "number", "tracks": "mapping instrument -> pattern name, list of pattern names (played in turn, repeating to fill the section), or {pattern, transpose, vel, gain_db}" } },
        "arrangement": "list of section names, or {section, repeat} entries (default: all sections in order)",
    });
    let pack = json!({
        "kind": "\"pack\"",
        "name": "string",
        "output_dir": "directory for the rendered files (default ./out)",
        "defaults": "{sr, bits, quality} applied to every asset",
        "assets": [{
            "id": "file name stem",
            "source": "path or builtin:<id> of a patch or score",
            "set": "mapping of patch parameter overrides",
            "seed": "integer", "variations": "integer: number of round-robin variations",
            "loop": "true/false", "stems": "true/false (scores)", "sections": "true/false (scores)",
            "profile": "lint profile to check", "duration": "seconds (patches)",
        }],
    });
    match kind.as_deref() {
        Some("patch") => ok(json!({ "patch": patch })),
        Some("score") => ok(json!({ "score": score })),
        Some("pack") => ok(json!({ "pack": pack })),
        None => ok(json!({ "patch": patch, "score": score, "pack": pack, "see_also": ["aas nodes", "aas nodes --functions", "aas profiles", "aas guide"] })),
        Some(other) => Err(Error::usage(format!("unknown schema '{other}'")).hint("use patch, score or pack")),
    }
}

fn cmd_theory(cmd: TheoryCmd) -> Result<Output> {
    match cmd {
        TheoryCmd::Key { name } => {
            let k = Key::parse(&name)?;
            ok(json!({ "key": k.name(), "notes": k.note_names(), "diatonic_chords": k.diatonic_chords(), "tip": "progressions built from these chords stay in key" }))
        }
        TheoryCmd::Chord { symbol, octave } => {
            let c = Chord::parse(&symbol)?;
            let midi = c.close(octave);
            ok(json!({ "chord": symbol, "notes": c.note_names(octave), "midi": midi, "hz": midi.iter().map(|m| round_to(midi_to_hz(*m as f64), 2)).collect::<Vec<_>>() }))
        }
        TheoryCmd::Note { name } => {
            let m = parse_note(&name).ok_or_else(|| Error::usage(format!("cannot read note '{name}'")).hint("note names look like C4, F#3, Bb2 (C4 is middle C)"))?;
            ok(json!({ "note": midi_to_name(m), "midi": m, "hz": round_to(midi_to_hz(m as f64), 3) }))
        }
        TheoryCmd::Scales => ok(json!({ "scales": SCALES.iter().map(|s| json!({"name": s.0, "semitones": s.1})).collect::<Vec<_>>() })),
    }
}

struct PackItem {
    id: String,
    source: String,
    common: RenderCommon,
}

struct PackSpec {
    name: String,
    output_dir: Option<String>,
    items: Vec<PackItem>,
}

fn parse_pack(d: &Doc) -> Result<PackSpec> {
    let o = as_obj(&d.value, "a pack")?;
    check_keys(o, &["kind", "name", "description", "output_dir", "defaults", "assets"])?;
    let empty = Map::new();
    let defaults = match o.get("defaults") {
        Some(v) => as_obj(v, "'defaults'").map_err(|e| e.at("defaults"))?,
        None => &empty,
    };
    check_keys(defaults, &["sr", "bits", "quality"]).map_err(|e| e.within("defaults"))?;
    let sr = get_f64(defaults, "sr")?.unwrap_or(48000.0) as u32;
    let bits = match defaults.get("bits") {
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::String(s)) => s.clone(),
        _ => "24".into(),
    };
    let quality = get_str(defaults, "quality")?.unwrap_or("standard").to_string();
    let assets = o.get("assets").and_then(|a| a.as_array()).ok_or_else(|| Error::invalid("pack needs an 'assets' list").at("assets").hint("e.g. assets: [{id: click, source: builtin:sfx/ui_click}]"))?;
    let mut items = Vec::new();
    for (i, a) in assets.iter().enumerate() {
        let path = format!("assets.{i}");
        let ao = as_obj(a, "an asset").map_err(|e| e.at(path.clone()))?;
        check_keys(ao, &["id", "source", "set", "seed", "variations", "loop", "stems", "sections", "profile", "duration"]).map_err(|e| e.within(&path))?;
        let source = get_str(ao, "source").map_err(|e| e.within(&path))?.ok_or_else(|| Error::invalid("asset needs a 'source'").at(path.clone()))?.to_string();
        let id = get_str(ao, "id").map_err(|e| e.within(&path))?.map(String::from).unwrap_or_else(|| Path::new(source.trim_start_matches("builtin:")).file_stem().unwrap().to_string_lossy().to_string());
        if id.is_empty() || id.contains(['/', '\\']) {
            return Err(Error::invalid("asset id must be a plain file name without slashes").at(format!("{path}.id")));
        }
        if items.iter().any(|x: &PackItem| x.id == id) {
            return Err(Error::invalid(format!("asset id '{id}' is used twice")).at(format!("{path}.id")).hint("give each asset a unique id"));
        }
        let mut set = Vec::new();
        if let Some(sv) = ao.get("set") {
            let so = as_obj(sv, "'set'").map_err(|e| e.at(format!("{path}.set")))?;
            for k in so.keys() {
                let v = get_f64(so, k).map_err(|e| e.within(&format!("{path}.set")))?.ok_or_else(|| Error::invalid("value must be a number").at(format!("{path}.set.{k}")))?;
                set.push(format!("{k}={v}"));
            }
        }
        let flag = |k: &str| ao.get(k).and_then(|b| b.as_bool()).unwrap_or(false);
        let source_abs = if source.starts_with("builtin:") || Path::new(&source).is_absolute() { source.clone() } else { d.base_dir.join(&source).display().to_string() };
        items.push(PackItem {
            id,
            source: source_abs,
            common: RenderCommon {
                output: None,
                set,
                seed: get_f64(ao, "seed").map_err(|e| e.within(&path))?.map(|s| s as u64),
                variations: get_f64(ao, "variations").map_err(|e| e.within(&path))?.unwrap_or(1.0) as usize,
                sr,
                bits: bits.clone(),
                quality: quality.clone(),
                looped: flag("loop"),
                duration: get_f64(ao, "duration").map_err(|e| e.within(&path))?,
                stems: flag("stems"),
                sections: flag("sections"),
                profile: get_str(ao, "profile").map_err(|e| e.within(&path))?.map(String::from),
                no_analysis: false,
                probe: false,
                raw: false,
            },
        });
    }
    if items.is_empty() {
        return Err(Error::invalid("pack has no assets").at("assets"));
    }
    Ok(PackSpec { name: get_str(o, "name")?.unwrap_or("pack").to_string(), output_dir: get_str(o, "output_dir")?.map(String::from), items })
}

fn cmd_pack(a: &PackArgs) -> Result<Output> {
    let started = Instant::now();
    let d = doc::load(&a.manifest)?;
    if d.kind != DocKind::Pack {
        return Err(Error::usage("this document is not a pack manifest").hint("a pack has `kind: pack` and an `assets:` list; use `aas render` for patches and scores"));
    }
    let spec = parse_pack(&d)?;
    let out_dir = a.output.clone().unwrap_or_else(|| match &spec.output_dir {
        Some(o) if Path::new(o).is_absolute() => PathBuf::from(o),
        Some(o) => d.base_dir.join(o),
        None => PathBuf::from("out"),
    });
    let mut assets = Vec::new();
    let (mut failed, mut lint_errors) = (0, 0);
    for item in &spec.items {
        let mut common = item.common.clone();
        common.output = Some(out_dir.join(format!("{}.wav", item.id)).display().to_string());
        match cmd_render(&item.source, &common) {
            Ok(Output::Json(v, _)) => {
                let errs = v["outputs"].as_array().map_or(0, |o| o.iter().filter(|f| f["lint"]["passed"] == json!(false)).count());
                lint_errors += errs;
                assets.push(json!({ "id": item.id, "source": item.source, "ok": true, "outputs": v["outputs"], "warnings": v.get("warnings").cloned().unwrap_or(json!([])) }));
            }
            Ok(Output::Text(_)) => {}
            Err(e) => {
                failed += 1;
                assets.push(json!({ "id": item.id, "source": item.source, "ok": false, "error": e.to_json() }));
            }
        }
    }
    let manifest = json!({
        "pack": spec.name,
        "generator": format!("aas {}", env!("CARGO_PKG_VERSION")),
        "assets": assets,
    });
    let mpath = out_dir.join("manifest.json");
    std::fs::create_dir_all(&out_dir).map_err(|e| Error::io(format!("cannot create {}: {e}", out_dir.display())))?;
    std::fs::write(&mpath, serde_json::to_string_pretty(&manifest).unwrap()).map_err(|e| Error::io(format!("cannot write {}: {e}", mpath.display())))?;
    let v = json!({
        "command": "pack",
        "pack": spec.name,
        "output_dir": out_dir.display().to_string(),
        "manifest": mpath.display().to_string(),
        "assets_total": spec.items.len(),
        "assets_failed": failed,
        "files_failing_lint": lint_errors,
        "assets": manifest["assets"],
        "render_time_s": round_to(started.elapsed().as_secs_f64(), 2),
    });
    Ok(Output::Json(v, if failed > 0 { 1 } else if lint_errors > 0 { 5 } else { 0 }))
}
