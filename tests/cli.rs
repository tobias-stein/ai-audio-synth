//! End-to-end tests that drive the `aas` binary the way an agent would.

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_aas")
}

fn run(args: &[&str]) -> (i32, Value) {
    let out = Command::new(bin()).args(args).output().expect("failed to run aas");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let json: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("stdout is not JSON ({e}) for {args:?}:\n{text}"));
    (out.status.code().unwrap_or(-1), json)
}

fn run_stdin(args: &[&str], input: &str) -> (i32, Value) {
    use std::io::Write;
    let mut child = Command::new(bin()).args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(input.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    (out.status.code().unwrap_or(-1), serde_json::from_slice(&out.stdout).expect("stdout is not JSON"))
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("aas_test_{}_{}", std::process::id(), name));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn s(p: &Path) -> String {
    p.display().to_string()
}

fn library_items(category: &str) -> Vec<Value> {
    let (code, v) = run(&["library", "list", "--category", category]);
    assert_eq!(code, 0);
    v["items"].as_array().unwrap().clone()
}

fn issue_codes(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter(|i| i["severity"] == "error").map(|i| i["code"].as_str().unwrap().to_string()).collect()).unwrap_or_default()
}

#[test]
fn every_sfx_recipe_renders_and_passes_its_own_profile() {
    let dir = tmp("sfx");
    let items = library_items("sfx");
    assert!(items.len() >= 30, "expected a substantial recipe library, found {}", items.len());
    for item in items {
        let id = item["id"].as_str().unwrap();
        assert!(!item["description"].as_str().unwrap().is_empty(), "{id} has no description");
        assert!(item["profile"].is_string(), "{id} has no lint profile");
        let out = dir.join("x.wav");
        let (code, v) = run(&["render", &format!("builtin:{id}"), "-o", &s(&out)]);
        assert_eq!(code, 0, "{id}: {v}");
        let o = &v["outputs"][0];
        assert_eq!(o["lint"]["passed"], true, "{id} fails its own profile: {}", o["lint"]);
        assert!(issue_codes(&o["analysis"]["issues"]).is_empty(), "{id}: {}", o["analysis"]["issues"]);
        assert!(o.get("warnings").is_none(), "{id} renders with warnings: {}", o["warnings"]);
        let tp = o["analysis"]["true_peak_db"].as_f64().unwrap();
        assert!(tp <= -0.9, "{id}: true peak {tp}");
    }
}

#[test]
fn recipes_survive_parameter_extremes() {
    let dir = tmp("extremes");
    for item in library_items("sfx") {
        let id = item["id"].as_str().unwrap();
        let Some(params) = item["params"].as_array() else { continue };
        for p in params {
            for bound in ["min", "max"] {
                let Some(val) = p[bound].as_f64() else { continue };
                let set = format!("{}={}", p["name"].as_str().unwrap(), val);
                let out = dir.join("x.wav");
                let mut args = vec!["render".to_string(), format!("builtin:{id}"), "-o".into(), s(&out), "--quality".into(), "draft".into(), "--set".into(), set.clone()];
                if item["loop"] == true {
                    args.extend(["--duration".to_string(), "3".to_string()]);
                }
                let refs: Vec<&str> = args.iter().map(|a| a.as_str()).collect();
                let (code, v) = run(&refs);
                assert_eq!(code, 0, "{id} --set {set}: {v}");
                let codes = issue_codes(&v["outputs"][0]["analysis"]["issues"]);
                assert!(codes.is_empty(), "{id} --set {set}: {codes:?}");
            }
        }
    }
}

#[test]
fn every_instrument_and_drum_renders_cleanly() {
    let dir = tmp("inst");
    for cat in ["instruments", "drums"] {
        let items = library_items(cat);
        assert!(items.len() >= 15, "{cat}: {}", items.len());
        for item in items {
            let id = item["id"].as_str().unwrap();
            let out = dir.join("x.wav");
            let (code, v) = run(&["render", &format!("builtin:{id}"), "-o", &s(&out), "--raw"]);
            assert_eq!(code, 0, "{id}: {v}");
            let a = &v["outputs"][0]["analysis"];
            assert!(issue_codes(&a["issues"]).is_empty(), "{id}: {}", a["issues"]);
            // un-normalized level must leave headroom for mixing several parts
            let peak = a["peak_db"].as_f64().unwrap();
            assert!((-24.0..=-2.0).contains(&peak), "{id}: raw peak {peak} dBFS is outside the mixing range");
        }
    }
}

#[test]
fn every_example_validates() {
    for item in library_items("examples") {
        let id = item["id"].as_str().unwrap();
        let (code, v) = run(&["validate", &format!("builtin:{id}")]);
        assert_eq!(code, 0, "{id}: {v}");
        assert_eq!(v["valid"], true);
        assert!(v.get("warnings").is_none(), "{id} has warnings: {}", v["warnings"]);
    }
}

#[test]
fn score_renders_with_expected_tempo_key_and_loudness() {
    let dir = tmp("score");
    let out = dir.join("m.wav");
    let (code, v) = run(&["render", "builtin:examples/minimal_score", "-o", &s(&out), "--loop", "--stems"]);
    assert_eq!(code, 0, "{v}");
    let o = &v["outputs"][0];
    assert_eq!(o["musical_length_samples"], 460800); // 4 bars at 100 bpm, 48 kHz
    assert_eq!(o["duration_s"], 9.6);
    let a = &o["analysis"];
    assert!((a["lufs"].as_f64().unwrap() + 16.0).abs() < 0.6, "{a}");
    assert!(a["true_peak_db"].as_f64().unwrap() <= -0.95);
    assert!((a["tempo_bpm"].as_f64().unwrap() - 100.0).abs() < 2.0);
    assert_eq!(a["key"], "C minor");
    assert_eq!(o["lint"]["profile"], "music-loop");
    assert_eq!(o["lint"]["passed"], true, "{}", o["lint"]);
    let stems = v["outputs"][1]["files"].as_array().unwrap();
    assert_eq!(stems.len(), 4); // drums, bass, pad, hall bus
    for st in stems {
        assert!(Path::new(st["path"].as_str().unwrap()).exists());
    }
    // the loop is seamless
    let (_, an) = run(&["analyze", &s(&out)]);
    assert_eq!(an["analyzed_as_loop"], true);
    assert_eq!(an["loop_seam"]["audible"], false, "{}", an["loop_seam"]);
}

#[test]
fn rendering_is_deterministic_and_seeds_differ() {
    let dir = tmp("det");
    let (a, b, c) = (dir.join("a.wav"), dir.join("b.wav"), dir.join("c.wav"));
    for (p, seed) in [(&a, "7"), (&b, "7"), (&c, "8")] {
        let (code, _) = run(&["sfx", "explosion", "-o", &s(p), "--seed", seed, "--no-analysis"]);
        assert_eq!(code, 0);
    }
    let (ba, bb, bc) = (std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap(), std::fs::read(&c).unwrap());
    assert_eq!(ba, bb, "same seed must give identical files");
    assert_ne!(ba, bc, "different seeds must differ");
}

#[test]
fn variations_write_numbered_files() {
    let dir = tmp("var");
    let (code, v) = run(&["sfx", "sword_swing", "-o", &format!("{}/", s(&dir)), "--variations", "3"]);
    assert_eq!(code, 0, "{v}");
    let outs = v["outputs"].as_array().unwrap();
    assert_eq!(outs.len(), 3);
    for (i, o) in outs.iter().enumerate() {
        assert!(o["path"].as_str().unwrap().ends_with(&format!("sword_swing_{:02}.wav", i + 1)));
        assert_eq!(o["seed"], (i + 1) as u64);
    }
}

#[test]
fn level_does_not_depend_on_quality() {
    let dir = tmp("quality");
    let mut lufs = Vec::new();
    for q in ["draft", "standard", "high"] {
        let (code, v) = run(&["sfx", "gunshot", "-o", &s(&dir.join("g.wav")), "--raw", "--quality", q]);
        assert_eq!(code, 0, "{v}");
        lufs.push(v["outputs"][0]["analysis"]["lufs"].as_f64().unwrap());
    }
    assert!((lufs[0] - lufs[1]).abs() < 1.0 && (lufs[1] - lufs[2]).abs() < 0.5, "loudness changes with quality: {lufs:?}");
}

#[test]
fn errors_are_structured_and_actionable() {
    let cases: &[(&str, &str, &str)] = &[
        ("nodes: {a: {type: osk}}\nduration: 1", "nodes.a.type", "osc"),
        ("nodes: {a: {type: osc, frq: 3}}\nduration: 1", "nodes.a.frq", "freq"),
        ("nodes: {a: {type: lowpass, in: b, cutoff: 100}, out: {type: gain, in: a}}\nduration: 1", "nodes.a", "is not a node"),
        ("nodes: {a: {type: lowpass, in: b, cutoff: 100}, b: {type: gain, in: a}}\nduration: 1\noutput: b", "nodes", "cycle"),
        ("nodes: {out: {type: osc, wave: saw, unison: 99}}\nduration: 1", "nodes.out.unison", "maximum"),
        ("nodes: {out: {type: osc, freq: \"440 *\"}}\nduration: 1", "nodes.out.freq", "expression"),
        ("nodes: {out: {type: osc}}", "duration", "duration"),
        ("nodes: {out: {type: osc}}\nduration: 1\nnormalise: none", "normalise", "normalize"),
    ];
    for (doc, path, needle) in cases {
        let (code, v) = run_stdin(&["render", "-", "-o", "/dev/null"], doc);
        assert_eq!(code, 3, "{doc}: {v}");
        assert_eq!(v["ok"], false);
        let e = &v["error"];
        assert!(e["path"].as_str().unwrap_or("").starts_with(path), "{doc}: path {}", e["path"]);
        let text = format!("{} {}", e["message"].as_str().unwrap_or(""), e["hint"].as_str().unwrap_or(""));
        assert!(text.contains(needle), "{doc}: expected '{needle}' in: {text}");
    }
    // score errors
    let score = "tempo: 120\ninstruments: {b: {patch: builtin:instruments/bass_sub}}\npatterns: {p: {notes: \"C2:1 D2:1 | E2:4\"}}\nsections: {m: {bars: 1, tracks: {b: p}}}";
    let (code, v) = run_stdin(&["validate", "-"], score);
    assert_eq!(code, 3);
    assert!(v["error"]["message"].as_str().unwrap().contains("bar 1"), "{v}");
    assert_eq!(v["error"]["path"], "patterns.p.notes");
    let score = "instruments: {b: {patch: builtin:instruments/bass_subb}}\nsections: {m: {bars: 1}}";
    let (_, v) = run_stdin(&["validate", "-"], score);
    assert!(v["error"]["hint"].as_str().unwrap().contains("bass_sub"), "{v}");
    // usage errors are JSON too
    let (code, v) = run(&["render"]);
    assert_eq!(code, 2);
    assert_eq!(v["error"]["code"], "usage_error");
    let (code, v) = run(&["analyze", "/nonexistent.wav"]);
    assert_eq!(code, 4);
    assert_eq!(v["error"]["code"], "io_error");
}

#[test]
fn json_documents_and_out_of_key_warnings() {
    let doc = r#"{"tempo": 120, "key": "C major",
      "instruments": {"lead": {"patch": "builtin:instruments/lead_square"}},
      "patterns": {"p": {"notes": "C4:1 F#4:1 G4:2"}},
      "sections": {"m": {"bars": 1, "tracks": {"lead": "p"}}}}"#;
    let (code, v) = run_stdin(&["validate", "-"], doc);
    assert_eq!(code, 0, "{v}");
    assert!(v["warnings"][0].as_str().unwrap().contains("F#4"), "{v}");
}

#[test]
fn lint_fails_with_exit_code_and_fix() {
    let dir = tmp("lint");
    let out = dir.join("boom.wav");
    let (code, _) = run(&["sfx", "explosion", "-o", &s(&out), "--no-analysis"]);
    assert_eq!(code, 0);
    let (code, v) = run(&["lint", &s(&out), "--profile", "ui"]);
    assert_eq!(code, 5);
    assert_eq!(v["passed"], false);
    let issues = v["issues"].as_array().unwrap();
    assert!(issues.iter().any(|i| i["code"] == "too_long"));
    assert!(issues.iter().all(|i| !i["fix"].as_str().unwrap().is_empty()));
    let (code, v) = run(&["lint", &s(&out), "--profile", "impact"]);
    assert_eq!(code, 0, "{v}");
}

#[test]
fn compare_points_toward_the_reference() {
    let dir = tmp("cmp");
    let (dark, bright) = (dir.join("dark.wav"), dir.join("bright.wav"));
    run(&["sfx", "ui_click", "-o", &s(&dark), "--set", "pitch=500", "--no-analysis"]);
    run(&["sfx", "ui_click", "-o", &s(&bright), "--set", "pitch=4000", "--no-analysis"]);
    let (code, v) = run(&["compare", &s(&dark), &s(&bright)]);
    assert_eq!(code, 0);
    let hints = v["how_to_get_closer"].as_array().unwrap().iter().map(|h| h.as_str().unwrap()).collect::<Vec<_>>().join(" | ");
    assert!(hints.contains("darker") || hints.contains("lower"), "{hints}");
    let (_, same) = run(&["compare", &s(&dark), &s(&dark)]);
    assert!(same["similarity"].as_f64().unwrap() > 99.0);
    assert!(v["similarity"].as_f64().unwrap() < same["similarity"].as_f64().unwrap());
}

#[test]
fn pack_writes_manifest() {
    let dir = tmp("pack");
    let manifest = dir.join("pack.yaml");
    std::fs::write(&manifest, "kind: pack\nname: t\nassets:\n  - {id: click, source: builtin:sfx/ui_click, variations: 2}\n  - {id: bad, source: builtin:sfx/nope}\n").unwrap();
    let out = dir.join("out");
    let (code, v) = run(&["pack", &s(&manifest), "-o", &s(&out)]);
    assert_eq!(code, 1, "one asset is broken, so the pack must report failure: {v}");
    assert_eq!(v["assets_failed"], 1);
    assert!(out.join("click_01.wav").exists() && out.join("click_02.wav").exists());
    let m: Value = serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m["assets"].as_array().unwrap().len(), 2);
    assert_eq!(m["assets"][1]["ok"], false);
}

#[test]
fn reference_commands_work() {
    let (code, v) = run(&["nodes"]);
    assert_eq!(code, 0);
    assert!(v["nodes"]["source"].as_array().unwrap().len() >= 5);
    let (_, v) = run(&["nodes", "lowpas"]);
    assert!(v["error"]["hint"].as_str().unwrap().contains("lowpass"));
    let (code, v) = run(&["theory", "key", "D minor"]);
    assert_eq!(code, 0);
    assert_eq!(v["diatonic_chords"][0], "Dm");
    let (code, v) = run(&["schema"]);
    assert_eq!(code, 0);
    assert!(v["score"]["patterns"].is_object());
    let guide = Command::new(bin()).arg("guide").output().unwrap();
    assert!(String::from_utf8_lossy(&guide.stdout).contains("# aas: guide for agents"));
}
