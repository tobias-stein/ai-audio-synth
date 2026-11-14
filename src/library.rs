//! The built-in library of sound-effect recipes, instruments, drums and
//! example scores, embedded in the binary.

use crate::error::{Error, Result};
use crate::util::did_you_mean;
use include_dir::{include_dir, Dir};

static LIBRARY: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/library");

pub struct Entry {
    /// Id without extension, e.g. `sfx/explosion`.
    pub id: String,
    pub text: &'static str,
}

fn walk(dir: &'static Dir<'static>, out: &mut Vec<Entry>) {
    for f in dir.files() {
        let path = f.path();
        if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            if let Some(text) = f.contents_utf8() {
                let id = path.with_extension("").to_string_lossy().replace('\\', "/");
                out.push(Entry { id, text });
            }
        }
    }
    for d in dir.dirs() {
        walk(d, out);
    }
}

pub fn entries() -> Vec<Entry> {
    let mut v = Vec::new();
    walk(&LIBRARY, &mut v);
    v.sort_by(|a, b| a.id.cmp(&b.id));
    v
}

pub fn get(id: &str) -> Result<&'static str> {
    let id = id.trim_end_matches(".yaml");
    let all = entries();
    if let Some(e) = all.iter().find(|e| e.id == id) {
        return Ok(e.text);
    }
    // allow a bare name when it is unambiguous: "explosion" -> "sfx/explosion"
    let matches: Vec<&Entry> = all.iter().filter(|e| e.id.rsplit('/').next() == Some(id)).collect();
    if matches.len() == 1 {
        return Ok(matches[0].text);
    }
    let mut err = Error::new("unknown_name", format!("no built-in document with id '{id}'"));
    err.hint = Some(match did_you_mean(id, all.iter().map(|e| e.id.as_str())) {
        Some(s) => format!("did you mean 'builtin:{s}'? Run `aas library list` to see all ids"),
        None => "run `aas library list` to see all ids".into(),
    });
    Err(err)
}
