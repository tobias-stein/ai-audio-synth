//! Loading documents (patches, scores, packs) from files, stdin or the
//! built-in library, in YAML or JSON.

use crate::error::{Error, Result};
use crate::library;
use serde_json::Value;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub enum DocKind {
    Patch,
    Score,
    Pack,
}

pub struct Doc {
    pub value: Value,
    pub kind: DocKind,
    /// Human-readable origin: a path, `builtin:...` or `stdin`.
    pub origin: String,
    /// Directory relative paths inside the document resolve against.
    pub base_dir: PathBuf,
}

pub fn parse_text(text: &str, origin: &str) -> Result<Value> {
    let trimmed = text.trim_start();
    let v: Value = if trimmed.starts_with('{') {
        serde_json::from_str(text).map_err(|e| Error::invalid(format!("{origin}: invalid JSON: {e}")).hint("fix the syntax error at the reported line and column"))?
    } else {
        serde_yaml_ng::from_str(text).map_err(|e| Error::invalid(format!("{origin}: invalid YAML: {e}")).hint("check indentation and quote strings containing ':' or '#'; expressions such as \"freq * 2\" must be quoted"))?
    };
    if !v.is_object() {
        return Err(Error::invalid(format!("{origin}: the document must be a mapping at the top level")));
    }
    Ok(v)
}

pub fn detect_kind(v: &Value, origin: &str) -> Result<DocKind> {
    let obj = v.as_object().unwrap();
    if let Some(k) = obj.get("kind") {
        return match k.as_str() {
            Some("patch") => Ok(DocKind::Patch),
            Some("score") => Ok(DocKind::Score),
            Some("pack") => Ok(DocKind::Pack),
            _ => Err(Error::invalid(format!("{origin}: 'kind' must be patch, score or pack")).at("kind")),
        };
    }
    if obj.contains_key("nodes") {
        Ok(DocKind::Patch)
    } else if obj.contains_key("instruments") || obj.contains_key("arrangement") || obj.contains_key("sections") {
        Ok(DocKind::Score)
    } else if obj.contains_key("assets") {
        Ok(DocKind::Pack)
    } else {
        Err(Error::invalid(format!("{origin}: cannot tell what kind of document this is")).hint("add `kind: patch` (a sound), `kind: score` (music) or `kind: pack` (a batch of assets)"))
    }
}

/// Load a document from a path, `builtin:<id>` or `-` (stdin).
pub fn load(source: &str) -> Result<Doc> {
    load_from(source, Path::new("."))
}

pub fn load_from(source: &str, base_dir: &Path) -> Result<Doc> {
    let (text, origin, base) = if source == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s).map_err(|e| Error::io(format!("cannot read stdin: {e}")))?;
        (s, "stdin".to_string(), PathBuf::from("."))
    } else if let Some(id) = source.strip_prefix("builtin:") {
        (library::get(id)?.to_string(), source.to_string(), PathBuf::from("."))
    } else {
        let p = if Path::new(source).is_absolute() { PathBuf::from(source) } else { base_dir.join(source) };
        let text = std::fs::read_to_string(&p).map_err(|e| {
            let mut err = Error::io(format!("cannot read {}: {e}", p.display()));
            if library::get(source).is_ok() {
                err = err.hint(format!("there is a built-in with this id: use 'builtin:{source}'"));
            } else {
                err = err.hint("pass a path to a .yaml/.json document, or builtin:<id> (see `aas library list`)");
            }
            err
        })?;
        let base = p.parent().map(|d| d.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
        (text, p.display().to_string(), base)
    };
    let value = parse_text(&text, &origin)?;
    let kind = detect_kind(&value, &origin)?;
    Ok(Doc { value, kind, origin, base_dir: base })
}
