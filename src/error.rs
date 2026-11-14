//! Structured errors. Every error carries a stable code, the path of the
//! offending field and, where possible, a hint telling the caller what to change.

use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct Error {
    pub code: &'static str,
    pub message: String,
    pub path: Option<String>,
    pub hint: Option<String>,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Error { code, message: message.into(), path: None, hint: None }
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new("invalid_document", message)
    }
    pub fn io(message: impl Into<String>) -> Self {
        Self::new("io_error", message)
    }
    pub fn usage(message: impl Into<String>) -> Self {
        Self::new("usage_error", message)
    }
    pub fn at(mut self, path: impl Into<String>) -> Self {
        if self.path.is_none() {
            self.path = Some(path.into());
        }
        self
    }
    /// Prefix the path with a parent segment (used when bubbling up).
    pub fn within(mut self, parent: &str) -> Self {
        self.path = Some(match self.path.take() {
            Some(p) if !p.is_empty() => format!("{parent}.{p}"),
            _ => parent.to_string(),
        });
        self
    }
    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
    pub fn exit_code(&self) -> i32 {
        match self.code {
            "usage_error" => 2,
            "invalid_document" | "unknown_name" | "expression_error" => 3,
            "io_error" | "unsupported_format" => 4,
            "lint_failed" => 5,
            _ => 1,
        }
    }
    pub fn to_json(&self) -> Value {
        let mut e = json!({ "code": self.code, "message": self.message });
        if let Some(p) = &self.path {
            e["path"] = json!(p);
        }
        if let Some(h) = &self.hint {
            e["hint"] = json!(h);
        }
        e
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(p) = &self.path {
            write!(f, "{}: ", p)?;
        }
        write!(f, "{}", self.message)?;
        if let Some(h) = &self.hint {
            write!(f, " (hint: {})", h)?;
        }
        Ok(())
    }
}
