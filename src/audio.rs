//! The audio buffer type shared by the engine, the mixer and the analyzers.

use crate::util::{db_to_lin, lin_to_db};

/// A mono or stereo block of 64-bit float samples.
#[derive(Clone, Debug, Default)]
pub struct Sig {
    pub l: Vec<f64>,
    pub r: Option<Vec<f64>>,
}

impl Sig {
    pub fn mono(l: Vec<f64>) -> Self {
        Sig { l, r: None }
    }
    pub fn stereo(l: Vec<f64>, r: Vec<f64>) -> Self {
        debug_assert_eq!(l.len(), r.len());
        Sig { l, r: Some(r) }
    }
    pub fn silence(n: usize, stereo: bool) -> Self {
        Sig { l: vec![0.0; n], r: if stereo { Some(vec![0.0; n]) } else { None } }
    }
    pub fn len(&self) -> usize {
        self.l.len()
    }
    pub fn is_empty(&self) -> bool {
        self.l.is_empty()
    }
    pub fn is_stereo(&self) -> bool {
        self.r.is_some()
    }
    pub fn channels(&self) -> usize {
        if self.r.is_some() {
            2
        } else {
            1
        }
    }
    pub fn right(&self) -> &[f64] {
        self.r.as_deref().unwrap_or(&self.l)
    }
    /// Mono downmix (average of channels).
    pub fn to_mono(&self) -> Vec<f64> {
        match &self.r {
            None => self.l.clone(),
            Some(r) => self.l.iter().zip(r).map(|(a, b)| 0.5 * (a + b)).collect(),
        }
    }
    pub fn into_stereo(self) -> Sig {
        match self.r {
            Some(_) => self,
            None => {
                let r = self.l.clone();
                Sig { l: self.l, r: Some(r) }
            }
        }
    }
    pub fn chans(&self) -> Vec<&[f64]> {
        match &self.r {
            None => vec![&self.l],
            Some(r) => vec![&self.l, r],
        }
    }
    /// Apply a per-channel function producing a new signal. `f` receives the
    /// channel index so stateful processors can decorrelate channels if they wish.
    pub fn map_channels<F: FnMut(usize, &[f64]) -> Vec<f64>>(&self, mut f: F) -> Sig {
        let l = f(0, &self.l);
        let r = self.r.as_ref().map(|r| f(1, r));
        Sig { l, r }
    }
    pub fn scale(&mut self, g: f64) {
        for x in self.l.iter_mut() {
            *x *= g;
        }
        if let Some(r) = self.r.as_mut() {
            for x in r.iter_mut() {
                *x *= g;
            }
        }
    }
    pub fn peak(&self) -> f64 {
        let mut p = 0.0f64;
        for c in self.chans() {
            for x in c {
                p = p.max(x.abs());
            }
        }
        p
    }
    pub fn peak_db(&self) -> f64 {
        lin_to_db(self.peak())
    }
    pub fn resize(&mut self, n: usize) {
        self.l.resize(n, 0.0);
        if let Some(r) = self.r.as_mut() {
            r.resize(n, 0.0);
        }
    }
    /// Mix `other` into self starting at sample `offset`, growing if needed.
    pub fn mix_in(&mut self, other: &Sig, offset: usize, gain: f64) {
        let need = offset + other.len();
        if other.is_stereo() && !self.is_stereo() {
            self.r = Some(self.l.clone());
        }
        if self.len() < need {
            self.resize(need);
        }
        for (i, x) in other.l.iter().enumerate() {
            self.l[offset + i] += x * gain;
        }
        if let Some(r) = self.r.as_mut() {
            let src = other.right();
            for (i, x) in src.iter().enumerate() {
                r[offset + i] += x * gain;
            }
        }
    }
    /// Index of the last sample louder than `threshold_db` relative to full scale.
    pub fn last_audible(&self, threshold_db: f64) -> Option<usize> {
        let th = db_to_lin(threshold_db);
        let mut last = None;
        for c in self.chans() {
            if let Some(i) = c.iter().rposition(|x| x.abs() > th) {
                last = Some(last.map_or(i, |l: usize| l.max(i)));
            }
        }
        last
    }
    pub fn has_non_finite(&self) -> bool {
        self.chans().iter().any(|c| c.iter().any(|x| !x.is_finite()))
    }
    /// Linear fade over the first `n_in` and last `n_out` samples.
    pub fn fade(&mut self, n_in: usize, n_out: usize) {
        let n = self.len();
        let apply = |c: &mut Vec<f64>| {
            let ni = n_in.min(n);
            for i in 0..ni {
                c[i] *= i as f64 / ni as f64;
            }
            let no = n_out.min(n);
            for i in 0..no {
                // raised-cosine fade out sounds smoother than linear on tails
                let x = (i as f64 + 1.0) / no as f64;
                c[n - no + i] *= 0.5 + 0.5 * (std::f64::consts::PI * x).cos();
            }
        };
        apply(&mut self.l);
        if let Some(r) = self.r.as_mut() {
            apply(r);
        }
    }
}
