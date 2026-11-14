//! DSP building blocks. Everything here is offline, deterministic and works
//! on whole buffers at the engine's internal (oversampled) rate.

pub mod dynamics;
pub mod filter;
pub mod fx;
pub mod osc;
pub mod physical;
pub mod resample;
pub mod reverb;

use std::rc::Rc;

/// A node parameter: either constant for the whole render or one value per sample.
#[derive(Clone, Debug)]
pub enum P {
    C(f64),
    B(Rc<Vec<f64>>),
}

impl P {
    #[inline(always)]
    pub fn at(&self, i: usize) -> f64 {
        match self {
            P::C(v) => *v,
            P::B(b) => b[i],
        }
    }
    pub fn is_const(&self) -> bool {
        matches!(self, P::C(_))
    }
    pub fn first(&self) -> f64 {
        match self {
            P::C(v) => *v,
            P::B(b) => b.first().copied().unwrap_or(0.0),
        }
    }
    pub fn max(&self) -> f64 {
        match self {
            P::C(v) => *v,
            P::B(b) => b.iter().cloned().fold(f64::MIN, f64::max),
        }
    }
}
