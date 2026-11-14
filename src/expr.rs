//! A small arithmetic expression language used for every numeric field of a
//! patch. Expressions can reference other nodes (audio-rate signals), patch
//! parameters and built-in variables such as `t`, `freq`, `vel` and `gate`.

use crate::audio::Sig;
use crate::error::{Error, Result};
use crate::util::{did_you_mean, Rng};
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum Expr {
    Num(f64),
    Var(String),
    Neg(Box<Expr>),
    Bin(char, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}

#[derive(Clone, Debug)]
pub enum Val {
    C(f64),
    S(Rc<Sig>),
}

impl Val {
    pub fn as_const(&self) -> Option<f64> {
        match self {
            Val::C(v) => Some(*v),
            Val::S(_) => None,
        }
    }
}

/// name, min args, max args
const FUNCS: &[(&str, usize, usize)] = &[
    ("sin", 1, 1),
    ("cos", 1, 1),
    ("tan", 1, 1),
    ("tanh", 1, 1),
    ("abs", 1, 1),
    ("sqrt", 1, 1),
    ("exp", 1, 1),
    ("ln", 1, 1),
    ("log2", 1, 1),
    ("log10", 1, 1),
    ("floor", 1, 1),
    ("ceil", 1, 1),
    ("round", 1, 1),
    ("sign", 1, 1),
    ("db", 1, 1),
    ("mtof", 1, 1),
    ("semi", 1, 1),
    ("cents", 1, 1),
    ("min", 2, 2),
    ("max", 2, 2),
    ("pow", 2, 2),
    ("mod", 2, 2),
    ("gt", 2, 2),
    ("lt", 2, 2),
    ("clamp", 3, 3),
    ("mix", 3, 3),
    ("smoothstep", 3, 3),
    ("rand", 0, 2),
    ("choose", 1, 8),
];

pub const FUNCTION_DOCS: &[(&str, &str)] = &[
    ("sin(x) cos(x) tan(x) tanh(x)", "trigonometric and hyperbolic functions (radians)"),
    ("abs(x) sqrt(x) exp(x) ln(x) log2(x) log10(x)", "elementary math"),
    ("floor(x) ceil(x) round(x) sign(x)", "rounding and sign"),
    ("min(a,b) max(a,b) pow(a,b) mod(a,b)", "two-argument math; a ^ b also works for powers"),
    ("clamp(x,lo,hi)", "limit x to the range lo..hi"),
    ("mix(a,b,t)", "linear blend: a when t=0, b when t=1"),
    ("smoothstep(lo,hi,x)", "smooth 0..1 ramp as x goes from lo to hi"),
    ("gt(a,b) lt(a,b)", "1 when a>b (or a<b), else 0"),
    ("db(x)", "decibels to linear gain: db(-6) = 0.501"),
    ("mtof(m)", "MIDI note number to Hz: mtof(69) = 440"),
    ("semi(x)", "semitones to frequency ratio: semi(12) = 2"),
    ("cents(x)", "cents to frequency ratio"),
    ("rand() rand(lo,hi)", "one seeded random constant per render; changes with --seed, which is how variations differ"),
    ("choose(a,b,...)", "pick one of the arguments at random per render (seeded)"),
];

pub const CONSTANTS: &[(&str, f64)] = &[("pi", std::f64::consts::PI), ("tau", std::f64::consts::TAU), ("e", std::f64::consts::E)];

struct Parser<'a> {
    src: &'a str,
    chars: Vec<char>,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: impl Into<String>) -> Error {
        Error::new("expression_error", format!("{} in expression \"{}\" at position {}", msg.into(), self.src, self.pos))
    }
    fn ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }
    fn peek(&mut self) -> Option<char> {
        self.ws();
        self.chars.get(self.pos).copied()
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expr(&mut self) -> Result<Expr> {
        let mut lhs = self.term()?;
        loop {
            match self.peek() {
                Some(c @ ('+' | '-')) => {
                    self.pos += 1;
                    let rhs = self.term()?;
                    lhs = Expr::Bin(c, Box::new(lhs), Box::new(rhs));
                }
                _ => return Ok(lhs),
            }
        }
    }
    fn term(&mut self) -> Result<Expr> {
        let mut lhs = self.unary()?;
        loop {
            match self.peek() {
                Some(c @ ('*' | '/')) => {
                    self.pos += 1;
                    let rhs = self.unary()?;
                    lhs = Expr::Bin(c, Box::new(lhs), Box::new(rhs));
                }
                _ => return Ok(lhs),
            }
        }
    }
    fn unary(&mut self) -> Result<Expr> {
        if self.eat('-') {
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        if self.eat('+') {
            return self.unary();
        }
        self.power()
    }
    fn power(&mut self) -> Result<Expr> {
        let base = self.atom()?;
        if self.eat('^') {
            let exp = self.unary()?; // right associative
            return Ok(Expr::Bin('^', Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }
    fn atom(&mut self) -> Result<Expr> {
        match self.peek() {
            None => Err(self.err("unexpected end")),
            Some('(') => {
                self.pos += 1;
                let e = self.expr()?;
                if !self.eat(')') {
                    return Err(self.err("expected ')'"));
                }
                Ok(e)
            }
            Some(c) if c.is_ascii_digit() || c == '.' => {
                let start = self.pos;
                while self.pos < self.chars.len() && (self.chars[self.pos].is_ascii_digit() || self.chars[self.pos] == '.') {
                    self.pos += 1;
                }
                if self.pos < self.chars.len() && (self.chars[self.pos] == 'e' || self.chars[self.pos] == 'E') {
                    let save = self.pos;
                    self.pos += 1;
                    if self.pos < self.chars.len() && (self.chars[self.pos] == '+' || self.chars[self.pos] == '-') {
                        self.pos += 1;
                    }
                    if self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
                        while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
                            self.pos += 1;
                        }
                    } else {
                        self.pos = save;
                    }
                }
                let s: String = self.chars[start..self.pos].iter().collect();
                s.parse::<f64>().map(Expr::Num).map_err(|_| self.err(format!("bad number '{s}'")))
            }
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                let start = self.pos;
                while self.pos < self.chars.len() && (self.chars[self.pos].is_ascii_alphanumeric() || self.chars[self.pos] == '_') {
                    self.pos += 1;
                }
                let name: String = self.chars[start..self.pos].iter().collect();
                if self.eat('(') {
                    let mut args = Vec::new();
                    if !self.eat(')') {
                        loop {
                            args.push(self.expr()?);
                            if self.eat(',') {
                                continue;
                            }
                            if self.eat(')') {
                                break;
                            }
                            return Err(self.err("expected ',' or ')'"));
                        }
                    }
                    let Some(&(_, lo, hi)) = FUNCS.iter().find(|f| f.0 == name) else {
                        let names: Vec<&str> = FUNCS.iter().map(|f| f.0).collect();
                        let mut e = self.err(format!("unknown function '{name}'"));
                        e.hint = Some(match did_you_mean(&name, names.iter().copied()) {
                            Some(s) => format!("did you mean '{s}'? Run `aas nodes --functions` for the list"),
                            None => "run `aas nodes --functions` for the list of functions".into(),
                        });
                        return Err(e);
                    };
                    if args.len() < lo || args.len() > hi {
                        return Err(self.err(format!("function '{name}' takes {} argument(s), got {}", if lo == hi { lo.to_string() } else { format!("{lo} to {hi}") }, args.len())));
                    }
                    if name == "rand" && args.len() == 1 {
                        return Err(self.err("rand takes zero or two arguments: rand() or rand(lo, hi)"));
                    }
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            Some(c) => Err(self.err(format!("unexpected character '{c}'"))),
        }
    }
}

impl Expr {
    pub fn parse(src: &str) -> Result<Expr> {
        let mut p = Parser { src, chars: src.chars().collect(), pos: 0 };
        let e = p.expr()?;
        if p.peek().is_some() {
            return Err(p.err("unexpected trailing input"));
        }
        Ok(e)
    }

    /// Names of variables this expression reads.
    pub fn idents(&self, out: &mut Vec<String>) {
        match self {
            Expr::Num(_) => {}
            Expr::Var(n) => {
                if !out.contains(n) {
                    out.push(n.clone())
                }
            }
            Expr::Neg(a) => a.idents(out),
            Expr::Bin(_, a, b) => {
                a.idents(out);
                b.idents(out);
            }
            Expr::Call(_, args) => {
                for a in args {
                    a.idents(out)
                }
            }
        }
    }

    pub fn is_literal(&self) -> Option<f64> {
        match self {
            Expr::Num(v) => Some(*v),
            Expr::Neg(a) => a.is_literal().map(|v| -v),
            _ => None,
        }
    }

    pub fn eval(&self, env: &dyn Fn(&str) -> Option<Val>, rng: &mut Rng) -> Result<Val> {
        Ok(match self {
            Expr::Num(v) => Val::C(*v),
            Expr::Var(n) => match env(n) {
                Some(v) => v,
                None => match CONSTANTS.iter().find(|c| c.0 == n) {
                    Some(c) => Val::C(c.1),
                    None => return Err(Error::new("unknown_name", format!("unknown name '{n}' in expression"))),
                },
            },
            Expr::Neg(a) => map1(&a.eval(env, rng)?, |x| -x),
            Expr::Bin(op, a, b) => {
                let (a, b) = (a.eval(env, rng)?, b.eval(env, rng)?);
                match op {
                    '+' => map2(&a, &b, |x, y| x + y),
                    '-' => map2(&a, &b, |x, y| x - y),
                    '*' => map2(&a, &b, |x, y| x * y),
                    '/' => map2(&a, &b, |x, y| if y == 0.0 { 0.0 } else { x / y }),
                    _ => map2(&a, &b, safe_pow),
                }
            }
            Expr::Call(name, args) => {
                if name == "rand" {
                    let u = rng.uniform();
                    if args.is_empty() {
                        return Ok(Val::C(u));
                    }
                    let (lo, hi) = (args[0].eval(env, rng)?, args[1].eval(env, rng)?);
                    return Ok(map2(&lo, &hi, move |l, h| l + (h - l) * u));
                }
                if name == "choose" {
                    let k = (rng.uniform() * args.len() as f64) as usize;
                    return args[k.min(args.len() - 1)].eval(env, rng);
                }
                let mut v = Vec::with_capacity(args.len());
                for a in args {
                    v.push(a.eval(env, rng)?);
                }
                match name.as_str() {
                    "sin" => map1(&v[0], f64::sin),
                    "cos" => map1(&v[0], f64::cos),
                    "tan" => map1(&v[0], f64::tan),
                    "tanh" => map1(&v[0], f64::tanh),
                    "abs" => map1(&v[0], f64::abs),
                    "sqrt" => map1(&v[0], |x| x.max(0.0).sqrt()),
                    "exp" => map1(&v[0], f64::exp),
                    "ln" => map1(&v[0], |x| x.max(1e-300).ln()),
                    "log2" => map1(&v[0], |x| x.max(1e-300).log2()),
                    "log10" => map1(&v[0], |x| x.max(1e-300).log10()),
                    "floor" => map1(&v[0], f64::floor),
                    "ceil" => map1(&v[0], f64::ceil),
                    "round" => map1(&v[0], f64::round),
                    "sign" => map1(&v[0], |x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }),
                    "db" => map1(&v[0], |x| 10f64.powf(x / 20.0)),
                    "mtof" => map1(&v[0], |x| 440.0 * 2f64.powf((x - 69.0) / 12.0)),
                    "semi" => map1(&v[0], |x| 2f64.powf(x / 12.0)),
                    "cents" => map1(&v[0], |x| 2f64.powf(x / 1200.0)),
                    "min" => map2(&v[0], &v[1], f64::min),
                    "max" => map2(&v[0], &v[1], f64::max),
                    "pow" => map2(&v[0], &v[1], safe_pow),
                    "mod" => map2(&v[0], &v[1], |x, y| if y == 0.0 { 0.0 } else { x.rem_euclid(y) }),
                    "gt" => map2(&v[0], &v[1], |x, y| if x > y { 1.0 } else { 0.0 }),
                    "lt" => map2(&v[0], &v[1], |x, y| if x < y { 1.0 } else { 0.0 }),
                    "clamp" => map3(&v[0], &v[1], &v[2], |x, lo, hi| x.max(lo).min(hi)),
                    "mix" => map3(&v[0], &v[1], &v[2], |a, b, t| a + (b - a) * t),
                    "smoothstep" => map3(&v[0], &v[1], &v[2], |lo, hi, x| {
                        let t = if hi == lo { 0.0 } else { ((x - lo) / (hi - lo)).clamp(0.0, 1.0) };
                        t * t * (3.0 - 2.0 * t)
                    }),
                    _ => unreachable!("function table and evaluator out of sync"),
                }
            }
        })
    }
}

fn safe_pow(x: f64, y: f64) -> f64 {
    let r = if x < 0.0 && y.fract() != 0.0 { -((-x).powf(y)) } else { x.powf(y) };
    if r.is_finite() {
        r
    } else {
        0.0
    }
}

pub fn map1(a: &Val, f: impl Fn(f64) -> f64) -> Val {
    match a {
        Val::C(v) => Val::C(f(*v)),
        Val::S(s) => Val::S(Rc::new(s.map_channels(|_, c| c.iter().map(|x| f(*x)).collect()))),
    }
}

pub fn map2(a: &Val, b: &Val, f: impl Fn(f64, f64) -> f64) -> Val {
    match (a, b) {
        (Val::C(x), Val::C(y)) => Val::C(f(*x, *y)),
        (Val::S(s), Val::C(y)) => Val::S(Rc::new(s.map_channels(|_, c| c.iter().map(|x| f(*x, *y)).collect()))),
        (Val::C(x), Val::S(s)) => Val::S(Rc::new(s.map_channels(|_, c| c.iter().map(|y| f(*x, *y)).collect()))),
        (Val::S(p), Val::S(q)) => {
            let n = p.len().min(q.len());
            let l: Vec<f64> = (0..n).map(|i| f(p.l[i], q.l[i])).collect();
            let r = if p.is_stereo() || q.is_stereo() {
                let (pr, qr) = (p.right(), q.right());
                Some((0..n).map(|i| f(pr[i], qr[i])).collect())
            } else {
                None
            };
            Val::S(Rc::new(Sig { l, r }))
        }
    }
}

pub fn map3(a: &Val, b: &Val, c: &Val, f: impl Fn(f64, f64, f64) -> f64) -> Val {
    if let (Val::C(x), Val::C(y), Val::C(z)) = (a, b, c) {
        return Val::C(f(*x, *y, *z));
    }
    let vals = [a, b, c];
    let n = vals.iter().filter_map(|v| if let Val::S(s) = v { Some(s.len()) } else { None }).min().unwrap_or(0);
    let stereo = vals.iter().any(|v| matches!(v, Val::S(s) if s.is_stereo()));
    let get = |v: &Val, ch: usize, i: usize| -> f64 {
        match v {
            Val::C(x) => *x,
            Val::S(s) => {
                if ch == 0 {
                    s.l[i]
                } else {
                    s.right()[i]
                }
            }
        }
    };
    let l: Vec<f64> = (0..n).map(|i| f(get(a, 0, i), get(b, 0, i), get(c, 0, i))).collect();
    let r = if stereo { Some((0..n).map(|i| f(get(a, 1, i), get(b, 1, i), get(c, 1, i))).collect()) } else { None };
    Val::S(Rc::new(Sig { l, r }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(src: &str) -> f64 {
        let e = Expr::parse(src).unwrap();
        let mut rng = Rng::new(1);
        let env = |n: &str| -> Option<Val> {
            match n {
                "x" => Some(Val::C(3.0)),
                _ => None,
            }
        };
        e.eval(&env, &mut rng).unwrap().as_const().unwrap()
    }

    #[test]
    fn arithmetic_and_precedence() {
        assert_eq!(ev("1 + 2 * 3"), 7.0);
        assert_eq!(ev("(1 + 2) * 3"), 9.0);
        assert_eq!(ev("-x ^ 2"), -9.0);
        assert_eq!(ev("2 ^ 3 ^ 2"), 512.0);
        assert_eq!(ev("10 / 4 - 0.5"), 2.0);
        assert!((ev("1.5e2") - 150.0).abs() < 1e-12);
        assert!((ev("mtof(69)") - 440.0).abs() < 1e-9);
        assert!((ev("db(-6)") - 0.5012).abs() < 1e-3);
        assert_eq!(ev("clamp(x, 0, 1)"), 1.0);
        assert!((ev("2 * pi") - std::f64::consts::TAU).abs() < 1e-12);
    }

    #[test]
    fn signals_broadcast() {
        let e = Expr::parse("a * 2 + b").unwrap();
        let a = Rc::new(Sig::mono(vec![1.0, 2.0, 3.0]));
        let b = Rc::new(Sig::stereo(vec![1.0, 1.0, 1.0], vec![0.0, 0.0, 0.0]));
        let env = move |n: &str| -> Option<Val> {
            match n {
                "a" => Some(Val::S(a.clone())),
                "b" => Some(Val::S(b.clone())),
                _ => None,
            }
        };
        let mut rng = Rng::new(1);
        let Val::S(s) = e.eval(&env, &mut rng).unwrap() else { panic!() };
        assert_eq!(s.l, vec![3.0, 5.0, 7.0]);
        assert_eq!(s.r.as_ref().unwrap(), &vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn errors_are_helpful() {
        let e = Expr::parse("sinn(1)").unwrap_err();
        assert!(e.hint.unwrap().contains("sin"));
        assert!(Expr::parse("1 +").is_err());
        assert!(Expr::parse("clamp(1)").is_err());
        assert!(Expr::parse("2 $ 3").is_err());
    }

    #[test]
    fn rand_is_seeded() {
        let e = Expr::parse("rand(10, 20)").unwrap();
        let env = |_: &str| -> Option<Val> { None };
        let a = e.eval(&env, &mut Rng::new(5)).unwrap().as_const().unwrap();
        let b = e.eval(&env, &mut Rng::new(5)).unwrap().as_const().unwrap();
        let c = e.eval(&env, &mut Rng::new(6)).unwrap().as_const().unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!((10.0..20.0).contains(&a));
    }
}
