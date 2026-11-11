use crate::optimization::{compile_variables, evaluate_value, Instruction};
use std::io::Write;

#[derive(Clone, Copy, Debug)]
struct Point {
    xy: [f64; 2],
    value: f64,
}
struct Evaluator {
    program: Vec<Instruction>,
    stack: Vec<f64>,
}
impl Evaluator {
    fn at(&mut self, xy: [f64; 2]) -> Option<Point> {
        if xy.iter().any(|v| !v.is_finite() || v.abs() > 1e100) {
            return None;
        }
        Some(Point {
            xy,
            value: evaluate_value(&self.program, xy, &mut self.stack)?,
        })
    }
}
struct Sample {
    phase: &'static str,
    alpha: f64,
    xy: [f64; 2],
    value: Option<f64>,
}
struct Search {
    direction: [f64; 2],
    status: &'static str,
    accepted: Option<usize>,
    samples: Vec<Sample>,
}
impl Search {
    fn probe(
        &mut self,
        e: &mut Evaluator,
        start: Point,
        alpha: f64,
        phase: &'static str,
        limit: usize,
    ) -> Option<usize> {
        if self.samples.len() > limit {
            return None;
        }
        let xy = [
            start.xy[0] + alpha * self.direction[0],
            start.xy[1] + alpha * self.direction[1],
        ];
        // With |alpha|, |direction| <= 1e100 this arithmetic remains finite.
        let value = e.at(xy).map(|p| p.value);
        self.samples.push(Sample {
            phase,
            alpha,
            xy,
            value,
        });
        Some(self.samples.len() - 1)
    }
    fn f(&self, i: usize) -> f64 {
        self.samples[i].value.unwrap_or(f64::INFINITY)
    }
}

fn line_search(
    e: &mut Evaluator,
    start: Point,
    direction: [f64; 2],
    initial: f64,
    tolerance: f64,
    limit: usize,
) -> Search {
    let mut s = Search {
        direction,
        status: "evaluation_limit",
        accepted: None,
        samples: vec![Sample {
            phase: "start",
            alpha: 0.0,
            xy: start.xy,
            value: Some(start.value),
        }],
    };
    if direction == [0.0, 0.0] {
        s.status = "zero_direction";
        s.accepted = Some(0);
        return s;
    }
    let mut ends = [0usize; 2];
    // The line is unrestricted: both signs of alpha are eligible.
    for (side, sign) in [1.0, -1.0].iter().enumerate() {
        let mut alpha = sign * initial;
        loop {
            let Some(i) = s.probe(e, start, alpha, "bracket", limit) else {
                return s;
            };
            if s.samples[i].value.is_some() {
                ends[side] = i;
                break;
            }
            let next = 0.5 * alpha;
            if next == 0.0 || next == alpha || s.samples[i].xy == start.xy {
                s.status = "not_bracketed";
                return s;
            }
            alpha = next;
        }
    }
    let (mut a, mut b, mut c);
    if s.f(ends[0]) >= start.value && s.f(ends[1]) >= start.value {
        a = ends[1];
        b = 0;
        c = ends[0];
    } else {
        a = 0;
        b = if s.f(ends[0]) <= s.f(ends[1]) {
            ends[0]
        } else {
            ends[1]
        };
        let mut boundary: Option<f64> = None;
        loop {
            let ba = s.samples[b].alpha;
            let next = if let Some(edge) = boundary {
                ba + 0.5 * (edge - ba)
            } else {
                ba + 2.0 * (ba - s.samples[a].alpha)
            };
            if !next.is_finite() || next.abs() > 1e100 || next == ba || boundary == Some(next) {
                s.status = "not_bracketed";
                return s;
            }
            let phase = if boundary.is_some() {
                "domain_refine"
            } else {
                "expand"
            };
            let Some(i) = s.probe(e, start, next, phase, limit) else {
                return s;
            };
            if s.samples[i].value.is_none() {
                boundary = Some(next);
                continue;
            }
            if s.f(i) >= s.f(b) {
                c = i;
                break;
            }
            a = b;
            b = i;
        }
        if s.samples[a].alpha > s.samples[c].alpha {
            std::mem::swap(&mut a, &mut c);
        }
    }
    // Brent minimization: safeguarded parabolic interpolation, golden-section fallback.
    // Only values of f are used; x is always a valid, nonincreasing best sample.
    let (mut lo, mut hi) = (s.samples[a].alpha, s.samples[c].alpha);
    let (mut x, mut w, mut v) = (b, b, b);
    let (mut d, mut previous) = (0.0f64, 0.0f64);
    loop {
        let xa = s.samples[x].alpha;
        let midpoint = 0.5 * lo + 0.5 * hi;
        let tol = (tolerance * (1.0 + xa.abs())).max(8.0 * f64::EPSILON * (1.0 + xa.abs()));
        if (xa - midpoint).abs() <= 2.0 * tol - 0.5 * (hi - lo) {
            s.status = "minimum";
            s.accepted = Some(x);
            return s;
        }
        let mut parabolic = false;
        if previous.abs() > tol {
            let wa = s.samples[w].alpha;
            let va = s.samples[v].alpha;
            let r = (xa - wa) * (s.f(x) - s.f(v));
            let q = (xa - va) * (s.f(x) - s.f(w));
            let mut p = (xa - va) * q - (xa - wa) * r;
            let mut q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            let old = previous;
            previous = d;
            if p.is_finite()
                && q.is_finite()
                && q > 0.0
                && p.abs() < (0.5 * q * old).abs()
                && p > q * (lo - xa)
                && p < q * (hi - xa)
            {
                d = p / q;
                let u = xa + d;
                if u - lo < 2.0 * tol || hi - u < 2.0 * tol {
                    d = tol.copysign(midpoint - xa);
                }
                parabolic = true;
            }
        }
        if !parabolic {
            previous = if xa >= midpoint { lo - xa } else { hi - xa };
            d = 0.3819660112501051 * previous;
        }
        let ua = xa + if d.abs() >= tol { d } else { tol.copysign(d) };
        if ua == xa || ua <= lo || ua >= hi {
            s.status = "precision_limit";
            return s;
        }
        let Some(u) = s.probe(
            e,
            start,
            ua,
            if parabolic { "parabolic" } else { "golden" },
            limit,
        ) else {
            return s;
        };
        if s.f(u) < s.f(x) {
            if ua >= xa {
                lo = xa;
            } else {
                hi = xa;
            }
            v = w;
            w = x;
            x = u;
        } else {
            if ua < xa {
                lo = ua;
            } else {
                hi = ua;
            }
            if s.f(u) <= s.f(w) || w == x {
                v = w;
                w = u;
            } else if s.f(u) <= s.f(v) || v == x || v == w {
                v = u;
            }
        }
    }
}

struct Cycle {
    d1: [f64; 2],
    d2: [f64; 2],
    d3: [f64; 2],
    change: f64,
    distance: f64,
}

pub fn run(
    mut lines: std::str::Lines<'_>,
    mut out: impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let f: Vec<&str> = lines
        .next()
        .ok_or("Missing Powell options.")?
        .split('\t')
        .collect();
    if f.len() != 11 {
        return Err("Expected initial pair, cycle tolerance/budget, line probe/tolerance/budget and four plot bounds.".into());
    }
    let x0: f64 = f[0].parse()?;
    let y0: f64 = f[1].parse()?;
    let tolerance: f64 = f[2].parse()?;
    let limit: usize = f[3].parse()?;
    let initial: f64 = f[4].parse()?;
    let line_tolerance: f64 = f[5].parse()?;
    let line_limit: usize = f[6].parse()?;
    let bounds = [
        f[7].parse::<f64>()?,
        f[8].parse()?,
        f[9].parse()?,
        f[10].parse()?,
    ];
    if [x0, y0]
        .iter()
        .chain(bounds.iter())
        .any(|v| !v.is_finite() || v.abs() > 1e50)
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || !(1..=100_000).contains(&limit)
        || !initial.is_finite()
        || initial <= 0.0
        || initial > 1e50
        || !line_tolerance.is_finite()
        || line_tolerance <= 0.0
        || line_tolerance >= 1.0
        || !(1..=2048).contains(&line_limit)
        || limit * 3 * line_limit > 1_000_000
        || bounds[0] >= bounds[1]
        || bounds[2] >= bounds[3]
    {
        return Err("Invalid Powell inputs or trace budget.".into());
    }
    let program = compile_variables(lines.next().ok_or("Missing Powell function.")?, true)?;
    let mut e = Evaluator {
        program,
        stack: Vec::new(),
    };
    let start = e
        .at([x0, y0])
        .ok_or("f(x₀,y₀) is outside its real domain or numerical limit.")?;
    let mut points = vec![start];
    let mut searches = Vec::new();
    let mut cycles = Vec::new();
    let (mut d1, mut d2) = ([1.0, 0.0], [0.0, 1.0]);
    let mut status = "iteration_limit";
    'outer: for _ in 0..limit {
        let start = *points.last().unwrap();
        let mut d3 = [0.0, 0.0];
        for stage in 0..3 {
            if stage == 2 {
                let p = points.last().unwrap();
                d3 = [p.xy[0] - start.xy[0], p.xy[1] - start.xy[1]];
            }
            let direction = match stage {
                0 => d1,
                1 => d2,
                _ => d3,
            };
            if direction.iter().any(|v| !v.is_finite() || v.abs() > 1e100) {
                status = "numerical_limit";
                break 'outer;
            }
            let s = line_search(
                &mut e,
                *points.last().unwrap(),
                direction,
                initial,
                line_tolerance,
                line_limit,
            );
            let next = s.accepted.map(|i| Point {
                xy: s.samples[i].xy,
                value: s.samples[i].value.unwrap(),
            });
            status = match s.status {
                "evaluation_limit" => "line_search_limit",
                "not_bracketed" => "powell_line_failed",
                "precision_limit" => "line_search_precision",
                _ => "iteration_limit",
            };
            searches.push(s);
            if let Some(p) = next {
                points.push(p);
            } else {
                break 'outer;
            }
        }
        let end = points.last().unwrap();
        let change = (start.value - end.value).abs();
        let distance = (end.xy[0] - start.xy[0]).hypot(end.xy[1] - start.xy[1]);
        cycles.push(Cycle {
            d1,
            d2,
            d3,
            change,
            distance,
        });
        // Exactly the direction shift in the supplied notes; no largest-decrease test.
        d1 = d2;
        d2 = d3;
        if change < tolerance {
            status = "powell_function_tolerance";
            break;
        }
        if distance < tolerance {
            status = "powell_position_tolerance";
            break;
        }
    }
    writeln!(out, "PWR\t{status}")?;
    for (i, p) in points.iter().enumerate() {
        writeln!(
            out,
            "PWP\t{i}\t{:.17e}\t{:.17e}\t{:.17e}",
            p.xy[0], p.xy[1], p.value
        )?;
    }
    for (i, c) in cycles.iter().enumerate() {
        writeln!(
            out,
            "PWC\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}",
            i + 1,
            c.d1[0],
            c.d1[1],
            c.d2[0],
            c.d2[1],
            c.d3[0],
            c.d3[1],
            c.change,
            c.distance
        )?;
    }
    for (i, s) in searches.iter().enumerate() {
        let accepted = s.accepted.map_or("_".into(), |v| v.to_string());
        writeln!(
            out,
            "PWL\t{i}\t{:.17e}\t{:.17e}\t{}\t{accepted}",
            s.direction[0], s.direction[1], s.status
        )?;
        for (j, p) in s.samples.iter().enumerate() {
            let value = p.value.map_or("_".into(), |v| format!("{v:.17e}"));
            writeln!(
                out,
                "PWS\t{i}\t{j}\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{value}",
                p.phase, p.alpha, p.xy[0], p.xy[1]
            )?;
        }
    }
    for j in 0..49 {
        for i in 0..49 {
            let xy = [
                bounds[0] + (bounds[1] - bounds[0]) * i as f64 / 48.0,
                bounds[2] + (bounds[3] - bounds[2]) * j as f64 / 48.0,
            ];
            let v = e.at(xy).map_or("_".into(), |p| format!("{:.17e}", p.value));
            writeln!(out, "PWG\t{}\t{v}", j * 49 + i)?;
        }
    }
    out.flush()?;
    Ok(())
}
