use crate::optimization::{compile_variables, Instruction};
use std::io::Write;

const PAIRS: [(usize, usize); 3] = [(0, 0), (0, 1), (1, 1)];

#[derive(Clone, Copy)]
struct Jet {
    value: f64,
    gradient: [f64; 2],
    hessian: [f64; 3], // xx, xy (= yx), yy
    constant: bool,
}

impl Jet {
    fn constant(value: f64) -> Self {
        Self {
            value,
            gradient: [0.0; 2],
            hessian: [0.0; 3],
            constant: true,
        }
    }

    fn compose(a: Self, value: f64, slope: f64, curvature: f64) -> Self {
        if a.constant {
            return Self::constant(value);
        }
        Self {
            value,
            gradient: a.gradient.map(|g| slope * g),
            hessian: std::array::from_fn(|k| {
                let (i, j) = PAIRS[k];
                slope * a.hessian[k] + curvature * a.gradient[i] * a.gradient[j]
            }),
            constant: false,
        }
    }

    fn product(a: Self, b: Self) -> Self {
        Self {
            value: a.value * b.value,
            gradient: std::array::from_fn(|i| a.gradient[i] * b.value + a.value * b.gradient[i]),
            hessian: std::array::from_fn(|k| {
                let (i, j) = PAIRS[k];
                a.hessian[k] * b.value
                    + a.gradient[i] * b.gradient[j]
                    + a.gradient[j] * b.gradient[i]
                    + a.value * b.hessian[k]
            }),
            constant: a.constant && b.constant,
        }
    }

    fn valid(&self) -> bool {
        std::iter::once(&self.value)
            .chain(self.gradient.iter())
            .chain(self.hessian.iter())
            .all(|v| v.is_finite() && v.abs() <= 1e100)
    }
}

fn evaluate(
    program: &[Instruction],
    point: [f64; 2],
    stack: &mut Vec<Jet>,
    derivatives: bool,
) -> Option<Jet> {
    stack.clear();
    for instruction in program {
        let mut result = match instruction {
            Instruction::Constant(value) => Jet::constant(*value),
            Instruction::X | Instruction::Y => {
                let axis = usize::from(matches!(instruction, Instruction::Y));
                let mut jet = Jet::constant(point[axis]);
                jet.constant = false;
                jet.gradient[axis] = 1.0;
                jet
            }
            Instruction::Unary(op) => {
                let a = stack.pop()?;
                let (value, slope, curvature) = match op.as_str() {
                    "neg" => (-a.value, -1.0, 0.0),
                    "sin" => (a.value.sin(), a.value.cos(), -a.value.sin()),
                    "cos" => (a.value.cos(), -a.value.sin(), -a.value.cos()),
                    "tan" => (
                        a.value.tan(),
                        1.0 / a.value.cos().powi(2),
                        2.0 * a.value.tan() / a.value.cos().powi(2),
                    ),
                    "exp" => (a.value.exp(), a.value.exp(), a.value.exp()),
                    "ln" => (a.value.ln(), 1.0 / a.value, -1.0 / a.value / a.value),
                    "sqrt" => (
                        a.value.sqrt(),
                        0.5 / a.value.sqrt(),
                        -0.25 / a.value / a.value.sqrt(),
                    ),
                    _ => return None,
                };
                Jet::compose(a, value, slope, curvature)
            }
            Instruction::Binary(op) => {
                let b = stack.pop()?;
                let a = stack.pop()?;
                match op.as_str() {
                    "add" | "sub" => {
                        let sign = if op == "add" { 1.0 } else { -1.0 };
                        Jet {
                            value: a.value + sign * b.value,
                            gradient: std::array::from_fn(|i| a.gradient[i] + sign * b.gradient[i]),
                            hessian: std::array::from_fn(|i| a.hessian[i] + sign * b.hessian[i]),
                            constant: a.constant && b.constant,
                        }
                    }
                    "mul" => Jet::product(a, b),
                    "div" => {
                        let value = a.value / b.value;
                        let gradient: [f64; 2] = std::array::from_fn(|i| {
                            (a.gradient[i] - value * b.gradient[i]) / b.value
                        });
                        Jet {
                            value,
                            gradient,
                            hessian: std::array::from_fn(|k| {
                                let (i, j) = PAIRS[k];
                                (a.hessian[k]
                                    - value * b.hessian[k]
                                    - gradient[i] * b.gradient[j]
                                    - gradient[j] * b.gradient[i])
                                    / b.value
                            }),
                            constant: a.constant && b.constant,
                        }
                    }
                    "pow" => {
                        let value = a.value.powf(b.value);
                        if !derivatives || a.constant && b.constant || b.constant && b.value == 0.0
                        {
                            let mut jet = Jet::constant(value);
                            jet.constant = a.constant && b.constant || b.constant && b.value == 0.0;
                            jet
                        } else if b.constant {
                            let slope = b.value * a.value.powf(b.value - 1.0);
                            let curvature = if b.value == 1.0 {
                                0.0
                            } else {
                                b.value * (b.value - 1.0) * a.value.powf(b.value - 2.0)
                            };
                            Jet::compose(a, value, slope, curvature)
                        } else if a.value > 0.0 {
                            let log_a = Jet::compose(
                                a,
                                a.value.ln(),
                                1.0 / a.value,
                                -1.0 / a.value / a.value,
                            );
                            let exponent = Jet::product(b, log_a);
                            Jet::compose(exponent, value, value, value)
                        } else {
                            return None;
                        }
                    }
                    _ => return None,
                }
            }
        };
        if !derivatives {
            result.gradient = [0.0; 2];
            result.hessian = [0.0; 3];
        }
        if !result.valid() {
            return None;
        }
        stack.push(result);
    }
    stack.pop()
}

struct Point {
    coordinates: [f64; 2],
    jet: Jet,
    norm: f64,
    scale: f64,
    determinant: f64,
}

impl Point {
    fn new(coordinates: [f64; 2], jet: Jet) -> Self {
        let norm = jet.gradient[0].hypot(jet.gradient[1]);
        let scale = jet.hessian.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        let determinant = if scale == 0.0 {
            0.0
        } else {
            (jet.hessian[0] / scale) * (jet.hessian[2] / scale) - (jet.hessian[1] / scale).powi(2)
        };
        Self {
            coordinates,
            jet,
            norm,
            scale,
            determinant,
        }
    }

    fn delta(&self) -> Option<[f64; 2]> {
        if self.scale == 0.0 || self.determinant == 0.0 {
            return None;
        }
        let [a, b, c] = self.jet.hessian.map(|v| v / self.scale);
        let [gx, gy] = self.jet.gradient.map(|v| v / self.scale);
        // Equivalent to -H^-1 gradient; scale both sides of H*s=gradient first.
        Some([
            -(c * gx - b * gy) / self.determinant,
            -(a * gy - b * gx) / self.determinant,
        ])
    }
}

pub fn run(
    mut lines: std::str::Lines<'_>,
    mut out: impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let fields: Vec<&str> = lines
        .next()
        .ok_or("Missing 2D Newton options.")?
        .split('\t')
        .collect();
    if fields.len() != 8 {
        return Err("Expected x0, y0, tolerance, iteration limit and four plot bounds.".into());
    }
    let x0: f64 = fields[0].parse()?;
    let y0: f64 = fields[1].parse()?;
    let tolerance: f64 = fields[2].parse()?;
    let limit: usize = fields[3].parse()?;
    let bounds: Vec<f64> = fields[4..]
        .iter()
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    if [x0, y0]
        .iter()
        .chain(bounds.iter())
        .any(|v| !v.is_finite() || v.abs() > 1e50)
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || !(1..=100_000).contains(&limit)
        || bounds[0] >= bounds[1]
        || bounds[2] >= bounds[3]
    {
        return Err("Invalid 2D Newton settings.".into());
    }
    let program = compile_variables(lines.next().ok_or("Missing 2D Newton function.")?, true)?;
    if lines.next().is_some() {
        return Err("Unexpected 2D Newton input.".into());
    }
    let mut stack = Vec::with_capacity(program.len());
    let initial = evaluate(&program, [x0,y0], &mut stack, true)
        .ok_or("f, both partials and all Hessian entries must be real, finite and within the numerical limit at starting (x0,y0).")?;
    let mut points = vec![Point::new([x0, y0], initial)];
    let mut deltas = Vec::new();
    let status = loop {
        let current = points.last().unwrap();
        if current.norm <= tolerance {
            break "gradient_tolerance";
        }
        if points.len() - 1 == limit {
            break "iteration_limit";
        }
        let Some(delta) = current.delta() else {
            break "singular_hessian";
        };
        let next = [
            current.coordinates[0] + delta[0],
            current.coordinates[1] + delta[1],
        ];
        if delta
            .iter()
            .chain(next.iter())
            .any(|v| !v.is_finite() || v.abs() > 1e100)
        {
            break "numerical_limit";
        }
        if next == current.coordinates {
            break "stagnation";
        }
        if let Some(jet) = evaluate(&program, next, &mut stack, true) {
            points.push(Point::new(next, jet));
            deltas.push(delta);
        } else {
            break "numerical_limit";
        }
    };
    writeln!(out, "N2R\t{status}")?;
    for (k, p) in points.iter().enumerate() {
        let [x, y] = p.coordinates;
        let [gx, gy] = p.jet.gradient;
        let [xx, xy, yy] = p.jet.hessian;
        let delta = deltas
            .get(k)
            .map(|d| format!("{:.17e}\t{:.17e}", d[0], d[1]))
            .unwrap_or("_\t_".into());
        writeln!(out, "N2P\t{k}\t{x:.17e}\t{y:.17e}\t{:.17e}\t{gx:.17e}\t{gy:.17e}\t{:.17e}\t{xx:.17e}\t{xy:.17e}\t{yy:.17e}\t{:.17e}\t{:.17e}\t{delta}", p.jet.value,p.norm,p.scale,p.determinant)?;
    }
    const GRID: usize = 49;
    for row in 0..GRID {
        for col in 0..GRID {
            let point = [
                bounds[0] + (bounds[1] - bounds[0]) * (col as f64 + 0.5) / GRID as f64,
                bounds[2] + (bounds[3] - bounds[2]) * (row as f64 + 0.5) / GRID as f64,
            ];
            let index = row * GRID + col;
            if let Some(jet) = evaluate(&program, point, &mut stack, false) {
                writeln!(out, "N2C\t{index}\t{:.17e}", jet.value)?;
            } else {
                writeln!(out, "N2C\t{index}\t_")?;
            }
        }
    }
    out.flush()?;
    Ok(())
}
