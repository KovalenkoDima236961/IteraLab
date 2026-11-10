//! Gradient and Newton methods, following the supplied one- and two-dimensional slides.

use std::io::Write;

#[derive(Clone, Copy)]
struct Dual {
    value: f64,
    derivative: f64,
    second: f64,
    constant: bool,
}

pub(super) enum Instruction {
    Constant(f64),
    X,
    Y,
    Unary(String),
    Binary(String),
}

fn compile(source: &str) -> Result<Vec<Instruction>, String> {
    compile_variables(source, false)
}

pub(super) fn compile_variables(source: &str, allow_y: bool) -> Result<Vec<Instruction>, String> {
    let mut depth = 0usize;
    let mut program = Vec::new();
    for token in source.split_whitespace() {
        let instruction = if let Some(number) = token.strip_prefix("c:") {
            let value: f64 = number.parse().map_err(|_| "Invalid constant.")?;
            if !value.is_finite() || value.abs() > 1e100 {
                return Err("Invalid constant.".into());
            }
            depth += 1;
            Instruction::Constant(value)
        } else if token == "x" {
            depth += 1;
            Instruction::X
        } else if token == "y" && allow_y {
            depth += 1;
            Instruction::Y
        } else if matches!(token, "neg" | "sin" | "cos" | "tan" | "exp" | "ln" | "sqrt")
            && depth >= 1
        {
            Instruction::Unary(token.into())
        } else if matches!(token, "add" | "sub" | "mul" | "div" | "pow") && depth >= 2 {
            depth -= 1;
            Instruction::Binary(token.into())
        } else {
            return Err("Invalid function instruction.".into());
        };
        program.push(instruction);
        if program.len() > 200 {
            return Err("Function is too complex.".into());
        }
    }
    if depth != 1 {
        return Err("Function must produce one value.".into());
    }
    Ok(program)
}

/// Function-only interpretation for derivative-free methods, including domain boundaries.
pub(super) fn evaluate_value(
    program: &[Instruction],
    point: [f64; 2],
    stack: &mut Vec<f64>,
) -> Option<f64> {
    stack.clear();
    for instruction in program {
        let value = match instruction {
            Instruction::Constant(value) => *value,
            Instruction::X => point[0],
            Instruction::Y => point[1],
            Instruction::Unary(op) => {
                let a = stack.pop()?;
                match op.as_str() {
                    "neg" => -a,
                    "sin" => a.sin(),
                    "cos" => a.cos(),
                    "tan" => a.tan(),
                    "exp" => a.exp(),
                    "ln" => a.ln(),
                    "sqrt" => a.sqrt(),
                    _ => return None,
                }
            }
            Instruction::Binary(op) => {
                let b = stack.pop()?;
                let a = stack.pop()?;
                match op.as_str() {
                    "add" => a + b,
                    "sub" => a - b,
                    "mul" => a * b,
                    "div" => a / b,
                    "pow" => a.powf(b),
                    _ => return None,
                }
            }
        };
        if !value.is_finite() || value.abs() > 1e100 {
            return None;
        }
        stack.push(value);
    }
    stack.pop()
}

fn evaluate(program: &[Instruction], x: f64, stack: &mut Vec<Dual>, order: u8) -> Option<Dual> {
    evaluate_xy(program, [x, 0.0], stack, order, 0)
}

fn evaluate_xy(
    program: &[Instruction],
    point: [f64; 2],
    stack: &mut Vec<Dual>,
    order: u8,
    axis: usize,
) -> Option<Dual> {
    stack.clear();
    for instruction in program {
        let result = match instruction {
            Instruction::Constant(value) => Dual {
                value: *value,
                derivative: 0.0,
                second: 0.0,
                constant: true,
            },
            Instruction::X | Instruction::Y => {
                let coordinate = usize::from(matches!(instruction, Instruction::Y));
                Dual {
                    value: point[coordinate],
                    derivative: if order > 0 && coordinate == axis {
                        1.0
                    } else {
                        0.0
                    },
                    second: 0.0,
                    constant: false,
                }
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
                    "ln" => (a.value.ln(), 1.0 / a.value, -1.0 / a.value.powi(2)),
                    "sqrt" => (
                        a.value.sqrt(),
                        0.5 / a.value.sqrt(),
                        -0.25 / (a.value * a.value.sqrt()),
                    ),
                    _ => return None,
                };
                Dual {
                    value,
                    derivative: if a.constant || order == 0 {
                        0.0
                    } else {
                        slope * a.derivative
                    },
                    second: if a.constant || order < 2 {
                        0.0
                    } else {
                        curvature * a.derivative.powi(2) + slope * a.second
                    },
                    constant: a.constant,
                }
            }
            Instruction::Binary(op) => {
                let b = stack.pop()?;
                let a = stack.pop()?;
                let (value, derivative, second) = match op.as_str() {
                    "add" => (
                        a.value + b.value,
                        a.derivative + b.derivative,
                        a.second + b.second,
                    ),
                    "sub" => (
                        a.value - b.value,
                        a.derivative - b.derivative,
                        a.second - b.second,
                    ),
                    "mul" => (
                        a.value * b.value,
                        a.derivative * b.value + a.value * b.derivative,
                        if order < 2 {
                            0.0
                        } else {
                            a.second * b.value
                                + 2.0 * a.derivative * b.derivative
                                + a.value * b.second
                        },
                    ),
                    "div" => {
                        let value = a.value / b.value;
                        let derivative = if order == 0 {
                            0.0
                        } else {
                            (a.derivative * b.value - a.value * b.derivative) / b.value.powi(2)
                        };
                        let second = if order < 2 {
                            0.0
                        } else {
                            (a.second - value * b.second - 2.0 * derivative * b.derivative)
                                / b.value
                        };
                        (value, derivative, second)
                    }
                    "pow" => {
                        let value = a.value.powf(b.value);
                        let (derivative, second) = if order == 0
                            || a.constant && b.constant
                            || b.constant && b.value == 0.0
                        {
                            (0.0, 0.0)
                        } else if b.constant {
                            let slope = b.value * a.value.powf(b.value - 1.0);
                            let curvature = if b.value == 1.0 {
                                0.0
                            } else {
                                b.value * (b.value - 1.0) * a.value.powf(b.value - 2.0)
                            };
                            (
                                slope * a.derivative,
                                if order < 2 {
                                    0.0
                                } else {
                                    curvature * a.derivative.powi(2) + slope * a.second
                                },
                            )
                        } else if a.value > 0.0 {
                            let log_slope = a.derivative / a.value;
                            let slope = b.derivative * a.value.ln() + b.value * log_slope;
                            let curvature = b.second * a.value.ln()
                                + 2.0 * b.derivative * log_slope
                                + b.value * (a.second / a.value - log_slope.powi(2));
                            (
                                value * slope,
                                if order < 2 {
                                    0.0
                                } else {
                                    value * (slope.powi(2) + curvature)
                                },
                            )
                        } else {
                            return None;
                        };
                        (value, derivative, second)
                    }
                    _ => return None,
                };
                Dual {
                    value,
                    derivative,
                    second,
                    constant: a.constant && b.constant,
                }
            }
        };
        if !result.value.is_finite()
            || !result.derivative.is_finite()
            || result.value.abs() > 1e100
            || result.derivative.abs() > 1e100
            || !result.second.is_finite()
            || result.second.abs() > 1e100
        {
            return None;
        }
        stack.push(result);
    }
    stack.pop()
}

/// Opaque first-order 2D evaluator shared with the normalized-vector method.
pub(super) struct GradientEvaluator {
    program: Vec<Instruction>,
    scratch: Vec<Dual>,
    values: Vec<f64>,
}

impl GradientEvaluator {
    pub(super) fn new(source: &str) -> Result<Self, String> {
        let program = compile_variables(source, true)?;
        Ok(Self {
            scratch: Vec::with_capacity(program.len()),
            values: Vec::with_capacity(program.len()),
            program,
        })
    }

    pub(super) fn evaluate(&mut self, point: [f64; 2]) -> Option<[f64; 4]> {
        let dx = evaluate_xy(&self.program, point, &mut self.scratch, 1, 0)?;
        let dy = evaluate_xy(&self.program, point, &mut self.scratch, 1, 1)?;
        let norm = dx.derivative.hypot(dy.derivative);
        norm.is_finite()
            .then_some([dx.value, dx.derivative, dy.derivative, norm])
    }

    pub(super) fn value(&mut self, point: [f64; 2]) -> Option<f64> {
        evaluate_value(&self.program, point, &mut self.values)
    }
}

pub fn run(
    mut lines: std::str::Lines<'_>,
    mut out: impl Write,
    newton: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let fields: Vec<&str> = lines
        .next()
        .ok_or("Missing gradient options.")?
        .split('\t')
        .collect();
    if fields.len() != if newton { 5 } else { 6 } {
        return Err("Invalid optimization option count.".into());
    }
    let mut options = fields.iter();
    let x0: f64 = options.next().unwrap().parse()?;
    let alpha: f64 = if newton {
        1.0
    } else {
        options.next().unwrap().parse()?
    };
    let tolerance: f64 = options.next().unwrap().parse()?;
    let limit: usize = options.next().unwrap().parse()?;
    let left: f64 = options.next().unwrap().parse()?;
    let right: f64 = options.next().unwrap().parse()?;
    if [x0, left, right]
        .iter()
        .any(|v| !v.is_finite() || v.abs() > 1e50)
        || !alpha.is_finite()
        || alpha <= 0.0
        || alpha > 1e50
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || if newton {
            limit > 99_999
        } else {
            !(1..=100_000).contains(&limit)
        }
        || left >= right
    {
        return Err("Invalid optimization settings.".into());
    }
    let program = compile(lines.next().ok_or("Missing function.")?)?;
    let mut supplied = Vec::new();
    if newton {
        for _ in 0..2 {
            let source = lines.next().ok_or("Missing derivative source.")?;
            supplied.push(if source == "_" {
                None
            } else {
                Some(compile(source)?)
            });
        }
    }
    if lines.next().is_some() {
        return Err("Unexpected optimization input.".into());
    }
    let mut stack = Vec::with_capacity(program.len());
    let evaluate_point = |x, stack: &mut Vec<Dual>| -> Option<Dual> {
        let order = if !newton {
            1
        } else if supplied[1].is_none() {
            2
        } else if supplied[0].is_none() {
            1
        } else {
            0
        };
        let mut point = evaluate(&program, x, stack, order)?;
        if newton {
            if let Some(first) = &supplied[0] {
                point.derivative = evaluate(first, x, stack, 0)?.value;
            }
            if let Some(second) = &supplied[1] {
                point.second = evaluate(second, x, stack, 0)?.value;
            }
        }
        Some(point)
    };
    let initial = evaluate_point(x0, &mut stack).ok_or(
        "The function and required derivatives must be real, finite and within the numerical limit at starting x0.",
    )?;
    let mut points = vec![(x0, initial)];
    let status = loop {
        let &(x, current) = points.last().unwrap();
        if !newton && current.derivative.abs() <= tolerance {
            break "gradient_tolerance";
        }
        if points.len() - 1 == limit + usize::from(newton) {
            break "iteration_limit";
        }
        if newton && current.second == 0.0 {
            break "zero_curvature";
        }
        // Both slide recurrences use derivatives at the OLD point.
        let delta = if newton {
            -current.derivative / current.second
        } else {
            -alpha * current.derivative
        };
        let next = x + delta;
        if !delta.is_finite() || !next.is_finite() || delta.abs() > 1e100 || next.abs() > 1e100 {
            break "numerical_limit";
        }
        if !newton && next == x {
            break "stagnation";
        }
        if let Some(value) = evaluate_point(next, &mut stack) {
            points.push((next, value));
        } else {
            break "numerical_limit";
        }
        // Newton tests the actual difference AFTER the update, strictly <.
        if newton && (next - x).abs() < tolerance {
            break "step_tolerance";
        }
    };
    writeln!(out, "{}\t{status}", if newton { "NR" } else { "GR" })?;
    for (k, (x, point)) in points.iter().enumerate() {
        if newton {
            writeln!(
                out,
                "NP\t{k}\t{x:.17e}\t{:.17e}\t{:.17e}\t{:.17e}",
                point.value, point.derivative, point.second
            )?;
        } else {
            writeln!(
                out,
                "GP\t{k}\t{x:.17e}\t{:.17e}\t{:.17e}",
                point.value, point.derivative
            )?;
        }
    }
    for i in 0..=400 {
        let x = left + (right - left) * (i as f64 / 400.0);
        if let Some(point) = evaluate(&program, x, &mut stack, 0) {
            writeln!(out, "GC\t{x:.17e}\t{:.17e}", point.value)?;
        } else {
            writeln!(out, "GC\t{x:.17e}\t_")?;
        }
    }
    out.flush()?;
    Ok(())
}

/// Simultaneous, unnormalized fixed-rate descent in the negative Euclidean gradient.
pub fn run_steepest(
    mut lines: std::str::Lines<'_>,
    mut out: impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let fields: Vec<&str> = lines
        .next()
        .ok_or("Missing 2D options.")?
        .split('\t')
        .collect();
    if fields.len() != 9 {
        return Err(
            "Expected x0, y0, alpha, tolerance, iteration limit and four plot bounds.".into(),
        );
    }
    let x0: f64 = fields[0].parse()?;
    let y0: f64 = fields[1].parse()?;
    let alpha: f64 = fields[2].parse()?;
    let tolerance: f64 = fields[3].parse()?;
    let limit: usize = fields[4].parse()?;
    let bounds: Vec<f64> = fields[5..]
        .iter()
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    if [x0, y0]
        .iter()
        .chain(bounds.iter())
        .any(|v| !v.is_finite() || v.abs() > 1e50)
        || !alpha.is_finite()
        || alpha <= 0.0
        || alpha > 1e50
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || !(1..=100_000).contains(&limit)
        || bounds[0] >= bounds[1]
        || bounds[2] >= bounds[3]
    {
        return Err("Invalid 2D steepest-descent settings.".into());
    }
    let program = compile_variables(lines.next().ok_or("Missing 2D function.")?, true)?;
    if lines.next().is_some() {
        return Err("Unexpected 2D input.".into());
    }
    let mut stack = Vec::with_capacity(program.len());
    let gradient = |point, stack: &mut Vec<Dual>| -> Option<(f64, f64, f64, f64)> {
        // Two forward automatic-differentiation passes, both at the same old (x, y).
        let dx = evaluate_xy(&program, point, stack, 1, 0)?;
        let dy = evaluate_xy(&program, point, stack, 1, 1)?;
        let norm = dx.derivative.hypot(dy.derivative);
        if !norm.is_finite() {
            return None;
        }
        Some((dx.value, dx.derivative, dy.derivative, norm))
    };
    let initial = gradient([x0, y0], &mut stack).ok_or("f(x,y) and both partial derivatives must be real, finite and within the numerical limit at starting (x0,y0).")?;
    let mut points = vec![([x0, y0], initial)];
    let status = loop {
        let &(point, (_, gx, gy, norm)) = points.last().unwrap();
        if norm <= tolerance {
            break "gradient_tolerance";
        }
        if points.len() - 1 == limit {
            break "iteration_limit";
        }
        let delta = [-alpha * gx, -alpha * gy];
        // Compute both new coordinates before committing either coordinate.
        let next = [point[0] + delta[0], point[1] + delta[1]];
        if delta
            .iter()
            .chain(next.iter())
            .any(|v| !v.is_finite() || v.abs() > 1e100)
        {
            break "numerical_limit";
        }
        if next == point {
            break "stagnation";
        }
        if let Some(value) = gradient(next, &mut stack) {
            points.push((next, value));
        } else {
            break "numerical_limit";
        }
    };
    writeln!(out, "G2R\t{status}")?;
    for (k, (point, (value, gx, gy, norm))) in points.iter().enumerate() {
        writeln!(
            out,
            "G2P\t{k}\t{:.17e}\t{:.17e}\t{value:.17e}\t{gx:.17e}\t{gy:.17e}\t{norm:.17e}",
            point[0], point[1]
        )?;
    }
    const GRID: usize = 49;
    for row in 0..GRID {
        for col in 0..GRID {
            let x = bounds[0] + (bounds[1] - bounds[0]) * (col as f64 + 0.5) / GRID as f64;
            let y = bounds[2] + (bounds[3] - bounds[2]) * (row as f64 + 0.5) / GRID as f64;
            let index = row * GRID + col;
            if let Some(value) = evaluate_xy(&program, [x, y], &mut stack, 0, 0) {
                writeln!(out, "G2C\t{index}\t{:.17e}", value.value)?;
            } else {
                writeln!(out, "G2C\t{index}\t_")?;
            }
        }
    }
    out.flush()?;
    Ok(())
}
