//! Literal slide: reflection, expansion, inside-only contraction, then shrink.

use crate::optimization::{compile_variables, evaluate_value, Instruction};
use std::io::Write;

#[derive(Clone, Copy, PartialEq)]
struct Vertex {
    coordinates: [f64; 2],
    value: f64,
}

struct Snapshot {
    simplex: [Vertex; 3],
    spread: f64,
    diameter: f64,
}

impl Snapshot {
    fn new(mut simplex: [Vertex; 3]) -> Self {
        // Stable ordering preserves the existing row order when objective values tie.
        simplex.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());
        let spread = simplex[2].value - simplex[0].value;
        let mut diameter = 0.0_f64;
        for i in 0..3 {
            for j in i + 1..3 {
                diameter = diameter.max(
                    (simplex[i].coordinates[0] - simplex[j].coordinates[0])
                        .hypot(simplex[i].coordinates[1] - simplex[j].coordinates[1]),
                );
            }
        }
        Self {
            simplex,
            spread,
            diameter,
        }
    }
}

struct Step {
    action: &'static str,
    centroid: [f64; 2],
    reflection: Vertex,
    expansion: Option<Vertex>,
    contraction: Option<Vertex>,
    shrunk_b: Option<Vertex>,
    shrunk_c: Option<Vertex>,
}

struct Failure {
    stage: &'static str,
    coordinates: [f64; 2],
}

fn trial(
    program: &[Instruction],
    coordinates: [f64; 2],
    stage: &'static str,
    stack: &mut Vec<f64>,
) -> Result<Vertex, Failure> {
    if coordinates
        .iter()
        .any(|v| !v.is_finite() || v.abs() > 1e100)
    {
        return Err(Failure { stage, coordinates });
    }
    evaluate_value(program, coordinates, stack)
        .map(|value| Vertex { coordinates, value })
        .ok_or(Failure { stage, coordinates })
}

fn iteration(
    program: &[Instruction],
    simplex: [Vertex; 3],
    stack: &mut Vec<f64>,
    standard: bool,
) -> Result<(Step, Snapshot), Failure> {
    let [a, b, c] = simplex;
    let centroid =
        std::array::from_fn(|i| a.coordinates[i] + 0.5 * (b.coordinates[i] - a.coordinates[i]));
    // Reflection alpha=1, always using the OLD worst vertex C.
    let reflection = trial(
        program,
        std::array::from_fn(|i| centroid[i] + (centroid[i] - c.coordinates[i])),
        "reflection",
        stack,
    )?;
    let mut step = Step {
        action: "reflect",
        centroid,
        reflection,
        expansion: None,
        contraction: None,
        shrunk_b: None,
        shrunk_c: None,
    };
    let after;
    if a.value <= reflection.value && reflection.value < b.value {
        after = [a, b, reflection];
    } else if reflection.value < a.value {
        // Expansion gamma=2. Equality selects the reflection, exactly as on the slide.
        let expansion = trial(
            program,
            std::array::from_fn(|i| centroid[i] + 2.0 * (reflection.coordinates[i] - centroid[i])),
            "expansion",
            stack,
        )?;
        step.expansion = Some(expansion);
        if expansion.value < reflection.value {
            step.action = "expand";
            after = [a, b, expansion];
        } else {
            step.action = "reflect_after_expand";
            after = [a, b, reflection];
        }
    } else {
        // f(R)>=f(B): the supplied slide always contracts toward OLD C, rho=0.5.
        let outside = standard && reflection.value < c.value;
        let target = if outside { reflection } else { c };
        let contraction = trial(
            program,
            std::array::from_fn(|i| centroid[i] + 0.5 * (target.coordinates[i] - centroid[i])),
            "contraction",
            stack,
        )?;
        step.contraction = Some(contraction);
        if (outside && contraction.value <= reflection.value)
            || (!outside && contraction.value < c.value)
        {
            step.action = if outside {
                "contract_outside"
            } else {
                "contract"
            };
            after = [a, b, contraction];
        } else {
            // Both shrink vertices use the same old A, sigma=0.5; commit atomically.
            let shrunk_b = trial(
                program,
                std::array::from_fn(|i| {
                    a.coordinates[i] + 0.5 * (b.coordinates[i] - a.coordinates[i])
                }),
                "shrink_B",
                stack,
            )?;
            let shrunk_c = trial(
                program,
                std::array::from_fn(|i| {
                    a.coordinates[i] + 0.5 * (c.coordinates[i] - a.coordinates[i])
                }),
                "shrink_C",
                stack,
            )?;
            step.action = "shrink";
            step.shrunk_b = Some(shrunk_b);
            step.shrunk_c = Some(shrunk_c);
            after = [a, shrunk_b, shrunk_c];
        }
    }
    Ok((step, Snapshot::new(after)))
}

fn vertex_text(vertex: Vertex) -> String {
    format!(
        "{:.17e}\t{:.17e}\t{:.17e}",
        vertex.coordinates[0], vertex.coordinates[1], vertex.value
    )
}

fn optional_vertex(vertex: Option<Vertex>) -> String {
    vertex.map(vertex_text).unwrap_or("_\t_\t_".into())
}

pub fn run(
    mut lines: std::str::Lines<'_>,
    mut out: impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let fields: Vec<&str> = lines
        .next()
        .ok_or("Missing Nelder-Mead options.")?
        .split('\t')
        .collect();
    if fields.len() != 6 && fields.len() != 8 {
        return Err("Expected tolerance, iteration limit and four plot bounds.".into());
    }
    let tolerance: f64 = fields[0].parse()?;
    let limit: usize = fields[1].parse()?;
    let variant = fields.get(6).copied().unwrap_or("slides");
    let stopping = fields.get(7).copied().unwrap_or("function");
    if !["slides", "standard"].contains(&variant) || !["function", "position"].contains(&stopping) {
        return Err("Unknown Nelder-Mead variant or stopping rule.".into());
    }
    let bounds: Vec<f64> = fields[2..6]
        .iter()
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    if !tolerance.is_finite()
        || tolerance <= 0.0
        || !(1..=100_000).contains(&limit)
        || bounds.iter().any(|v| !v.is_finite() || v.abs() > 1e50)
        || bounds[0] >= bounds[1]
        || bounds[2] >= bounds[3]
    {
        return Err("Invalid Nelder-Mead options.".into());
    }
    let mut initial = Vec::new();
    for _ in 0..3 {
        let values: Vec<f64> = lines
            .next()
            .ok_or("Missing initial simplex vertex.")?
            .split('\t')
            .map(|v| v.parse())
            .collect::<Result<_, _>>()?;
        if values.len() != 2 || values.iter().any(|v| !v.is_finite() || v.abs() > 1e50) {
            return Err("Initial simplex: three finite 2D vertices with coordinates at most 1e50 are required.".into());
        }
        initial.push([values[0], values[1]]);
    }
    let u = [initial[1][0] - initial[0][0], initial[1][1] - initial[0][1]];
    let v = [initial[2][0] - initial[0][0], initial[2][1] - initial[0][1]];
    let scale = u
        .iter()
        .chain(v.iter())
        .fold(0.0_f64, |a, b| a.max(b.abs()));
    if scale == 0.0 || (u[0] / scale) * (v[1] / scale) - (u[1] / scale) * (v[0] / scale) == 0.0 {
        return Err(
            "The initial simplex must have three non-collinear vertices at working precision."
                .into(),
        );
    }
    let program = compile_variables(lines.next().ok_or("Missing function.")?, true)?;
    if lines.next().is_some() {
        return Err("Unexpected Nelder-Mead input.".into());
    }
    let mut stack = Vec::with_capacity(program.len());
    let mut vertices = Vec::new();
    for coordinates in initial {
        let value = evaluate_value(&program, coordinates, &mut stack)
            .ok_or("f(x,y) must be real, finite and within the numerical limit at all three initial vertices.")?;
        vertices.push(Vertex { coordinates, value });
    }
    let mut points = vec![Snapshot::new([vertices[0], vertices[1], vertices[2]])];
    let mut steps = Vec::new();
    let mut failure = None;
    let status = loop {
        let current = points.last().unwrap();
        // Explicit interpretation of the slide's qualitative 'small changes in f'.
        let best = current.simplex[0].coordinates;
        let radius = current
            .simplex
            .iter()
            .map(|v| (v.coordinates[0] - best[0]).hypot(v.coordinates[1] - best[1]))
            .fold(0., f64::max);
        if stopping == "position" && radius < tolerance {
            break "simplex_tolerance";
        }
        if stopping == "function" && current.spread <= tolerance {
            break "function_tolerance";
        }
        if steps.len() == limit {
            break "iteration_limit";
        }
        match iteration(&program, current.simplex, &mut stack, variant == "standard") {
            Ok((step, after)) => {
                if after.simplex == current.simplex {
                    break "stagnation";
                }
                steps.push(step);
                points.push(after);
            }
            Err(attempt) => {
                failure = Some(attempt);
                break "numerical_limit";
            }
        }
    };
    writeln!(out, "NMR\t{status}")?;
    for (k, p) in points.iter().enumerate() {
        writeln!(
            out,
            "NMP\t{k}\t{:.17e}\t{:.17e}\t{}\t{}\t{}",
            p.spread,
            p.diameter,
            vertex_text(p.simplex[0]),
            vertex_text(p.simplex[1]),
            vertex_text(p.simplex[2])
        )?;
    }
    for (k, s) in steps.iter().enumerate() {
        writeln!(
            out,
            "NMT\t{k}\t{}\t{:.17e}\t{:.17e}\t{}\t{}\t{}\t{}\t{}",
            s.action,
            s.centroid[0],
            s.centroid[1],
            vertex_text(s.reflection),
            optional_vertex(s.expansion),
            optional_vertex(s.contraction),
            optional_vertex(s.shrunk_b),
            optional_vertex(s.shrunk_c)
        )?;
    }
    if let Some(failed) = failure {
        writeln!(
            out,
            "NMF\t{}\t{:.17e}\t{:.17e}",
            failed.stage, failed.coordinates[0], failed.coordinates[1]
        )?;
    }
    const GRID: usize = 49;
    for row in 0..GRID {
        for col in 0..GRID {
            let point = [
                bounds[0] + (bounds[1] - bounds[0]) * (col as f64 + 0.5) / GRID as f64,
                bounds[2] + (bounds[3] - bounds[2]) * (row as f64 + 0.5) / GRID as f64,
            ];
            let index = row * GRID + col;
            if let Some(value) = evaluate_value(&program, point, &mut stack) {
                writeln!(out, "NMC\t{index}\t{value:.17e}")?;
            } else {
                writeln!(out, "NMC\t{index}\t_")?;
            }
        }
    }
    out.flush()?;
    Ok(())
}
