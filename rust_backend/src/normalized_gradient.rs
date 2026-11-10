//! S=-gradient/||gradient||; numerically solve phi'(gamma)=0, then P_next=P+gamma*S.

use crate::optimization::GradientEvaluator;
use std::io::Write;

#[derive(Clone, Copy)]
struct Point {
    coordinates: [f64; 2],
    value: f64,
    gradient: [f64; 2],
    norm: f64,
}

impl Point {
    fn evaluate(evaluator: &mut GradientEvaluator, coordinates: [f64; 2]) -> Option<Self> {
        if coordinates
            .iter()
            .any(|v| !v.is_finite() || v.abs() > 1e100)
        {
            return None;
        }
        let [value, gx, gy, norm] = evaluator.evaluate(coordinates)?;
        Some(Self {
            coordinates,
            value,
            gradient: [gx, gy],
            norm,
        })
    }
}

struct Sample {
    phase: &'static str,
    gamma: f64,
    coordinates: [f64; 2],
    point: Option<Point>,
    derivative: Option<f64>,
}
struct Search {
    direction: [f64; 2],
    status: &'static str,
    accepted: Option<usize>,
    samples: Vec<Sample>,
}

fn line_search(
    evaluator: &mut GradientEvaluator,
    start: Point,
    initial_gamma: f64,
    tolerance: f64,
    limit: usize,
    normalize: bool,
) -> Search {
    // Scaling before normalization also handles subnormal gradient magnitudes.
    let scale = start.gradient[0].abs().max(start.gradient[1].abs());
    let scaled = start.gradient.map(|v| v / scale);
    let length = scaled[0].hypot(scaled[1]);
    let direction = if normalize {
        scaled.map(|v| -v / length)
    } else {
        start.gradient.map(|v| -v)
    };
    let initial_derivative = start.gradient[0] * direction[0] + start.gradient[1] * direction[1];
    let mut report = Search {
        direction,
        status: "evaluation_limit",
        accepted: None,
        samples: vec![Sample {
            phase: "start",
            gamma: 0.0,
            coordinates: start.coordinates,
            point: Some(start),
            derivative: Some(initial_derivative),
        }],
    };
    let mut low = 0.0;
    let mut high: Option<(f64, bool)> = None; // bool: valid positive-derivative endpoint.
    let mut gamma = initial_gamma;
    let mut phase = "bracket";
    for _ in 0..limit {
        let coordinates = [
            start.coordinates[0] + gamma * direction[0],
            start.coordinates[1] + gamma * direction[1],
        ];
        let point = Point::evaluate(evaluator, coordinates);
        let derivative = point.map(|p| p.gradient[0] * direction[0] + p.gradient[1] * direction[1]);
        report.samples.push(Sample {
            phase,
            gamma,
            coordinates,
            point,
            derivative,
        });
        if let (Some(point), Some(derivative)) = (point, derivative) {
            if derivative == 0.0 || (derivative / initial_derivative.abs()).abs() <= tolerance {
                if point.coordinates == start.coordinates {
                    report.status = "precision_limit";
                } else if point.value > start.value {
                    report.status = "objective_increase";
                } else {
                    report.status = "stationary";
                    report.accepted = Some(report.samples.len() - 1);
                }
                return report;
            }
            if derivative < 0.0 {
                low = gamma;
            } else {
                high = Some((gamma, true));
            }
        } else {
            // An invalid trial bounds the searched domain; refine within it.
            high = Some((gamma, false));
        }
        if let Some((upper, valid_upper)) = high {
            let next = low + 0.5 * (upper - low);
            if next == low || next == upper {
                report.status = if valid_upper {
                    "precision_limit"
                } else {
                    "not_bracketed"
                };
                return report;
            }
            gamma = next;
            phase = if valid_upper {
                "bisect"
            } else {
                "domain_refine"
            };
        } else {
            if gamma >= 1e100 {
                report.status = "not_bracketed";
                return report;
            }
            gamma = (2.0 * gamma).min(1e100);
            phase = "bracket";
        }
    }
    report
}

pub fn run(
    mut lines: std::str::Lines<'_>,
    mut out: impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let fields: Vec<&str> = lines
        .next()
        .ok_or("Missing normalized-gradient options.")?
        .split('\t')
        .collect();
    if fields.len() != 11 && fields.len() != 12 {
        return Err("Expected initial pair, gradient tolerance, update budget, initial gamma, line tolerance, line budget and four plot bounds.".into());
    }
    let x0: f64 = fields[0].parse()?;
    let y0: f64 = fields[1].parse()?;
    let tolerance: f64 = fields[2].parse()?;
    let limit: usize = fields[3].parse()?;
    let gamma: f64 = fields[4].parse()?;
    let line_tolerance: f64 = fields[5].parse()?;
    let line_limit: usize = fields[6].parse()?;
    let normalize = match fields.get(11).copied().unwrap_or("1") {
        "1" => true,
        "0" => false,
        _ => return Err("Normalize direction must be 0 or 1.".into()),
    };
    let bounds: Vec<f64> = fields[7..11]
        .iter()
        .map(|v| v.parse())
        .collect::<Result<_, _>>()?;
    if [x0, y0]
        .iter()
        .chain(bounds.iter())
        .any(|v| !v.is_finite() || v.abs() > 1e50)
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || !(1..=100_000).contains(&limit)
        || !gamma.is_finite()
        || gamma <= 0.0
        || gamma > 1e50
        || !line_tolerance.is_finite()
        || line_tolerance <= 0.0
        || line_tolerance >= 1.0
        || !(1..=2048).contains(&line_limit)
        || limit.saturating_mul(line_limit) > 1_000_000
        || bounds[0] >= bounds[1]
        || bounds[2] >= bounds[3]
    {
        return Err(
            "Invalid normalized-gradient options or excessive line-search trace budget.".into(),
        );
    }
    let mut evaluator = GradientEvaluator::new(
        lines
            .next()
            .ok_or("Missing normalized-gradient function.")?,
    )?;
    if lines.next().is_some() {
        return Err("Unexpected normalized-gradient input.".into());
    }
    let initial=Point::evaluate(&mut evaluator,[x0,y0]).ok_or("f(x,y) and both partial derivatives must be real, finite and within the numerical limit at starting (x0,y0).")?;
    let mut points = vec![initial];
    let mut searches = Vec::new();
    let status = loop {
        let current = *points.last().unwrap();
        // This test MUST precede normalization: zero gradients are never divided.
        if current.norm <= tolerance {
            break "gradient_tolerance";
        }
        if points.len() - 1 == limit {
            break "iteration_limit";
        }
        let search = line_search(
            &mut evaluator,
            current,
            gamma,
            line_tolerance,
            line_limit,
            normalize,
        );
        let failure = match search.status {
            "stationary" => None,
            "evaluation_limit" => Some("line_search_limit"),
            "not_bracketed" => Some("line_search_failed"),
            "precision_limit" => Some("line_search_precision"),
            "objective_increase" => Some("line_search_increase"),
            _ => unreachable!(),
        };
        let next = search.accepted.map(|i| search.samples[i].point.unwrap());
        searches.push(search);
        if let Some(status) = failure {
            break status;
        }
        points.push(next.unwrap());
    };
    writeln!(out, "NGR\t{status}")?;
    for (k, p) in points.iter().enumerate() {
        writeln!(
            out,
            "NGP\t{k}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}",
            p.coordinates[0], p.coordinates[1], p.value, p.gradient[0], p.gradient[1], p.norm
        )?;
    }
    for (k, search) in searches.iter().enumerate() {
        let accepted = search.accepted.map(|i| i.to_string()).unwrap_or("_".into());
        writeln!(
            out,
            "NGL\t{k}\t{:.17e}\t{:.17e}\t{}\t{accepted}",
            search.direction[0], search.direction[1], search.status
        )?;
        for (n, s) in search.samples.iter().enumerate() {
            let values = if let Some(point) = s.point {
                format!(
                    "{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}",
                    point.value,
                    point.gradient[0],
                    point.gradient[1],
                    s.derivative.unwrap()
                )
            } else {
                "_\t_\t_\t_".into()
            };
            writeln!(
                out,
                "NGS\t{k}\t{n}\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{values}",
                s.phase, s.gamma, s.coordinates[0], s.coordinates[1]
            )?;
        }
    }
    const GRID: usize = 49;
    for row in 0..GRID {
        for col in 0..GRID {
            let point = [
                bounds[0] + (bounds[1] - bounds[0]) * (col as f64 + 0.5) / GRID as f64,
                bounds[2] + (bounds[3] - bounds[2]) * (row as f64 + 0.5) / GRID as f64,
            ];
            let index = row * GRID + col;
            if let Some(value) = evaluator.value(point) {
                writeln!(out, "NGC\t{index}\t{value:.17e}")?;
            } else {
                writeln!(out, "NGC\t{index}\t_")?;
            }
        }
    }
    out.flush()?;
    Ok(())
}
