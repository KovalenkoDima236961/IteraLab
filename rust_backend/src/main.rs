#[path = "lib.rs"]
mod core;

use core::{Model, Options, Sample};
use std::io::{self, BufWriter, Read, Write};

fn numbers(line: &str) -> Result<Vec<f64>, String> {
    line.split(',')
        .map(|s| s.parse().map_err(|_| format!("Invalid number: {s}")))
        .collect()
}

fn vector(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn samples(lines: std::str::Lines<'_>) -> Result<Vec<Sample>, String> {
    lines
        .map(|line| {
            let mut row = numbers(line)?;
            let target = row.pop().ok_or("Empty pattern.")?;
            Ok(Sample {
                inputs: row,
                target,
            })
        })
        .collect()
}

fn hidden_weights(model: &core::madaline::Model) -> Vec<f64> {
    model
        .hidden
        .iter()
        .flat_map(|u| u.weights.iter().copied())
        .collect()
}

fn hidden_biases(model: &core::madaline::Model) -> Vec<f64> {
    model.hidden.iter().map(|u| u.bias).collect()
}

fn integers<T: std::fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn run_madaline(
    mut lines: std::str::Lines<'_>,
    options: Options,
) -> Result<(), Box<dyn std::error::Error>> {
    use core::madaline;
    let units: usize = lines.next().ok_or("Missing hidden unit count.")?.parse()?;
    if !(1..=madaline::MAX_HIDDEN_UNITS).contains(&units) {
        return Err(format!(
            "Use 1 to {} hidden ADALINE units.",
            madaline::MAX_HIDDEN_UNITS
        )
        .into());
    }
    let mut hidden = Vec::new();
    for _ in 0..units {
        hidden.push(Model {
            weights: numbers(lines.next().ok_or("Missing hidden weights.")?)?,
            bias: lines.next().ok_or("Missing hidden bias.")?.parse()?,
        });
    }
    let output_weights = numbers(lines.next().ok_or("Missing fixed output weights.")?)?;
    let output_bias = lines.next().ok_or("Missing fixed output bias.")?.parse()?;
    let samples = samples(lines)?;
    let result = madaline::train(
        &samples,
        madaline::Model {
            hidden,
            output_weights,
            output_bias,
        },
        options,
    )?;
    let mut out = BufWriter::new(io::stdout().lock());
    let (failed_epoch, failed_pattern) = result.failed_at.unwrap_or((0, 0));
    writeln!(
        out,
        "R\t{}\t{}\t{}\t{}",
        result.status.name(),
        result.epochs.len(),
        failed_epoch,
        failed_pattern
    )?;
    for s in &result.steps {
        writeln!(
            out,
            "MS\t{}\t{}\t{}\t{}\t{:.17e}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.17e}",
            s.epoch,
            s.pattern,
            vector(&s.response.nets),
            integers(&s.response.hidden_outputs),
            s.response.net,
            s.response.output,
            s.rule,
            integers(&s.selected.iter().map(|&v| u8::from(v)).collect::<Vec<_>>()),
            vector(&s.errors),
            vector(&hidden_weights(&s.before)),
            vector(&hidden_biases(&s.before)),
            vector(
                &s.delta_weights
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>()
            ),
            vector(&s.delta_biases),
            vector(&hidden_weights(&s.after)),
            vector(&hidden_biases(&s.after)),
            s.max_change
        )?;
    }
    for e in &result.epochs {
        writeln!(
            out,
            "ME\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{}\t{}\t{}",
            e.number,
            e.max_change,
            e.net_change,
            e.sse,
            e.correct,
            vector(&hidden_weights(&e.model)),
            vector(&hidden_biases(&e.model))
        )?;
    }
    for (index, sample) in samples.iter().enumerate() {
        let p = result.model.response(&sample.inputs);
        writeln!(
            out,
            "MP\t{}\t{}\t{}\t{:.17e}\t{}",
            index + 1,
            vector(&p.nets),
            integers(&p.hidden_outputs),
            p.net,
            p.output
        )?;
    }
    let (sse, correct) = madaline::diagnostics(&samples, &result.model);
    writeln!(
        out,
        "MM\t{}\t{}\t{:.17e}\t{}",
        vector(&hidden_weights(&result.model)),
        vector(&hidden_biases(&result.model)),
        sse,
        correct
    )?;
    out.flush()?;
    Ok(())
}

fn run_hebbian(mut lines: std::str::Lines<'_>) -> Result<(), Box<dyn std::error::Error>> {
    use core::hebbian;
    let eta: f64 = lines
        .next()
        .ok_or("Missing Hebbian learning rate eta.")?
        .parse()?;
    let samples = samples(lines)?;
    let result = hebbian::train(&samples, eta)?;
    let mut out = BufWriter::new(io::stdout().lock());
    let (failed_epoch, failed_pattern) = result.failed_at.unwrap_or((0, 0));
    writeln!(
        out,
        "R\t{}\t{}\t{}\t{}",
        result.status.name(),
        result.epochs.len(),
        failed_epoch,
        failed_pattern
    )?;
    for s in &result.steps {
        writeln!(
            out,
            "HS\t{}\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{}\t{}\t{}",
            s.epoch,
            s.pattern,
            s.delta_bias,
            s.before.bias,
            s.after.bias,
            s.max_change,
            vector(&s.before.weights),
            vector(&s.delta_weights),
            vector(&s.after.weights)
        )?;
    }
    for e in &result.epochs {
        writeln!(
            out,
            "E\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{}\t{}\t{:.17e}",
            e.number,
            e.max_change,
            e.net_change,
            e.sse,
            e.correct,
            vector(&e.model.weights),
            e.model.bias
        )?;
    }
    for (index, sample) in samples.iter().enumerate() {
        let net = result.model.net(&sample.inputs);
        writeln!(
            out,
            "P\t{}\t{:.17e}\t{}",
            index + 1,
            net,
            core::activate(net)
        )?;
    }
    let (sse, correct) = hebbian::diagnostics(&samples, &result.model);
    writeln!(
        out,
        "M\t{}\t{:.17e}\t{:.17e}\t{}",
        vector(&result.model.weights),
        result.model.bias,
        sse,
        correct
    )?;
    out.flush()?;
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().take(1_048_577).read_to_string(&mut input)?;
    if input.len() > 1_048_576 {
        return Err("Request is too large.".into());
    }
    let mut lines = input.lines();
    let protocol = lines.next().ok_or("Missing protocol.")?;
    if protocol == "HEBBIAN1" {
        return run_hebbian(lines);
    }
    if protocol != "ADALINE1" && protocol != "MADALINE2" {
        return Err("Expected ADALINE1, MADALINE2 or HEBBIAN1 request.".into());
    }
    let fields: Vec<&str> = lines
        .next()
        .ok_or("Missing options.")?
        .split('\t')
        .collect();
    if fields.len() != 3 {
        return Err("Expected alpha, tolerance, max epochs.".into());
    }
    let options = Options {
        alpha: fields[0].parse()?,
        tolerance: fields[1].parse()?,
        max_epochs: fields[2].parse()?,
    };
    if protocol == "MADALINE2" {
        return run_madaline(lines, options);
    }
    let weights = numbers(lines.next().ok_or("Missing weights.")?)?;
    let bias = lines.next().ok_or("Missing bias.")?.parse()?;
    let samples = samples(lines)?;
    let result = core::train(&samples, Model { weights, bias }, options)?;
    let mut out = BufWriter::new(io::stdout().lock());
    let (failed_epoch, failed_pattern) = result.failed_at.unwrap_or((0, 0));
    writeln!(
        out,
        "R\t{}\t{}\t{}\t{}",
        result.status.name(),
        result.epochs.len(),
        failed_epoch,
        failed_pattern
    )?;
    for s in &result.steps {
        writeln!(
            out,
            "S\t{}\t{}\t{:.17e}\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{}\t{}\t{}",
            s.epoch,
            s.pattern,
            s.net,
            s.output,
            s.error,
            s.delta_bias,
            s.before.bias,
            s.after.bias,
            s.max_change,
            vector(&s.before.weights),
            vector(&s.delta_weights),
            vector(&s.after.weights)
        )?;
    }
    for e in &result.epochs {
        writeln!(
            out,
            "E\t{}\t{:.17e}\t{:.17e}\t{:.17e}\t{}\t{}\t{:.17e}",
            e.number,
            e.max_change,
            e.net_change,
            e.sse,
            e.correct,
            vector(&e.model.weights),
            e.model.bias
        )?;
    }
    for (index, sample) in samples.iter().enumerate() {
        let net = result.model.net(&sample.inputs);
        writeln!(
            out,
            "P\t{}\t{:.17e}\t{}",
            index + 1,
            net,
            core::activate(net)
        )?;
    }
    let (sse, correct) = core::diagnostics(&samples, &result.model);
    writeln!(
        out,
        "M\t{}\t{:.17e}\t{:.17e}\t{}",
        vector(&result.model.weights),
        result.model.bias,
        sse,
        correct
    )?;
    out.flush()?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
