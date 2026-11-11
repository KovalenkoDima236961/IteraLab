//! Explicit file CLI, kept separate from the no-argument GUI protocol.
use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
pub const ALGORITHMS: &[(&str, &str)] = &[
    (
        "nelder-mead",
        "2D; variant=slides|standard; stopping=function|position; no learning rate",
    ),
    (
        "gradient-1d",
        "1D fixed rate; learning_rate_mode=provided|calculated (positive quadratic)",
    ),
    (
        "steepest-fixed",
        "2D fixed rate; learning_rate_mode=provided|calculated (positive-definite quadratic)",
    ),
    (
        "steepest-line-search",
        "2D negative gradient; step calculated by line search each iteration; initial_step is a probe",
    ),
    (
        "normalized-gradient",
        "2D unit negative gradient; step calculated by line search each iteration; initial_step is a probe",
    ),
    (
        "powell",
        "2D direction-shift cycles; initial_step is a line-search probe; no learning rate",
    ),
    ("newton-2d", "2D full Hessian; no learning rate"),
    (
        "perceptron",
        "mode=binary|bipolar|bipolar_error|band; learning_rate_mode=provided only",
    ),
    (
        "adaline",
        "Sequential LMS; learning_rate_mode=provided|calculated (augmented per-row norm)",
    ),
    (
        "hebb",
        "Zero-initialized, one ordered pass; learning_rate_mode=provided only",
    ),
    (
        "mcculloch-pitts",
        "lecture08 feed-forward binary network; fixed weights/thresholds; no learning rate",
    ),
];

fn help() -> String {
    let mut s = String::from("Rust calculation CLI\n\nUsage: exam-adaline ALGORITHM --input INPUT.txt --output REPORT.txt\n       exam-adaline --list\n       exam-adaline [ALGORITHM] --help\n\nNo arguments: existing stdin/stdout GUI protocol (no file access).\n\nAlgorithms and modes:\n");
    for (name, description) in ALGORITHMS {
        let _ = writeln!(s, "  {name}: {description}");
    }
    s.push_str("\nInput: key = value settings, then named numeric [sections]. # comments and blank lines allowed.\nSee every_example_gui/CLI.md and every_example_gui/cli_templates/ for every field and runnable inputs.\nExit 0: calculation status saved, including budget/numerical stops. Exit 1: invocation, input or I/O error.\n");
    s
}

pub fn run() -> Result<()> {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h" || args[0] == "--list") {
        print!("{}", help());
        return Ok(());
    }
    let algorithm = args
        .first()
        .and_then(|s| s.to_str())
        .ok_or("Missing algorithm; use --help.")?;
    if !ALGORITHMS.iter().any(|(name, _)| *name == algorithm) {
        return Err(format!("Unknown algorithm '{algorithm}'; use --list.").into());
    }
    if args.len() == 2 && (args[1] == "--help" || args[1] == "-h") {
        print!("{}", help());
        return Ok(());
    }
    let (mut input, mut output) = (None, None);
    let mut i = 1;
    while i < args.len() {
        let slot = match args[i].to_str() {
            Some("--input") => &mut input,
            Some("--output") => &mut output,
            _ => {
                return Err(format!(
                    "Unexpected argument '{}'; use --help.",
                    args[i].to_string_lossy()
                )
                .into())
            }
        };
        if slot.is_some() {
            return Err(format!("Duplicate argument '{}'.", args[i].to_string_lossy()).into());
        }
        let value = args
            .get(i + 1)
            .filter(|v| !v.is_empty() && !v.to_string_lossy().starts_with("--"))
            .ok_or("--input and --output each require a path.")?;
        *slot = Some(PathBuf::from(value));
        i += 2;
    }
    let input = input.ok_or("Missing --input path.")?;
    let output = output.ok_or("Missing --output path.")?;
    if let (Ok(a), Ok(b)) = (fs::canonicalize(&input), fs::canonicalize(&output)) {
        if a == b {
            return Err("Input and output must be different files.".into());
        }
    }
    let mut source = String::new();
    fs::File::open(&input)
        .map_err(|e| format!("Cannot read input '{}': {e}", input.display()))?
        .take(1_048_577)
        .read_to_string(&mut source)
        .map_err(|e| format!("Cannot read input '{}': {e}", input.display()))?;
    if source.len() > 1_048_576 {
        return Err("Input file exceeds 1 MiB.".into());
    }
    let prepared = prepare(algorithm, &source)?;
    let mut records = Vec::new();
    crate::dispatch(&prepared.protocol, &mut records)?;
    let records = String::from_utf8(records)?;
    let report = report(algorithm, &input, &source, &prepared, &records)?;
    atomic_write(&output, report.as_bytes())
        .map_err(|e| format!("Cannot write output '{}': {e}", output.display()))?;
    println!("Saved {} report to {}", algorithm, output.display());
    Ok(())
}

struct Document {
    settings: BTreeMap<String, String>,
    sections: BTreeMap<String, Vec<Vec<f64>>>,
}

fn numeric_row(text: &str) -> std::result::Result<Vec<f64>, String> {
    let values = text
        .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .filter(|v| !v.is_empty())
        .map(|v| {
            v.parse::<f64>()
                .map_err(|_| format!("Invalid number '{v}'."))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return Err("Use finite numeric values.".into());
    }
    Ok(values)
}

impl Document {
    fn parse(source: &str) -> Result<Self> {
        let mut doc = Self {
            settings: BTreeMap::new(),
            sections: BTreeMap::new(),
        };
        let mut section: Option<String> = None;
        for (i, raw) in source
            .strip_prefix('\u{feff}')
            .unwrap_or(source)
            .lines()
            .enumerate()
        {
            let line = raw.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let error = |s: &str| format!("Input line {}: {s}", i + 1);
            if line.starts_with('[') && line.ends_with(']') {
                let name = line[1..line.len() - 1].trim();
                if name.is_empty() || doc.sections.contains_key(name) {
                    return Err(error("empty or duplicate section").into());
                }
                doc.sections.insert(name.into(), Vec::new());
                section = Some(name.into());
            } else if let Some((key, value)) = line.split_once('=') {
                if section.is_some() {
                    return Err(error("settings must precede all data sections").into());
                }
                let (key, value) = (key.trim(), value.trim());
                if key.is_empty()
                    || value.is_empty()
                    || doc.settings.insert(key.into(), value.into()).is_some()
                {
                    return Err(error("empty setting or duplicate key").into());
                }
            } else if let Some(name) = &section {
                let row = numeric_row(line).map_err(|e| error(&e))?;
                let rows = doc.sections.get_mut(name).unwrap();
                if row.len() > 129 || rows.len() >= 128 {
                    return Err(error("data section exceeds supported dimensions").into());
                }
                if rows.first().is_some_and(|first| first.len() != row.len()) {
                    return Err(error("inconsistent row dimensions").into());
                }
                rows.push(row);
            } else {
                return Err(error("expected key = value or [section]").into());
            }
        }
        Ok(doc)
    }
    fn text(&mut self, key: &str) -> Result<String> {
        self.settings
            .remove(key)
            .ok_or_else(|| format!("Missing required setting '{key}'.").into())
    }
    fn number(&mut self, key: &str) -> Result<f64> {
        let text = self.text(key)?;
        let value: f64 = text
            .parse()
            .map_err(|_| format!("'{key}' must be a number."))?;
        if !value.is_finite() {
            return Err(format!("'{key}' must be finite.").into());
        }
        Ok(value)
    }
    fn count(&mut self, key: &str) -> Result<usize> {
        self.text(key)?
            .parse()
            .map_err(|_| format!("'{key}' must be a nonnegative integer.").into())
    }
    fn vector(&mut self, key: &str, length: Option<usize>) -> Result<Vec<f64>> {
        let v = numeric_row(&self.text(key)?)?;
        if length.is_some_and(|n| n != v.len()) {
            return Err(format!("'{key}' requires {} values.", length.unwrap()).into());
        }
        Ok(v)
    }
    fn rows(&mut self, key: &str) -> Result<Vec<Vec<f64>>> {
        let rows = self
            .sections
            .remove(key)
            .ok_or_else(|| format!("Missing required [{key}] section."))?;
        if rows.is_empty() {
            return Err(format!("[{key}] must contain at least one row.").into());
        }
        Ok(rows)
    }
    fn finish(&self) -> Result<()> {
        if let Some(key) = self.settings.keys().next() {
            return Err(format!("Unknown or inapplicable setting '{key}'.").into());
        }
        if let Some(key) = self.sections.keys().next() {
            return Err(format!("Unknown or inapplicable section [{key}].").into());
        }
        Ok(())
    }
}

fn joined(values: &[f64], separator: &str) -> String {
    values
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(separator)
}
fn rows_text(rows: &[Vec<f64>]) -> String {
    rows.iter()
        .map(|r| joined(r, ","))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) struct Prepared {
    pub protocol: String,
    pub selection: String,
}

fn rate(
    doc: &mut Document,
    algorithm: &str,
    expression: Option<&str>,
    samples: &[Vec<f64>],
) -> Result<(f64, String)> {
    let mode = doc.text("learning_rate_mode")?;
    let (alpha, detail) = match mode.as_str() {
        "provided" => (
            doc.number("learning_rate")?,
            "rule = supplied value used exactly as parsed (f64)\n".into(),
        ),
        "calculated" => {
            if doc.settings.contains_key("learning_rate") {
                return Err("Conflicting learning_rate in calculated mode; remove it.".into());
            }
            match algorithm {
                "gradient-1d" | "steepest-fixed" => crate::cli_expression::quadratic_rate(expression.unwrap(), if algorithm == "gradient-1d" { 1 } else { 2 })?,
                "adaline" => {
                    // Include the trained bias as the last coordinate; no mean/batch bound.
                    let norms: Vec<f64> = samples.iter().map(|r| 1. + r[..r.len()-1].iter().map(|v| v*v).sum::<f64>()).collect();
                    let maximum = norms.iter().copied().fold(0., f64::max);
                    let alpha = 1. / maximum;
                    if !maximum.is_finite() || !alpha.is_finite() || alpha <= 0. { return Err("Cannot calculate a finite positive sequential ADALINE rate.".into()); }
                    (alpha, format!("rule = 1 / max_k(1 + ||x_k||^2), calculated once and fixed\nassumption = sequential linear LMS with trained bias; per-row squared loss / 2\naugmented_squared_norms_in_row_order = {}\nmax_augmented_squared_norm = {maximum:.17e}\nEach row has 0 < alpha*||[x,1]||^2 <= 1; this does not promise convergence on inconsistent data.\n", joined(&norms, ", ")))
                }
                _ => return Err(format!("calculated learning-rate mode is unsupported for {algorithm}; use provided mode.").into()),
            }
        }
        _ => return Err("learning_rate_mode must be provided or calculated.".into()),
    };
    if alpha <= 0. || alpha > 1e50 || !alpha.is_finite() {
        return Err("learning_rate must be positive, finite and at most 1e50.".into());
    }
    Ok((
        alpha,
        format!("learning_rate_mode = {mode}\n{detail}resolved_learning_rate = {alpha:.17e}\n"),
    ))
}

pub(super) fn prepare(algorithm: &str, source: &str) -> Result<Prepared> {
    let mut d = Document::parse(source)?;
    let mut selection = "Learning rate: not applicable.\n".to_string();
    let protocol = match algorithm {
        "gradient-1d"
        | "steepest-fixed"
        | "steepest-line-search"
        | "normalized-gradient"
        | "powell"
        | "newton-2d"
        | "nelder-mead" => {
            let dimensions = if algorithm == "gradient-1d" { 1 } else { 2 };
            let expression = crate::cli_expression::compile(&d.text("objective")?, dimensions)?;
            let tolerance = d.number("tolerance")?;
            let limit = d.count("max_iterations")?;
            let bounds = d.vector("plot_bounds", Some(dimensions * 2))?;
            if algorithm == "nelder-mead" {
                let variant = d.text("variant")?;
                let stopping = d.text("stopping")?;
                let simplex = d.rows("simplex")?;
                if simplex.len() != 3 || simplex[0].len() != 2 {
                    return Err("[simplex] requires exactly three rows of two coordinates.".into());
                }
                format!("NELDER_MEAD2_V1\n{tolerance:.17e}\t{limit}\t{}\t{variant}\t{stopping}\n{}\n{expression}\n", joined(&bounds, "\t"), simplex.iter().map(|r| joined(r, "\t")).collect::<Vec<_>>().join("\n"))
            } else {
                let x0 = d.number("x0")?;
                if algorithm == "gradient-1d" {
                    let (alpha, detail) = rate(&mut d, algorithm, Some(&expression), &[])?;
                    selection = detail;
                    format!("GRADIENT1_V1\n{x0:.17e}\t{alpha:.17e}\t{tolerance:.17e}\t{limit}\t{}\n{expression}\n", joined(&bounds, "\t"))
                } else {
                    let y0 = d.number("y0")?;
                    let tail = format!("{tolerance:.17e}\t{limit}");
                    if algorithm == "steepest-fixed" {
                        let (alpha, detail) = rate(&mut d, algorithm, Some(&expression), &[])?;
                        selection = detail;
                        format!("STEEPEST2_V1\n{x0:.17e}\t{y0:.17e}\t{alpha:.17e}\t{tail}\t{}\n{expression}\n", joined(&bounds, "\t"))
                    } else if algorithm == "newton-2d" {
                        format!(
                            "NEWTON2_V1\n{x0:.17e}\t{y0:.17e}\t{tail}\t{}\n{expression}\n",
                            joined(&bounds, "\t")
                        )
                    } else {
                        let probe = d.number("initial_step")?;
                        let line_tolerance = d.number("line_tolerance")?;
                        let line_limit = d.count("max_line_evaluations")?;
                        let rule = if algorithm == "powell" {
                            "existing derivative-free signed line minimization along the current Powell direction"
                        } else if algorithm == "normalized-gradient" {
                            "existing directional-derivative bracketing/bisection along the unit negative gradient"
                        } else {
                            "existing directional-derivative bracketing/bisection along the unnormalized negative gradient"
                        };
                        selection = format!("step_selection = per_iteration_line_search\nrule = {rule}\ninitial_search_probe = {probe:.17e}\nA new accepted step is calculated for each search; see accepted_index and its trial.\nNo separately fixed learning rate is supplied or calculated before execution.\n");
                        if algorithm == "normalized-gradient" {
                            selection.push_str("direction_rule = S_k = -gradient(f)(P_k) / ||gradient(f)(P_k)||\nline_function = phi(gamma) = f(P_k + gamma*S_k)\nstep_condition = phi'(gamma_k) = 0, approximated using the existing relative line tolerance\nupdate_rule = P_(k+1) = P_k + gamma_k*S_k\n");
                        } else if algorithm == "steepest-line-search" {
                            selection.push_str("direction_rule = d_k = -gradient(f)(X_k)\nline_function = phi(alpha) = f(X_k + alpha*d_k)\nstep_condition = phi'(alpha_k) = 0, approximated using the existing relative line tolerance\nupdate_rule = X_(k+1) = X_k + alpha_k*d_k\n");
                        }
                        let opts = format!("{x0:.17e}\t{y0:.17e}\t{tail}\t{probe:.17e}\t{line_tolerance:.17e}\t{line_limit}\t{}", joined(&bounds, "\t"));
                        if algorithm == "powell" {
                            format!("POWELL2_V1\n{opts}\n{expression}\n")
                        } else {
                            format!(
                                "NORMALIZED2_V1\n{opts}\t{}\n{expression}\n",
                                u8::from(algorithm == "normalized-gradient")
                            )
                        }
                    }
                }
            }
        }
        "adaline" | "perceptron" | "hebb" => {
            let samples = d.rows("samples")?;
            if samples[0].len() < 2 {
                return Err("[samples] rows need inputs followed by target.".into());
            }
            let (alpha, detail) = rate(&mut d, algorithm, None, &samples)?;
            selection = detail;
            if algorithm == "hebb" {
                format!("HEBBIAN1\n{alpha:.17e}\n{}\n", rows_text(&samples))
            } else {
                let weights = d.vector("initial_weights", None)?;
                let bias = d.number("initial_bias")?;
                let epochs = d.count("max_epochs")?;
                let opts = if algorithm == "adaline" {
                    let tolerance = d.number("tolerance")?;
                    format!("ADALINE1\n{alpha:.17e}\t{tolerance:.17e}\t{epochs}")
                } else {
                    let mode = d.text("mode")?;
                    let theta = if mode == "band" {
                        d.number("theta")?
                    } else {
                        0.
                    };
                    format!("PERCEPTRON1\n{mode}\n{alpha:.17e}\t{theta:.17e}\t{epochs}")
                };
                format!(
                    "{opts}\n{}\n{bias:.17e}\n{}\n",
                    joined(&weights, ","),
                    rows_text(&samples)
                )
            }
        }
        "mcculloch-pitts" => {
            let patterns = d.rows("patterns")?;
            let weights = d.rows("weights")?;
            let thresholds = vec![d.vector("thresholds", None)?];
            let expected = if d.sections.contains_key("expected") {
                d.rows("expected")?
            } else {
                Vec::new()
            };
            let mut p = String::from("LECTURE08_V1\nmcp\n0\t0\tbinary\n");
            for matrix in [
                &patterns,
                &thresholds,
                &weights,
                &Vec::new(),
                &Vec::new(),
                &expected,
            ] {
                writeln!(
                    p,
                    "{}\t{}",
                    matrix.len(),
                    matrix.first().map_or(0, Vec::len)
                )?;
                writeln!(
                    p,
                    "{}",
                    joined(&matrix.iter().flatten().copied().collect::<Vec<_>>(), ",")
                )?;
            }
            p
        }
        _ => return Err(format!("Unknown algorithm '{algorithm}'.").into()),
    };
    d.finish()?;
    if protocol.len() > 1_048_576 {
        return Err("Resolved backend request exceeds 1 MiB.".into());
    }
    Ok(Prepared {
        protocol,
        selection,
    })
}

struct TempReport(PathBuf);
impl Drop for TempReport {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut attempt = 0u32;
    let (temporary, mut file) = loop {
        let candidate = parent.join(format!(".exam-report-{}-{attempt}.tmp", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => break (TempReport(candidate), file),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && attempt < 1000 => {
                attempt += 1
            }
            Err(e) => return Err(e),
        }
    };
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary.0, path)?;
    Ok(())
}

fn termination(status: &str) -> (&'static str, &'static str) {
    match status {
        "gradient_tolerance" => (
            "converged",
            "Gradient magnitude/norm <= tolerance at the recorded point.",
        ),
        "function_tolerance" => ("converged", "Sorted simplex objective spread <= tolerance."),
        "simplex_tolerance" => (
            "converged",
            "Maximum distance from the best simplex vertex < tolerance.",
        ),
        "powell_function_tolerance" => (
            "converged",
            "Absolute objective change after a full Powell cycle < tolerance.",
        ),
        "powell_position_tolerance" => (
            "converged",
            "Displacement after a full Powell cycle < tolerance.",
        ),
        "tolerance_met" => (
            "converged",
            "Largest individual coefficient change over a full epoch < tolerance.",
        ),
        "weights_stable" => (
            "converged",
            "No weight or bias changed anywhere in the completed epoch.",
        ),
        "pass_complete" => (
            "normal_completion",
            "One ordered Hebbian pass and final verification completed; accuracy is separate.",
        ),
        "normal_completion" => (
            "normal_completion",
            "All binary patterns evaluated through the fixed network; no convergence test applies.",
        ),
        "iteration_limit" | "epoch_limit" => (
            "budget_exhausted",
            "Requested update/epoch budget exhausted without satisfying the stopping test.",
        ),
        "line_search_limit" => (
            "budget_exhausted",
            "Line-search trial budget exhausted; no fallback step was introduced.",
        ),
        "numerical_limit" => (
            "numerical_failure",
            "Existing domain/finite-magnitude guard rejected an update; last valid state retained.",
        ),
        "singular_hessian" => (
            "numerical_failure",
            "Existing Newton system detected a singular Hessian; last valid point retained.",
        ),
        "stagnation" => (
            "numerical_failure",
            "Candidate state rounds to the current state while the stopping test fails.",
        ),
        "line_search_precision" => (
            "numerical_failure",
            "Existing line search reached its working-precision limit.",
        ),
        "line_search_failed" | "powell_line_failed" => (
            "search_failure",
            "Existing search could not bracket an acceptable line solution.",
        ),
        "line_search_increase" => (
            "search_failure",
            "Stationary line trial increased the objective and was rejected.",
        ),
        _ => (
            "backend_status",
            "See the unmodified backend status and complete trace.",
        ),
    }
}

fn columns(tag: &str, mcp: bool) -> Option<(&'static str, &'static str)> {
    Some(match tag {
        "R" => ("Training termination", "status completed_epochs failed_epoch failed_pattern"),
        "S" if mcp => ("Neuron calculations", "title detail referenced_matrices"),
        "S" | "PS" => ("Training updates", "epoch pattern net output error delta_bias before_bias after_bias max_change before_weights delta_weights after_weights changed"),
        "HS" => ("Hebbian updates", "epoch pattern delta_bias before_bias after_bias max_change before_weights delta_weights after_weights"),
        "E" | "PE" => ("Epoch summaries", "epoch max_change net_change SSE correct weights bias mistakes changed_visits"),
        "P" => ("Final fixed-model predictions", "pattern net output"),
        "M" if mcp => ("Network matrices", "name rows columns row_major_values"),
        "M" => ("Final model", "weights bias SSE correct"),
        "L08" => ("Network operation", "operation"),
        "D" => ("Network diagnostics", "name value"),
        "N" => ("Network notes", "name text"),
        "END" => ("Network report complete", ""),
        "GR" | "G2R" | "N2R" | "NMR" | "NGR" | "PWR" => ("Termination", "status"),
        "GP" => ("Gradient iterations", "iteration x objective derivative"),
        "G2P" | "NGP" => ("Gradient iterations", "iteration x y objective gradient_x gradient_y gradient_norm"),
        "N2P" => ("Newton iterations", "iteration x y objective gradient_x gradient_y gradient_norm H_xx H_xy H_yy Hessian_scale scaled_determinant delta_x delta_y"),
        "NMP" => ("Sorted simplex history", "iteration function_spread diameter A_x A_y A_f B_x B_y B_f C_x C_y C_f"),
        "NMT" => ("Simplex changes and trials", "iteration action centroid_x centroid_y reflection_x reflection_y reflection_f expansion_x expansion_y expansion_f contraction_x contraction_y contraction_f shrink_B_x shrink_B_y shrink_B_f shrink_C_x shrink_C_y shrink_C_f"),
        "NMF" => ("Rejected simplex trial", "stage x y"),
        "NGL" | "PWL" => ("Line searches", "search_index direction_x direction_y status accepted_index"),
        "NGS" => ("Line-search trials", "search_index trial_index phase step x y objective gradient_x gradient_y directional_derivative"),
        "PWP" => ("Powell moves", "move_index x y objective"),
        "PWC" => ("Completed Powell cycles", "cycle d1_x d1_y d2_x d2_y d3_x d3_y absolute_function_change displacement"),
        "PWS" => ("Powell line-search trials", "search_index trial_index phase step x y objective"),
        "GC" => ("Objective plot samples (not iterations)", "x objective"),
        "G2C" | "N2C" | "NMC" | "NGC" | "PWG" => ("Objective grid samples (not iterations)", "cell_index objective"),
        _ => return None,
    })
}

fn report(
    algorithm: &str,
    input: &Path,
    source: &str,
    prepared: &Prepared,
    records: &str,
) -> Result<String> {
    let mcp = algorithm == "mcculloch-pitts";
    let status = if mcp {
        "normal_completion"
    } else {
        records
            .lines()
            .next()
            .and_then(|l| l.split('\t').nth(1))
            .ok_or("Backend returned no status.")?
    };
    let (category, reason) = termination(status);
    let mut s = format!("Rust calculation report\nAlgorithm: {algorithm}\nInput: {}\nStatus: {status}\nOutcome: {category}\nReason: {reason}\n\nLearning-rate / search selection\n{}", input.display(), prepared.selection);
    let settings = Document::parse(source)?.settings;
    if algorithm == "perceptron" {
        writeln!(s, "Activation/update mode: {}", settings["mode"])?;
    } else if algorithm == "nelder-mead" {
        writeln!(
            s,
            "Branch variant: {}\nStopping rule: {}",
            settings["variant"], settings["stopping"]
        )?;
    }
    let point_tag = match algorithm {
        "gradient-1d" => "GP",
        "steepest-fixed" => "G2P",
        "newton-2d" => "N2P",
        "nelder-mead" => "NMP",
        "normalized-gradient" | "steepest-line-search" => "NGP",
        "powell" => "PWP",
        _ => "",
    };
    if !point_tag.is_empty() {
        let points: Vec<&str> = records
            .lines()
            .filter(|l| l.split('\t').next() == Some(point_tag))
            .collect();
        writeln!(
            s,
            "\nCompleted updates/moves: {}",
            points.len().saturating_sub(1)
        )?;
        if algorithm == "powell" {
            writeln!(
                s,
                "Completed cycles: {}",
                records.lines().filter(|l| l.starts_with("PWC\t")).count()
            )?;
        }
        if let Some(last) = points.last() {
            let (_, names) = columns(point_tag, false).unwrap();
            s.push_str("Final result:\n");
            for (name, value) in names.split_whitespace().zip(last.split('\t').skip(1)) {
                writeln!(s, "  {name} = {value}")?;
            }
        }
    } else if mcp {
        let first = records
            .lines()
            .find(|l| l.starts_with("M\tX\t"))
            .ok_or("Missing network input matrix.")?;
        writeln!(
            s,
            "\nCompleted patterns: {}\nFinal results: net and output matrices below.",
            first.split('\t').nth(2).unwrap()
        )?;
        for name in ["net", "output"] {
            let line = records
                .lines()
                .find(|l| l.starts_with(&format!("M\t{name}\t")))
                .ok_or("Missing network result matrix.")?;
            let fields: Vec<_> = line.split('\t').collect();
            let width: usize = fields[3].parse()?;
            let values: Vec<_> = fields[4].split(',').collect();
            writeln!(s, "Final network {name} (one row per input pattern):")?;
            for (index, row) in values.chunks(width).enumerate() {
                writeln!(s, "  pattern {}: {}", index + 1, row.join(", "))?;
            }
        }
    } else {
        let meta: Vec<_> = records.lines().next().unwrap().split('\t').collect();
        writeln!(s, "\nCompleted epochs/passes: {}\nFailed epoch/pattern: {}/{} (0/0 means none)\nFinal results: Final model and Final fixed-model predictions below.", meta[2], meta[3], meta[4])?;
        writeln!(
            s,
            "Recorded pattern visits: {}",
            records
                .lines()
                .filter(|l| matches!(l.split('\t').next(), Some("S" | "PS" | "HS")))
                .count()
        )?;
        let model = records
            .lines()
            .find(|l| l.starts_with("M\t"))
            .ok_or("Missing final learning model.")?;
        s.push_str("Final model:\n");
        for (name, value) in ["weights", "bias", "SSE", "correct"]
            .iter()
            .zip(model.split('\t').skip(1))
        {
            writeln!(s, "  {name} = {value}")?;
        }
        if status == "numerical_limit" {
            s.push_str("Predictions describe the last valid partial model; verification/training is incomplete.\n");
        }
    }
    s.push_str(
        "\nInput settings and data (original order)\n--------------------------------------\n",
    );
    s.push_str(source);
    s.push_str("\n\nComplete calculation trace\nNumbers retain the backend's 17 digits after the decimal in scientific notation.\nTab-separated columns; vectors/matrices use comma-separated values. '_' means no value/trial.\nIteration/search indices start at 0; epochs, patterns and cycles start at 1.\nGrid samples use explicit plot_bounds: 49x49 cell centers, except Powell uses 49x49 endpoints-inclusive samples (all row-major); 1D uses 401 endpoints-inclusive samples.\nSearch accepted_index refers to trial_index within the same search, including the initial trial.\nConvergence records the existing stopping test; it does not certify a global minimum or classification accuracy.\n");
    let mut previous = "";
    for line in records.lines() {
        let tag = line.split('\t').next().unwrap();
        let (title, names) = columns(tag, mcp)
            .ok_or_else(|| format!("Unrecognized backend report record '{tag}'."))?;
        if tag != previous {
            writeln!(s, "\n{title}")?;
            // Shared learning records have different optional tail columns.
            let actual = line.split('\t').count() - 1;
            writeln!(
                s,
                "record\t{}",
                names
                    .split_whitespace()
                    .take(actual)
                    .collect::<Vec<_>>()
                    .join("\t")
            )?;
            previous = tag;
        }
        writeln!(s, "{line}")?;
    }
    s.push_str("\nResolved backend request (reproducibility)\n--- BEGIN BACKEND REQUEST ---\n");
    s.push_str(&prepared.protocol);
    s.push_str("--- END BACKEND REQUEST ---\n");
    Ok(s)
}
