use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

static SERIAL: AtomicUsize = AtomicUsize::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "exam cli tests {} {}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_exam-adaline"))
}
fn template(relative: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../cli_templates")
            .join(relative),
    )
    .unwrap()
}
fn calculate(algorithm: &str, source: &str) -> String {
    let scratch = Scratch::new();
    let input = scratch.file("input with spaces.txt");
    let output = scratch.file("output with spaces.txt");
    fs::write(&input, source).unwrap();
    let result = bin()
        .args([algorithm, "--input"])
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{algorithm}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stderr.is_empty());
    fs::read_to_string(output).unwrap()
}
fn request(report: &str) -> &str {
    report
        .split_once("--- BEGIN BACKEND REQUEST ---\n")
        .unwrap()
        .1
        .split_once("--- END BACKEND REQUEST ---\n")
        .unwrap()
        .0
}
fn backend(protocol: &str) -> Output {
    let mut child = bin()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(protocol.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}
fn records(report: &str) -> String {
    let trace = report
        .split_once("Complete calculation trace\n")
        .unwrap()
        .1
        .split_once("Resolved backend request (reproducibility)\n")
        .unwrap()
        .0;
    trace
        .lines()
        .filter(|line| (line.contains('\t') && !line.starts_with("record\t")) || *line == "END")
        .map(|l| format!("{l}\n"))
        .collect()
}
fn fail(algorithm: &str, source: &str, message: &str) {
    let scratch = Scratch::new();
    fs::write(scratch.file("input.txt"), source).unwrap();
    fs::write(scratch.file("keep.txt"), "previous complete report").unwrap();
    let result = bin()
        .current_dir(&scratch.0)
        .args([algorithm, "--input", "input.txt", "--output", "keep.txt"])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(result.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&result.stderr).contains(message),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        fs::read_to_string(scratch.file("keep.txt")).unwrap(),
        "previous complete report"
    );
    assert_eq!(
        fs::read_dir(&scratch.0).unwrap().count(),
        2,
        "partial file left behind"
    );
}

#[test]
fn every_template_preserves_every_backend_record_and_termination() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../cli_templates");
    let mut seen = Vec::new();
    for group in ["optimization", "learning", "networks"] {
        for entry in fs::read_dir(root.join(group)).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|p| p.to_str()) != Some("txt") {
                continue;
            }
            let name = path.file_stem().unwrap().to_str().unwrap();
            let algorithm = [
                "nelder-mead",
                "gradient-1d",
                "steepest-fixed",
                "steepest-line-search",
                "normalized-gradient",
                "powell",
                "newton-2d",
                "perceptron",
                "adaline",
                "hebb",
                "mcculloch-pitts",
            ]
            .into_iter()
            .find(|a| name.starts_with(a))
            .unwrap();
            let source = fs::read_to_string(&path).unwrap();
            let report = calculate(algorithm, &source);
            let legacy = backend(request(&report));
            assert!(legacy.status.success());
            assert_eq!(
                records(&report),
                String::from_utf8(legacy.stdout).unwrap(),
                "{name}"
            );
            assert!(report.contains(&source), "original row order/settings lost");
            assert!(
                report.contains("Status:")
                    && report.contains("Outcome:")
                    && report.contains("Reason:")
            );
            assert!(report.contains("Final result"));
            seen.push(name.to_string());
        }
    }
    assert_eq!(seen.len(), 20);
    for variant in ["binary", "bipolar", "bipolar_error", "band"] {
        assert!(seen.contains(&format!("perceptron-{variant}")));
    }
    for variant in ["slides", "standard"] {
        for stop in ["function", "position"] {
            assert!(seen.contains(&format!("nelder-mead-{variant}-{stop}")));
        }
    }
}

#[test]
fn calculated_and_provided_rates_have_identical_full_numerical_records() {
    for (algorithm, file) in [
        ("gradient-1d", "optimization/gradient-1d-calculated.txt"),
        (
            "steepest-fixed",
            "optimization/steepest-fixed-calculated.txt",
        ),
        ("adaline", "learning/adaline-calculated.txt"),
    ] {
        let source = template(file);
        let calculated = calculate(algorithm, &source);
        let rate = calculated
            .lines()
            .find_map(|l| l.strip_prefix("resolved_learning_rate = "))
            .unwrap();
        let provided = calculate(
            algorithm,
            &source.replace(
                "learning_rate_mode = calculated",
                &format!("learning_rate_mode = provided\nlearning_rate = {rate}"),
            ),
        );
        assert_eq!(records(&calculated), records(&provided));
        assert_eq!(request(&calculated), request(&provided));
        let numeric: f64 = rate.parse().unwrap();
        let expected = match algorithm {
            "gradient-1d" => 0.5,
            "steepest-fixed" => 1. / (5. + 2_f64.sqrt()),
            _ => 1. / 3.,
        };
        assert!((numeric - expected).abs() < 1e-15);
        assert!(calculated.contains("calculated once and fixed"));
        if algorithm == "adaline" {
            assert!(
                calculated.contains("augmented_squared_norms_in_row_order")
                    && calculated.contains("trained bias")
            );
        } else {
            assert!(
                calculated.contains("lambda_max")
                    && calculated.contains("positive-definite quadratic")
            );
        }
    }
}

#[test]
fn optimized_steepest_step_is_recalculated_each_iteration() {
    let source = template("optimization/steepest-line-search.txt")
        .replace("max_iterations = 200", "max_iterations = 2");
    let report = calculate("steepest-line-search", &source);
    assert!(report.contains("step_selection = per_iteration_line_search"));
    assert!(report.contains("unnormalized negative gradient"));
    assert!(!report.contains("Learning rate: not applicable."));
    assert!(!report.contains("resolved_learning_rate ="));
    let trace = records(&report);
    let mut accepted = Vec::new();
    for search in 0..2 {
        let header = trace
            .lines()
            .find(|l| l.starts_with(&format!("NGL\t{search}\t")))
            .unwrap();
        let index = header.split('\t').nth(5).unwrap();
        let trial = trace
            .lines()
            .find(|l| l.starts_with(&format!("NGS\t{search}\t{index}\t")))
            .unwrap();
        accepted.push(trial.split('\t').nth(4).unwrap().parse::<f64>().unwrap());
    }
    // Hand-worked CV5 fixture: phi_0(alpha)=36*alpha^2-20*alpha.
    assert!((accepted[0] - 5. / 18.).abs() < 1e-9);
    assert!((accepted[1] - 5. / 32.).abs() < 1e-8);
    let fixed = calculate(
        "steepest-fixed",
        &template("optimization/steepest-fixed-calculated.txt"),
    );
    let fixed_rate = fixed
        .lines()
        .find_map(|l| l.strip_prefix("resolved_learning_rate = "))
        .unwrap()
        .parse::<f64>()
        .unwrap();
    assert!((fixed_rate - 1. / (5. + 2_f64.sqrt())).abs() < 1e-15);
    assert!((accepted[0] - fixed_rate).abs() > 0.1);
}

#[test]
fn normalized_gradient_matches_the_unit_vector_slide_and_recomputes_gamma() {
    let source = template("optimization/normalized-gradient.txt")
        .replace("max_iterations = 200", "max_iterations = 2");
    let report = calculate("normalized-gradient", &source);
    assert!(report.contains("direction_rule = S_k = -gradient(f)(P_k) / ||gradient(f)(P_k)||"));
    assert!(report.contains("line_function = phi(gamma) = f(P_k + gamma*S_k)"));
    assert!(report.contains("update_rule = P_(k+1) = P_k + gamma_k*S_k"));
    assert!(!report.contains("resolved_learning_rate ="));
    let trace = records(&report);
    let points: Vec<Vec<f64>> = trace
        .lines()
        .filter(|l| l.starts_with("NGP\t"))
        .map(|l| l.split('\t').skip(2).map(|v| v.parse().unwrap()).collect())
        .collect();
    let searches: Vec<Vec<&str>> = trace
        .lines()
        .filter(|l| l.starts_with("NGL\t"))
        .map(|l| l.split('\t').collect())
        .collect();
    assert_eq!(searches.len(), 2);
    let mut gammas = Vec::new();
    for (k, search) in searches.iter().enumerate() {
        let sx: f64 = search[2].parse().unwrap();
        let sy: f64 = search[3].parse().unwrap();
        assert!((sx.hypot(sy) - 1.).abs() < 1e-14);
        let old = &points[k];
        assert!((sx + old[3] / old[5]).abs() < 1e-14);
        assert!((sy + old[4] / old[5]).abs() < 1e-14);
        let trial = trace
            .lines()
            .find(|l| l.starts_with(&format!("NGS\t{k}\t{}\t", search[5])))
            .unwrap();
        let gamma: f64 = trial.split('\t').nth(4).unwrap().parse().unwrap();
        assert!((points[k + 1][0] - (old[0] + gamma * sx)).abs() < 1e-14);
        assert!((points[k + 1][1] - (old[1] + gamma * sy)).abs() < 1e-14);
        gammas.push(gamma);
    }
    assert!((gammas[0] - 17. * 17_f64.sqrt() / 26.).abs() < 1e-9);
    assert!((gammas[0] - gammas[1]).abs() > 1.);
    assert!((points[1][0] - 113. / 26.).abs() < 1e-9);
    assert!((points[1][1] - 31. / 13.).abs() < 1e-9);
    let stationary = source
        .replace("objective = x^2-4*x+y^2-y-x*y", "objective = x^2+y^2")
        .replace("x0 = 5", "x0 = 0")
        .replace("y0 = 5", "y0 = 0");
    let stopped = calculate("normalized-gradient", &stationary);
    assert!(stopped.contains("Status: gradient_tolerance"));
    assert!(!records(&stopped).contains("NGL\t"));
}

#[test]
fn adaline_rate_includes_bias_and_uses_maximum_not_mean_or_batch_spectrum() {
    let source = template("learning/adaline-calculated.txt").replace(
        "1, 1, 1\n1, -1, -1\n-1, 1, -1\n-1, -1, -1",
        "2, 1, -1\n3, 2, 1\n4, 3, 1",
    );
    let report = calculate("adaline", &source);
    let rate: f64 = report
        .lines()
        .find_map(|l| l.strip_prefix("resolved_learning_rate = "))
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(rate, 1. / 26.);
    assert!(report.contains("max_augmented_squared_norm = 2.60000000000000000e1"));
    let zeros = source.replace("2, 1, -1\n3, 2, 1\n4, 3, 1", "0, 0, 1");
    assert!(calculate("adaline", &zeros).contains("resolved_learning_rate = 1.00000000000000000e0"));
}

#[test]
fn missing_conflicting_unsupported_and_inapplicable_rates_are_errors() {
    let source = template("learning/adaline-provided.txt");
    for (old, new, message) in [
        ("learning_rate_mode = provided\n", "", "learning_rate_mode"),
        ("learning_rate = 0.2\n", "", "learning_rate"),
        (
            "learning_rate_mode = provided",
            "learning_rate_mode = calculated",
            "Conflicting",
        ),
        (
            "learning_rate_mode = provided",
            "learning_rate_mode = guessed",
            "provided or calculated",
        ),
        ("learning_rate = 0.2", "learning_rate = 0", "learning_rate"),
        ("learning_rate = 0.2", "learning_rate = NaN", "finite"),
    ] {
        fail("adaline", &source.replace(old, new), message);
    }
    for (algorithm, file) in [
        ("hebb", "learning/hebb.txt"),
        ("perceptron", "learning/perceptron-binary.txt"),
    ] {
        let source = template(file)
            .replace(
                "learning_rate_mode = provided",
                "learning_rate_mode = calculated",
            )
            .replace("learning_rate = 1\n", "");
        fail(algorithm, &source, "unsupported");
    }
    for (algorithm, file) in [
        ("newton-2d", "optimization/newton-2d.txt"),
        (
            "nelder-mead",
            "optimization/nelder-mead-slides-function.txt",
        ),
        ("powell", "optimization/powell.txt"),
        (
            "normalized-gradient",
            "optimization/normalized-gradient.txt",
        ),
        (
            "steepest-line-search",
            "optimization/steepest-line-search.txt",
        ),
        ("mcculloch-pitts", "networks/mcculloch-pitts.txt"),
    ] {
        fail(
            algorithm,
            &format!("learning_rate_mode = calculated\n{}", template(file)),
            "inapplicable",
        );
    }
    let source = template("optimization/gradient-1d-calculated.txt");
    for expr in ["exp(x)+x^2", "-x^2", "x", "x^4", "1/x"] {
        fail(
            "gradient-1d",
            &source.replace("objective = x^2", &format!("objective = {expr}")),
            "requires a structurally quadratic",
        );
    }
    fail(
        "steepest-fixed",
        &template("optimization/steepest-fixed-calculated.txt")
            .replace("2*x^2+x*y+3*y^2-4*x+2*y", "x^2-y^2"),
        "positive-definite",
    );
}

#[test]
fn malformed_data_and_backend_validations_preserve_the_existing_report() {
    let source = template("learning/adaline-provided.txt");
    for (bad, message) in [
        (
            source.replace("initial_weights = 0, 0", "initial_weights = 0"),
            "match the weights",
        ),
        (source.replace("1, 1, 1", "1, 1, 0"), "bipolar"),
        (source.replace("1, 1, 1", "1, 1"), "dimensions"),
        (source.replace("1, 1, 1", "x, 1, 1"), "Invalid number"),
        (
            source.replace("max_epochs = 2", "max_epochs = 0"),
            "Maximum epochs",
        ),
        (
            source.replace("max_epochs = 2", "max_epochs = 1.5"),
            "integer",
        ),
        (
            source.replace("max_epochs = 2", "max_epochs = 100000"),
            "trace supports",
        ),
        (
            source.replace("[samples]", "[samples]\n[samples]"),
            "duplicate section",
        ),
        (format!("tolerance = 1\n{source}"), "duplicate key"),
        (format!("misspelled = 1\n{source}"), "inapplicable setting"),
        (format!("{source}\n[unused]\n1\n"), "inapplicable section"),
        (format!("{source}\nextra = 1\n"), "settings must precede"),
        ("[samples]\n".into(), "at least one row"),
    ] {
        fail("adaline", &bad, message);
    }
    fail(
        "perceptron",
        &template("learning/perceptron-binary.txt")
            .replace("learning_rate = 1", "learning_rate = 1.1"),
        "rate",
    );
    fail(
        "perceptron",
        &format!(
            "theta = 0.2\n{}",
            template("learning/perceptron-binary.txt")
        ),
        "inapplicable setting",
    );
    fail(
        "perceptron",
        &template("learning/perceptron-band.txt").replace("theta = 0.2", "theta = -1"),
        "Theta",
    );
    fail(
        "nelder-mead",
        &template("optimization/nelder-mead-slides-function.txt")
            .replace("2, 2\n2, 1\n1, 2", "0, 0\n1, 1\n2, 2"),
        "non-collinear",
    );
    fail(
        "mcculloch-pitts",
        &template("networks/mcculloch-pitts.txt").replace("1, 1, 0, 0, 0", "1, 1, 1, 0, 0"),
        "same or later neuron",
    );
    for expr in ["x+", "sin()", "2x", "__import__('os')", "y"] {
        fail(
            "gradient-1d",
            &template("optimization/gradient-1d-provided.txt")
                .replace("objective = x^2", &format!("objective = {expr}")),
            "expression",
        );
    }
    fail(
        "newton-2d",
        &template("optimization/newton-2d.txt")
            .replace("objective = x^2+2*y^2", "objective = ln(-1)+x+y"),
        "starting",
    );
}

#[test]
fn statuses_distinguish_convergence_budget_numerical_failure_and_normal_completion() {
    let source = template("optimization/gradient-1d-provided.txt");
    let budget = calculate(
        "gradient-1d",
        &source.replace("max_iterations = 200", "max_iterations = 1"),
    );
    assert!(budget.contains("Status: iteration_limit\nOutcome: budget_exhausted"));
    let numerical = calculate(
        "gradient-1d",
        &source
            .replace("objective = x^2", "objective = ln(x)")
            .replace("learning_rate = 0.1", "learning_rate = 100"),
    );
    assert!(numerical.contains("Status: numerical_limit\nOutcome: numerical_failure"));
    assert!(calculate(
        "gradient-1d",
        &template("optimization/gradient-1d-calculated.txt")
    )
    .contains("Outcome: converged"));
    assert!(calculate("hebb", &template("learning/hebb.txt"))
        .contains("Status: pass_complete\nOutcome: normal_completion"));
    assert!(
        calculate("mcculloch-pitts", &template("networks/mcculloch-pitts.txt"))
            .contains("Status: normal_completion\nOutcome: normal_completion")
    );
    let limited_search = calculate(
        "normalized-gradient",
        &template("optimization/normalized-gradient.txt")
            .replace("max_line_evaluations = 256", "max_line_evaluations = 1"),
    );
    assert!(limited_search.contains("Status: line_search_limit\nOutcome: budget_exhausted"));
    let singular = calculate(
        "newton-2d",
        &template("optimization/newton-2d.txt").replace("objective = x^2+2*y^2", "objective = x+y"),
    );
    assert!(singular.contains("Status: singular_hessian\nOutcome: numerical_failure"));
    let partial = calculate(
        "hebb",
        "learning_rate_mode = provided\nlearning_rate = 1e50\n[samples]\n1e50, 1\n1e50, 1\n",
    );
    assert!(partial.contains("verification/training is incomplete"));
}

#[test]
fn invocation_and_file_errors_use_stderr_and_leave_no_partial_reports() {
    for args in [
        vec!["unknown"],
        vec!["adaline"],
        vec!["adaline", "--input"],
        vec!["adaline", "--bogus"],
        vec!["adaline", "--input", "a", "--input", "b"],
    ] {
        let r = bin().args(args).output().unwrap();
        assert_eq!(r.status.code(), Some(1));
        assert!(r.stdout.is_empty());
        assert!(!r.stderr.is_empty());
    }
    for args in [vec!["--help"], vec!["--list"], vec!["perceptron", "--help"]] {
        let r = bin().args(args).output().unwrap();
        assert!(r.status.success());
        assert!(r.stderr.is_empty());
        assert!(String::from_utf8_lossy(&r.stdout).contains("bipolar_error"));
    }
    let scratch = Scratch::new();
    fs::write(scratch.file("input.txt"), template("learning/hebb.txt")).unwrap();
    fs::create_dir(scratch.file("directory.txt")).unwrap();
    for (input, output, message) in [
        ("missing.txt", "output.txt", "Cannot read"),
        ("directory.txt", "output.txt", "Cannot read"),
        ("input.txt", "missing/output.txt", "Cannot write"),
        ("input.txt", "directory.txt", "Cannot write"),
        ("input.txt", "input.txt", "different files"),
    ] {
        let r = bin()
            .current_dir(&scratch.0)
            .args(["hebb", "--input", input, "--output", output])
            .output()
            .unwrap();
        assert_eq!(r.status.code(), Some(1));
        assert!(r.stdout.is_empty());
        assert!(String::from_utf8_lossy(&r.stderr).contains(message));
        assert_eq!(fs::read_dir(&scratch.0).unwrap().count(), 2);
    }
    let r = bin()
        .current_dir(&scratch.0)
        .args([
            "hebb",
            "--input",
            "input.txt",
            "--output",
            "relative report.txt",
        ])
        .output()
        .unwrap();
    assert!(r.status.success());
    assert!(scratch.file("relative report.txt").is_file());
    fs::write(scratch.file("input.txt"), vec![b'x'; 1_048_577]).unwrap();
    let r = bin()
        .current_dir(&scratch.0)
        .args([
            "hebb",
            "--input",
            "input.txt",
            "--output",
            "relative report.txt",
        ])
        .output()
        .unwrap();
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr).contains("1 MiB"));
    assert!(fs::read_to_string(scratch.file("relative report.txt"))
        .unwrap()
        .starts_with("Rust calculation report"));
}

#[test]
fn comments_blank_lines_bom_crlf_and_separators_preserve_row_order() {
    let source = template("learning/hebb.txt");
    let normal = calculate("hebb", &source);
    let variant = format!(
        "\u{feff}\n# header\n\n{}",
        source
            .replace("1, 1, 1", "1; 1\t1 # first pattern")
            .replace('\n', "\r\n")
    );
    assert_eq!(records(&normal), records(&calculate("hebb", &variant)));
    assert_eq!(request(&normal), request(&calculate("hebb", &variant)));
    let reversed = source.replace(
        "1, 1, 1\n-1, 1, -1\n1, -1, -1\n-1, -1, -1",
        "-1, -1, -1\n1, -1, -1\n-1, 1, -1\n1, 1, 1",
    );
    assert_ne!(records(&normal), records(&calculate("hebb", &reversed)));
}

#[test]
fn no_argument_error_protocol_remains_unchanged() {
    let r = backend("WRONG\n");
    assert_eq!(r.status.code(), Some(1));
    assert!(r.stdout.is_empty());
    assert_eq!(
        String::from_utf8(r.stderr).unwrap(),
        "Expected ADALINE1, MADALINE2, HEBBIAN1 or PERCEPTRON1 request.\n"
    );
}

#[test]
fn provided_rates_and_nonzero_initialization_are_passed_without_adjustment() {
    let source = template("learning/adaline-provided.txt")
        .replace("learning_rate = 0.2", "learning_rate = 0.12345678901234567")
        .replace("initial_weights = 0, 0", "initial_weights = 0.125, -0.25")
        .replace("initial_bias = 0", "initial_bias = 0.0625");
    let report = calculate("adaline", &source);
    let lines: Vec<_> = request(&report).lines().collect();
    let alpha: f64 = lines[1].split('\t').next().unwrap().parse().unwrap();
    assert_eq!(alpha.to_bits(), 0.12345678901234567_f64.to_bits());
    assert_eq!(lines[2], "1.25000000000000000e-1,-2.50000000000000000e-1");
    assert_eq!(lines[3], "6.25000000000000000e-2");
    assert_eq!(
        records(&report),
        String::from_utf8(backend(request(&report)).stdout).unwrap()
    );
    let first: Vec<_> = records(&report)
        .lines()
        .find(|l| l.starts_with("S\t"))
        .unwrap()
        .split('\t')
        .map(str::to_string)
        .collect();
    assert_eq!(first[3].parse::<f64>().unwrap(), -0.0625);
    assert_eq!(first[10], lines[2]);
}

#[test]
fn invalid_utf8_does_not_replace_output_and_success_replaces_a_complete_report() {
    let scratch = Scratch::new();
    let output = scratch.file("out.txt");
    fs::write(&output, "keep").unwrap();
    fs::write(scratch.file("bad.txt"), [0xff, 0xfe]).unwrap();
    let r = bin()
        .current_dir(&scratch.0)
        .args(["hebb", "--input", "bad.txt", "--output", "out.txt"])
        .output()
        .unwrap();
    assert_eq!(r.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&r.stderr).contains("Cannot read input"));
    assert_eq!(fs::read_to_string(&output).unwrap(), "keep");
    fs::write(scratch.file("bad.txt"), template("learning/hebb.txt")).unwrap();
    let r = bin()
        .current_dir(&scratch.0)
        .args(["hebb", "--output", "out.txt", "--input", "bad.txt"])
        .output()
        .unwrap();
    assert!(r.status.success());
    assert!(fs::read_to_string(&output)
        .unwrap()
        .ends_with("--- END BACKEND REQUEST ---\n"));
    assert_eq!(fs::read_dir(&scratch.0).unwrap().count(), 2);
}

#[cfg(unix)]
#[test]
fn permission_failures_preserve_existing_files() {
    use std::os::unix::fs::PermissionsExt;
    let scratch = Scratch::new();
    let input = scratch.file("input.txt");
    fs::write(&input, template("learning/hebb.txt")).unwrap();
    fs::set_permissions(&input, fs::Permissions::from_mode(0o000)).unwrap();
    // Privileged test users may bypass mode bits; only assert an actual denial.
    if fs::File::open(&input).is_err() {
        let r = bin()
            .args(["hebb", "--input"])
            .arg(&input)
            .arg("--output")
            .arg(scratch.file("out.txt"))
            .output()
            .unwrap();
        assert_eq!(r.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&r.stderr).contains("Cannot read"));
        assert!(!scratch.file("out.txt").exists());
    }
    fs::set_permissions(&input, fs::Permissions::from_mode(0o600)).unwrap();
    let directory = scratch.file("read only");
    fs::create_dir(&directory).unwrap();
    let output = directory.join("keep.txt");
    fs::write(&output, "keep").unwrap();
    fs::set_permissions(&directory, fs::Permissions::from_mode(0o555)).unwrap();
    let probe = directory.join("probe");
    if fs::File::create(&probe).is_err() {
        let r = bin()
            .args(["hebb", "--input"])
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .output()
            .unwrap();
        assert_eq!(r.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&r.stderr).contains("Cannot write"));
        assert_eq!(fs::read_to_string(&output).unwrap(), "keep");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
    }
    fs::set_permissions(&directory, fs::Permissions::from_mode(0o755)).unwrap();
}
