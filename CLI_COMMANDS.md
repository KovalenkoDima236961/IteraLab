# CLI commands for every algorithm

Run these commands from the project root (`math`). On your current machine:

```sh
cd /Users/dimon228/Desktop/math
```

Each calculation command below can be copied independently. Cargo builds the
executable automatically if needed. Every command uses a supplied `.txt`
template and saves a complete `.txt` report in the current directory. Change
`--input` to your own input file and `--output` to your preferred report path.
Quote paths containing spaces. An existing report at the output path is replaced
after successful calculation and report generation.

This guide covers all **11 CLI algorithm names and 20 runnable templates**.
For input fields, calculation rules, and report details, see [CLI.md](CLI.md).

## Help and available algorithms

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- --help
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- --list
```

Algorithm-specific help uses the algorithm name followed by `--help`, for example:

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- normalized-gradient --help
```

## 1. Nelder–Mead — 2D

Choose the branch variant and stopping rule through the corresponding input
template. `function` uses objective spread; `position` uses simplex size.

### Slide branches, function stopping

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  nelder-mead \
  --input every_example_gui/cli_templates/optimization/nelder-mead-slides-function.txt \
  --output "nelder-mead-slides-function-report.txt"
```

### Slide branches, position stopping

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  nelder-mead \
  --input every_example_gui/cli_templates/optimization/nelder-mead-slides-position.txt \
  --output "nelder-mead-slides-position-report.txt"
```

### Standard branches, function stopping

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  nelder-mead \
  --input every_example_gui/cli_templates/optimization/nelder-mead-standard-function.txt \
  --output "nelder-mead-standard-function-report.txt"
```

### Standard branches, position stopping

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  nelder-mead \
  --input every_example_gui/cli_templates/optimization/nelder-mead-standard-position.txt \
  --output "nelder-mead-standard-position-report.txt"
```

## 2. Gradient Method — 1D, fixed rate

### Rate provided in the input file

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  gradient-1d \
  --input every_example_gui/cli_templates/optimization/gradient-1d-provided.txt \
  --output "gradient-1d-provided-report.txt"
```

### Fixed rate calculated before execution

Calculates `1/lambda_max(H)` once for a supported positive quadratic.

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  gradient-1d \
  --input every_example_gui/cli_templates/optimization/gradient-1d-calculated.txt \
  --output "gradient-1d-calculated-report.txt"
```

## 3. Steepest Gradient Descent — 2D, fixed rate

This variant keeps one rate for the entire run. For a new optimized step each
iteration, use algorithm 4 below.

### Rate provided in the input file

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  steepest-fixed \
  --input every_example_gui/cli_templates/optimization/steepest-fixed-provided.txt \
  --output "steepest-fixed-provided-report.txt"
```

### Fixed rate calculated before execution

Calculates `1/lambda_max(H)` once for a supported positive-definite quadratic.

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  steepest-fixed \
  --input every_example_gui/cli_templates/optimization/steepest-fixed-calculated.txt \
  --output "steepest-fixed-calculated-report.txt"
```

## 4. Steepest Descent — 2D, unnormalized vector and line search

Uses `d_k = -gradient(f)(X_k)` and calculates a new accepted `alpha_k` by line
search each iteration. This is the optimized-step method from your handwritten
steepest-descent notes. `initial_step` is only the starting search probe.

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  steepest-line-search \
  --input every_example_gui/cli_templates/optimization/steepest-line-search.txt \
  --output "steepest-line-search-report.txt"
```

## 5. Gradient Method — 2D, normalized vector and line search

Uses `S_k = -gradient(f)(P_k)/||gradient(f)(P_k)||` and calculates a new accepted
`gamma_k` by line search each iteration. This is **gradientná metóda s
normalizovaným vektorom** from the supplied slide. `initial_step` is only the
starting search probe.

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  normalized-gradient \
  --input every_example_gui/cli_templates/optimization/normalized-gradient.txt \
  --output "normalized-gradient-report.txt"
```

## 6. Powell's Method — 2D

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  powell \
  --input every_example_gui/cli_templates/optimization/powell.txt \
  --output "powell-report.txt"
```

## 7. Newton's Method — 2D

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  newton-2d \
  --input every_example_gui/cli_templates/optimization/newton-2d.txt \
  --output "newton-2d-report.txt"
```

## 8. Perceptron

The activation/update mode is selected in the input file. All four modes use
a provided learning rate.

### Binary — 0/1, error-based updates

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  perceptron \
  --input every_example_gui/cli_templates/learning/perceptron-binary.txt \
  --output "perceptron-binary-report.txt"
```

### Bipolar sign — −1/+1, target-based updates

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  perceptron \
  --input every_example_gui/cli_templates/learning/perceptron-bipolar.txt \
  --output "perceptron-bipolar-report.txt"
```

### Bipolar error — −1/+1, error-based updates

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  perceptron \
  --input every_example_gui/cli_templates/learning/perceptron-bipolar_error.txt \
  --output "perceptron-bipolar-error-report.txt"
```

### Threshold band — −1/0/+1 activation

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  perceptron \
  --input every_example_gui/cli_templates/learning/perceptron-band.txt \
  --output "perceptron-band-report.txt"
```

## 9. ADALINE

### Rate provided in the input file

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  adaline \
  --input every_example_gui/cli_templates/learning/adaline-provided.txt \
  --output "adaline-provided-report.txt"
```

### Fixed rate calculated before execution

Calculates `1/max_k(1 + ||x_k||^2)` once, including the trained bias coordinate.

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  adaline \
  --input every_example_gui/cli_templates/learning/adaline-calculated.txt \
  --output "adaline-calculated-report.txt"
```

## 10. Hebbian Learning / Hebb

Uses a provided rate, zero initialization, and one ordered training pass.

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  hebb \
  --input every_example_gui/cli_templates/learning/hebb.txt \
  --output "hebb-report.txt"
```

## 11. McCulloch–Pitts network

Uses the existing Lecture 08 network implementation with fixed weights and
thresholds. The template contains the three-neuron XOR circuit.

```sh
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  mcculloch-pitts \
  --input every_example_gui/cli_templates/networks/mcculloch-pitts.txt \
  --output "mcculloch-pitts-report.txt"
```

## Run the compiled executable directly

Build once:

```sh
cargo build --offline --manifest-path every_example_gui/rust_backend/Cargo.toml
```

Then replace the `cargo run ... --` prefix of any calculation command above
with `every_example_gui/rust_backend/target/debug/exam-adaline`, for example:

```sh
every_example_gui/rust_backend/target/debug/exam-adaline normalized-gradient \
  --input every_example_gui/cli_templates/optimization/normalized-gradient.txt \
  --output "normalized-gradient-report.txt"
```

Alternatively, build directly with `rustc` without Cargo:

```sh
rustc --edition=2021 -O every_example_gui/rust_backend/src/main.rs -o /tmp/exam-cli
/tmp/exam-cli normalized-gradient \
  --input every_example_gui/cli_templates/optimization/normalized-gradient.txt \
  --output "normalized-gradient-report.txt"
```

## Exit status

- `0`: a complete report was saved, including runs stopped by calculation or
  numerical limits; also used for successful help/list commands. Check the
  report's `Status`, `Outcome`, and `Reason` to see how the calculation ended.
- `1`: invalid command/input or a file-operation failure; the error is printed
  to stderr.

Invoke the executable with an algorithm argument for file-based calculations.
Launching it with no arguments selects the existing GUI stdin/stdout protocol
and waits for a protocol request.
