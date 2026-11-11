# Rust file CLI

For copy-and-paste commands for every algorithm and template, see
[CLI_COMMANDS.md](CLI_COMMANDS.md).

The file CLI runs the existing numerical implementations in Rust and writes a
complete text report. It requires neither Python nor Tk, has no third-party
dependencies, and works offline. All commands below run from the `math` directory.

## Build and run

```sh
cargo build --offline --manifest-path every_example_gui/rust_backend/Cargo.toml
cargo run --offline --manifest-path every_example_gui/rust_backend/Cargo.toml -- \
  nelder-mead \
  --input every_example_gui/cli_templates/optimization/nelder-mead-slides-function.txt \
  --output "Nelder Mead report.txt"
```

The existing direct compiler workflow also builds the same CLI:

```sh
rustc --edition=2021 -O every_example_gui/rust_backend/src/main.rs -o /tmp/exam-cli
/tmp/exam-cli --help
/tmp/exam-cli --list
/tmp/exam-cli perceptron --help
/tmp/exam-cli adaline \
  --input every_example_gui/cli_templates/learning/adaline-calculated.txt \
  --output "ADALINE report.txt"
```

Use `exam-cli.exe` or the Cargo `.exe` path on Windows. Both paths can be relative
to the current working directory or absolute. Quote paths containing spaces.
Both `--input` and `--output` are required, in either order; repeated flags,
unknown arguments, and unknown algorithms are errors. There is no interactive
menu: the algorithm argument makes the selection explicit.

**No arguments retains the original stdin/stdout GUI protocol**, including its
format, validation, and exit behavior. The GUI continues to launch that path,
with no CLI input-file reads or report writes. Existing Newton 1D, MADALINE,
and other Lecture 08 GUI operations remain available through that protocol;
they are outside this file CLI's requested algorithm set.

## Algorithms and distinct modes

| Command | Required variant settings | Rate / step selection |
| --- | --- | --- |
| `nelder-mead` | `variant = slides` or `standard`; `stopping = function` or `position` | None |
| `gradient-1d` | Fixed, unnormalized 1D gradient | Provided; calculated for certified positive quadratics |
| `steepest-fixed` | Fixed, unnormalized 2D negative gradient | Provided; calculated for certified positive-definite quadratics |
| `steepest-line-search` | Existing `normalize = false` path | New step calculated by line search each iteration; `initial_step` is a probe |
| `normalized-gradient` | Existing `normalize = true` path | New step calculated by line search each iteration; `initial_step` is a probe |
| `powell` | Existing three-search direction-shift cycles | New signed step calculated for each line search; `initial_step` is a probe |
| `newton-2d` | Existing full Hessian and gradient-norm stopping | None |
| `perceptron` | `mode = binary`, `bipolar`, `bipolar_error`, or `band` | Provided only, in every mode |
| `adaline` | Existing sequential, linear-response LMS | Provided; calculated with the per-row augmented-norm rule |
| `hebb` | Existing zero-initialized, one-pass supervised Hebbian rule | Provided only |
| `mcculloch-pitts` | Existing `lecture08` binary feed-forward network | None |

Nelder–Mead `slides` always contracts toward the old worst vertex when reflection
is not better than the second-best vertex. `standard` also permits the existing
outside contraction. `function` stops at objective spread **≤ tolerance**;
`position` stops when maximum distance from the best vertex is **< tolerance**.
Both selections are mandatory in the file format; the protocol's existing
defaults are unchanged. Reflection/expansion/contraction/shrink coefficients
remain the backend's fixed `1, 2, 0.5, 0.5`; they are not CLI settings.

### Steepest descent with an optimized step

For the lecture rule `d_k = -gradient(f)(X_k)`, followed by minimizing
`phi_k(alpha) = f(X_k + alpha*d_k)`, select **`steepest-line-search`** and its
matching template. It calculates a new accepted `alpha_k` each iteration using
the existing derivative-based search. `initial_step = 1` only initializes the
search; it does not force the accepted step to be 1.

`steepest-fixed` with `learning_rate_mode = calculated` is a separate fixed-rate
variant: the new CLI parameter selector computes `1/lambda_max(H)` once before
running the unchanged fixed-rate backend. This does not implement the lecture's
per-iteration line minimization, and the two rate rules are not interchangeable.
For the supplied CV5 objective from `(0,0)`, optimized steepest descent's first
step is `5/18`, whereas the calculated fixed rate is `1/(5+sqrt(2))`.

The preserved backend is not an exact transcription of every option in the
lecture images: it stops at gradient norm **<= tolerance** or its update budget,
and uses directional-derivative bracketing/bisection. It does not offer the
images' strict gradient inequality, displacement/function-change stopping
options, explicit second-derivative minimum test, or Golden Section fallback.
The CLI does not add or substitute these numerical behaviors.

### Gradient method with a normalized vector

Select **`normalized-gradient`** with
`cli_templates/optimization/normalized-gradient.txt` for the normalized-vector
lecture slide. The existing backend follows:

```text
g_k = gradient(f)(P_k)
S_k = -g_k / ||g_k||
phi(gamma) = f(P_k + gamma*S_k)
Find gamma_k from phi'(gamma_k) = 0 using the existing numerical line search.
P_(k+1) = P_k + gamma_k*S_k
```

It recomputes the unit direction and accepted `gamma_k` each iteration.
`initial_step` is the starting gamma probe for the search, not a fixed step
or a rate selected with `1/lambda_max(H)`. The gradient stopping test runs before
normalization, so an already stationary point does not divide by a zero norm.
The existing scaled normalization preserves the unit direction for small or
large gradients; no new normalization formula or numerical guard is introduced.

The distinction from `steepest-line-search` is the direction: the latter uses
the unnormalized `d_k=-g_k`. Their scalar steps therefore have different
meanings. Reports explicitly show the direction, line function, stationary-step
condition and update formula, plus every recorded direction/trial and accepted
index. For the normalized template's first search, `g=(1,4)`,
`S=(-1,-4)/sqrt(17)` and `gamma=17*sqrt(17)/26`, giving
`P_1=(113/26,31/13)` within the existing search tolerance.

### Perceptron modes

Perceptron variants retain their different updates:

| `mode` | Targets; activation | Mistake update multiplier |
| --- | --- | --- |
| `binary` | 0/1; 1 at `net >= 0` | `rate * (target - output)` |
| `bipolar` | −1/+1; +1 at `net >= 0` | `rate * target` |
| `bipolar_error` | −1/+1; +1 at `net >= 0` | `rate * (target - output)` |
| `band` | −1/+1; output 0 on the inclusive `[−theta, theta]` band | `rate * target` |

Every mode still needs a complete unchanged epoch to stop with `weights_stable`.
The CLI preserves row order and every explicitly supplied initial coefficient.

## Input syntax

Files are UTF-8 text; a leading UTF-8 BOM and CRLF line endings are accepted.
Names and values such as modes are case-sensitive. Put **all settings before
all data sections**:

```text
# Named settings, with blank lines or inline comments
mode = band
learning_rate_mode = provided
learning_rate = 1
initial_weights = 0, 0
initial_bias = 0
max_epochs = 100
theta = 0.2

[samples]
1, 1, 1     # x1, x2, target
0, 1, -1
1, 0, -1
0, 0, -1
```

`#` starts a comment through the end of the line. Numeric vectors and section
rows accept commas, semicolons, or whitespace (including repeated separators).
Use decimal points and scientific notation such as `1e-6`. Do not quote values,
write column-name headers, or include thousands separators. Every row of a
section must have the same number of values. Duplicate keys/sections, empty
required sections, nonfinite numbers, unknown fields, and fields inapplicable
to the selected algorithm are errors. Files and resolved protocol requests
are limited to 1 MiB each.

**No user-selectable numeric field has an implicit CLI default.** Supply every
field listed as required below. The few fixed/optional conventions are stated
explicitly; no GUI model defaults are silently substituted.

### Optimization fields

All optimization commands require:

| Field | Meaning and accepted values |
| --- | --- |
| `objective` | Infix expression, 1–1000 bytes, at most 200 backend instructions, nesting depth at most 30 |
| `tolerance` | Positive finite number; method's existing stopping test, described below |
| `max_iterations` | Integer 1–100000; committed-update budget, or full-cycle budget for Powell |
| `plot_bounds` | Two values `xmin, xmax` in 1D; four values `xmin, xmax, ymin, ymax` in 2D. Each finite, magnitude ≤ 1e50; each minimum strictly below its maximum. These generate the backend's recorded objective samples and **never constrain the optimizer** |

All except Nelder–Mead require `x0`; all 2D methods except Nelder–Mead also
require `y0`. Initial coordinates must be finite and magnitude ≤ 1e50.
Nelder–Mead instead requires `[simplex]`, exactly three rows of two finite
coordinates, magnitude ≤ 1e50, non-collinear at backend working precision;
the rows are the supplied A/B/C order before the existing stable sort. Its
`variant` and `stopping` settings are described above.

`gradient-1d` and `steepest-fixed` require `learning_rate_mode` and, in provided
mode, `learning_rate`. They stop at absolute derivative/Euclidean gradient
norm **≤ tolerance**. The rate is fixed throughout execution.

`steepest-line-search`, `normalized-gradient`, and `powell` additionally require:

| Field | Meaning and accepted values |
| --- | --- |
| `initial_step` | Positive finite initial line-search probe ≤ 1e50; not a learning rate or an accepted step |
| `line_tolerance` | Strictly between 0 and 1; relative directional-derivative residual for the gradient searches, existing scalar-position tolerance for Powell |
| `max_line_evaluations` | Integer 1–2048; existing trial budget per search |

Gradient line methods require `max_iterations * max_line_evaluations <= 1000000`;
Powell requires `3 * max_iterations * max_line_evaluations <= 1000000`. Search
traces include the initial point in addition to the subsequent trials.
Gradient line methods stop at gradient norm **≤ tolerance**. Powell tests
absolute function change, then displacement, both **< tolerance**, only after
a full three-search cycle. Its initial directions remain `(1,0)` and `(0,1)`;
its third direction remains the raw displacement. Newton 2D uses its existing
gradient norm **≤ tolerance** test and unmodified singularity/guard behavior.

Expressions support `x` (and `y` for 2D), `pi`, `e`, `+ - * / ^` or `**`,
parentheses, unary signs, and one-argument `sin`, `cos`, `tan`, `exp`, `ln`/`log`,
`sqrt`. Multiplication must be explicit (`2*x`). Power is right-associative and
binds more tightly than unary signs: `-x^2` means `-(x^2)`; `2^-2` is allowed.
Trigonometric functions use radians; `log` is natural logarithm. Constants
must be finite and magnitude ≤ 1e100. The parser translates to the existing
allowlisted backend instructions; it does not evaluate user code or replace
automatic derivatives. Existing domain rules, including the positive-base
requirement for variable exponents, are preserved.

### Learning fields

All learning commands require `[samples]`, with one input vector followed by
its target per row, and `learning_rate_mode`. Provided mode requires
`learning_rate`. Rates must be positive finite numbers ≤ 1e50; **Perceptron
additionally requires rate ≤ 1**. Use 1–32 inputs and 1–128 patterns. Inputs,
initial weights, and initial bias must be finite and magnitude ≤ 1e50.
Targets are exactly −1/+1 except binary perceptron's 0/1.

| Algorithm | Additional required fields | Fixed/optional conventions |
| --- | --- | --- |
| `adaline` | `initial_weights` (one per input), `initial_bias`, positive finite `tolerance`, integer `max_epochs` 1–100000 | Strict largest individual change **< tolerance**, only after a full epoch |
| `perceptron` | `initial_weights`, `initial_bias`, integer `max_epochs` 1–100000, `mode`; **band also requires `theta`** | `theta` is finite in [0, 1e50]. Other modes have fixed theta=0 and reject a `theta` setting. No tolerance field |
| `hebb` | None | All weights and bias start at zero; exactly one ordered pass. Rejects initialization, tolerance, theta, and epoch settings |

For iterative learners the existing limits are
`patterns * max_epochs <= 100000` and
`patterns * max_epochs * inputs <= 1000000`. No visits are dropped from the
trace. ADALINE SSE uses linear responses; Hebbian/perceptron SSE is
classification SSE. Fixed-model prediction records are separate from online
training responses. A numerical stop can leave a partial epoch/pass; its
predictions describe the last valid model, not completed verification.

### McCulloch–Pitts fields

Required settings/sections:

| Field | Meaning |
| --- | --- |
| `thresholds` | One numeric threshold per neuron, in execution order |
| `[patterns]` | Rows of binary inputs, exactly 0 or 1 |
| `[weights]` | One row per neuron, columns ordered as external inputs then all neurons |
| `[expected]` | Optional expected output rows, one value per neuron; omit the entire section when unused. Existing backend comparisons are literal, without recoding |

For `n` inputs and `m` neurons use an `m × (n+m)` weight matrix and `m`
thresholds. The existing matrix limits require `n+m <= 32`, `m <= 32`, and
1–128 patterns. All weights/thresholds/expected values are finite with magnitude
≤ 1e50. Weights from the same or later neurons must be zero. Each neuron
computes `net >= threshold`, making its binary output available only to later
neurons. This is exactly the existing `lecture08` implementation: there is no
training, bias setting, activation selection, tolerance, or learning rate.

## Learning-rate selection

Choose explicitly:

```text
learning_rate_mode = provided
learning_rate = 0.1
```

or, only on a supported method:

```text
learning_rate_mode = calculated
```

Provided mode passes the parsed `f64` value unchanged; it is never clamped,
rounded for computation, or overridden by the calculated rule. In calculated
mode, specifying `learning_rate` is a conflict. Missing/unknown modes, missing
provided rates, and learning-rate fields on methods without one are errors.

### Fixed gradient methods: reciprocal largest curvature

`gradient-1d` and `steepest-fixed` support calculated mode only when the parser
can certify a quadratic with a constant positive-definite Hessian `H`. Select
the **actual rate `alpha = 1 / lambda_max(H)`**, once before dispatch. For 1D
`a*x^2+b*x+c`, this is `1/(2*a)` with `a > 0`. For 2D
`a*x^2+b*x*y+c*y^2+d*x+e*y+f`, use
`H = [[2*a,b],[b,2*c]]` and its largest eigenvalue. The eigenvalues are computed
with scaling; a nonpositive minimum eigenvalue or unrepresentable rate is
rejected. Reports include all three independent Hessian entries, both
eigenvalues, the assumption, rule, and resolved rate.

This is grounded in the project's existing fixed negative-gradient recurrence
and quadratic practical presets (04/CV5). For an SPD quadratic,
`z_next = (I-alpha*H) z`; with this choice the eigenfactors lie in `[0,1)`.
For CV5's `2*x^2+x*y+3*y^2-4*x+2*y`, the eigenvalues are `5 ± sqrt(2)`, so
the selected rate is `1/(5+sqrt(2))`, approximately `0.1559037581576915`.

Certification accepts constants, variables, addition, subtraction, negation,
products with degree at most two, division by nonzero constants, and powers
0/1/2 when the result remains degree at most two. It conservatively rejects
transcendental operations, higher powers, variable denominators, affine/flat
objectives, and indefinite/nonconvex quadratics. Even an equivalent quadratic
written with unsupported operations must use provided mode or be rewritten.
No local Hessian sample is treated as a global curvature bound. There is no
automatic rate for general nonquadratic objectives, and no adaptive schedule.

### Sequential ADALINE: maximum augmented per-row curvature

For each ordered sample form `z_k = [x_k, 1]`, including the trained bias.
Compute `B = max_k(1 + sum_i x_ki^2)` and choose the **actual fixed rate
`alpha = 1/B`**. Reports include every augmented squared norm in row order,
`B`, the rule, assumptions, and resolved rate. The bipolar AND template gives
`B=3` and rate `1/3`. The housing rows `(2,1), (3,2), (4,3)` give
`B=max(6,14,26)=26` and rate `1/26`.

This rule follows the existing sequential update
`theta_next = theta + alpha*(t-z_k·theta)*z_k`, i.e. descent on each row's
`(t-z_k·theta)^2/2`. Each row Hessian is `z_k z_k^T`; with the selected fixed
rate, `0 < alpha*||z_k||^2 <= 1`, so each row's linear update is nonexpansive
and its residual factor is in `[0,1)`. This explicitly accounts for sample
order, per-row updates, and bias. It does **not** promise a single fixed-model
solution or tolerance convergence for inconsistent data (which can approach
a cycle). The existing largest-update stopping rule remains authoritative.
Zero external inputs still give `B=1` because bias is trained.

This is not the batch/mean-LMS spectrum bound in the Lecture 08 correlation
worksheet. The practical rate values already supplied in 07/CV7/CV8 are
available unchanged through provided mode; calculated mode is an explicitly
selected parameter rule, not a claim that those examples prescribed this rate.

### Unsupported calculations and search steps

Calculated mode is explicitly unsupported for **every perceptron mode and
Hebbian Learning**. The project supplies literal rates for these rules, with
no selected-rate calculation that this CLI can justify. They require provided
mode; no formula is guessed.

Normalized gradient and optimized steepest descent keep their directional
derivative bracketing/bisection; Powell keeps its derivative-free signed line
minimization. `initial_step` initializes each existing search. The step actually
accepted by a search appears in its referenced trial, and can differ each
iteration. These algorithms reject learning-rate fields, as do Newton,
Nelder–Mead and McCulloch–Pitts.

## Every runnable template

The 20 templates use existing project/practical presets, with explicit settings.
To run any template, first build and set these shell variables:

```sh
cargo build --offline --manifest-path every_example_gui/rust_backend/Cargo.toml
BIN=every_example_gui/rust_backend/target/debug/exam-adaline
T=every_example_gui/cli_templates

"$BIN" nelder-mead --input "$T/optimization/nelder-mead-slides-function.txt" --output nm-slides-function-report.txt
"$BIN" nelder-mead --input "$T/optimization/nelder-mead-slides-position.txt" --output nm-slides-position-report.txt
"$BIN" nelder-mead --input "$T/optimization/nelder-mead-standard-function.txt" --output nm-standard-function-report.txt
"$BIN" nelder-mead --input "$T/optimization/nelder-mead-standard-position.txt" --output nm-standard-position-report.txt
"$BIN" gradient-1d --input "$T/optimization/gradient-1d-provided.txt" --output gradient-provided-report.txt
"$BIN" gradient-1d --input "$T/optimization/gradient-1d-calculated.txt" --output gradient-calculated-report.txt
"$BIN" steepest-fixed --input "$T/optimization/steepest-fixed-provided.txt" --output steepest-provided-report.txt
"$BIN" steepest-fixed --input "$T/optimization/steepest-fixed-calculated.txt" --output steepest-calculated-report.txt
"$BIN" steepest-line-search --input "$T/optimization/steepest-line-search.txt" --output steepest-search-report.txt
"$BIN" normalized-gradient --input "$T/optimization/normalized-gradient.txt" --output normalized-report.txt
"$BIN" powell --input "$T/optimization/powell.txt" --output powell-report.txt
"$BIN" newton-2d --input "$T/optimization/newton-2d.txt" --output newton-report.txt
"$BIN" perceptron --input "$T/learning/perceptron-binary.txt" --output perceptron-binary-report.txt
"$BIN" perceptron --input "$T/learning/perceptron-bipolar.txt" --output perceptron-bipolar-report.txt
"$BIN" perceptron --input "$T/learning/perceptron-bipolar_error.txt" --output perceptron-error-report.txt
"$BIN" perceptron --input "$T/learning/perceptron-band.txt" --output perceptron-band-report.txt
"$BIN" adaline --input "$T/learning/adaline-provided.txt" --output adaline-provided-report.txt
"$BIN" adaline --input "$T/learning/adaline-calculated.txt" --output adaline-calculated-report.txt
"$BIN" hebb --input "$T/learning/hebb.txt" --output hebb-report.txt
"$BIN" mcculloch-pitts --input "$T/networks/mcculloch-pitts.txt" --output network-report.txt
```

## Reports, file writes, and exit statuses

Reports contain the algorithm/mode, original settings/data, selected rate rule
and intermediate values, unchanged backend status, a plain-language reason,
outcome category, completed epoch/update/cycle counts, and final results. Full
trace tables have descriptive titles and tab-separated column headers; their
data rows retain the exact backend records and full precision, including
unchanged visits, epoch summaries, predictions, sorted simplexes, trials,
search directions/accepted indices, Hessians, rejected trial information,
and all objective plot samples. Network reports include every weighted term,
threshold calculation and output. No calculations are repeated to construct
the report. A resolved backend request appendix permits exact reproduction
through the existing stdin protocol.

Numbers retain the existing scientific notation with 17 digits after the
decimal; vectors/matrix data use commas. Matrix dimensions accompany row-major
values; final network net/output matrices are also displayed one pattern per
row. `_` means an unavailable value or unattempted trial. Epochs, patterns and
cycles start at 1; optimization points and line searches/trials start at 0.
`accepted_index` refers to `trial_index` in that same search, including its
initial-point trial. Grid indices are row-major: 49×49 cell centers, except
Powell's existing endpoints-inclusive 49×49 grid; 1D records 401 samples.

| Exit | Meaning |
| --- | --- |
| `0` | A complete report was saved, **including** a calculation that stopped at a budget or numerical/search limit; also help/list success |
| `1` | Invalid invocation/input, initial-domain/validation failure, unreadable file/invalid UTF-8, or output-file failure. Error on stderr, no success message |

`Status` always preserves the backend's literal value. `Outcome` groups
`gradient_tolerance`, `function_tolerance`, `simplex_tolerance`, Powell tolerance
statuses, `tolerance_met`, and `weights_stable` as `converged` according to their
existing tests; this does not certify a global minimum or perfect accuracy.
`iteration_limit`, `epoch_limit`, and `line_search_limit` are `budget_exhausted`.
`numerical_limit`, `singular_hessian`, `stagnation`, and `line_search_precision`
are `numerical_failure`; failed bracketing/increased-objective search statuses
are `search_failure`. Hebbian `pass_complete` and network `normal_completion`
are ordinary `normal_completion` without a convergence claim. The network label
is CLI metadata: its existing backend has no iterative termination status.

The entire input is parsed/validated and the calculation/report completed
before opening a temporary output file. The report is written and synced to
a uniquely created file in the destination directory, then renamed over the
destination. Ordinary read/write/rename failures leave the existing report
intact and clean up the temporary file; they never expose a partial report
under the requested output name. The parent directory must already exist and
be writable. Input and output may not refer to the same file. Atomic replacement
uses the platform's filesystem rename semantics; it replaces a destination
symlink itself, rather than writing through it. Power-loss durability and
cleanup after forced process termination are not promised.

## Verification

```sh
cargo test --offline --manifest-path every_example_gui/rust_backend/Cargo.toml
python3 -m unittest discover -s every_example_gui/tests -v
```

Rust integration tests execute every template, compare every report record and
status with direct backend dispatch, verify the selected-rate formulas and
calculated/provided trace identity, and exercise malformed data, incompatible
settings, file failures and report preservation. Python CLI integration tests
build directly with `rustc`, compare all template traces against independently
constructed existing Python adapter requests, and compare parsed results with
the GUI's existing optimized Rust/cache path. Existing numerical and Python
integration tests remain part of these commands. Optional GUI tests still
require `EXAM_GUI_TESTS=1` and a desktop display.

See [CLI review notes](CLI_REVIEW_NOTES.md) for the missing audit document and
existing conventions deliberately left unchanged.
