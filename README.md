# Exam lab · ADALINE, MADALINE and Hebbian Learning

A local desktop worksheet for the supplied **ADALINE, MADALINE Rule I (MRI), and Hebbian Learning** algorithms.
**Rust performs all training, stopping checks, SSE and predictions.** Python's
standard-library Tk interface presents the inputs and worked calculations.
No third-party Python packages, Rust crates, network service, or API key is needed.

## Run

Requires **Python 3.10+ with Tk** and **Rust (`rustc` on PATH)**. From `math`:

```sh
python3 every_example_gui/main.py
```

Or use `python3 -m every_example_gui.main`. An absolute path to `main.py` also works
from another directory. The first calculation compiles the Rust executable;
later calculations reuse a cache keyed by Rust source and compiler version in
`.cache/rust/`. No Cargo download is involved. On Linux, Tk may require your
distribution's `python3-tk` package. `python3 -m tkinter` checks the Tk installation.

## Use

1. Select **ADALINE**, **MADALINE (MRI)**, or **Hebbian Learning** in the algorithm selector, then choose
   **AND, OR, NAND, NOR, XOR, NOT**, or edit the training pairs directly.
   Each row contains `x1, x2, …, target`. Targets must be **−1 or +1**.
   Presets default to bipolar inputs; binary inputs are an explicit alternative.
   Custom finite real inputs also work with the same equations. A CSV import
   accepts an optional `x1,x2,target` header. Commas, semicolons and whitespace
   are accepted as separators; numbers use a decimal point.
2. Set one initial weight per input, bias, learning rate α, tolerance and maximum
   epochs. **Use zeros** or **Random · seed 42** fills explicit editable values.
   The seeded option gives reproducible small values in `[−0.1, 0.1]`.
   For MADALINE, choose **Hidden ADALINE units** (1–32; default 2). Each unit has
   its own input-weight vector and bias. Enter one fixed output weight per unit
   plus the output bias `b_out`. Changing the count preserves existing hidden
   inputs, adds fields for new units, and restores output defaults: every
   `v_j = 0.5`, `b_out = (H-1)/2`. **Use OR output defaults** restores those values
   at any time. Switching algorithms resets trainable initial
   values to zeros for ADALINE or seeded random values for MADALINE; the dataset
   stays in the editor. Loading a MADALINE preset also uses seeded random values.
   **Hebbian Learning** starts all weights and bias at zero on every run, as in
   the notes. These initial fields are read-only, and random initialization,
   tolerance, and maximum epochs are hidden. Enter its positive learning rate **η**;
   the procedure makes exactly one ordered pass, followed by verification.
3. Press **Run** for your selected algorithm or **Ctrl/⌘+Enter**. Compilation and training run in a
   worker; **Cancel calculation** terminates the process and retains the previous
   completed result. Editing inputs while a run is active affects the next run.
4. **Worked steps** substitutes the numbers into every formula, one pattern at a
   time. Use the buttons, slider or **Alt+Left/Right**. End-of-epoch patterns show
   the exact stopping test, or the final verification for Hebbian Learning.
5. **Full training trace** shows all inputs, old weights, net input, class, error,
   deltas and new weights. Select a row to update the worked calculation;
   double-click to open it. Use pagination or jump to an epoch. **Export all CSV**
   writes every recorded update, including rows on other pages.
6. **Epochs & learning** shows SSE and largest-update charts plus a paginated
   table of every complete epoch. Double-click an epoch to inspect its last
   pattern. **Export epochs CSV** saves its full history.
   Hebbian results rename this tab **Pass summary** and export its single
   completed pass with **Export pass CSV**.
7. **Final model** shows the separating equation and fixed-model predictions.
   **Algorithm** documents the supplied steps and the meaning of each diagnostic.

Display precision is selectable; CSV files preserve full floating-point precision.
Results and exports always use the input snapshot captured for that run. Editing
the form, importing data, an invalid input, or cancelling does not alter that result.

## ADALINE calculation

For **every** pattern, in the displayed order, with no shuffling:

```text
x_i = s_i
y_in = b + sum(x_i * w_i)
y = +1 if y_in >= 0, otherwise -1
e = t - y_in
b_new = b_old + alpha * e
w_i_new = w_i_old + alpha * e * x_i
```

Every update uses the same **pre-update linear `y_in`**. Even a correctly classified
pattern updates its weights. The new model is used by the next pattern.

After a complete epoch, stop only when the **largest absolute individual weight
or bias update across all patterns is strictly less than tolerance**. Net change
between epoch endpoints is displayed separately. There is no substitute accuracy,
SSE, or endpoint-change stopping rule.

Default settings: **α = 0.2, ε = 0.001, 100 epochs, zero initial weights and bias**.
Truth-table order is true/true, true/false, false/true, false/false; NOT visits
true then false. Outputs and targets always use bipolar values and threshold zero.

The outcome is explicit:

- `tolerance_met`: the literal strict stopping condition passed.
- `epoch_limit`: the calculation budget ended before the tolerance test passed.
  This can happen with perfect classification: constant-rate online ADALINE can
  cycle within an epoch even after epoch endpoints become stationary.
- `numerical_limit`: an attempted update had a non-finite intermediate or magnitude
  above `1e100`. That update is rejected; the last valid model, committed updates,
  and failed epoch/pattern remain available. Incomplete epochs are not included
  in the epoch table or charts.

SSE is calculated from the entire training set using one fixed model at each epoch
endpoint. XOR is intentionally available to explore a nonseparable truth table;
one affine separator cannot correctly classify all its patterns.

## MADALINE Rule I (MRI)

The supplied Rule I is implemented directly in Rust with **H configurable hidden
ADALINE units (1–32)**. The two-unit diagram is the default case:

```text
z_in_j = b_j + sum_i(x_i * w_ij)       j = 1 ... H
z_j = +1 if z_in_j >= 0, otherwise -1
y_in = b_out + sum_j(z_j * v_j)
y = +1 if y_in >= 0, otherwise -1
```

Step 7 has exactly the three cases in the image:

- **y = t:** no weights or biases change.
- **y ≠ t, t = +1:** select one hidden unit with the smallest `abs(z_in_j)`.
  Update only its bias and input weights, using `alpha*(1-z_in_j)` and
  `alpha*(1-z_in_j)*x_i`. The search includes every hidden unit; ties select
  the lowest-index tied unit deterministically.
- **y ≠ t, t = −1:** select every hidden unit whose `z_in_j > 0`. Each selected
  unit uses its own old linear response: `alpha*(-1-z_in_j)` for bias and
  `alpha*(-1-z_in_j)*x_i` for weights. All units' selections are made before
  any update is applied.

The output weights **v1 … vH and b_out remain fixed**, as in Step 0. The page does
not specify their numbers, so the editable defaults are **v_j = 0.5** and
**b_out = (H-1)/2**, implementing bipolar OR on all hidden activations. For two
units these are the previous `v1 = v2 = b3 = 0.5`. For three units, enter three
output weights; the default bias is `1`. Enter different fixed values if your
exercise gives them; the update cases stay as written.

“Positive net input” is interpreted literally as **strictly greater than zero**.
At exactly zero, activation is +1 but the unit is excluded from the negative
target update. Zero initialization can therefore stop with errors. The GUI
shows that outcome and defaults to seeded small random hidden coefficients.

Step 8 is tested after each full pass through the table. **`weights_stable`**
means no individual hidden weight or bias change occurred in that epoch.
**`tolerance_met`** means the largest absolute individual change across all hidden
units and patterns was strictly below the chosen acceptable level ε.
**`epoch_limit`** counts complete epochs (Step 2). Classification accuracy and
the net difference between epoch endpoints do not replace these tests.

The MADALINE diagnostics use **classification SSE = sum((t-y)^2)** on the whole
dataset with the fixed model at the epoch endpoint: each wrong bipolar output
contributes 4. Its chart and CSV identify this metric explicitly. A stable model
may still have errors; accuracy is reported separately. The traces include every
pattern visit, even correct patterns where the recorded deltas are zero.

**Worked XOR check:** choose 2 hidden units, XOR, α = 0.25, and enter Z1 weights `0.125, -0.25`,
b1 = `0.0625`, Z2 weights `-0.375, 0.125`, b2 = `0.1875`. Keep the output defaults.
The first epoch updates both hidden units on its fourth pattern. After the second
epoch the fixed model predicts all four patterns correctly; the third full epoch
has no changes and reports `weights_stable`. These values are checked against
an independent, hand-worked two-epoch fixture in the tests.

The original screenshot has two inputs; the GUI supports the same rule with
1–32 input coordinates and 1–32 hidden units. `w_ij` means input `i`
to hidden unit `j`, so Z1 has `w11, w21` and Z2 has `w12, w22` for two inputs.
Worked steps show the selected units, their own errors, every coefficient update,
and the Step 8 test. Final predictions include every hidden response and output.
Weight labels use underscores when either index can have multiple digits
(`w1_11` versus `w11_1`) so every CSV header stays unique. CSV exports include
the hidden count, every unit's state and updates, the fixed output constants
(`v1_fixed` … `vH_fixed`, `b_out_fixed`), and the immutable initial model.
Changing the form's hidden count leaves previously calculated results and their
exported dimensions intact until a new run completes.

## Hebbian Learning

This is the supervised Hebbian procedure in the supplied Slovak notes:

1. Choose **η > 0**. Initialize **every weight and bias to zero**.
2. Visit each training pair `(x, t)` once, in its displayed order.
3. Compute `delta_w_i = eta * x_i * t` and `delta_b = eta * t`.
4. Add these deltas to the existing weights and bias before the next row.
5. After that **one pass**, verify every pattern with one fixed final model.

There is no prediction-error term, no skipping correctly classified patterns,
and no tolerance/epoch loop. Every press of Run starts again from zero. Existing
ADALINE/MADALINE initialization or stopping settings do not affect Hebbian runs.
The shared learning-rate field retains its value when switching algorithms and
is labeled η for Hebbian Learning.

The page does not specify an activation function. Verification uses the GUI's
existing **bipolar threshold**: `y_in = b + sum(x_i*w_i)`, then `y = +1` when
`y_in >= 0`, otherwise `-1`. Targets are bipolar (`-1`, `+1`). Binary, bipolar,
and custom finite real inputs are accepted, with 1–32 coordinates and up to
128 patterns. Verification does not change the training updates.

**`pass_complete`** records that all pairs were processed and verification ran.
Classification accuracy is separate: a complete pass need not classify every
pattern correctly. For bipolar XOR, the final weights and bias are zero and
the zero tie predicts +1, giving 2/4 correct. The diagnostic classification SSE
is `sum((t-y)^2)`, so each wrong bipolar prediction contributes 4.

**Worked AND example:** choose bipolar AND and η = 1. In the usual row order,
the post-update `(w1, w2, b)` values are `(1,1,1)`, `(0,2,0)`, `(1,1,-1)`, and
`(2,2,-2)`. All four verification rows use the final `(2,2,-2)` model, producing
linear responses `2, -2, -2, -6` and correct bipolar classifications.

Worked steps substitute η, inputs and targets into every delta, then show every
coefficient addition. The last pattern shows Step 5 verification for the entire
dataset. The full trace CSV includes the input snapshot, zero initial model,
and all before/delta/after coefficients. The pass CSV records its endpoint,
accuracy and classification SSE. Both exports preserve full floating-point precision.

The existing numerical guard rejects a whole unsafe pattern update, preserves
the previously committed prefix, and reports `numerical_limit`. Such a run has
no completed pass-summary row. Its displayed predictions describe the last
valid partial model and are explicitly labeled as incomplete verification.

## Calculation limits

To keep complete traces practical, requests support 1–32 inputs, 1–128 patterns,
at most 100,000 pattern updates, and at most 1,000,000 input coordinates across
those updates, counting every hidden unit for MADALINE. Requests above those limits
are rejected before running. No trace
rows are silently dropped. Inputs and initial coefficients are bounded at `1e50`
in magnitude. Each compiler/training process has a two-minute timeout.

## Source and verification

- `rust_backend/src/lib.rs`: numerical implementation and hand-calculation tests.
- `rust_backend/src/madaline.rs`: MADALINE Rule I and its branch/boundary tests.
- `rust_backend/src/hebbian.rs`: zero-initialized, one-pass Hebbian training and verification.
- `rust_backend/src/main.rs`: validated line protocol for the Python adapter.
- `backend.py`: immutable run data, Rust build/process management, CSV exports.
- `app.py`: interface, substituted formulas, navigation and canvas charts.
- `madaline.py`, `madaline_view.py`: MADALINE run data, protocol, CSV and worked views.
- `hebbian.py`, `hebbian_view.py`: Hebbian run data, protocol, CSV and worked views.
- `main.py`: portable launcher. Other algorithms can be added as separate Rust
  modules and worksheet views without moving calculation loops into Python.

```sh
cargo test --offline --lib --manifest-path every_example_gui/rust_backend/Cargo.toml
python3 -m unittest discover -s every_example_gui/tests -v
EXAM_GUI_TESTS=1 python3 -m unittest every_example_gui.tests.test_gui -v
```

The final command requires a desktop display. Tests check hand-worked updates,
strict stopping and bias handling, custom initialization/dimensions, divergence,
input errors, exports, navigation, cancellation and immutable results. When the
previous exercise's results are present, integration tests compare every recorded
value in its first two epochs and its final models/diagnostics for all eight
binary/bipolar-input cases with bipolar targets. Existing exercises are untouched.
MADALINE tests also cover both update directions, ties, exact-zero nets, unchanged
correct patterns, strict acceptable-change checks, fixed output values, numerical
failure, and switching between both algorithms with previous results retained.
Variable-count checks exercise 1, 3, 4, 5, 12 and 32 hidden units, later-unit
selection, ties between later units, full-network stopping checks, output defaults,
CSV dimensions, unambiguous coefficient names, and GUI resizing.
Hebbian checks cover the hand-worked AND trace, target-based updates, one-pass
completion with imperfect accuracy, custom dimensions, order, fresh zero starts,
numerical limits, exports, and switching between all algorithm views.
