# Worksheet guide

Detailed desktop instructions, equations, worked examples, stopping rules,
numerical limits, exports, and test coverage for Exam Lab.

See the [project overview](README.md) for setup and architecture,
[CLI commands](CLI_COMMANDS.md) for runnable examples, and
[CLI reference](CLI.md) for file syntax and parameter selection. Commands
in this guide run from the repository root (`math`).

---

## Use

Select **Learning** for the neuron and Lecture 08 tools below, or **Optimization**
for the gradient worksheet described in the next section.

Use **Load practical exercise** to load the supplied learning tables with their original row order, initialization, rates and epoch budgets. Optimization and matrix/network tools have corresponding source-labelled examples. Restart the running GUI after updating the source.

1. Select **ADALINE**, **MADALINE (MRI)**, **Hebbian Learning**, or **Perceptron** in the algorithm selector, then choose
   **AND, OR, NAND, NOR, XOR, NOT**, or edit the training pairs directly.
   Each row contains `x1, x2, …, target`. Binary Perceptron targets are **0 or 1**;
   other modes use **−1 or +1**.
   Presets default to bipolar inputs, or binary inputs for binary Perceptron;
   the input encoding can also be chosen independently.
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
   **Perceptron** offers **Binary (0/1)**, **Bipolar sign (−1/+1)**,
   **Bipolar error update (−1/+1)**, and **Threshold band (−1/0/+1)**. Selection starts from zero with rate 1. Its
   tolerance control is hidden; set a maximum epoch budget and, in band mode,
   a nonnegative **θ**. Its stopping rule requires a full pass without changes.
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

## Practical exercises and optimized steepest descent

The source-labelled presets reproduce inputs from `../practical_examples`.
The numerical and practical integration tests below check these fixtures. The
previously referenced `PRACTICAL_EXAMPLES_AUDIT.md` is absent from this checkout;
see [CLI review notes](CLI_REVIEW_NOTES.md) for that inspection limitation and
retained mathematical conventions.

Choose **Steepest Descent · Line search · 2D** for CV5's optimized step:
`d=−∇f(P)`, `φ(α)=f(P+αd)`, solve `φ′(α)=0`, then update `P+=αd`.
This is separate from fixed-rate descent and normalized-gradient descent.
The initial scalar control is a search probe; the accepted scalar is found anew.
The complete search is recorded with the same limits and failure handling as the
normalized method. The relative residual is `|φ′(α)|/|φ′(0)|`; the outer stop is
`||∇f||₂ ≤ ε`. CSV identifies this method as `LINE_STEEPEST2D`.

## Optimization · Gradient Method in one dimension

1. Select **Optimization** in the persistent workspace selector, then
   **Gradient Method · 1D** in its method selector. Enter `f(x)`,
   starting estimate `x₀`, fixed positive step length `α`, gradient tolerance `ε`
   and maximum iterations, or load one of the four examples.
2. Press **Find minimum · Gradient 1D** or **Ctrl/⌘+Enter**. Rust evaluates the
   objective and its analytic derivative with forward automatic differentiation.
   Every update uses exactly the supplied slide's recurrence:
   **`x[k+1] = x[k] − α f′(x[k])`**. The rate stays fixed throughout the run.
3. Inspect **Result & plots**, navigate **Worked steps** with its buttons, slider
   or **Alt+Left/Right**, and select any row in the paginated **All iterations**
   table. **Export all CSV** saves every estimate at full precision, together
   with the original function and run parameters. The initial estimate is `k=0`;
   the final row has no further committed update.
4. Switch back to **Learning** whenever needed. Each workspace has its own
   worker, cancellation and results; changing workspace does not reset a form
   or stop its active calculation. There is one Tk root and one event loop.

The slide gives the recurrence without a stopping rule. This worksheet explicitly
stops when **`|f′(x[k])| ≤ ε`**, including at the initial point, or at the iteration
budget. A small step alone is not convergence. A stationary point can also be a
maximum or flat point on a nonconvex function, so gradient tolerance does not
prove a local or global minimum. Large rates can oscillate or increase `f`;
the worksheet displays those values without adjusting the slide's algorithm.
For `f(x)=x²`, `x₀=4`, `α=0.1`, the first estimates are `4, 3.2, 2.56, 2.048`.

Function syntax: `x`, `pi`, `e`, `+ - * / ^` (or `**`), parentheses and
`sin`, `cos`, `tan`, `exp`, `ln`/`log`, `sqrt` with one argument. Multiplication
must be explicit (`2*x`); trigonometric arguments use radians and `log` is
natural logarithm. Variable exponents require a positive base. Expressions are
parsed with an allowlist and never executed as Python code. No new dependencies
are needed. Functions, derivatives and intermediate values must be real, finite
and at most `1e100` in magnitude. A failed update is rejected with
`numerical_limit`, preserving the last valid point. `stagnation` reports a new
estimate that rounds to the current one while the gradient test fails. An invalid
initial function/derivative reports an error and retains the previous result.

Plot bounds affect only the sampled function view; they never constrain the
iteration. Domain/numerical failures leave plot gaps. Histories may be downsampled
for drawing, while tables and exports retain up to 100,001 estimates. Display
precision changes formatting only. **Algorithm** documents all conventions.

## Optimization · Newton's Method in one dimension

Choose **Newton's Method · 1D** in the optimization method selector. Enter `f(x)`,
`x₀`, positive tolerance `ε`, and `Nmax`. Supply `f′(x)` and `f″(x)` expressions
or leave either field blank to calculate that derivative of `f` automatically
in Rust. Supplied derivative expressions are used literally and must match the
objective. The gradient method's learning rate is hidden and never used by Newton.

The worksheet follows the supplied image:

```text
For i = 0, 1, 2, …, Nmax (inclusive):
    Evaluate f′(x_i) and f″(x_i).
    x_next = x_i − f′(x_i) / f″(x_i)
    If |x_next − x_i| < ε: stop.
```

The tolerance test is strictly **`<`** and runs **after** each update; equality
continues. It uses the actual difference of successive estimates, without a
gradient stopping test or damping. The form labels `Nmax` as the **last loop
index**, matching the image's inclusive `i=0…Nmax`: `Nmax=0` permits one update,
`Nmax=1` permits two, and `Nmax=99,999` permits the trace limit of 100,000 updates.
The initial estimate is row `i=0`; each committed update appends the next estimate.

For `f(x)=(x−2)²+1`, `x₀=6`, Newton computes `x₁=6−8/2=2`. Its step is 4, so
it continues for `ε=1e−6`; `x₂=2−0/2=2` then passes the zero-step test.
An already stationary start with nonzero curvature still performs its zero update
before stopping. A zero `f″` makes the quotient undefined and reports
**Zero second derivative**, even when `f′` is also zero. Small nonzero curvature
is used literally; large or invalid steps can reach the numerical/domain guard.
Negative curvature is also used literally. A passed step test alone does not
prove a minimum, a stationary point, or global optimality; inspect both derivatives.

**Worked steps** substitutes the first and second derivatives at the old point,
the quotient, new estimate and exact strict test. **All iterations** and
**Export all CSV** include both derivatives, the quotient step, actual absolute
step and whether its test passed. Exports retain the immutable function,
derivative sources, tolerance, inclusive Nmax and full precision. **Result & plots**
shares the function and objective-history views with Gradient; plot bounds remain
view settings. Automatic second derivatives use the chain rule, without finite
differences or new dependencies.

Each optimization method retains its own form, completed result and navigation.
You may switch methods or return to Learning while a run finishes; it is saved
under the method that started it. One optimization calculation runs at a time,
independently of the learning workspace. Cancellation or input/runtime errors
retain that method's previous completed result.

## Optimization · Steepest Gradient Descent in two dimensions

Choose **Steepest Gradient Descent · 2D** in the same optimization method selector.
Enter a differentiable `f(x,y)`, starting coordinates `x₀` and `y₀`, fixed positive
step length `α`, gradient tolerance `ε`, and maximum iterations, or load one of
the four examples. The existing allowlisted expression syntax now accepts `y`
for this method; use explicit multiplication such as `x*y`. Both analytic partial
derivatives are calculated in Rust with forward automatic differentiation.
One-dimensional methods retain their `x`-only input conventions.

Every update follows the supplied slide:

```text
gradient = (∂f/∂x(x_k, y_k), ∂f/∂y(x_k, y_k))
x_next = x_k − α * gradient.x
y_next = y_k − α * gradient.y
Commit both coordinates together.
```

Both partials use the **same old pair**. The gradient is not normalized, and the
algorithm never performs a line search or changes `α`. For `f=x²+xy+y²`,
`(x₀,y₀)=(2,1)` and `α=0.1`, the gradient is `(5,4)` and the new pair is
`(1.5,0.6)`. The next gradient is `(3.6,2.7)`, giving `(1.14,0.33)`.

The slide specifies no stopping test. This worksheet explicitly stops when
**`||∇f(x_k,y_k)||₂ ≤ ε`**, including at the initial point, or after the selected
maximum number of updates. Small coordinate steps alone are not convergence.
Passing the gradient norm test does not prove a minimum; a stationary point can
also be a maximum or saddle. Small rates can converge slowly; large rates can
overshoot, oscillate or increase the objective, and those values remain visible.

**Result & plots** shows a sampled function heatmap with the optimization path
and an objective-history chart. The heatmap's horizontal axis is `x`, vertical
axis is `y`, pale cells are lower sampled values, darker cells higher ones, and
gray cells are invalid function samples. Selecting a trace row or navigating
**Worked steps** highlights its pair. Plot ranges are view settings and never
constrain the iterations. Long histories are downsampled only for drawing.

**Worked steps** substitutes both old-point partial derivatives, the negative
gradient, both coordinate steps and the next pair. The paginated **All iterations**
table and **Export all CSV** retain every pair, objective, both partials, gradient
norm, both steps and next pair at full precision, with the immutable run inputs.
The last row has no outgoing update. There are at most 100,000 updates and
100,001 recorded estimates.

If either coordinate, partial derivative, function evaluation or coordinate step
is nonfinite, outside the required domain, or above the numerical guard of
`1e100` in magnitude, the **entire attempted pair is rejected**. The last valid
pair remains. Floating-point stagnation means both new coordinates round to
their old values while the gradient norm test fails. Invalid inputs, a bad
starting domain, and cancellation retain the previous completed result.
Switch between all optimization methods and Learning at any time; forms,
completed results and navigation remain attached to their original method in
the same application window.

## Optimization · Newton's Method in two dimensions

Choose **Newton's Method · 2D** in the optimization method selector. Enter a
twice-differentiable `f(x,y)`, starting coordinates, gradient tolerance and maximum
iterations, or load an example. Rust calculates the gradient and full symmetric
Hessian, including the mixed derivative, using second-order automatic
differentiation. The learning-rate and scalar-derivative fields are hidden and
unused in this method.

The supplied slide's update is implemented as the complete matrix operation:

```text
g = [f_x(x_k,y_k), f_y(x_k,y_k)]
H = [[f_xx, f_xy], [f_yx, f_yy]] at (x_k,y_k)
Solve H s = g                       # s = H^(-1) g
(x_next, y_next) = (x_k, y_k) − s
Commit both coordinates together.
```

All entries and both gradient components come from the **same old pair**.
Mixed terms participate in both coordinate updates. There is no learning rate,
damping, line search, diagonal approximation, or substitution of another Hessian.
For the coupled example `f=(x−2)²+(x−2)(y+1)+2(y+1)²` at `(6,3)`,
`g=(12,20)` and `H=[[2,1],[1,4]]`. The system gives `s=(4,4)`, reaching the
known minimum `(2,−1)` in one update.

The linear system is algebraically the slide's inverse-Hessian update. Rust
scales both sides by `scale=max(abs(H_ij))` and solves the 2×2 system using
`det(H/scale)`. This avoids underflow in the raw determinant for very small
Hessians. A zero scale or zero scaled determinant reports **Singular Hessian**
at working precision and commits no step. Small nonzero determinants are used
literally; no pseudoinverse or regularization is substituted. Indefinite and
negative definite invertible Hessians are also used as written.

The 2D Newton slide specifies no stopping rule, so this worksheet uses the same
explicit **`||∇f||₂ ≤ ε`** criterion and update budget as the 2D gradient method.
This includes the initial pair; a stationary start performs zero updates without
attempting an inverse. It is independent of the 1D Newton slide's strict
post-update step test. Maximum iterations counts updates (1–100,000), and a small
step alone is not convergence. Passing the gradient test can identify a saddle
or maximum; curvature classification is displayed as a diagnostic and never
changes the recurrence. Newton can increase the objective or cycle.

**Worked steps** displays the complete Hessian, mixed partials, inverse formula,
numeric system and simultaneous coordinate updates. **All iterations** and
**Export all CSV** include every pair, gradient, all four Hessian entries,
Hessian scale, scaled determinant, actual Newton steps and next pair. Exports
use the immutable inputs and full precision; the last row has no outgoing step.
The existing function heatmap, optimization path and objective-history plots
also work with Newton. Plot ranges affect only the view.

Function, gradient and Hessian entries and intermediate expression results must
be real, finite and at most `1e100` in magnitude. Both new coordinates and
attempted steps have the same guard. A failed attempt rejects the entire pair,
retaining the last valid one. Invalid inputs, bad starting domains and
cancellation retain the previous completed result. All eight optimization methods
keep their own forms, results and navigation in the same application window.

## Optimization · Nelder–Mead in two dimensions

Choose **Nelder–Mead · 2D** in the same optimization method selector. Enter
`f(x,y)` and all three initial vertices **A, B, C**, or load an example. Rust
evaluates only function values; no gradients or Hessians are calculated.
The initial simplex must form a non-collinear triangle at working precision.
Initial vertex labels identify input rows; **A/B/C become sorted roles** at each
iteration, with `f(A) ≤ f(B) ≤ f(C)`. Ties preserve their existing order.

The default **slides** branch option implements the supplied image with fixed coefficients
**reflection α=1, expansion γ=2, contraction ρ=0.5, shrink σ=0.5**:

```text
Sort A, B, C by objective value.
M = (A+B)/2                         # Exclude worst vertex C.
R = M + (M−C)
If f(A) ≤ f(R) < f(B): replace C with R.
If f(R) < f(A):
    E = M + 2*(R−M)
    Replace C with E if f(E) < f(R), otherwise with R.
If f(R) ≥ f(B):
    K = M + 0.5*(OLD C−M)           # Inside contraction, as in the image.
    If f(K) < f(OLD C): replace C with K.
    Otherwise shrink both OLD B and OLD C toward OLD A:
        B_new = A + 0.5*(B−A)
        C_new = A + 0.5*(C−A)
Re-sort before repeating.
```

In **slides** mode, contraction is always **inside**, including when `f(B) ≤ f(R) < f(C)`. Select **standard** for the CV4Utorok rule: in that interval contract outside with `K=(M+R)/2`, accepting it when `f(K) ≤ f(R)`; otherwise shrink. Both variants retain the inside branch when `f(R) ≥ f(C)`.
Equality with `f(B)` triggers contraction; equal expansion/reflection values
select reflection; equal contraction/old-worst values trigger shrink. Shrink
has no additional improvement test. The gradient method's editable learning
rate is hidden and unused; Nelder–Mead's α is its fixed reflection coefficient.

The image says to continue until changes in `f` are small without specifying a
precise numerical criterion. The worksheet explicitly uses the current
simplex's **`f(C)−f(A) ≤ ε`** test, including at the initial simplex, plus a
maximum iteration budget. A best value that stays unchanged does not itself
stop the method. The **simplex diameter** is reported as a diagnostic. Select **position** stopping to reproduce CV4Utorok: `max_j distance(vertex_j,best) < ε` (strict), independently of function spread. Flat
objectives and equal-valued vertices can pass this function test while the
simplex remains large, even away from a minimum; passing it does not establish
spatial accuracy or optimality. An update that leaves the whole sorted simplex
unchanged at floating-point precision reports stagnation separately.

**Worked steps** substitutes the sorted vertices, centroid, reflection and any
expansion/contraction/shrink points, shows the exact comparisons and selected
branch, and displays the re-sorted next simplex. The heatmap overlays the
selected simplex with solid edges, its next simplex with dashed edges and
visible M/R/E/K trial markers. The right chart shows the **best objective**
for every simplex, including iterations where that best value stays the same.
Plot bounds affect only the view.

**All iterations** and **Export all CSV** retain every sorted simplex, function
spread, diameter, branch, centroid, all evaluated trial points (including rejected
expansion or contraction points and both shrink points), and the next simplex.
Exports preserve original unsorted input vertices, fixed coefficients and run
parameters at full precision. The final row has no outgoing update.

Function values, intermediate expression values, trial coordinates and committed
coordinates must be real, finite and at most `1e100` in magnitude. Initial
coordinates and plot bounds are limited to `1e50`. If any **required** trial
fails, the entire attempted iteration is rejected: an invalid expansion does
not silently accept reflection, and a failed second shrink point does not commit
the first. The failed stage and coordinates remain available in the result and
CSV. An invalid initial simplex/function, input error or cancellation retains
the previous completed run. Each of the eight optimization methods retains its
own form, result and navigation in the single application window.

## Optimization · Gradient method with normalized vector

Choose **Normalized Gradient · 2D** for the supplied **GRADIENTNÁ METÓDA S
NORMALIZOVANÝM VEKTOROM**. This is a separate method with its own inputs, result
and worked steps. It follows the slide:

```text
g_k = ∇f(P_k)
S_k = −g_k / ||g_k||₂
φ(γ) = f(P_k + γ*S_k)
Find positive γ_k with φ′(γ_k) = 0.
P_next = P_k + γ_k*S_k
```

Rust calculates the gradient analytically with first-order automatic
differentiation, normalizes its direction, and determines a new `γ_k` at every
iteration. Both coordinate updates use the same old pair and direction. The
fixed learning-rate field is hidden and unused. Because `S_k` is a unit vector,
`γ_k` is the step's Euclidean length, up to floating-point rounding.

The slide gives a stationary equation without specifying a general symbolic
solver. The worksheet solves it **numerically** using
`φ′(γ)=∇f(P_k+γ*S_k)·S_k`: start at the entered positive **initial γ probe**,
double it until a negative-to-positive derivative bracket is found, then bisect.
Invalid function/derivative trials bound the searched domain; the search can
refine toward valid interior points. Each trial is recorded, including invalid
ones. A trial is accepted only when **`|φ′(γ)| / ||g_k||₂ ≤ line tolerance`**
(or the derivative is exactly zero), it gives a different representable point,
and its objective does not exceed the old one. A failed search never substitutes
a fixed or merely decreasing step.

This numerical search assumes a continuous line derivative. Convex or unimodal
line functions have the intended stationary minimum. Sampling nonconvex lines
can skip roots, and a stationary root need not be the first or global minimum;
an objective-increasing stationary trial is rejected explicitly. The display
and exports label the numerical tolerance, rather than claiming a general
closed-form exact solution.

For `f=x²+y²`, `P₀=(3,4)`, the gradient is `(6,8)`, its norm is `10`, and
`S₀=(−0.6,−0.8)`. Thus `φ(γ)=25−10γ+γ²`, whose derivative vanishes at **γ=5**.
The update reaches `(0,0)` in one step. From probe 1, the recorded search tries
`1,2,4,8`, then bisection tries `6,5`.

The image specifies no outer stopping test. The worksheet uses the existing 2D
convention **`||∇f(P_k)||₂ ≤ ε`** or the outer update budget. It checks the
gradient **before normalization**, so a zero-gradient start performs zero
updates without division. This outer test is independent of the relative line
tolerance. A stationary point can still be a maximum or saddle.

Controls include the initial γ probe, relative line tolerance (strictly between
0 and 1), and maximum trial evaluations per search (1–2048). The `γ=0` row reuses
the old gradient and does not consume a trial evaluation. The complete trace
budget requires `outer iteration limit × line trial limit ≤ 1,000,000`.
Separate outcomes identify exhausted line budgets, a missing forward stationary
bracket, floating-point resolution limits and an objective-increasing stationary
trial. These outcomes retain the last outer point and the failed search trace.
They do not report convergence. Initial data errors or cancellation retain the
previous completed result.

**Worked steps** substitutes the gradient, its norm, normalized vector, line
function, directional derivative, accepted γ and both new coordinates. The
method's **Line search** tab shows every trial for the selected outer estimate,
with pagination. Select a row to highlight its point on the heatmap; the plot
also shows a unit-direction arrow and accepted γ point. **Export all CSV** saves
every outer point and accepted step. **Export line searches CSV** saves all line
searches, including samples on other pages/outer iterations and invalid trials.
Both exports use immutable run inputs and full precision. Plot bounds affect
the view only. All eight optimization methods remain in the same application.

## Optimization · Powell's Method · 2D

Select **Powell's Method · 2D**. The first preset loads the supplied exercise:
`f(x,y)=x^2+2*y^2+2*x*y-4*x-6*y`, starting at `(0,0)`.
The implementation follows the supplied direction-shift variant. Initialize
`d₁=(1,0)`, `d₂=(0,1)`, and in each cycle save its starting point `Q₀`:

1. Minimize `f(Q₀+αd₁)` over signed α, giving `Q₁`.
2. Minimize `f(Q₁+αd₂)`, giving `Q₂`.
3. Form the raw displacement `d₃=Q₂−Q₀`, without normalization.
4. Minimize `f(Q₂+αd₃)`, giving `Q₃`.
5. Shift directions to `(old d₂, d₃)` and start the next cycle at `Q₃`.

This uses function values only. It does not use the largest-decrease direction
replacement or an extrapolation test found in other Powell variants. The notes
permit any 1D minimizer; this implementation brackets a minimum, then uses
**Brent minimization**, with safeguarded quadratic interpolation and
golden-section fallback. Both signs of the initial α probe are eligible; the
probe is not a fixed step length. Domain-invalid probes are retained and refined
toward valid samples. A failed search commits no trial; successful earlier moves
of an unfinished cycle remain in the trace. An exactly zero direction records
`α=0` without trial evaluations.

After each **complete cycle**, stop when either strict test passes:
`|f(Q₀)−f(Q₃)| < ε` or `||Q₃−Q₀||₂ < ε`, or when the maximum number of complete
cycles has been performed. Both diagnostics are displayed; function change takes
precedence if both tests pass. Small changes do not prove a minimum, particularly
on flat regions or with dependent directions. The line solver assumes a local
continuous, unimodal function; arbitrary nonconvex functions need not give a
global line minimum. Flat or rounded function values can hide further progress.

The exercise's reference trace is built into the Algorithm tab:

| Point | x | y | f(x,y) |
| --- | --- | --- | --- |
| X₀ | 0 | 0 | 0 |
| X₁ | 2 | 0 | −4 |
| X₂ | 2 | 1/2 | −9/2 |
| X₃ | 22/13 | 11/26 | −121/26 |
| X₄ | 22/13 | 17/26 | −1609/338 |
| X₅ | 250/169 | 203/338 | −21241/4394 |
| X₆ | 1 | 1 | −5 |

The minimum is reached after **two cycles**. Set the cycle limit to `2` to show
exactly this trace. With the default larger budget, a third cycle confirms the
small-change stopping criterion. Calculated values match these fractions within
floating-point precision.

The slider/table navigates individual line moves. **Worked steps** substitutes
the line function, direction, signed α, both new coordinates, direction shift
and cycle stopping tests. The plots show every move, with colored arrows for the
selected cycle. **Powell line search** shows every trial of the selected move,
with pagination and probe highlighting. **Export all CSV** saves all moves and
cycle diagnostics; **Export line searches CSV** saves all samples across all
searches, including a failed last search. Both use immutable completed inputs
and full precision.

Line tolerance controls α resolution, `τ(1+|α|)`, with a floor of eight machine
epsilons at that scale. Set a per-search trial budget (1–2048); the complete trace
requires `3 × cycle limit × trial limit ≤ 1,000,000`. Initial coordinates, probe
and plot bounds are limited to magnitude `1e50`; visited coordinates, expression
intermediates and direction components to `1e100`. Plot bounds affect only the
view. Each method keeps its state in the same application window.

## Lecture 08 · matrix and associative-memory tools

Choose one of the **Lecture 08 · …** entries in the existing algorithm
selector. They use a separate matrix workspace with the same styling. Returning
to the previously selected neuron worksheet preserves its inputs and completed
result. The original scalar **Hebbian Learning** remains its own algorithm with
its original learning rate, bias, targets and activation.

1. Enter one numeric pattern per row, separated by commas, semicolons or spaces,
   or paste a JSON array of rows. Each editor also imports numeric CSV/TSV/text
   or JSON files (no column headers). **Load lecture example** fills the selected
   tool's example; **Clear inputs** clears its editors and displayed matrix result.
2. Press **Calculate**, or **Train memory** for associative Hebbian learning.
   **Ctrl/⌘+Enter** also runs the selected tool. Each Lecture 08 tool retains its
   own input form when switching between them.
3. Inspect **Results**, navigate **Worked steps**, or select any intermediate
   matrix in **All matrices**. Large tables have pages; wide matrices scroll
   horizontally. **Lecture notes** explains formulas, dimensions and conventions.
4. **Export matrices CSV** saves every intermediate/final matrix and numerical
   scalar, including matrices on other pages. **Export full report JSON** also
   includes the captured inputs, notes and all worked steps. Display precision
   does not round calculations or exported values.
5. After training or constructing outer products, **Use trained W for recall**
   copies the calculated model, separate bias and training patterns to Recall.
   Edit the query rows for new inputs, then Calculate. The transfer uses the last
   completed result even if the training form has since been edited.

All matrices use **patterns as rows**: `X` and `S` are `P × n`, `T` is `P × m`,
and `W` is `n × m`. Recall accepts `Q × n` query rows and returns `Q × m` outputs.
Dimensions are inferred from the editors and validated before calculation.
Inputs support 1–128 patterns and 1–32 input/output coordinates; supplied `W`
has 1–32 rows and columns. Values must be finite, with magnitude at most `1e50`
for training/query values and `1e150` for supplied recall weights/bias.

| Selector entry | Calculation and displayed results |
| --- | --- |
| Lecture 08 · Correlation | Every `x_pᵀx_p`, their sum, `R=XᵀX/P`, eigenvalues/vectors, maximum eigenvalue, numerical residuals, determinant checks and the lecture learning-rate bound. No centering or implicit bias column. |
| Lecture 08 · Hebbian memory | Start `W=0`; each pair adds `ΔW[i,j]=s[i]t[j]`. Show each delta and updated matrix. If enabled, a separate bias starts at zero and adds `t` after each pair. One pass, no learning-rate multiplier. |
| Lecture 08 · Outer products | Show every `s_pᵀt_p`, accumulated sum and final `W=SᵀT`. Shares the associative Hebbian accumulation engine, with bias disabled, and independently checks the direct matrix product. |
| Lecture 08 · Orthogonality | Show `G=SSᵀ`, squared norms and every distinct pair's dot product and orthogonality result. Default absolute tolerance is 0 (exact zero); an explicit tolerance is optional. Zero vectors are identified. |
| Lecture 08 · Recall | Show each `x_i W[i,j]` term, `XW`, optional separately added bias and elementwise outputs using binary, sign or bipolar activation. Expected targets are optional and compared literally, with matched elements and complete rows reported. |

**Learning-rate conventions:** the tool displays both the literal note formula
`α < λmax/2` and the corrected mean LMS stability bound **`0 < α < 2/λmax`**.
The latter follows from `error_next=(I−αR)error` for mean LMS updates. For
mean squared error without a half factor, the gradient has a factor 2 and the
bound is `α < 1/λmax`. These are batch/mean bounds, not a guarantee for the
sequential constant-rate ADALINE trainer. Include a constant input column when
analyzing a trainable bias. With `R=0`, the inputs impose no curvature bound and
cannot identify the weights. No diagnostic automatically changes a training rate.
For the four binary input rows, `λmax=0.75`: the literal note formula gives
0.375, while the mean LMS upper bound is 8/3.
For 1–2 coordinates, Worked steps expands `det(R−λI)=0`. Larger matrices use
scaled symmetric Jacobi diagonalization with an explicit convergence check;
eigenvector residuals and scaled determinants verify the computed roots.

**Recall conventions:** select **binary** (`1` if `z>0`, otherwise `0`),
**sign** (`−1/0/+1`, with zero mapped to zero, as in 08_cvicenie), or
**bipolar** (`+1` if `z>=0`, otherwise `−1`, as in CV8Siete).
Training verification, query recall, report exports and **Use trained W for recall**
all use and preserve this selection. Bias is off by default and remains separate.

**Matrix product** shows `AB`, `Aᵀ`, `Bᵀ` and every dot-product contribution.
**McCulloch–Pitts network** accepts binary pattern rows, a neuron-weight matrix
with one row per neuron and columns for inputs followed by neurons, and one row
of thresholds. Self/later-neuron weights must be zero. Each neuron emits `1`
when its weighted sum is at least its threshold; otherwise `0`. Outputs appear
in neuron order, with the final neuron in the last column. Source presets include
AND, OR, NAND, NOT, the three-neuron Exercise 6 network, and XOR via OR/NAND/AND.

The supplied associative example produces
`W=[[2,0],[1,0],[0,1],[0,2]]`. Inputs `(1,0,0,0)` and `(1,1,0,0)` give weighted
sums `(2,0)` and `(3,0)` and both predict `(1,0)`. Training/outer-product results
also show final-model recall and the cross-talk contribution
`sum_{p≠k}(s_k·s_p)t_p`. Nonorthogonal inputs introduce cross-talk terms but may
still recall correctly after activation. Orthogonal nonzero inputs remove those
terms; a zero input cannot distinguish a nonzero target without bias. General
finite real training targets are accepted, but the specified binary activation
does not reproduce arbitrary real target values.

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
- **y ≠ t, t = −1:** select every hidden unit whose `z_in_j >= 0`. Each selected
  unit uses its own old linear response: `alpha*(-1-z_in_j)` for bias and
  `alpha*(-1-z_in_j)*x_i` for weights. All units' selections are made before
  any update is applied.

The output weights **v1 … vH and b_out remain fixed**, as in Step 0. The page does
not specify their numbers, so the editable defaults are **v_j = 0.5** and
**b_out = (H-1)/2**, implementing bipolar OR on all hidden activations. For two
units these are the previous `v1 = v2 = b3 = 0.5`. For three units, enter three
output weights; the default bias is `1`. Enter different fixed values if your
exercise gives them; the update cases stay as written.

At exactly zero, activation is +1, so that unit is included in the negative-target
update. This corrects the previous `>0` boundary, which incorrectly stopped the
zero-initialized AND task. It matches the worked two-epoch table in
07_cvicenie, page 10. Small random hidden coefficients remain the default.

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

## Perceptron

The image's two activation/update combinations and an additional bipolar sign
perceptron are separate variants, recorded in every run snapshot and export.

| Variant | Targets and activation | Weight/bias update |
| --- | --- | --- |
| Binary (upper section) | Targets `0,1`; `y=1` at `net>=0`, otherwise `0` | `e=t-y`; `delta_w_i=eta*e*x_i`, `delta_b=eta*e` |
| Bipolar · error update (CV6) | Targets `-1,+1`; `y=+1` at `net>=0`, otherwise `-1` | `e=t-y`; `delta_w_i=alpha*e*x_i`, `delta_b=alpha*e` (mistakes give `e=±2`) |
| Bipolar sign | Targets `-1,+1`; `y=+1` at `net>=0`, otherwise `-1` | Only when `y!=t`: `delta_w_i=alpha*t*x_i`, `delta_b=alpha*t` |
| Threshold band (lower Steps 0–6) | Targets `-1,+1`; `y=+1` above θ, `0` on `[-theta,theta]`, `-1` below −θ | Only when `y!=t`: `delta_w_i=alpha*t*x_i`, `delta_b=alpha*t` |

All variants calculate `net = b + sum(x_i*w_i)` from the old model. All deltas for one
pattern use that same response and are committed before visiting the next row.
Correct predictions leave all weights and bias unchanged. In the bipolar sign and band variants,
an opposite-class prediction gives `e=2` or `-2`, but the update still uses the
**target** rather than that error. At exactly zero, binary output is `1`, bipolar
sign output is `+1`, and band output is `0` when θ = 0. Bipolar sign never
predicts a neutral class. Both `-theta` and `theta` are inside the threshold band.

Choose `0 < rate <= 1`, as specified in Step 0. Switching to Perceptron defaults
to zero initial weights/bias and rate 1; initial coefficients remain editable.
Theta defaults to zero and is editable only for band mode. The screenshot has
two inputs, and the same formula supports the GUI's 1–32 input coordinates.
Custom finite real inputs are accepted with targets matching the selected mode.

Presets and CSV import support binary `0/1` and bipolar `-1/+1` targets. Choosing a Perceptron
variant reloads the selected truth-table preset in its default input encoding
and target labels. Switching back to another algorithm reloads preset targets
as bipolar. **Custom training data is retained without automatic recoding**;
input validation identifies labels incompatible with the selected rule.

Step 6 runs only after a full pass. **`weights_stable`** means no weight or bias
changed at any point in that pass. Neither final accuracy nor equality of the
epoch's starting and ending weights replaces this test: changes can cancel.
Even after an epoch first reaches perfect classification, the algorithm performs
the unchanged verification epoch required by the notes. An epoch limit reports
`epoch_limit`, including for nonseparable XOR. The existing numerical guard
reports `numerical_limit` and retains the last valid model and committed prefix.

**Binary AND check:** keep the default row order `(1,1), (1,0), (0,1), (0,0)`.
With zero initialization and rate 1, the final model is `w=(1,2), b=-3`.
Epoch 7 ends with all four predictions correct but contains updates; epoch 8
makes no changes and stops. The first pattern at net zero is already correct
and produces no update.

**Bipolar AND check:** select **Bipolar (−1/+1)**, zero initialization, and rate 1.
The default bipolar truth table finishes epoch 1 with `w=(0,0), b=-2`, reaches
`w=(1,1), b=-1` with all predictions correct in epoch 2, then stops after the
unchanged third epoch. The negative-target mistake at net zero records error
`-2` but applies the target-based bias delta `-1`. This convention follows the
lecture's Rosenblatt update; it does not multiply the update by the ±2 error.

Worked steps show the selected activation, classification error, applicable
update formula, substituted deltas, and the complete-pass stopping check. Epoch
tables distinguish **Mistakes** (incorrect predictions during the online pass),
**Changed visits** (patterns that actually changed coefficients), and **Correct**
(accuracy of the fixed epoch-end model). Classification SSE is `sum((t-y)^2)`:
it equals the number of binary errors, and each bipolar sign error contributes 4;
in band mode a wrong output contributes
1 for the neutral class or 4 for an opposite class. These are diagnostics only.

Both CSV exports include the mode, learning rate, θ, epoch budget, and
full-precision values. The trace also includes the initial model, unchanged
correct patterns, and a `weights_changed` flag. Existing results retain their original variant and labels
when you edit the form or switch algorithms. Float calculations use Rust `f64`;
changing display precision does not round the computation.

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
- `rust_backend/src/perceptron.rs`: binary, bipolar sign, and threshold-band perceptron rules.
- `rust_backend/src/lecture08.rs`: independent matrix calculations, scaled eigensolver,
  multi-output associative memory, recall, validation and numerical tests.
- `rust_backend/src/optimization.rs`: one-dimensional gradient and Newton methods,
  2D steepest descent, first/second automatic derivatives, numerical/domain guards
  and function samples.
- `rust_backend/src/newton2d.rs`: direct propagation of the full 2D Hessian,
  scaled Newton system, singularity/domain guards and function samples.
- `rust_backend/src/nelder_mead.rs`: sorted simplexes, the supplied reflection,
  expansion, inside contraction and shrink rules, complete trial records and guards.
- `rust_backend/src/normalized_gradient.rs`: normalized negative gradients,
  directional-derivative bracketing/bisection, domain refinement and full line traces.
- `rust_backend/src/powell.rs`: the supplied three-search direction-shift cycle,
  derivative-free Brent line minimization, cycle stopping tests and full trial records.
- `rust_backend/src/main.rs`: validated line protocol for the Python adapter.
- `rust_backend/src/cli.rs`, `cli_expression.rs`: explicit file CLI, strict input
  parsing, parameter selection, shared dispatch and complete text reports.
- `rust_backend/tests/cli.rs`, `tests/test_cli.py`: all-template CLI verification,
  selected-rate checks, file errors and full-trace comparisons with GUI adapters.
- `backend.py`: immutable run data, Rust build/process management, CSV exports.
- `ui/app.py`: interface, substituted formulas, navigation and canvas charts.
- `algorithms/madaline.py`, `ui/madaline_view.py`: MADALINE run data, protocol, CSV and worked views.
- `algorithms/hebbian.py`, `ui/hebbian_view.py`: Hebbian run data, protocol, CSV and worked views.
- `algorithms/perceptron.py`, `ui/perceptron_view.py`: Perceptron variants, protocol, CSV and worked views.
- `algorithms/lecture08.py`, `ui/lecture08_gui.py`: matrix requests/reports, imports/exports and
  the separate Lecture 08 workspace. Numerical loops remain in Rust.
- `algorithms/optimization.py`, `ui/optimization_gui.py`: allowlisted function input, immutable
  gradient/Newton results, CSV exports and the optimization workspace.
- `ui/newton_view.py`: the supplied Newton algorithm, substitutions and strict step tests.
- `algorithms/steepest.py`, `ui/steepest_view.py`: 2D objective requests/results, CSV exports,
  simultaneous-update substitutions, gradient norms and heatmap/path charts.
- `algorithms/newton2d.py`, `ui/newton2d_view.py`: full-Hessian Newton requests/results,
  matrix substitutions, curvature diagnostics and complete CSV exports.
- `algorithms/nelder_mead.py`, `ui/nelder_mead_view.py`: initial-simplex inputs, immutable
  simplex/trial records, branch substitutions, triangle overlays and full CSV export.
- `algorithms/normalized_gradient.py`, `ui/normalized_gradient_view.py`: independent normalized
  method inputs/results, line-search tables, numerical substitutions, probe
  overlays and separate outer/line CSV exports.
- `algorithms/powell.py`, `ui/powell_view.py`: independent Powell inputs/results, supplied-exercise
  preset and reference, cycle substitutions, movement/probe plots and complete exports.
- `examples/learning.py`: editable learning presets transcribed from the practical exercises.
- `main.py`: portable launcher. Other algorithms can be added as separate Rust
  modules and worksheet views without moving calculation loops into Python.

```sh
cargo test --offline --manifest-path every_example_gui/rust_backend/Cargo.toml
cargo test --offline --lib --manifest-path every_example_gui/rust_backend/Cargo.toml
cargo test --offline --bin exam-adaline lecture08::tests --manifest-path every_example_gui/rust_backend/Cargo.toml
python3 -m unittest discover -s every_example_gui/tests -v
EXAM_GUI_TESTS=1 python3 -m unittest discover -s every_example_gui/tests -v
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
Perceptron checks cover the hand-worked binary AND trace, the required unchanged
epoch, threshold boundaries, target-versus-error updates, cancelling updates,
custom initialization, XOR limits, binary CSV input, zero-update charts, mode
switching and immutable exports. When `du_7_prednaska_1/results` is present,
all six saved threshold-band lecture traces are compared value by value using
their original input order, learning rate and theta.
Bipolar sign tests verify the zero tie, absence of neutral predictions,
target-based deltas, a hand-worked AND trace, unchanged-pass stopping, target
validation, mode switching, and exports retaining their original variant.
Lecture 08 tests verify the supplied correlation/rate and associative examples,
general real matrices with multiple outputs, per-pattern updates, independent
outer-product agreement, bias isolation, strict-zero recall, orthogonality and
cross-talk, singular/zero/scaled spectra, eigenvector invariants, input/dimension
errors, full-precision exports, cancellation and GUI state preservation.
Optimization tests check hand-worked gradient estimates, analytic derivative
fixtures, stationary starts, exact tolerance boundaries, oscillation, divergence,
domain failures, stagnation, safe expression parsing, complete exports and
switching between learning, matrix and optimization views in one window.
Newton checks include analytic second-derivative fixtures, supplied derivatives,
hand-worked updates, post-update strict tolerance, inclusive Nmax, zero/tiny and
negative curvature, cycles, numerical/domain failures, complete CSV data,
per-method state preservation and completion after switching methods.
Two-dimensional checks cover coupled hand calculations, analytic partial
derivatives, a Rosenbrock fixture, Euclidean norm stopping, fixed-rate oscillation,
atomic rejection of invalid pairs, both-coordinate stagnation, heatmap sampling,
complete exports, pagination and switching among all eight optimization methods.
2D Newton checks verify coupled hand calculations, nonlinear Hessian updates,
analytic mixed derivatives, a Rosenbrock step, scaled systems with underflowing
raw determinants, small mixed terms beside large diagonals, singular and
indefinite Hessians, cycles, atomic domain rejection, complete matrix exports and
background-result isolation in the desktop workspace.
Nelder–Mead checks include hand-worked paths through every branch, strict equality
boundaries, inside contraction when reflection improves on C, atomic rejection
at all five trial stages, stable sorting, function-spread stopping, geometry and
stagnation limits, derivative-free nonsmooth objectives, complete trial exports,
simplex overlays, pagination and method/background-result preservation.
Normalized-vector checks cover a hand-worked 3–4–5 direction and line root,
variable γ on elliptic/coupled quadratics, unit directions, relative line
residuals, objective scaling, zero-gradient starts, domain refinement, failed
searches without fallback steps, line precision limits, full trial exports,
line-search pagination and method/background-state isolation.
Powell checks reproduce all six supplied exercise moves and signed α values,
verify raw displacement directions and the exact direction shift, strict
post-cycle function/position tests, zero directions, nonsmooth functions,
Rosenbrock, domain refinement, incomplete-cycle failures, complete exports,
line-search pagination, cancellation and method/background-state isolation.
