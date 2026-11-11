from collections import Counter

from ..algorithms.optimization import STATUS
from ..algorithms.nelder_mead import ACTIONS
from . import steepest_view

NOTES = """NELDER–MEAD · TWO DIMENSIONS · SUPPLIED ALGORITHM

Two branch conventions are available. "slides" reproduces the inside-only contraction in the Exercise 4 slides. "standard" also implements the outside contraction in CV4Utorok: if f(B) ≤ f(R) < f(C), K=(M+R)/2, accept when f(K) ≤ f(R), otherwise shrink. For f(R) ≥ f(C), contract inside as below. These variants can legitimately produce different intermediate simplexes.

Two stopping rules are available. "function" uses spread ≤ ε as below. "position" reproduces CV4Utorok: max_j ||vertex_j−best||₂ < ε (strict). The reported full pairwise diameter is a separate diagnostic. Choose the rule required by the exercise.

Enter f(x,y), three non-collinear initial vertices, a function tolerance ε and an iteration budget. This is a derivative-free method: Rust evaluates only f, without gradients or Hessians. The initial vertex labels identify the input rows. At EVERY iteration the roles A, B and C are reassigned by sorting f(A) ≤ f(B) ≤ f(C). Ties preserve their existing order.

The image's coefficients are fixed:
    reflection α=1, expansion γ=2, contraction ρ=0.5, shrink σ=0.5.

1. Sort the simplex: f(A) ≤ f(B) ≤ f(C).
2. Compute M=(A+B)/2, excluding the worst point C.
3. Reflect: R=M+α(M−C).
   If f(A) ≤ f(R) < f(B), replace C with R.
4. If f(R) < f(A), expand: E=M+γ(R−M).
   If f(E) < f(R), replace C with E; otherwise replace C with R.
5. If f(R) ≥ f(B), contract INSIDE: K=M+ρ(C−M), using the OLD C.
   If f(K) < f(C), replace C with K.
   Otherwise shrink BOTH non-best vertices toward the OLD A:
       B′=A+σ(B−A), C′=A+σ(C−A).
   A remains unchanged. Evaluate both new points and commit the simplex together.
Re-sort the new simplex before the next iteration.

These are the slide's exact branches. In particular, f(R)=f(A) can take reflection; f(R)=f(B) takes inside contraction; equal expansion/reflection values select R; equal contraction/old-worst values trigger shrink. The image uses inside contraction even when f(B) ≤ f(R) < f(C). No outside-contraction branch is substituted. Shrink has no additional improvement test and may increase the worst objective value.

Explicit stopping convention
The image says to repeat until changes in f are small, without defining a numerical test. This worksheet explicitly uses the current sorted simplex's objective spread:
    spread=f(C)−f(A) ≤ ε.
The test includes the initial simplex and runs again after each committed iteration. Maximum iterations counts complete simplex updates (1–100,000). A best value that stays unchanged does not itself stop the algorithm. The simplex diameter is reported as a diagnostic and does not enter this stopping rule. Flat objectives and equal-valued vertices can pass the function test with a large simplex, including away from a minimum; this test does not establish spatial accuracy or optimality. If an attempted update leaves the entire sorted simplex unchanged at floating-point precision, the status is stagnation rather than tolerance convergence.

Function and numerical limits
Use x, y, pi, e, + − * / ^ or **, parentheses, sin, cos, tan, exp, ln/log and sqrt. Multiplication must be explicit. Trigonometric functions use radians; log is natural logarithm. No derivative is required: sqrt(x²+y²) is valid at the origin. Function values and intermediate expression values, trial coordinates and committed coordinates must be finite and at most 1e100 in magnitude. Initial coordinates and plot bounds have magnitude at most 1e50. Rust rejects duplicate/collinear initial vertices at working precision using a scaled area check.
If a REQUIRED trial point is invalid, the entire attempted iteration is rejected and the previous simplex is retained. An invalid expansion is not silently replaced by reflection. A failed second shrink point does not commit the first shrink point. The failed stage and coordinates are displayed and exported. An invalid initial vertex/function or cancelled calculation retains the previous completed run.

Worked steps and plots
Every completed iteration records M, R, any E or K, both shrink points when applicable, the exact comparison and selected branch. The plot overlays the selected old simplex (solid), next simplex (dashed) and visible trial points on the function heatmap. A/B/C are the CURRENT sorted roles. The right chart shows the best objective at each simplex, including repeated best values. Plot bounds affect only the view, not the iterations. Long histories are downsampled only for drawing.
All iterations and CSV exports retain every sorted simplex, objective spread, diameter, branch and evaluated trial point. Exports also include the original unsorted input vertices, fixed coefficients and immutable run parameters at full precision. The final row has no outgoing update. Switch methods or return to Learning at any time; each method's inputs, completed result and navigation remain in the same application window.
"""

TRACE_COLUMNS = [
    "k",
    "Action",
    "f(C)−f(A)",
    "Diameter",
    *[f"{v} {col}" for v in "ABC" for col in ("x", "y", "f")],
]


def _point(f, vertex):
    return f"({f(vertex.x)}, {f(vertex.y)}), f = {f(vertex.value)}"


def render_summary(panel):
    r, f = panel.result, panel.fmt
    p = r.final
    initial = "; ".join(
        f"{name}=({f(v[0])},{f(v[1])})" for name, v in zip("ABC", r.request.simplex)
    )
    panel.snapshot.set(
        f"{r.request.algorithm_name}: f(x,y) = {r.request.expression}  ·  initial {initial}  ·  ε = {f(r.request.tolerance)}"
    )
    detail = {
        "function_tolerance": "The function spread test passed. Diameter is diagnostic; spatial accuracy is not established by this test.",
        "simplex_tolerance": "The maximum vertex distance to the best vertex is strictly below ε.",
        "iteration_limit": "The iteration budget ended before the function spread test passed.",
        "numerical_limit": "A required trial failed the numerical/domain guard; the attempted simplex update was rejected.",
        "stagnation": "The attempted update left the entire sorted simplex unchanged while the function test still fails.",
    }[r.status]
    counts = Counter(step.action for step in r.steps)
    branches = (
        ", ".join(
            f"{ACTIONS[action]}: {counts[action]}"
            for action in ACTIONS
            if counts[action]
        )
        or "No simplex updates"
    )
    failed = (
        f"\nFailed trial: {r.failure.stage} at ({f(r.failure.x)}, {f(r.failure.y)})"
        if r.failure
        else ""
    )
    panel.app._set_text(
        panel.summary_text,
        f"{STATUS[r.status]}\nBest estimate A: {_point(f, p.simplex[0])}\nCompleted iterations: {p.iteration:,}   ·   f(C)−f(A) = {f(p.spread)}\nBranches: {r.request.variant}; stopping: {r.request.stopping}; ε={f(r.request.tolerance)}\nSimplex diameter: {f(p.diameter)}; max distance to best: {f(p.best_radius)}\n{detail}{failed}\n{branches}",
    )


def worked_step(panel):
    r, f, i = panel.result, panel.fmt, panel.step_index
    p = r.points[i]
    a, b, c = p.simplex
    lines = [
        f"SIMPLEX k = {p.iteration}\nf(x,y) = {r.request.expression}",
        f"1 · SORT · current roles\nA (best): {_point(f, a)}\nB (middle): {_point(f, b)}\nC (worst): {_point(f, c)}\n{f(a.value)} ≤ {f(b.value)} ≤ {f(c.value)}",
        f"Position test: max_j ||vertex_j−best||₂ < ε\n{f(p.best_radius)} < {f(r.request.tolerance)} → {str(p.best_radius < r.request.tolerance).upper()}"
        if r.request.stopping == "position"
        else f"Function spread test: f(C)−f(A) ≤ ε\n{f(c.value)} − {f(a.value)} = {f(p.spread)}\n{f(p.spread)} ≤ {f(r.request.tolerance)} → {str(p.spread <= r.request.tolerance).upper()}\nDiameter = {f(p.diameter)} (diagnostic only)",
    ]
    if i < len(r.steps):
        step = r.steps[i]
        mx, my = step.centroid
        reflected = step.reflection
        lines.extend(
            [
                f"2 · CENTROID · exclude C\nM=(A+B)/2\nMₓ=({f(a.x)}+{f(b.x)})/2={f(mx)}\nMᵧ=({f(a.y)}+{f(b.y)})/2={f(my)}",
                f"3 · REFLECTION · α=1\nR=M+1×(M−C)\nR=({f(mx)},{f(my)})+1×[({f(mx)},{f(my)})−({f(c.x)},{f(c.y)})]\nR: {_point(f, reflected)}",
                f"f(A) ≤ f(R) < f(B)\n{f(a.value)} ≤ {f(reflected.value)} < {f(b.value)} → {str(a.value <= reflected.value < b.value).upper()}",
            ]
        )
        if step.expansion:
            e = step.expansion
            lines.append(
                f"4 · EXPANSION · f(R) < f(A) is TRUE; γ=2\nE=M+2×(R−M)\nE: {_point(f, e)}\nf(E) < f(R): {f(e.value)} < {f(reflected.value)} → {str(e.value < reflected.value).upper()}\nSelect {'E' if step.action == 'expand' else 'R'}; equality selects R."
            )
        if step.contraction:
            k = step.contraction
            outside = r.request.variant == "standard" and reflected.value < c.value
            lines.append(
                f"5 · OUTSIDE CONTRACTION · f(B) ≤ f(R) < f(C)\nK=M+0.5×(R−M)\nK: {_point(f, k)}\nf(K) ≤ f(R): {f(k.value)} ≤ {f(reflected.value)} → {str(k.value <= reflected.value).upper()}"
                if outside
                else f"5 · INSIDE CONTRACTION · f(R) ≥ f(B) is TRUE; ρ=0.5\nK=M+0.5×(OLD C−M)\nK: {_point(f, k)}\nf(K) < f(OLD C): {f(k.value)} < {f(c.value)} → {str(k.value < c.value).upper()}"
            )
        if step.action == "shrink":
            lines.append(
                f"SHRINK · σ=0.5 · keep OLD A\nB′=A+0.5×(B−A): {_point(f, step.shrunk_b)}\nC′=A+0.5×(C−A): {_point(f, step.shrunk_c)}\nCommit both shrink vertices together."
            )
        after = r.points[i + 1]
        lines.append(
            f"SELECTED BRANCH: {ACTIONS[step.action]}\nRe-sort the next simplex:\n"
            + "\n".join(
                f"{label}: {_point(f, v)}" for label, v in zip("ABC", after.simplex)
            )
        )
    else:
        if r.failure:
            lines.append(
                f"Failed {r.failure.stage} trial at ({f(r.failure.x)}, {f(r.failure.y)}).\nThe entire attempted iteration was rejected; this last valid simplex is retained."
            )
        lines.append(
            f"Stop: {STATUS[r.status]}. No further simplex update was committed."
        )
    return "\n\n".join(lines)


def trace_values(panel, index):
    r, f = panel.result, panel.fmt
    p = r.points[index]
    action = (
        ACTIONS[r.steps[index].action] if index < len(r.steps) else STATUS[r.status]
    )
    return [
        p.iteration,
        action,
        f(p.spread),
        f(p.diameter),
        *[f(x) for v in p.simplex for x in (v.x, v.y, v.value)],
    ]


def draw_chart(panel):
    left, right, top, bottom = steepest_view.draw_chart(
        panel, "BEST OBJECTIVE BY ITERATION"
    )
    c, r = panel.chart, panel.result
    request, index = r.request, panel.step_index

    def visible(x, y):
        return (
            request.plot_min <= x <= request.plot_max
            and request.plot_y_min <= y <= request.plot_y_max
        )

    def xy(x, y):
        return (
            left
            + (x - request.plot_min)
            / (request.plot_max - request.plot_min)
            * (right - left),
            bottom
            - (y - request.plot_y_min)
            / (request.plot_y_max - request.plot_y_min)
            * (bottom - top),
        )

    def triangle(vertices, color, tag, dashed=False):
        for a, b in zip(vertices, (*vertices[1:], vertices[0])):
            if visible(a.x, a.y) and visible(b.x, b.y):
                c.create_line(
                    *xy(a.x, a.y),
                    *xy(b.x, b.y),
                    fill=color,
                    width=2,
                    dash=(4, 3) if dashed else (),
                    tags=tag,
                )

    def mark(label, x, y, color, tag):
        if visible(x, y):
            px, py = xy(x, y)
            c.create_oval(
                px - 4, py - 4, px + 4, py + 4, fill=color, outline="white", tags=tag
            )
            c.create_text(
                px + 6,
                py - 8,
                text=label,
                anchor="w",
                fill=color,
                font=("Arial", 9, "bold"),
                tags=tag,
            )

    current = r.points[index].simplex
    triangle(current, "#74499b", "simplex")
    for label, v in zip("ABC", current):
        mark(label, v.x, v.y, "#74499b", "simplex_vertex")
    if index < len(r.steps):
        step = r.steps[index]
        triangle(r.points[index + 1].simplex, "#087e76", "next_simplex", True)
        mark("M", *step.centroid, "#617284", "centroid")
        for label, v, color in (
            ("R", step.reflection, "#b36b00"),
            ("E", step.expansion, "#d35336"),
            ("K", step.contraction, "#325caf"),
        ):
            if v is not None:
                mark(label, v.x, v.y, color, "trial_point")
