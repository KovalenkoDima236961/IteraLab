NOTES = """NEWTON'S METHOD · ONE DIMENSION · SUPPLIED ALGORITHM

Inputs: f(x), f′(x), f″(x), starting estimate x₀, tolerance ε, and Nmax.
Leave either derivative field blank to calculate that derivative of f automatically in Rust. Supplied derivative expressions are used literally; ensure they are the correct derivatives of f. There is no learning rate in this method.

Choose x₀. For i=0,1,2,…,Nmax (inclusive, as written on the slide):
    Evaluate f′(xᵢ) and f″(xᵢ).
    xᵢ₊₁ = xᵢ − f′(xᵢ) / f″(xᵢ)
    If |xᵢ₊₁ − xᵢ| < ε, stop.

The tolerance test is AFTER each update and strictly <; equality continues. It is the actual difference of the two floating-point estimates. A zero step passes this literal test. The method does not substitute a gradient test, modify the curvature, damp the step, backtrack, or reject negative curvature. Nmax is the last loop index, so Nmax=0 permits one update and Nmax=1 permits two. The trace is limited to 100,000 updates, so the largest Nmax is 99,999.

Example: f(x)=(x−2)²+1, f′(x)=2(x−2), f″(x)=2, x₀=6.
i=0: x₁=6−8/2=2; |2−6|=4. Continue for ε=1e−6.
i=1: x₂=2−0/2=2; |2−2|=0 < ε. Stop.
Starting at the minimum still performs the zero update before the tolerance test.

Starting points and minimum estimates
The slide flags sensitive starting points around extrema, saddle/inflection and critical points. The specific division failure is f″(xᵢ)=0: no update can be formed and the worksheet reports Zero second derivative, even if f′ is also zero. Small nonzero curvature is used literally and can produce a very large step. A valid minimum with positive nonzero curvature is a valid starting point.
A passed step test alone does not prove a minimum or a stationary point. Newton can approach a maximum when curvature is negative, can cycle, or can stop because rounding made the estimates equal. Inspect f′ and f″ and the function plot. Curvature and gradient are diagnostics; they do not change the slide's stopping rule.

Numerical limits
If the function, required derivatives, or attempted update are undefined, nonfinite or exceed 1e100 in magnitude, the attempt is rejected and the last valid point remains. Invalid starting values retain the previous completed result. Automatic derivatives use the chain rule through second order, without finite differences. Constant powers include their exact zero/one/two cases at x=0. Variable exponents require a positive base for differentiation.

Function syntax
x, pi, e, + − * / ^ or **, parentheses, sin, cos, tan, exp, ln/log and sqrt. Multiplication must be explicit: 2*x. Trigonometric arguments use radians; log is natural logarithm. Blank derivatives always mean derivatives of f, independently of any supplied derivative field.

Plots, navigation and exports
Plot bounds affect the view only; iterations may leave that range. Curve gaps indicate invalid/numerically limited function samples. Worked steps substitute f′ and f″ at the OLD estimate and show the strict test for each committed update. The last trace row has no outgoing update; its incoming step is explained in Worked steps. CSV exports include both derivative sources, every estimate, curvature, quotient step, actual absolute step, and strict test at full precision.
Switch between Gradient and Newton at any time. Each method retains its form, result and navigation position. A running optimization finishes under its original method even when another method is selected. Learning remains in the same application window.
"""


def summary(panel):
    r, f = panel.result, panel.fmt
    p = r.final
    step = abs(p.x - r.points[-2].x) if len(r.points) > 1 else None
    test = (
        f"|xᵢ₊₁ − xᵢ| = {f(step)} < ε: {step < r.request.tolerance}"
        if step is not None
        else "No update was committed; no step tolerance test was performed."
    )
    detail = {
        "step_tolerance": "The strict step test passed. This alone does not prove a minimum; inspect the derivatives.",
        "iteration_limit": "The inclusive i=0…Nmax loop ended before the strict step test passed.",
        "zero_curvature": "f″(xᵢ) = 0: the Newton quotient is undefined. No further update was committed.",
        "numerical_limit": "The next attempt exceeded a numerical limit or left the required domain; it was rejected.",
    }[r.status]
    return f"Estimate x = {f(p.x)}     f(x) = {f(p.value)}\nf′(x) = {f(p.gradient)}     f″(x) = {f(p.second_derivative)}\nCompleted updates: {p.iteration:,}   ·   {test}\n{detail}"


def worked_step(panel):
    r, f, index = panel.result, panel.fmt, panel.step_index
    p = r.points[index]
    first = r.request.first_derivative or "automatic derivative of f"
    second = r.request.second_derivative or "automatic second derivative of f"
    lines = [
        f"ESTIMATE i = {p.iteration}\nf(x) = {r.request.expression}\nf′(x): {first}\nf″(x): {second}",
        f"xᵢ = {f(p.x)}\nf(xᵢ) = {f(p.value)}\nf′(xᵢ) = {f(p.gradient)}\nf″(xᵢ) = {f(p.second_derivative)}",
    ]
    if index + 1 < len(r.points):
        nxt = r.points[index + 1]
        step = abs(nxt.x - p.x)
        lines.extend(
            [
                f"NEWTON UPDATE · using derivatives at the OLD point\nΔx = −f′(xᵢ) / f″(xᵢ)\n   = −({f(p.gradient)}) / ({f(p.second_derivative)})\n   = {f(-p.gradient / p.second_derivative)}",
                f"xᵢ₊₁ = xᵢ − f′(xᵢ) / f″(xᵢ)\n     = ({f(p.x)}) − ({f(p.gradient)}) / ({f(p.second_derivative)})\n     = {f(nxt.x)}",
                f"STOPPING TEST · after the update\n|xᵢ₊₁ − xᵢ| < ε\n|({f(nxt.x)}) − ({f(p.x)})| = {f(step)}\n{f(step)} < {f(r.request.tolerance)} → {str(step < r.request.tolerance).upper()}",
                "Step tolerance met; stop."
                if step < r.request.tolerance
                else "Strict tolerance not met (equality also continues).",
            ]
        )
        if step >= r.request.tolerance and index == r.request.max_iterations:
            lines.append(
                "The last permitted loop index Nmax was processed; stop at the iteration limit."
            )
    else:
        if index:
            step = abs(p.x - r.points[index - 1].x)
            lines.append(
                f"Last committed absolute step = {f(step)}\n{f(step)} < {f(r.request.tolerance)} → {str(step < r.request.tolerance).upper()}"
            )
        if r.status == "zero_curvature":
            lines.append(
                "Cannot divide by f″(xᵢ) = 0. The next Newton update is undefined."
            )
        from ..algorithms.optimization import STATUS

        lines.append(f"Stop: {STATUS[r.status]}. No further update was committed.")
    return "\n\n".join(lines)
