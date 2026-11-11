import math

from ..algorithms.optimization import STATUS

NOTES = """STEEPEST GRADIENT DESCENT · TWO DIMENSIONS

The supplied slide defines the gradient:
    ∇f(x,y) = (∂f/∂x(x,y), ∂f/∂y(x,y)).
It points in the direction of steepest increase in the Euclidean plane. Descent follows its negative.

The slide's fixed-step updates are:
    x[k+1] = x[k] − α ∂f/∂x(x[k], y[k])
    y[k+1] = y[k] − α ∂f/∂y(x[k], y[k])

Both partial derivatives use the SAME OLD pair (x[k],y[k]). Compute both new coordinates, then commit the pair together. The displayed y_x in the slide's first component is treated as y_k, consistent with its gradient definition and second component. There is no coordinate-by-coordinate recomputation, gradient normalization, line search, or rate adjustment. α remains the entered positive constant.

Inputs
Enter a differentiable f(x,y), initial x₀ and y₀, step length α, gradient tolerance ε and maximum iterations. Rust evaluates both analytic partial derivatives by forward automatic differentiation and the chain rule, without finite differences. Use x, y, pi, e, + − * / ^ or **, parentheses, sin, cos, tan, exp, ln/log and sqrt. Multiplication must be explicit, for example x*y. Trigonometric functions use radians; log is natural logarithm. Variable exponents require a positive base for differentiation. One-dimensional methods retain their original x-only syntax.

Stopping convention
The slide provides no stopping rule. This worksheet explicitly checks the Euclidean gradient norm:
    ||∇f||₂ = sqrt((∂f/∂x)² + (∂f/∂y)²) ≤ ε.
The test includes the initial point; an already stationary start performs zero updates. Otherwise the calculation performs at most the selected number of simultaneous updates (1–100,000). A small step alone is not convergence. Floating-point stagnation is reported only when BOTH new coordinates round to their old values while the gradient test fails. A stationary point can also be a maximum or a saddle; passing this test alone does not prove a minimum.

Worked coupled example
f(x,y)=x²+xy+y², (x₀,y₀)=(2,1), α=0.1.
∂f/∂x=2x+y=5; ∂f/∂y=x+2y=4.
x₁=2−0.1×5=1.5; y₁=1−0.1×4=0.6.
Both partials used (2,1), so y₁ is 0.6.
At (1.5,0.6), the next gradient is (3.6,2.7); the next pair is (1.14,0.33).

Step length and numerical limits
As the slide explains, small α can make progress slow, and large α can overshoot a minimum or diverge. Objective increases are recorded and do not trigger a different algorithm. The function must be differentiable on the visited points. Values, partial derivatives, intermediate expression results, coordinates and attempted coordinate steps must be finite and at most 1e100 in magnitude. If either coordinate or either partial fails, the entire attempted pair is rejected; the last valid pair remains. An invalid start, cancelled calculation or input error retains the previous completed result.

Plots and exports
The left plot is a sampled function heatmap with the optimization path. Its horizontal axis is x and vertical axis is y. Pale colors are lower sampled objective values, darker colors higher ones. Gray cells mark invalid samples. The selected point is highlighted; the right plot shows objective value versus iteration. Plot ranges affect only the view, never constrain the algorithm. Points outside the view remain in the full trace and objective history. Long paths are downsampled only for drawing.
Worked steps substitute BOTH partial derivatives at the old pair, show the negative gradient and each coordinate update. All iterations and CSV exports retain every pair, objective, both partials, gradient norm, both coordinate steps and next pair. The final row has no outgoing update. Exports use the immutable completed inputs and full precision.
Switch methods or return to Learning at any time. Each method retains its form, result and navigation in the same application window; background results stay with the method that started them.
"""

TRACE_COLUMNS = [
    "k",
    "xₖ",
    "yₖ",
    "f(xₖ,yₖ)",
    "∂f/∂x",
    "∂f/∂y",
    "||∇f||₂",
    "Δx",
    "Δy",
    "xₖ₊₁",
    "yₖ₊₁",
]


def render_summary(panel):
    r, f = panel.result, panel.fmt
    p, request = r.final, r.request
    panel.snapshot.set(
        f"{request.algorithm_name}: f(x,y) = {request.expression}  ·  (x₀,y₀) = ({f(request.x0)}, {f(request.y0)})  ·  α = {f(request.alpha)}  ·  ε = {f(request.tolerance)}"
    )
    detail = {
        "gradient_tolerance": "The gradient norm test passed. A stationary point alone does not prove a minimum.",
        "iteration_limit": "The update budget ended before the gradient norm test passed.",
        "numerical_limit": "The next pair failed the numerical/domain guard; neither coordinate was committed.",
        "stagnation": "Both coordinates round to their old values while the gradient norm test still fails.",
    }[r.status]
    increases = sum(b.value > a.value for a, b in zip(r.points, r.points[1:]))
    outside = sum(not in_view(q, request) for q in r.points)
    text = f"{STATUS[r.status]}\nEstimate (x,y) = ({f(p.x)}, {f(p.y)})     f(x,y) = {f(p.value)}\n∇f = ({f(p.gradient_x)}, {f(p.gradient_y)})     ||∇f||₂ = {f(p.gradient_norm)}\nCompleted updates: {p.iteration:,}   ·   ||∇f||₂ ≤ ε: {p.gradient_norm <= request.tolerance}\n{detail}\nObjective increases: {increases:,}   ·   Estimates outside heatmap: {outside:,}\nPlot bounds are view settings; they do not constrain either coordinate."
    panel.app._set_text(panel.summary_text, text)


def in_view(point, request):
    return (
        request.plot_min <= point.x <= request.plot_max
        and request.plot_y_min <= point.y <= request.plot_y_max
    )


def worked_step(panel):
    r, f, i = panel.result, panel.fmt, panel.step_index
    p, request = r.points[i], r.request
    lines = [
        f"ITERATION k = {p.iteration}\nf(x,y) = {request.expression}",
        f"OLD pair (xₖ,yₖ) = ({f(p.x)}, {f(p.y)})\nf(xₖ,yₖ) = {f(p.value)}",
        f"Partial derivatives at the SAME OLD pair\n∂f/∂x(xₖ,yₖ) = {f(p.gradient_x)}\n∂f/∂y(xₖ,yₖ) = {f(p.gradient_y)}\n∇f = ({f(p.gradient_x)}, {f(p.gradient_y)})\n−∇f = ({f(-p.gradient_x)}, {f(-p.gradient_y)})",
        f"Stopping test: ||∇f||₂ ≤ ε\nsqrt(({f(p.gradient_x)})² + ({f(p.gradient_y)})²) = {f(p.gradient_norm)}\n{f(p.gradient_norm)} ≤ {f(request.tolerance)} → {str(p.gradient_norm <= request.tolerance).upper()}",
    ]
    if i + 1 < len(r.points):
        nxt = r.points[i + 1]
        lines.extend(
            [
                f"Δx = −α ∂f/∂x = −({f(request.alpha)}) × ({f(p.gradient_x)}) = {f(-request.alpha * p.gradient_x)}\nΔy = −α ∂f/∂y = −({f(request.alpha)}) × ({f(p.gradient_y)}) = {f(-request.alpha * p.gradient_y)}",
                f"xₖ₊₁ = xₖ − α ∂f/∂x(xₖ,yₖ)\n     = ({f(p.x)}) − ({f(request.alpha)}) × ({f(p.gradient_x)})\n     = {f(nxt.x)}",
                f"yₖ₊₁ = yₖ − α ∂f/∂y(xₖ,yₖ)\n     = ({f(p.y)}) − ({f(request.alpha)}) × ({f(p.gradient_y)})\n     = {f(nxt.y)}",
                f"Commit BOTH coordinates together: ({f(nxt.x)}, {f(nxt.y)})\nf(xₖ₊₁,yₖ₊₁) = {f(nxt.value)}\nUse this new pair in iteration {nxt.iteration}. α stays {f(request.alpha)}.",
            ]
        )
    else:
        lines.append(f"Stop: {STATUS[r.status]}. No further update was committed.")
    return "\n\n".join(lines)


def trace_values(panel, index):
    r, f = panel.result, panel.fmt
    p = r.points[index]
    nxt = r.points[index + 1] if index + 1 < len(r.points) else None
    return [
        p.iteration,
        f(p.x),
        f(p.y),
        f(p.value),
        f(p.gradient_x),
        f(p.gradient_y),
        f(p.gradient_norm),
        f(-r.request.alpha * p.gradient_x) if nxt else "—",
        f(-r.request.alpha * p.gradient_y) if nxt else "—",
        f(nxt.x) if nxt else "—",
        f(nxt.y) if nxt else "—",
    ]


def draw_chart(panel, history_title="OBJECTIVE BY ITERATION"):
    c, r = panel.chart, panel.result
    request = r.request
    width, height = c.winfo_width(), c.winfo_height()
    ink, muted, teal = "#182d40", "#617284", "#087e76"
    left, right, top, bottom = 65, width / 2 - 22, 35, height - 55
    phase_rectangle = (left, right, top, bottom)
    c.create_text(
        left,
        12,
        text="FUNCTION HEATMAP & PATH",
        anchor="w",
        fill=muted,
        font=("Arial", 9, "bold"),
    )
    valid = [v for row in r.grid for v in row if v is not None]
    low, high = (min(valid), max(valid)) if valid else (0, 1)
    n = len(r.grid)
    colors = []
    for i in range(64):
        t = i / 63
        colors.append(
            "#"
            + "".join(
                f"{round(a + (b - a) * t):02x}"
                for a, b in zip((235, 247, 241), (35, 76, 113))
            )
        )
    for row, values in enumerate(r.grid):
        for col, value in enumerate(values):
            color = (
                "#e1e4e7"
                if value is None
                else colors[
                    0
                    if low == high
                    else max(0, min(63, int((value - low) / (high - low) * 63)))
                ]
            )
            c.create_rectangle(
                left + col / n * (right - left),
                bottom - (row + 1) / n * (bottom - top),
                left + (col + 1) / n * (right - left),
                bottom - row / n * (bottom - top),
                fill=color,
                outline="",
                tags="heatmap",
            )

    def xy(p):
        return (
            left
            + (p.x - request.plot_min)
            / (request.plot_max - request.plot_min)
            * (right - left),
            bottom
            - (p.y - request.plot_y_min)
            / (request.plot_y_max - request.plot_y_min)
            * (bottom - top),
        )

    stride = max(1, math.ceil(len(r.points) / 800))
    indices = list(range(0, len(r.points), stride))
    if indices[-1] != len(r.points) - 1:
        indices.append(len(r.points) - 1)
    previous = None
    for i in indices:
        p = r.points[i]
        if not in_view(p, request):
            previous = None
            continue
        point = xy(p)
        if previous:
            c.create_line(*previous, *point, fill="#c45521", width=2, tags="path")
        previous = point
    for label, p, color in (
        ("start", r.points[0], ink),
        ("final", r.final, teal),
        ("selected", r.points[panel.step_index], "#e8911b"),
    ):
        if in_view(p, request):
            px, py = xy(p)
            radius = 5 if label == "selected" else 3
            c.create_oval(
                px - radius,
                py - radius,
                px + radius,
                py + radius,
                fill=color,
                outline="white",
                tags=label,
            )
    c.create_rectangle(left, top, right, bottom, outline="#aab9c4")
    for x, label, anchor in (
        (left, request.plot_min, "w"),
        (right, request.plot_max, "e"),
    ):
        c.create_text(
            x,
            bottom + 12,
            text=f"x = {label:.4g}",
            anchor=anchor,
            fill=muted,
            font=("Arial", 9),
        )
    for y, label in ((bottom, request.plot_y_min), (top, request.plot_y_max)):
        c.create_text(
            left - 7,
            y,
            text=f"y {label:.3g}",
            anchor="e",
            fill=muted,
            font=("Arial", 9),
        )
    legend = (
        f"f: {low:.3g} (pale) → {high:.3g} (dark)"
        if valid
        else "No valid function samples"
    )
    c.create_text(
        (left + right) / 2, bottom + 34, text=legend, fill=muted, font=("Arial", 8)
    )
    # Complete objective history uses all iterates for scaling; drawing is downsampled.
    left, right = width / 2 + 65, width - 22
    c.create_text(
        left,
        12,
        text=history_title,
        anchor="w",
        fill=muted,
        font=("Arial", 9, "bold"),
    )
    low, high = min(p.value for p in r.points), max(p.value for p in r.points)
    pad = max(abs(low), abs(high), 1) * 0.05 if low == high else (high - low) * 0.08
    low, high = low - pad, high + pad

    def history_xy(p):
        return (
            left + p.iteration / max(1, r.final.iteration) * (right - left),
            bottom - (p.value - low) / (high - low) * (bottom - top),
        )

    for j in range(3):
        value = high - (high - low) * j / 2
        py = top + (bottom - top) * j / 2
        c.create_line(left, py, right, py, fill="#e3eaee")
        c.create_text(
            left - 7, py, text=f"{value:.3g}", anchor="e", fill=muted, font=("Arial", 9)
        )
    coordinates = [v for i in indices for v in history_xy(r.points[i])]
    if len(coordinates) > 2:
        c.create_line(*coordinates, fill=ink, width=2, tags="objective")
    for p in (r.final, r.points[panel.step_index]):
        px, py = history_xy(p)
        c.create_oval(px - 4, py - 4, px + 4, py + 4, fill=teal, outline="white")
    c.create_text(left, bottom + 12, text="0", anchor="w", fill=muted)
    c.create_text(
        right, bottom + 12, text=f"k = {r.final.iteration:,}", anchor="e", fill=muted
    )
    return phase_rectangle
