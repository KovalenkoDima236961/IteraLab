"""Normalized-vector formulas, complete line-search tables and probe overlays."""

import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..algorithms.normalized_gradient import NormalizedRequest, export_line_searches
from ..algorithms.optimization import STATUS
from . import steepest_view

PAGE_SIZE = 200
NOTES = """GRADIENTNÁ METÓDA S NORMALIZOVANÝM VEKTOROM · TWO DIMENSIONS

The supplied slide specifies:
1. Gradient at the old point: gₖ=∇f(Pₖ)=(fₓ(Pₖ),fᵧ(Pₖ)).
2. Normalized descending direction: Sₖ=−gₖ/||gₖ||₂.
3. Line function φ(γ)=f(Pₖ+γSₖ). Solve φ′(γ)=0 for γₖ>0.
4. New point: Pₖ₊₁=Pₖ+γₖSₖ.

This is a separate algorithm. The gradient is normalized and γ is determined anew by a line search at each iteration. Because S is a unit vector, γ is the step's Euclidean length, up to floating-point rounding. Both new coordinates use the same OLD point and direction. There is no editable fixed learning rate in this method.

Automatic derivatives
Rust calculates both analytic partial derivatives using first-order forward automatic differentiation. No Hessian or finite-difference step is required. The line derivative is
    φ′(γ)=∇f(Pₖ+γSₖ)·Sₖ.
At γ=0 it is −||gₖ||₂, apart from rounding, and is negative for a nonzero gradient. Scaling the two gradient components before normalization avoids loss of the direction for very small magnitudes.

Numerical interpretation of φ′(γ)=0
The slide does not give a general symbolic equation solver. This implementation uses a recorded, numerical derivative-root search:
• Start with the entered positive initial γ PROBE, not a fixed accepted step.
• Double γ while φ′ remains negative, until a positive derivative brackets a root.
• Bisect a bracket whose lower derivative is negative and upper derivative positive.
• Invalid function/derivative trials bound the searched domain; refine toward valid points inside that boundary. Every invalid trial remains in the table.
• Accept a positive trial only when |φ′(γ)|/||gₖ||₂ ≤ the relative line tolerance (or φ′ is exactly zero), it changes the representable point, and f does not increase.
The equation is therefore solved to the explicitly displayed numerical tolerance, not asserted to have a closed-form exact solution. A failed search never substitutes a fixed-rate or merely decreasing step.

The search assumes a continuous line derivative on the searched interval. For a convex or unimodal line function its stationary point gives the line minimum. On nonconvex functions sampling can skip roots, and a stationary root alone need not be the first or global minimum. A stationary trial with a larger objective is explicitly rejected. This procedure is local and does not prove global optimality.

Worked circular example
f=x²+y²; P₀=(3,4), g₀=(6,8), ||g₀||=10.
S₀=(−0.6,−0.8), φ(γ)=(3−0.6γ)²+(4−0.8γ)²=25−10γ+γ².
φ′(γ)=−10+2γ=0 gives γ₀=5.
P₁=(3,4)+5(−0.6,−0.8)=(0,0).
From initial probe 1, the bracket search tries 1,2,4,8; bisection tries 6, then 5 and finds the exact root in this example.

Outer stopping convention
The slide gives no outer stopping test. As in the existing 2D gradient workspace, this worksheet explicitly uses ||∇f(Pₖ)||₂ ≤ ε, including at the initial point, or the selected maximum number of outer updates. This test precedes normalization: zero gradients are never divided. It can identify a stationary maximum or saddle as well as a minimum. The line tolerance is a separate, dimensionless derivative residual and does not replace the outer gradient test.

Search outcomes
Evaluation limit: the per-search trial budget ended without an accepted stationary point.
No bracket: no forward stationary root was bracketed before a domain/numerical boundary or γ=1e100.
Precision limit: the bracket cannot be refined at floating-point precision, or the accepted point would round to the old pair.
Objective increase: a stationary trial increased f, so no outer update was committed.
These statuses retain the last valid outer point, direction and all trial records. They are not convergence. Invalid initial data or a cancelled run retains the previous completed result.

Controls and limits
Set f(x,y), x₀, y₀, outer tolerance and iteration limit, initial γ probe, relative line tolerance (strictly between 0 and 1), and maximum trial evaluations per line search (1–2048). The γ=0 row reuses the old point and does not consume a trial evaluation. The complete trace budget is at most 1,000,000 outer-budget × trial-budget evaluations.
Use x, y, pi, e, + − * / ^ or **, parentheses, sin, cos, tan, exp, ln/log and sqrt; multiplication is explicit, angles are radians and log is natural. Initial coordinates, plot bounds and initial γ probe have magnitude at most 1e50. Valid expression values, partial derivatives and visited coordinates have magnitude at most 1e100 and must be finite. Numerical/domain-invalid trials are recorded without function/derivative values. Plot ranges affect the view only, never the line search.

Worked steps displays g, its norm, S, φ, the chain-rule line derivative, accepted γ and both new coordinates. Line search shows every bracket/bisection/domain trial for the selected outer point, with pagination. Select a trial to highlight it on the heatmap. Export line searches CSV saves ALL searches, not only the selected point/page. Export all CSV saves the complete outer history with directions and accepted γ values. Both exports retain the immutable run parameters and full precision. Each algorithm keeps its own form and result in the same application window.
"""

TRACE_COLUMNS = [
    "k",
    "xₖ",
    "yₖ",
    "f(xₖ,yₖ)",
    "∂f/∂x",
    "∂f/∂y",
    "||∇f||₂",
    "Sₓ",
    "Sᵧ",
    "γₖ",
    "φ′(γₖ)",
    "Line trials",
    "Search status",
]

LINE_NOTES = """STEEPEST DESCENT WITH LINE SEARCH · CV5

At the old point P, calculate g=∇f(P) and the unnormalized direction d=−g.
Set φ(α)=f(P+αd); solve φ′(α)=∇f(P+αd)·d=0 for a positive step.
Commit P_next=P+αd. Recalculate the gradient and repeat.

The initial step control is a probe. Bracketing and bisection solve the line derivative to the selected relative residual |φ′(α)|/|φ′(0)|, with a recorded evaluation budget. A failure retains the last valid point and all trials. Acceptance also requires a changed point and a nonincreasing objective. This local search does not prove global optimality on a nonconvex line.

The outer test is ||∇f||₂ ≤ ε, before any search; the iteration budget counts committed updates. The scalar step differs from normalized-gradient γ: α multiplies the full negative gradient, and step distance is α||g||₂. On an exactly solved line the two parameterizations reach the same point.

CV5 example: f=2*x^2+x*y+3*y^2-4*x+2*y, P₀=(0,0).
g₀=(−4,2); d₀=(4,−2); φ(α)=36α²−20α.
α₀=5/18; P₁=(10/9,−5/9); f(P₁)=−25/9.
The true minimum is (26/23,−12/23), f=−64/23.

Both coordinates use the old point and direction. Plot bounds are view settings. Tables and CSV exports retain all outer points, directions, scalar steps and search trials at full precision.
"""


def current_search(panel):
    if isinstance(
        getattr(panel.result, "request", None), NormalizedRequest
    ) and panel.step_index < len(panel.result.searches):
        return panel.result.searches[panel.step_index]
    return None


def render_summary(panel):
    r, f = panel.result, panel.fmt
    p = r.final
    panel.snapshot.set(
        f"{r.request.algorithm_name}: f(x,y) = {r.request.expression}  ·  P₀=({f(r.request.x0)}, {f(r.request.y0)})  ·  ε={f(r.request.tolerance)}"
    )
    detail = {
        "gradient_tolerance": "The outer gradient norm test passed. Stationarity alone does not prove a minimum.",
        "iteration_limit": "The outer update budget ended before the gradient test passed.",
        "line_search_limit": "The line-search trial budget ended without an accepted stationary point.",
        "line_search_failed": "No forward stationary point was bracketed before the domain/numerical boundary.",
        "line_search_precision": "The line search reached floating-point resolution before an acceptable new point.",
        "line_search_increase": "The stationary line trial increased f and was rejected.",
    }[r.status]
    trials = sum(len(s.samples) - 1 for s in r.searches)
    panel.app._set_text(
        panel.summary_text,
        f"{STATUS[r.status]}\nEstimate (x,y)=({f(p.x)}, {f(p.y)})     f(x,y)={f(p.value)}\n∇f=({f(p.gradient_x)}, {f(p.gradient_y)})     ||∇f||₂={f(p.gradient_norm)}\nCompleted updates: {p.iteration:,}   ·   Line trial evaluations: {trials:,}\n{detail}\nLine residual tolerance: {f(r.request.line_tolerance)}; initial scalar probe: {f(r.request.initial_gamma)}\nDirection: {'normalized negative gradient' if r.request.normalize else 'full negative gradient d=−∇f'}. Each accepted scalar solves φ′≈0.",
    )


def worked_step(panel):
    r, f, i = panel.result, panel.fmt, panel.step_index
    p = r.points[i]
    lines = [
        f"ITERATION k={p.iteration}\nf(x,y)={r.request.expression}",
        f"OLD point Pₖ=({f(p.x)}, {f(p.y)})\n∇f(Pₖ)=({f(p.gradient_x)}, {f(p.gradient_y)})\n||∇f(Pₖ)||₂={f(p.gradient_norm)}",
        f"Outer stopping test: {f(p.gradient_norm)} ≤ {f(r.request.tolerance)} → {str(p.gradient_norm <= r.request.tolerance).upper()}",
    ]
    search = current_search(panel)
    if search:
        sx, sy = search.direction_x, search.direction_y
        lines.extend(
            [
                f"NORMALIZED DIRECTION\nSₖ=−∇f(Pₖ)/||∇f(Pₖ)||₂\n  =−({f(p.gradient_x)}, {f(p.gradient_y)})/{f(p.gradient_norm)}\n  =({f(sx)}, {f(sy)})"
                if r.request.normalize
                else f"STEEPEST DIRECTION (unnormalized)\ndₖ=−∇f(Pₖ)=({f(sx)}, {f(sy)})\nThe scalar γ below is the optimized αₖ multiplying dₖ.",
                f"LINE FUNCTION\nφ(γ)=f(Pₖ+γSₖ)\n    =f(({f(p.x)})+γ({f(sx)}), ({f(p.y)})+γ({f(sy)}))\nφ′(γ)=fₓ(Pₖ+γSₖ)({f(sx)})+fᵧ(Pₖ+γSₖ)({f(sy)})\nφ′(0)={f(search.samples[0].derivative)}",
                f"Solve φ′(γ)=0 numerically.\nAcceptance residual: |φ′(γ)|/|φ′(0)| ≤ {f(r.request.line_tolerance)}\nSearch status: {search.status}; trial evaluations: {len(search.samples) - 1}\nInspect every probe in the Line search tab.",
            ]
        )
        if search.accepted:
            a = search.accepted
            nxt = r.points[i + 1]
            lines.extend(
                [
                    f"ACCEPTED LINE STEP\nγₖ={f(a.gamma)}\nφ′(γₖ)={f(a.derivative)}\nφ(γₖ)={f(a.value)} ≤ φ(0)={f(p.value)}",
                    f"NEW POINT\nPₖ₊₁=Pₖ+γₖSₖ\nxₖ₊₁=({f(p.x)})+({f(a.gamma)})({f(sx)})={f(nxt.x)}\nyₖ₊₁=({f(p.y)})+({f(a.gamma)})({f(sy)})={f(nxt.y)}\nCommit both coordinates together; recompute the gradient at the new point.",
                ]
            )
        else:
            lines.append(
                "The search did not produce an accepted stationary line point. The old point is retained."
            )
    elif p.gradient_norm <= r.request.tolerance:
        lines.append(
            "The gradient test passed BEFORE normalization. No division or line search is needed."
        )
    else:
        lines.append(
            "The outer iteration budget is exhausted; no further line search was started."
        )
    if i == len(r.points) - 1:
        lines.append(f"Stop: {STATUS[r.status]}.")
    return "\n\n".join(lines)


def trace_values(panel, index):
    r, f = panel.result, panel.fmt
    p = r.points[index]
    s = r.searches[index] if index < len(r.searches) else None
    a = s.accepted if s else None
    return [
        p.iteration,
        f(p.x),
        f(p.y),
        f(p.value),
        f(p.gradient_x),
        f(p.gradient_y),
        f(p.gradient_norm),
        f(s.direction_x) if s else "—",
        f(s.direction_y) if s else "—",
        f(a.gamma) if a else "—",
        f(a.derivative) if a else "—",
        len(s.samples) - 1 if s else "—",
        s.status if s else "—",
    ]


def build_line_tab(panel):
    panel.line_tab = ttk.Frame(panel.notebook, padding=14, style="White.TFrame")
    panel.notebook.add(panel.line_tab, text="Line search")
    panel.notebook.hide(panel.line_tab)
    panel.line_summary = tk.StringVar(
        value="Run a gradient line-search method to inspect its trials."
    )
    ttk.Label(
        panel.line_tab,
        textvariable=panel.line_summary,
        background="white",
        wraplength=780,
    ).pack(fill="x", pady=(0, 10))
    bar = ttk.Frame(panel.line_tab, style="White.TFrame")
    bar.pack(fill="x", pady=(0, 10))
    panel.line_prev = ttk.Button(
        bar,
        text="← Page",
        command=lambda: change_line_page(panel, -1),
        state="disabled",
    )
    panel.line_prev.pack(side="left")
    panel.line_next = ttk.Button(
        bar, text="Page →", command=lambda: change_line_page(panel, 1), state="disabled"
    )
    panel.line_next.pack(side="left", padx=6)
    panel.line_page_text = tk.StringVar(value="No line trials")
    ttk.Label(bar, textvariable=panel.line_page_text, background="white").pack(
        side="left", padx=6
    )
    panel.line_export_button = ttk.Button(
        bar,
        text="Export line searches CSV…",
        command=lambda: save_lines(panel),
        state="disabled",
    )
    panel.line_export_button.pack(side="right")
    panel.line_tree = panel.app._table(panel.line_tab)
    panel.line_tree.bind("<<TreeviewSelect>>", lambda _: select_line_sample(panel))
    panel.line_page = 0
    panel.line_sample_index = None
    panel.line_context = None


def render_line_search(panel):
    context = (id(panel.result), panel.step_index)
    if panel.line_context != context:
        panel.line_page = 0
        panel.line_sample_index = None
        panel.line_context = context
    search = current_search(panel)
    samples = search.samples if search else ()
    panel.app._columns(
        panel.line_tree,
        ["n", "Phase", "γ", "x(γ)", "y(γ)", "φ(γ)", "fₓ", "fᵧ", "φ′(γ)", "Valid"],
    )
    if search:
        accepted = search.accepted
        chosen = (
            f"accepted γ={panel.fmt(accepted.gamma)}"
            if accepted
            else "no accepted step"
        )
        panel.line_summary.set(
            f"Outer k={search.iteration} · S=({panel.fmt(search.direction_x)}, {panel.fmt(search.direction_y)}) · {search.status} · {chosen}\nγ=0 reuses the old gradient; invalid trials bound the searched domain and have no function/derivative values."
        )
    else:
        panel.line_summary.set(
            f"No line search at outer k={panel.step_index}: the gradient test or outer iteration budget stopped before normalization."
        )
    start = panel.line_page * PAGE_SIZE
    for sample in samples[start : start + PAGE_SIZE]:
        panel.line_tree.insert(
            "",
            "end",
            iid=str(sample.number),
            values=[
                sample.number,
                sample.phase,
                panel.fmt(sample.gamma),
                panel.fmt(sample.x),
                panel.fmt(sample.y),
                *[
                    panel.fmt(v) if v is not None else "undefined"
                    for v in (
                        sample.value,
                        sample.gradient_x,
                        sample.gradient_y,
                        sample.derivative,
                    )
                ],
                "Yes" if sample.valid else "No",
            ],
        )
    total = max(1, math.ceil(len(samples) / PAGE_SIZE))
    panel.line_page_text.set(
        f"Page {panel.line_page + 1}/{total} · {len(samples):,} samples"
    )
    panel.line_prev.configure(state="normal" if panel.line_page else "disabled")
    panel.line_next.configure(
        state="normal" if panel.line_page + 1 < total else "disabled"
    )
    panel.line_export_button.configure(
        state="normal" if panel.result.searches else "disabled"
    )


def change_line_page(panel, delta):
    search = current_search(panel)
    if search:
        panel.line_page = max(
            0, min(panel.line_page + delta, (len(search.samples) - 1) // PAGE_SIZE)
        )
        render_line_search(panel)


def select_line_sample(panel):
    selection = panel.line_tree.selection()
    if selection:
        panel.line_sample_index = int(selection[0])
        panel.draw_chart()


def save_lines(panel):
    if (
        not isinstance(getattr(panel.result, "request", None), NormalizedRequest)
        or not panel.result.searches
    ):
        return
    path = filedialog.asksaveasfilename(
        parent=panel.root,
        title="Export all gradient line searches",
        initialfile="normalized_gradient_line_searches.csv"
        if panel.result.request.normalize
        else "steepest_line_searches.csv",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv")],
    )
    if path:
        try:
            export_line_searches(path, panel.result)
            panel.status.set(
                f"Saved {Path(path).name} · all line searches at full precision"
            )
        except OSError as exc:
            messagebox.showerror("Could not save CSV", str(exc), parent=panel.root)


def draw_chart(panel):
    left, right, top, bottom = steepest_view.draw_chart(panel)
    search = current_search(panel)
    if not search:
        return
    r, c = panel.result, panel.chart
    q = r.request
    p = r.points[panel.step_index]

    def visible(x, y):
        return q.plot_min <= x <= q.plot_max and q.plot_y_min <= y <= q.plot_y_max

    def xy(x, y):
        return (
            left + (x - q.plot_min) / (q.plot_max - q.plot_min) * (right - left),
            bottom
            - (y - q.plot_y_min) / (q.plot_y_max - q.plot_y_min) * (bottom - top),
        )

    end = (p.x + search.direction_x, p.y + search.direction_y)
    if visible(p.x, p.y) and visible(*end):
        c.create_line(
            *xy(p.x, p.y),
            *xy(*end),
            arrow="last",
            fill="#087e76",
            width=2,
            tags="normalized_direction",
        )
    for sample in search.samples[:: max(1, math.ceil(len(search.samples) / 120))]:
        if visible(sample.x, sample.y):
            px, py = xy(sample.x, sample.y)
            c.create_oval(
                px - 2,
                py - 2,
                px + 2,
                py + 2,
                fill="#325caf" if sample.valid else "#c45532",
                outline="",
                tags="line_probe",
            )
    for sample, color, tag in (
        (search.accepted, "#087e76", "accepted_gamma"),
        (
            search.samples[panel.line_sample_index]
            if panel.line_sample_index is not None
            and panel.line_sample_index < len(search.samples)
            else None,
            "#e8911b",
            "selected_probe",
        ),
    ):
        if sample and visible(sample.x, sample.y):
            px, py = xy(sample.x, sample.y)
            c.create_oval(
                px - 5, py - 5, px + 5, py + 5, fill=color, outline="white", tags=tag
            )
