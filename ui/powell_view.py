import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import steepest_view
from ..algorithms.optimization import STATUS
from ..algorithms.powell import PowellRequest, export_line_searches

PAGE_SIZE = 200
NOTES = """POWELLOVA METÓDA · TWO DIMENSIONS

This worksheet follows the supplied direction-shift algorithm.
Initialize X₀=(x₀,y₀), d₁=(1,0), d₂=(0,1).
In each cycle, save its starting point Q₀ and perform:
1. Q₁ = Q₀ + α₁d₁, minimizing f(Q₀+αd₁) along the whole line.
2. Q₂ = Q₁ + α₂d₂, minimizing f(Q₁+αd₂).
3. Form d₃ = Q₂ − Q₀. This is the raw displacement; do not normalize it.
4. Q₃ = Q₂ + α₃d₃, minimizing f(Q₂+αd₃).
5. Carry forward d₁ ← old d₂, d₂ ← d₃, and the point Q₀ ← Q₃.
Repeat. The third search starts at Q₂, not at the cycle's initial point.
There is no largest-decrease replacement, extrapolation test, gradient or Hessian in this variant. α may be positive, negative or zero. A zero displacement direction keeps the same point and records a zero_direction search.

Derivative-free line search
The notes allow any one-dimensional minimizer (including Golden Search). This implementation uses Brent minimization: bracket a line minimum using function values, then use safeguarded quadratic interpolation with golden-section steps when interpolation is unsuitable. Both signs of the initial α probe are tried; this control is a probe, not a fixed accepted step. When a probe leaves the real/numerical domain, refine toward valid samples. A valid three-point minimum bracket is required before minimization. Every actual trial is recorded, including invalid trials. The α=0 row reuses the starting value and consumes no trial evaluation.
The line tolerance controls numerical α resolution: τ(1+|α|), with a floor of 8 machine epsilons at that scale. This is an approximate local line minimum, not an exact symbolic solution. A continuous unimodal line function is the intended case; arbitrary nonconvex or discontinuous functions need not yield a global line minimum.
On a failed line search, no trial from that search is committed. Earlier successful moves in the unfinished cycle remain visible; only complete cycles count toward the stopping tests and direction update. Evaluation limits and domain/numerical/precision failures are reported separately from convergence.

Stopping criteria from the notes
After EACH COMPLETE three-search cycle, compare its Q₀ and Q₃:
• |f(Q₀)−f(Q₃)| < ε, OR
• sqrt((Q₃ₓ−Q₀ₓ)²+(Q₃ᵧ−Q₀ᵧ)²) < ε.
Both are strict tests and both are displayed. If both pass, the function-change status takes precedence. The cycle limit is a maximum number of complete cycles (k=1,…,Kmax); no extra cycle is started. No convergence test is made before the first cycle. A small change can occur away from a minimum, especially on flat regions or after directions become dependent. These tests do not prove a minimum.
The supplied exercise reaches its minimum after two cycles. With a larger cycle budget, a third cycle confirms the small-change test; reaching a point alone is not an early stopping test in the notes.

SUPPLIED EXERCISE · reference values
f(x,y)=x²+2y²+2xy−4x−6y; X₀=(0,0).
Cycle 1: d₁=(1,0), d₂=(0,1).
X₁=(2,0), α₁=2, f=−4.
X₂=(2,1/2), α₂=1/2, f=−9/2.
d₃=X₂−X₀=(2,1/2).
X₃=(22/13,11/26), α₃=−2/13, f=−121/26.
Cycle 2: d₁=(0,1), d₂=(2,1/2).
X₄=(22/13,17/26), α₄=3/13, f=−1609/338.
X₅=(250/169,203/338), α₅=−18/169, f=−21241/4394.
d₄=X₅−X₃=(−36/169,30/169).
X₆=(1,1), α₆=9/4, f=−5.
The preset loads this function, start and plotting region. Set the cycle limit to 2 to view exactly X₀ through X₆, or leave the default to include the stopping-confirmation cycle. Worked steps substitute the calculated values at the selected move.

Controls and display
Set f(x,y), its starting pair, cycle tolerance and cycle budget, initial α probe, line α tolerance and per-search trial budget. Directions always initialize as in the notes. No derivatives are evaluated, so supported nonsmooth functions such as sqrt(x²)+sqrt(y²) can be used. Use explicit multiplication, x/y, pi/e, + − * / ^, sin/cos/tan/exp/ln/log/sqrt; log is natural and angles are radians.
Initial coordinates/probe/plot bounds have magnitude at most 1e50. Visited coordinates, expression intermediates and direction components are limited to 1e100 and must be finite. At most 3 × cycle-budget × trial-budget = 1,000,000 line trials may be requested. Plot bounds change the view only.
Navigate individual moves Xᵢ with the slider, keyboard or table. Each full cycle has three moves. The Powell line search tab shows the selected move's complete trial history with pagination and probe highlighting. Export all CSV saves all committed points, directions, accepted α and complete-cycle diagnostics; Export line searches CSV saves every search and sample, including a failed final attempt. Both exports use the completed run's immutable inputs and full precision. Learning and the other methods keep their own forms and results in this same window.
"""

TRACE_COLUMNS = [
    "Move",
    "Cycle",
    "Search",
    "x",
    "y",
    "f(x,y)",
    "dₓ",
    "dᵧ",
    "α",
    "Next x",
    "Next y",
    "Trials",
    "Search status",
]


def current_search(panel):
    if isinstance(
        getattr(panel.result, "request", None), PowellRequest
    ) and panel.step_index < len(panel.result.searches):
        return panel.result.searches[panel.step_index]
    return None


def render_summary(panel):
    r, f = panel.result, panel.fmt
    p = r.final
    last = r.cycles[-1] if r.cycles else None
    panel.snapshot.set(
        f"{r.request.algorithm_name}: f(x,y)={r.request.expression} · X₀=({f(r.request.x0)}, {f(r.request.y0)}) · ε={f(r.request.tolerance)}"
    )
    diagnostics = (
        f"Last cycle: |Δf|={f(last.function_change)} < ε: {last.function_change < r.request.tolerance}; displacement={f(last.displacement)} < ε: {last.displacement < r.request.tolerance}"
        if last
        else "No complete cycle; no cycle stopping test was performed."
    )
    panel.app._set_text(
        panel.summary_text,
        f"{STATUS[r.status]}\nEstimate (x,y)=({f(p.x)}, {f(p.y)})     f(x,y)={f(p.value)}\nComplete cycles: {len(r.cycles):,} · Committed line moves: {p.iteration:,} · Trial evaluations: {sum(len(s.samples) - 1 for s in r.searches):,}\n{diagnostics}\nDirections initialize (1,0),(0,1); each cycle replaces them with (old d₂, Q₂−Q₀).\nFunction values only; a failed search retains the last committed move. Small changes alone do not prove a minimum.",
    )


def worked_step(panel):
    r, f, i = panel.result, panel.fmt, panel.step_index
    p = r.points[i]
    base = 3 * (i // 3)
    search = current_search(panel)
    lines = [
        f"POWELL · MOVE X{i}\nf(x,y)={r.request.expression}",
        f"Current point X{i}=({f(p.x)}, {f(p.y)}), f={f(p.value)}",
    ]
    if search:
        start = r.points[base]
        lines.append(
            f"CYCLE {search.cycle}, SEARCH {search.stage}/3\nCycle start Q₀=X{base}=({f(start.x)}, {f(start.y)})"
        )
        if search.stage == 3:
            lines.append(
                f"NEW DIRECTION\nd₃=Q₂−Q₀=X{i}−X{base}\n  =({f(p.x)}−{f(start.x)}, {f(p.y)}−{f(start.y)})\n  =({f(search.direction_x)}, {f(search.direction_y)})\nKeep this displacement without normalization."
            )
        lines.append(
            f"LINE MINIMIZATION\nd=({f(search.direction_x)}, {f(search.direction_y)})\nφ(α)=f(X{i}+αd)\n    =f(({f(p.x)})+α({f(search.direction_x)}), ({f(p.y)})+α({f(search.direction_y)}))\nFind a local minimum over signed α using function values.\nSearch status: {search.status}; trials: {len(search.samples) - 1}.\nInspect every trial in Powell line search."
        )
        a = search.accepted
        if a:
            lines.append(
                f"ACCEPTED MOVE\nα={f(a.alpha)}\nX{i + 1}=X{i}+αd\nx=({f(p.x)})+({f(a.alpha)})({f(search.direction_x)})={f(a.x)}\ny=({f(p.y)})+({f(a.alpha)})({f(search.direction_y)})={f(a.y)}\nf(X{i + 1})={f(a.value)} ≤ f(X{i})={f(p.value)}"
            )
            if search.status == "zero_direction":
                lines.append(
                    "The direction is exactly zero. This search keeps the current point, with α=0 and no trial evaluations."
                )
        else:
            lines.append(
                "No line minimum was accepted. No trial from this search was committed; the preceding successful moves remain in the trace."
            )
    c = (
        r.cycles[base // 3]
        if base // 3 < len(r.cycles)
        else (r.cycles[-1] if i == len(r.points) - 1 and r.cycles else None)
    )
    if c:
        start, end = r.points[3 * (c.number - 1)], r.points[3 * c.number]
        lines.append(
            f"COMPLETE CYCLE {c.number} · DIRECTION UPDATE\nd₁(new)=d₂(old)=({f(c.d2_x)}, {f(c.d2_y)})\nd₂(new)=d₃=({f(c.d3_x)}, {f(c.d3_y)})\nNext cycle starts at ({f(end.x)}, {f(end.y)})."
        )
        lines.append(
            f"POST-CYCLE STOPPING TESTS\n|f(Q₀)−f(Q₃)|=|{f(start.value)}−({f(end.value)})|={f(c.function_change)}\n{f(c.function_change)} < {f(r.request.tolerance)} → {str(c.function_change < r.request.tolerance).upper()}\n||Q₃−Q₀||₂={f(c.displacement)}\n{f(c.displacement)} < {f(r.request.tolerance)} → {str(c.displacement < r.request.tolerance).upper()}\nStop when either strict test passes, or when the cycle budget ends."
        )
    if i == len(r.points) - 1:
        lines.append(f"Stop: {STATUS[r.status]}. Complete cycles: {len(r.cycles)}.")
    return "\n\n".join(lines)


def trace_values(panel, index):
    r, f = panel.result, panel.fmt
    p = r.points[index]
    s = r.searches[index] if index < len(r.searches) else None
    a = s.accepted if s else None
    return [
        index,
        s.cycle if s else "—",
        s.stage if s else "—",
        f(p.x),
        f(p.y),
        f(p.value),
        f(s.direction_x) if s else "—",
        f(s.direction_y) if s else "—",
        f(a.alpha) if a else "—",
        f(a.x) if a else "—",
        f(a.y) if a else "—",
        len(s.samples) - 1 if s else "—",
        s.status if s else "—",
    ]


def build_line_tab(panel):
    panel.powell_line_tab = ttk.Frame(panel.notebook, padding=14, style="White.TFrame")
    panel.notebook.add(panel.powell_line_tab, text="Powell line search")
    panel.notebook.hide(panel.powell_line_tab)
    panel.powell_line_summary = tk.StringVar()
    ttk.Label(
        panel.powell_line_tab,
        textvariable=panel.powell_line_summary,
        background="white",
        wraplength=780,
    ).pack(fill="x", pady=(0, 10))
    bar = ttk.Frame(panel.powell_line_tab, style="White.TFrame")
    bar.pack(fill="x", pady=(0, 10))
    panel.powell_line_prev = ttk.Button(
        bar,
        text="← Page",
        command=lambda: change_line_page(panel, -1),
        state="disabled",
    )
    panel.powell_line_prev.pack(side="left")
    panel.powell_line_next = ttk.Button(
        bar, text="Page →", command=lambda: change_line_page(panel, 1), state="disabled"
    )
    panel.powell_line_next.pack(side="left", padx=6)
    panel.powell_line_page_text = tk.StringVar(value="No line trials")
    ttk.Label(bar, textvariable=panel.powell_line_page_text, background="white").pack(
        side="left", padx=6
    )
    panel.powell_line_export = ttk.Button(
        bar,
        text="Export line searches CSV…",
        command=lambda: save_lines(panel),
        state="disabled",
    )
    panel.powell_line_export.pack(side="right")
    panel.powell_line_tree = panel.app._table(panel.powell_line_tab)
    panel.powell_line_tree.bind("<<TreeviewSelect>>", lambda _: select_sample(panel))
    panel.powell_line_context = None
    panel.powell_line_page = 0
    panel.powell_sample_index = None


def render_line_search(panel):
    context = (id(panel.result), panel.step_index)
    if panel.powell_line_context != context:
        panel.powell_line_context = context
        panel.powell_line_page = 0
        panel.powell_sample_index = None
    search = current_search(panel)
    samples = search.samples if search else ()
    panel.app._columns(
        panel.powell_line_tree,
        ["n", "Phase", "α", "x(α)", "y(α)", "φ(α)", "Valid", "Accepted"],
    )
    panel.powell_line_summary.set(
        f"Cycle {search.cycle}, search {search.stage} · from X{search.iteration} · d=({panel.fmt(search.direction_x)}, {panel.fmt(search.direction_y)}) · {search.status}\nBoth signs of α are eligible. Invalid trials have no function value; α=0 reuses the old point."
        if search
        else "No outgoing line search at this point. The cycle stopping test or budget ended the calculation."
    )
    start = panel.powell_line_page * PAGE_SIZE
    for p in samples[start : start + PAGE_SIZE]:
        panel.powell_line_tree.insert(
            "",
            "end",
            iid=str(p.number),
            values=[
                p.number,
                p.phase,
                panel.fmt(p.alpha),
                panel.fmt(p.x),
                panel.fmt(p.y),
                panel.fmt(p.value) if p.valid else "undefined",
                "Yes" if p.valid else "No",
                "Yes" if p.number == search.accepted_index else "",
            ],
        )
    total = max(1, math.ceil(len(samples) / PAGE_SIZE))
    panel.powell_line_page_text.set(
        f"Page {panel.powell_line_page + 1}/{total} · {len(samples):,} samples"
    )
    panel.powell_line_prev.configure(
        state="normal" if panel.powell_line_page else "disabled"
    )
    panel.powell_line_next.configure(
        state="normal" if panel.powell_line_page + 1 < total else "disabled"
    )
    panel.powell_line_export.configure(
        state="normal" if panel.result.searches else "disabled"
    )


def change_line_page(panel, delta):
    s = current_search(panel)
    if s:
        panel.powell_line_page = max(
            0, min(panel.powell_line_page + delta, (len(s.samples) - 1) // PAGE_SIZE)
        )
        render_line_search(panel)


def select_sample(panel):
    selection = panel.powell_line_tree.selection()
    if selection:
        panel.powell_sample_index = int(selection[0])
        panel.draw_chart()


def save_lines(panel):
    if (
        not isinstance(getattr(panel.result, "request", None), PowellRequest)
        or not panel.result.searches
    ):
        return
    path = filedialog.asksaveasfilename(
        parent=panel.root,
        title="Export all Powell line searches",
        initialfile="powell_line_searches.csv",
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
    left, right, top, bottom = steepest_view.draw_chart(
        panel, history_title="OBJECTIVE BY LINE MOVE"
    )
    r, c = panel.result, panel.chart
    q = r.request

    def visible(x, y):
        return q.plot_min <= x <= q.plot_max and q.plot_y_min <= y <= q.plot_y_max

    def xy(x, y):
        return (
            left + (x - q.plot_min) / (q.plot_max - q.plot_min) * (right - left),
            bottom
            - (y - q.plot_y_min) / (q.plot_y_max - q.plot_y_min) * (bottom - top),
        )

    base = 3 * min(panel.step_index // 3, max(0, len(r.cycles) - 1))
    for i in range(base, min(base + 4, len(r.points))):
        p = r.points[i]
        if visible(p.x, p.y):
            px, py = xy(p.x, p.y)
            c.create_text(
                px + 5,
                py - 8,
                text=f"X{i}",
                anchor="sw",
                fill="#252d3c",
                tags="powell_point_label",
            )
        if i < base + 3 and i + 1 < len(r.points):
            nxt = r.points[i + 1]
            if visible(p.x, p.y) and visible(nxt.x, nxt.y):
                c.create_line(
                    *xy(p.x, p.y),
                    *xy(nxt.x, nxt.y),
                    fill=("#db603c", "#087e76", "#7541a3")[i % 3],
                    width=3,
                    arrow="last",
                    tags="powell_move",
                )
    s = current_search(panel)
    if s:
        for p in s.samples[:: max(1, math.ceil(len(s.samples) / 120))]:
            if visible(p.x, p.y):
                px, py = xy(p.x, p.y)
                c.create_oval(
                    px - 2,
                    py - 2,
                    px + 2,
                    py + 2,
                    fill="#325caf" if p.valid else "#c45532",
                    outline="",
                    tags="powell_probe",
                )
        n = panel.powell_sample_index
        if n is not None and n < len(s.samples):
            p = s.samples[n]
            if visible(p.x, p.y):
                px, py = xy(p.x, p.y)
                c.create_oval(
                    px - 5,
                    py - 5,
                    px + 5,
                    py + 5,
                    fill="#e8911b",
                    outline="white",
                    tags="powell_selected_probe",
                )
