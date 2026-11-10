"""Optimization workspace inside the single Exam lab Tk window."""

import math
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .app import BG, INK, MUTED, TEAL
from ..backend import Cancelled, InputError, number, train
from ..algorithms.optimization import (
    GradientRequest,
    NewtonRequest,
    GRADIENT,
    NEWTON,
    NEWTON2D,
    NELDER_MEAD,
    NORMALIZED,
    LINE_STEEPEST,
    POWELL,
    STEEPEST,
    METHODS,
    PRESETS,
    SOURCE_PRESET_OPTIONS,
    STATUS,
    export_trace,
)
from . import (
    newton_view,
    steepest_view,
    newton2d_view,
    nelder_mead_view,
    normalized_gradient_view,
    powell_view,
)
from ..algorithms.steepest import SteepestRequest, PRESETS as STEEPEST_PRESETS
from ..algorithms.newton2d import Newton2DRequest, PRESETS as NEWTON2D_PRESETS
from ..algorithms.nelder_mead import NelderMeadRequest, PRESETS as NELDER_MEAD_PRESETS
from ..algorithms.normalized_gradient import (
    NormalizedRequest,
    LineSteepestRequest,
    PRESETS as NORMALIZED_PRESETS,
)
from ..algorithms.powell import PowellRequest, PRESETS as POWELL_PRESETS

PAGE_SIZE = 200
NOTES = """GRADIENT METHOD · ONE DIMENSION

The supplied slide gives exactly this recurrence:
    x[k+1] = x[k] − α f′(x[k])

x[k] is the current estimate at iteration k.
α is the fixed positive step length (learning rate). The slide suggests values such as 0.01–0.5; the appropriate value depends on f.
f′(x[k]) is the slope of the function at the OLD estimate.
The minus sign moves opposite the slope, toward lower function values for a sufficiently small step.

1. Enter f(x), starting x₀ and α.
2. Evaluate f(x[k]) and f′(x[k]).
3. If |f′(x[k])| ≤ ε, stop.
4. Otherwise calculate Δx = −α f′(x[k]), then x[k+1] = x[k] + Δx.
5. Evaluate the new point and repeat, up to the chosen maximum number of updates.

The slide specifies the update, but no stopping test. This worksheet explicitly uses a gradient tolerance and an iteration budget. It never changes α, performs a line search, projects onto a plot interval, or substitutes a small-step stopping test. A tiny α can produce a tiny step while the gradient is still large.

The initial point is k=0. An already stationary start performs zero updates. The final row has no next update. A numerical/domain failure rejects that attempted update and keeps the last valid point. Floating-point stagnation means x[k+1] rounds to x[k] while the gradient tolerance still fails.

Stationary points and minima
Gradient tolerance identifies an approximate stationary point. On a nonconvex function it can also be a maximum or a flat point; this test alone does not prove a local or global minimum. Different starting points may approach different minima. A large α can oscillate or diverge. Objective increases are displayed and do not change the lecture rule.
For f(x)=x², x[k+1]=(1−2α)x[k]. From x₀=4 and α=0.1, the first estimates are 4, 3.2, 2.56, 2.048.

Function syntax
Use x, pi, e, +, −, *, /, ^ or **, parentheses, and sin, cos, tan, exp, ln (or log), sqrt. Use explicit multiplication: 2*x. Powers follow standard precedence: -x^2 means -(x^2). Trigonometric functions use radians. log is the natural logarithm.
Rust evaluates the expression and its analytic derivative together using forward automatic differentiation and the chain rule; no finite-difference step is used. Functions and derivatives must be real, finite and at most 1e100 in magnitude, including intermediate values. Variable exponents require a positive base. Nondifferentiable/domain-boundary expressions can be rejected.

Plot and trace
Plot bounds affect only the sampled function view, never the iteration. Points outside that interval remain in the complete trace and objective history. Gaps indicate invalid or numerically limited samples. Plotting may downsample long histories; tables and CSV retain all points. Select a trace row or navigate Worked steps to highlight an estimate. CSV exports preserve full precision and the immutable inputs of the completed run.

Use the persistent Learning / Optimization selector at any time. Each workspace keeps its form and completed result, and a calculation can finish while you use the other workspace. Ctrl/⌘+Enter runs the visible workspace. Alt+Left/Right navigates its worked steps.
"""


class OptimizationPanel:
    def __init__(self, app, parent):
        self.app, self.root = app, app.root
        self.frame = ttk.Frame(parent, padding=(18, 16, 18, 8))
        self.result = None
        self.method = tk.StringVar(value=GRADIENT)
        self.active_method = GRADIENT
        self.states = {}
        self.loading = False
        self.running_method = None
        self.formula = tk.StringVar(value="xₖ₊₁ = xₖ − α f′(xₖ)")
        self.heading = tk.StringVar(value="MINIMIZATION  /  FIXED-STEP GRADIENT METHOD")
        self.busy = self.closed = False
        self.events = queue.Queue()
        self.cancel_event = threading.Event()
        self.step_index = self.trace_page = 0
        self.chart_job = None
        self.status = tk.StringVar(
            value="Ready · enter f(x) and find a minimum estimate"
        )
        self.preset = tk.StringVar(value=next(iter(PRESETS)))
        self.precision = tk.StringVar(value="10")
        self.fields = {
            k: tk.StringVar(value=v)
            for k, v in {
                "expression": "(x - 2)^2 + 1",
                "x0": "6",
                "alpha": "0.1",
                "tolerance": "1e-6",
                "max_iterations": "200",
                "plot_min": "-2",
                "plot_max": "7",
                "first_derivative": "",
                "second_derivative": "",
                "y0": "3",
                "plot_y_min": "-3",
                "plot_y_max": "4",
                "b_x": "7",
                "b_y": "3",
                "c_x": "6",
                "c_y": "4",
                "initial_gamma": "1",
                "initial_step": "1",
                "line_tolerance": "1e-10",
                "max_line_evaluations": "512",
                "variant": "slides",
                "stopping": "function",
            }.items()
        }
        self.defaults = {key: var.get() for key, var in self.fields.items()}
        self._build()
        self._configure_method()
        for var in self.fields.values():
            var.trace_add("write", self._dirty)
        self.poll_job = self.root.after(60, self._poll)

    def _build(self):
        footer = ttk.Frame(self.frame, padding=(0, 6, 0, 0))
        footer.pack(side="bottom", fill="x")
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel").pack(
            side="left"
        )
        sidebar = ttk.Frame(self.frame, width=325)
        sidebar.pack(side="left", fill="y", padx=(0, 18))
        sidebar.pack_propagate(False)
        actions = ttk.Frame(sidebar, padding=(0, 10, 0, 0))
        actions.pack(side="bottom", fill="x")
        self.run_button = ttk.Button(
            actions,
            text="Find minimum · Gradient 1D",
            command=self.start,
            style="Run.TButton",
        )
        self.run_button.pack(fill="x")
        self.cancel_button = ttk.Button(
            actions, text="Cancel calculation", command=self.cancel, state="disabled"
        )
        self.cancel_button.pack(fill="x", pady=(6, 0))
        self.progress = ttk.Progressbar(actions, mode="indeterminate")
        self.progress.pack(fill="x", pady=(6, 0))
        canvas = tk.Canvas(sidebar, bg=BG, highlightthickness=0)
        scroll = ttk.Scrollbar(sidebar, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)
        form = ttk.Frame(canvas, padding=(0, 0, 10, 8))
        window = canvas.create_window((0, 0), window=form, anchor="nw")
        form.bind(
            "<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind(
            "<Configure>", lambda e: canvas.itemconfigure(window, width=e.width)
        )
        ttk.Label(form, text="OPTIMIZATION METHOD", style="Section.TLabel").pack(
            anchor="w"
        )
        selector = ttk.Combobox(
            form, textvariable=self.method, values=METHODS, state="readonly"
        )
        selector.pack(fill="x")
        selector.bind("<<ComboboxSelected>>", lambda _: self.select_method())
        ttk.Label(
            form, textvariable=self.formula, font=("Arial", 17), foreground=TEAL
        ).pack(anchor="w", pady=(4, 10))
        self.preset_box = ttk.Combobox(
            form, textvariable=self.preset, values=tuple(PRESETS), state="readonly"
        )
        self.preset_box.pack(fill="x")
        ttk.Button(form, text="Load example", command=self.load_example).pack(
            fill="x", pady=(5, 5)
        )
        self.field_rows, self.field_labels = {}, {}
        for key, label in (
            ("expression", "Function f(x)"),
            ("first_derivative", "f′(x) · blank = automatic"),
            ("second_derivative", "f″(x) · blank = automatic"),
            ("x0", "Starting estimate x₀"),
            ("y0", "Starting estimate y₀"),
            ("b_x", "Initial vertex B · x"),
            ("b_y", "Initial vertex B · y"),
            ("c_x", "Initial vertex C · x"),
            ("c_y", "Initial vertex C · y"),
            ("alpha", "Fixed step length α > 0"),
            ("tolerance", "Gradient tolerance ε · |f′(x)| ≤ ε"),
            ("max_iterations", "Maximum iterations"),
            ("initial_gamma", "Initial γ probe · line search"),
            ("initial_step", "Initial |α| probe · signed line search"),
            ("line_tolerance", "Relative line-derivative tolerance"),
            ("max_line_evaluations", "Maximum trial evaluations / line search"),
            ("plot_min", "Plot minimum x · view only"),
            ("plot_max", "Plot maximum x · view only"),
            ("plot_y_min", "Plot minimum y · view only"),
            ("plot_y_max", "Plot maximum y · view only"),
        ):
            row = ttk.Frame(form)
            row.pack(fill="x")
            self.field_rows[key] = row
            label_var = tk.StringVar(value=label)
            self.field_labels[key] = label_var
            ttk.Label(row, textvariable=label_var, style="Muted.TLabel").pack(
                anchor="w", pady=(8, 3)
            )
            ttk.Entry(row, textvariable=self.fields[key]).pack(fill="x")
        self.syntax_hint = ttk.Label(
            form,
            text="Use explicit multiplication: 2*x\nx, pi, e; + − * / ^; sin, cos, tan,\nexp, ln, log, sqrt. Blank derivatives\nare calculated automatically.",
            style="Muted.TLabel",
        )
        self.syntax_hint.pack(anchor="w", pady=(10, 4))
        self.simplex_options = ttk.Frame(form)
        for key, label, choices in (
            ("variant", "Nelder–Mead branches", ("slides", "standard")),
            ("stopping", "Nelder–Mead stopping", ("function", "position")),
        ):
            ttk.Label(self.simplex_options, text=label, style="Muted.TLabel").pack(
                anchor="w"
            )
            box = ttk.Combobox(
                self.simplex_options,
                textvariable=self.fields[key],
                values=choices,
                state="readonly",
            )
            box.pack(fill="x", pady=(0, 5))
            box.bind("<<ComboboxSelected>>", lambda _: self._configure_method())
        ttk.Label(form, text="Display significant digits", style="Muted.TLabel").pack(
            anchor="w", pady=(8, 3)
        )
        precision = ttk.Combobox(
            form,
            textvariable=self.precision,
            values=("6", "10", "15"),
            state="readonly",
        )
        precision.pack(fill="x")
        precision.bind("<<ComboboxSelected>>", lambda _: self._render())

        def wheel(event):
            canvas.yview_scroll(-1 if event.delta > 0 or event.num == 4 else 1, "units")

        wheel_widgets = (
            form,
            *form.winfo_children(),
            canvas,
            *[
                child
                for row in self.field_rows.values()
                for child in row.winfo_children()
            ],
        )
        for widget in wheel_widgets:
            for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(event, wheel, add="+")
        right = ttk.Frame(self.frame)
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(
            right,
            textvariable=self.heading,
            font=("Arial", 15, "bold"),
        ).pack(anchor="w", pady=(2, 8))
        self.snapshot = tk.StringVar(
            value="Run the example to inspect each estimate and its slope."
        )
        ttk.Label(
            right, textvariable=self.snapshot, style="Muted.TLabel", wraplength=800
        ).pack(fill="x", pady=(0, 10))
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)
        overview = ttk.Frame(self.notebook, padding=14, style="White.TFrame")
        self.notebook.add(overview, text="Result & plots")
        self.summary_text = self.app._text(overview, height=8)
        self.chart = tk.Canvas(overview, bg="white", highlightthickness=0)
        self.chart.pack(fill="both", expand=True, pady=(8, 0))
        self.chart.bind("<Configure>", self._queue_chart)
        self.walk_tab = ttk.Frame(self.notebook, padding=14, style="White.TFrame")
        self.notebook.add(self.walk_tab, text="Worked steps")
        nav = ttk.Frame(self.walk_tab, style="White.TFrame")
        nav.pack(fill="x", pady=(0, 10))
        self.prev_button = ttk.Button(
            nav,
            text="← Previous",
            command=lambda: self.go_step(self.step_index - 1),
            state="disabled",
        )
        self.prev_button.pack(side="left")
        self.next_button = ttk.Button(
            nav,
            text="Next →",
            command=lambda: self.go_step(self.step_index + 1),
            state="disabled",
        )
        self.next_button.pack(side="left", padx=6)
        self.step_label = tk.StringVar(value="No calculation yet")
        ttk.Label(nav, textvariable=self.step_label, background="white").pack(
            side="left", padx=8
        )
        self.slider = ttk.Scale(
            self.walk_tab, from_=0, to=1, command=self._slide, state="disabled"
        )
        self.slider.pack(fill="x", pady=(0, 10))
        self.worked_text = self.app._text(self.walk_tab)
        trace = ttk.Frame(self.notebook, padding=14, style="White.TFrame")
        self.notebook.add(trace, text="All iterations")
        bar = ttk.Frame(trace, style="White.TFrame")
        bar.pack(fill="x", pady=(0, 10))
        self.csv_button = ttk.Button(
            bar, text="Export all CSV…", command=self.save_csv, state="disabled"
        )
        self.csv_button.pack(side="right")
        self.page_prev = ttk.Button(
            bar, text="← Page", command=lambda: self.change_page(-1), state="disabled"
        )
        self.page_prev.pack(side="left")
        self.page_next = ttk.Button(
            bar, text="Page →", command=lambda: self.change_page(1), state="disabled"
        )
        self.page_next.pack(side="left", padx=6)
        self.page_text = tk.StringVar(value="No iterations")
        ttk.Label(bar, textvariable=self.page_text, background="white").pack(
            side="left", padx=5
        )
        self.trace_tree = self.app._table(trace)
        self.trace_tree.bind("<<TreeviewSelect>>", self._trace_selected)
        self.trace_tree.bind(
            "<Double-1>", lambda _: self.notebook.select(self.walk_tab)
        )
        notes_tab = ttk.Frame(self.notebook, padding=14, style="White.TFrame")
        self.notebook.add(notes_tab, text="Algorithm")
        self.notes_text = self.app._text(notes_tab)
        self.app._set_text(self.notes_text, NOTES)
        normalized_gradient_view.build_line_tab(self)
        powell_view.build_line_tab(self)

    def update_header(self):
        if self.app.workspace.get() == "Optimization":
            label = (
                "Powell 2D"
                if self.active_method == POWELL
                else "Steepest Line Search 2D"
                if self.active_method == LINE_STEEPEST
                else "Normalized Gradient 2D"
                if self.active_method == NORMALIZED
                else "Nelder–Mead 2D"
                if self.active_method == NELDER_MEAD
                else "Newton 2D"
                if self.active_method == NEWTON2D
                else "Steepest Descent 2D"
                if self.active_method == STEEPEST
                else "Newton 1D"
                if self.active_method == NEWTON
                else "Gradient 1D"
            )
            self.app.header_text.set(f"OPTIMIZATION  /  {self.active_method.upper()}")
            self.root.title(f"Exam lab · Optimization · {label}")

    def _configure_method(self):
        newton = self.active_method == NEWTON
        steepest = self.active_method == STEEPEST
        newton2d = self.active_method == NEWTON2D
        nelder_mead = self.active_method == NELDER_MEAD
        normalized = self.active_method in (NORMALIZED, LINE_STEEPEST)
        line_steepest = self.active_method == LINE_STEEPEST
        powell = self.active_method == POWELL
        dimension2 = steepest or newton2d or nelder_mead or normalized or powell
        self.formula.set(
            "Q₁=Q₀+α₁d₁ · Q₂=Q₁+α₂d₂\nd₃=Q₂−Q₀ · Q₃=Q₂+α₃d₃"
            if powell
            else "dₖ = −∇f(Pₖ)\nφ′(αₖ)=0 · Pₖ₊₁=Pₖ+αₖdₖ"
            if line_steepest
            else "Sₖ = −∇f(Pₖ)/||∇f(Pₖ)||₂\nφ′(γₖ)=0 · Pₖ₊₁=Pₖ+γₖSₖ"
            if normalized
            else "M = (A+B)/2\nα=1 · γ=2 · ρ=σ=0.5"
            if nelder_mead
            else "[xₖ₊₁,yₖ₊₁]ᵀ = [xₖ,yₖ]ᵀ\n− H(xₖ,yₖ)⁻¹ ∇f(xₖ,yₖ)"
            if newton2d
            else "xₖ₊₁ = xₖ − α ∂f/∂x(xₖ,yₖ)\nyₖ₊₁ = yₖ − α ∂f/∂y(xₖ,yₖ)"
            if steepest
            else "xᵢ₊₁ = xᵢ − f′(xᵢ) / f″(xᵢ)"
            if newton
            else "xₖ₊₁ = xₖ − α f′(xₖ)"
        )
        self.heading.set(
            "MINIMIZATION  /  POWELL'S METHOD · 2D"
            if powell
            else "MINIMIZATION  /  STEEPEST DESCENT WITH LINE SEARCH · 2D"
            if line_steepest
            else "MINIMIZATION  /  NORMALIZED GRADIENT · 2D"
            if normalized
            else "MINIMIZATION  /  NELDER–MEAD · 2D"
            if nelder_mead
            else "MINIMIZATION  /  NEWTON'S METHOD · 2D"
            if newton2d
            else "MINIMIZATION  /  STEEPEST DESCENT · 2D"
            if steepest
            else "MINIMIZATION  /  NEWTON'S METHOD · 1D"
            if newton
            else "MINIMIZATION  /  FIXED-STEP GRADIENT METHOD"
        )
        self.run_button.configure(
            text="Find minimum · Powell 2D"
            if powell
            else "Find minimum · Steepest line search"
            if line_steepest
            else "Find minimum · Normalized gradient"
            if normalized
            else "Find minimum · Nelder–Mead"
            if nelder_mead
            else "Find minimum · Newton 2D"
            if newton2d
            else "Find minimum · Steepest 2D"
            if steepest
            else "Find minimum · Newton 1D"
            if newton
            else "Find minimum · Gradient 1D"
        )
        self.field_labels["expression"].set(
            "Function f(x, y)" if dimension2 else "Function f(x)"
        )
        self.field_labels["tolerance"].set(
            "Cycle tolerance ε · |Δf| or distance < ε"
            if powell
            else "Simplex radius ε · max distance to best < ε"
            if nelder_mead and self.fields["stopping"].get() == "position"
            else "Function spread ε · f(C)−f(A) ≤ ε"
            if nelder_mead
            else "Gradient tolerance ε · ||∇f||₂ ≤ ε"
            if dimension2
            else "Step tolerance ε · |xᵢ₊₁ − xᵢ| < ε"
            if newton
            else "Gradient tolerance ε · |f′(x)| ≤ ε"
        )
        self.field_labels["max_iterations"].set(
            "Maximum complete cycles · 3 moves / cycle"
            if powell
            else "Last loop index Nmax · i=0…Nmax"
            if newton
            else "Maximum iterations"
        )
        self.simplex_options.pack_forget()
        if nelder_mead:
            self.simplex_options.pack(fill="x", before=self.syntax_hint)
        self.field_rows["alpha"].pack_forget()
        for key in ("first_derivative", "second_derivative"):
            self.field_rows[key].pack_forget()
            if newton:
                self.field_rows[key].pack(fill="x", before=self.field_rows["x0"])
        if not (newton or newton2d or nelder_mead or normalized or powell):
            self.field_rows["alpha"].pack(fill="x", before=self.field_rows["tolerance"])
        for key in ("y0", "plot_y_min", "plot_y_max"):
            self.field_rows[key].pack_forget()
        if dimension2:
            self.field_rows["y0"].pack(
                fill="x",
                before=self.field_rows["tolerance"]
                if newton2d or nelder_mead or normalized or powell
                else self.field_rows["alpha"],
            )
            for key in ("plot_y_min", "plot_y_max"):
                self.field_rows[key].pack(fill="x", before=self.syntax_hint)
        self.field_labels["x0"].set(
            "Initial vertex A · x" if nelder_mead else "Starting estimate x₀"
        )
        self.field_labels["y0"].set(
            "Initial vertex A · y" if nelder_mead else "Starting estimate y₀"
        )
        for key in ("b_x", "b_y", "c_x", "c_y"):
            self.field_rows[key].pack_forget()
            if nelder_mead:
                self.field_rows[key].pack(fill="x", before=self.field_rows["tolerance"])
        self.field_labels["line_tolerance"].set(
            "Line-search α resolution tolerance"
            if powell
            else "Relative line-derivative tolerance"
        )
        for key in (
            "initial_gamma",
            "initial_step",
            "line_tolerance",
            "max_line_evaluations",
        ):
            self.field_rows[key].pack_forget()
            if (normalized and key != "initial_step") or (
                powell and key != "initial_gamma"
            ):
                self.field_rows[key].pack(fill="x", before=self.field_rows["plot_min"])
        if normalized:
            self.notebook.add(self.line_tab, text="Line search")
        else:
            self.notebook.hide(self.line_tab)
        if powell:
            self.notebook.add(self.powell_line_tab, text="Powell line search")
        else:
            self.notebook.hide(self.powell_line_tab)
        self.preset_box.configure(values=tuple(self._presets()))
        self.syntax_hint.configure(
            text="Use explicit multiplication: x*y\n"
            + ("x, y, pi, e" if dimension2 else "x, pi, e")
            + "; + − * / ^; sin, cos, tan,\nexp, ln, log, sqrt.\n"
            + (
                "Function values only · no derivatives."
                if nelder_mead or powell
                else "Derivatives are automatic when blank."
            )
        )
        self.app._set_text(
            self.notes_text,
            powell_view.NOTES
            if powell
            else normalized_gradient_view.LINE_NOTES
            if line_steepest
            else normalized_gradient_view.NOTES
            if normalized
            else nelder_mead_view.NOTES
            if nelder_mead
            else newton2d_view.NOTES
            if newton2d
            else steepest_view.NOTES
            if steepest
            else newton_view.NOTES
            if newton
            else NOTES,
        )
        self.update_header()

    def _presets(self, method=None):
        method = method or self.active_method
        return (
            POWELL_PRESETS
            if method == POWELL
            else NORMALIZED_PRESETS
            if method in (NORMALIZED, LINE_STEEPEST)
            else NELDER_MEAD_PRESETS
            if method == NELDER_MEAD
            else NEWTON2D_PRESETS
            if method == NEWTON2D
            else STEEPEST_PRESETS
            if method == STEEPEST
            else PRESETS
        )

    def _two_dimensional_view(self):
        if self.result:
            if isinstance(self.result.request, PowellRequest):
                return powell_view
            if isinstance(self.result.request, NormalizedRequest):
                return normalized_gradient_view
            if isinstance(self.result.request, NelderMeadRequest):
                return nelder_mead_view
            if isinstance(self.result.request, Newton2DRequest):
                return newton2d_view
            if isinstance(self.result.request, SteepestRequest):
                return steepest_view
        return None

    def select_method(self):
        method = self.method.get()
        if method == self.active_method or method not in METHODS:
            return
        self.states[self.active_method] = {
            "fields": {key: var.get() for key, var in self.fields.items()},
            "preset": self.preset.get(),
            "result": self.result,
            "step_index": self.step_index,
            "trace_page": self.trace_page,
            "status": self.status.get(),
        }
        defaults = self.defaults.copy()
        presets = self._presets(method)
        if method in (
            STEEPEST,
            NEWTON2D,
            NELDER_MEAD,
            NORMALIZED,
            LINE_STEEPEST,
            POWELL,
        ):
            defaults.update(presets[next(iter(presets))])
        state = self.states.get(
            method,
            {
                "fields": defaults,
                "preset": next(iter(presets)),
                "result": None,
                "step_index": 0,
                "trace_page": 0,
                "status": "Ready · enter f(x) and find a minimum estimate",
            },
        )
        self.active_method = method
        self.loading = True
        try:
            for key, value in state["fields"].items():
                self.fields[key].set(value)
            self.preset.set(state["preset"])
        finally:
            self.loading = False
        self.result = state["result"]
        self.step_index, self.trace_page = state["step_index"], state["trace_page"]
        self.status.set(state["status"])
        self._configure_method()
        if self.result:
            self._render()
            try:
                if self.read_request() != self.result.request:
                    self._dirty()
            except InputError:
                self._dirty()
        else:
            self.snapshot.set(
                "Run the example to inspect each estimate and its derivatives."
            )
            for widget in (self.summary_text, self.worked_text):
                self.app._set_text(widget, "")
            self.trace_tree.delete(*self.trace_tree.get_children())
            self.chart.delete("all")
            for widget in (
                self.csv_button,
                self.prev_button,
                self.next_button,
                self.page_prev,
                self.page_next,
                self.slider,
            ):
                widget.configure(state="disabled")
            self.step_label.set("No calculation yet")
            self.page_text.set("No iterations")

    def fmt(self, value):
        return f"{value:.{int(self.precision.get())}g}"

    def _dirty(self, *_):
        if not self.loading and self.result:
            self.status.set(
                "Inputs edited · displayed result belongs to the last completed run"
            )

    def load_example(self):
        if self.active_method in (
            STEEPEST,
            NEWTON2D,
            NELDER_MEAD,
            NORMALIZED,
            LINE_STEEPEST,
            POWELL,
        ):
            values = {
                "tolerance": "1e-6",
                "max_iterations": "200",
                **(
                    {
                        "initial_step": "1",
                        "line_tolerance": "1e-10",
                        "max_line_evaluations": "256",
                    }
                    if self.active_method == POWELL
                    else {}
                ),
                **self._presets()[self.preset.get()],
            }
            for key, value in values.items():
                self.fields[key].set(value)
            if self.active_method in (NORMALIZED, LINE_STEEPEST):
                for key in ("initial_gamma", "line_tolerance", "max_line_evaluations"):
                    self.fields[key].set(self.defaults[key])
            return
        expression, x0, alpha, left, right = PRESETS[self.preset.get()]
        for key, value in zip(
            ("expression", "x0", "alpha", "plot_min", "plot_max"),
            (expression, x0, alpha, left, right),
        ):
            self.fields[key].set(value)
        self.fields["tolerance"].set("1e-6")
        self.fields["max_iterations"].set("200")
        self.fields["first_derivative"].set("")
        self.fields["second_derivative"].set("")
        for key, value in SOURCE_PRESET_OPTIONS.get(self.preset.get(), {}).items():
            self.fields[key].set(value)

    def read_request(self):
        try:
            limit = int(self.fields["max_iterations"].get())
        except ValueError as exc:
            raise InputError("Maximum iterations must be an integer.") from exc
        if self.active_method in (NORMALIZED, LINE_STEEPEST, POWELL):
            try:
                line_limit = int(self.fields["max_line_evaluations"].get())
            except ValueError as exc:
                raise InputError(
                    "Maximum line-search trial evaluations must be an integer."
                ) from exc
            powell = self.active_method == POWELL
            request_type = (
                PowellRequest
                if powell
                else LineSteepestRequest
                if self.active_method == LINE_STEEPEST
                else NormalizedRequest
            )
            probe = (
                ("initial_step", "Initial |α| probe")
                if powell
                else ("initial_gamma", "Initial γ probe")
            )
            request = request_type(
                expression=self.fields["expression"].get(),
                max_iterations=limit,
                max_line_evaluations=line_limit,
                **{
                    key: number(self.fields[key].get(), label)
                    for key, label in (
                        ("x0", "Starting x₀"),
                        ("y0", "Starting y₀"),
                        (
                            "tolerance",
                            "Cycle tolerance" if powell else "Outer gradient tolerance",
                        ),
                        probe,
                        ("line_tolerance", "Line-search tolerance"),
                        ("plot_min", "Plot minimum x"),
                        ("plot_max", "Plot maximum x"),
                        ("plot_y_min", "Plot minimum y"),
                        ("plot_y_max", "Plot maximum y"),
                    )
                },
            )
            request.validate()
            return request
        if self.active_method == NELDER_MEAD:
            request = NelderMeadRequest(
                expression=self.fields["expression"].get(),
                variant=self.fields["variant"].get(),
                stopping=self.fields["stopping"].get(),
                max_iterations=limit,
                simplex=tuple(
                    (
                        number(self.fields[x].get(), f"Vertex {label} x"),
                        number(self.fields[y].get(), f"Vertex {label} y"),
                    )
                    for label, x, y in (
                        ("A", "x0", "y0"),
                        ("B", "b_x", "b_y"),
                        ("C", "c_x", "c_y"),
                    )
                ),
                **{
                    key: number(self.fields[key].get(), label)
                    for key, label in (
                        ("tolerance", "Function spread tolerance ε"),
                        ("plot_min", "Plot minimum x"),
                        ("plot_max", "Plot maximum x"),
                        ("plot_y_min", "Plot minimum y"),
                        ("plot_y_max", "Plot maximum y"),
                    )
                },
            )
            request.validate()
            return request
        if self.active_method in (STEEPEST, NEWTON2D):
            request_type = (
                Newton2DRequest if self.active_method == NEWTON2D else SteepestRequest
            )
            rate = (
                {}
                if self.active_method == NEWTON2D
                else {"alpha": number(self.fields["alpha"].get(), "Step length α")}
            )
            request = request_type(
                expression=self.fields["expression"].get(),
                max_iterations=limit,
                **rate,
                **{
                    key: number(self.fields[key].get(), label)
                    for key, label in (
                        ("x0", "Starting x₀"),
                        ("y0", "Starting y₀"),
                        ("tolerance", "Tolerance ε"),
                        ("plot_min", "Plot minimum x"),
                        ("plot_max", "Plot maximum x"),
                        ("plot_y_min", "Plot minimum y"),
                        ("plot_y_max", "Plot maximum y"),
                    )
                },
            )
            request.validate()
            return request
        newton = self.active_method == NEWTON
        request_type = NewtonRequest if newton else GradientRequest
        derivative_inputs = (
            {
                key: self.fields[key].get()
                for key in ("first_derivative", "second_derivative")
            }
            if newton
            else {"alpha": number(self.fields["alpha"].get(), "Step length α")}
        )
        request = request_type(
            expression=self.fields["expression"].get(),
            max_iterations=limit,
            **derivative_inputs,
            **{
                key: number(self.fields[key].get(), label)
                for key, label in (
                    ("x0", "Starting x₀"),
                    ("tolerance", "Tolerance ε"),
                    ("plot_min", "Plot minimum"),
                    ("plot_max", "Plot maximum"),
                )
            },
        )
        request.validate()
        return request

    def start(self):
        if self.busy or self.closed:
            return
        try:
            request = self.read_request()
        except InputError as exc:
            messagebox.showerror(
                "Check optimization inputs", str(exc), parent=self.root
            )
            return
        self.busy = True
        self.running_method = self.active_method
        self.cancel_event = threading.Event()
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.start(12)
        self.status.set("Preparing Rust…")

        def worker():
            try:
                result = train(
                    request,
                    cancel=self.cancel_event,
                    progress=lambda text: self.events.put(("progress", text)),
                )
                self.events.put(("done", result))
            except Cancelled:
                self.events.put(("cancelled", None))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def cancel(self):
        if self.busy:
            self.cancel_event.set()
            self.cancel_button.configure(state="disabled")
            self.status.set("Cancelling calculation…")

    def _poll(self):
        if self.closed:
            return
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "progress":
                    if not self.cancel_event.is_set():
                        self.status.set(data)
                    continue
                self.busy = False
                self.progress.stop()
                self.run_button.configure(state="normal")
                self.cancel_button.configure(state="disabled")
                if self.cancel_event.is_set() or kind == "cancelled":
                    status = (
                        "Calculation cancelled · previous completed result retained"
                    )
                elif kind == "error":
                    status = "Calculation failed · previous completed result retained"
                    messagebox.showerror("Optimization failed", data, parent=self.root)
                else:
                    status = f"{STATUS[data.status]} · {data.final.iteration:,} updates"
                    if isinstance(data.request, PowellRequest):
                        status = f"{STATUS[data.status]} · {len(data.cycles):,} complete cycles · {data.final.iteration:,} line moves"
                    if self.active_method == self.running_method:
                        self.result = data
                        self.step_index = self.trace_page = 0
                        self._render()
                    else:
                        self.states[self.running_method].update(
                            result=data, step_index=0, trace_page=0
                        )
                if self.active_method == self.running_method:
                    self.status.set(status)
                    if kind == "done" and not self.cancel_event.is_set():
                        try:
                            if self.read_request() != data.request:
                                self._dirty()
                        except InputError:
                            self._dirty()
                else:
                    self.states[self.running_method]["status"] = status
                    self.status.set(f"{self.running_method}: {status}")
        except queue.Empty:
            pass
        self.poll_job = self.root.after(60, self._poll)

    def _render(self):
        if not self.result:
            return
        view2d = self._two_dimensional_view()
        if view2d:
            view2d.render_summary(self)
            self._render_controls()
            return
        r, f = self.result, self.fmt
        p = r.final
        newton = isinstance(r.request, NewtonRequest)
        rate = "" if newton else f"  ·  α = {f(r.request.alpha)}"
        self.snapshot.set(
            f"{r.request.algorithm_name}: f(x) = {r.request.expression}  ·  x₀ = {f(r.request.x0)}{rate}  ·  ε = {f(r.request.tolerance)}"
        )
        detail = {
            "gradient_tolerance": "The gradient tolerance passed. A stationary point alone does not prove a minimum.",
            "iteration_limit": "The update budget ended before the gradient tolerance passed.",
            "numerical_limit": "The next update exceeded the numerical limit or left the function/derivative domain; it was rejected.",
            "stagnation": "The next x rounds to the current x while the gradient tolerance still fails.",
        }.get(r.status, "")
        increases = sum(b.value > a.value for a, b in zip(r.points, r.points[1:]))
        outside = sum(
            not r.request.plot_min <= point.x <= r.request.plot_max
            for point in r.points
        )
        summary = (
            newton_view.summary(self)
            if newton
            else f"Estimate x = {f(p.x)}     f(x) = {f(p.value)}     f′(x) = {f(p.gradient)}\nCompleted updates: {p.iteration:,}   ·   |f′(x)| ≤ ε: {abs(p.gradient) <= r.request.tolerance}\n{detail}"
        )
        self.app._set_text(
            self.summary_text,
            f"{STATUS[r.status]}\n{summary}\nObjective increases: {increases:,}   ·   Estimates outside function plot: {outside:,}\nPlot bounds are view settings; they do not constrain x.",
        )
        self._render_controls()

    def _render_controls(self):
        r = self.result
        self.csv_button.configure(state="normal")
        self.slider.configure(state="normal", to=max(1, len(r.points) - 1))
        self._render_trace()
        self.go_step(self.step_index)

    def _slide(self, value):
        index = int(float(value))
        if index != self.step_index:
            self.go_step(index)

    def go_step(self, index):
        if not self.result:
            return
        r, f = self.result, self.fmt
        self.step_index = max(0, min(index, len(r.points) - 1))
        p = r.points[self.step_index]
        self.slider.set(self.step_index)
        self.prev_button.configure(state="normal" if self.step_index else "disabled")
        self.next_button.configure(
            state="normal" if self.step_index + 1 < len(r.points) else "disabled"
        )
        self.step_label.set(f"k = {p.iteration:,} / {r.final.iteration:,}")
        if isinstance(r.request, PowellRequest):
            self.step_label.set(f"Move X{p.iteration:,} / X{r.final.iteration:,}")
        if isinstance(r.request, NelderMeadRequest):
            self.step_label.set(f"Simplex k = {p.iteration:,} / {r.final.iteration:,}")
        view2d = self._two_dimensional_view()
        if view2d:
            self.app._set_text(self.worked_text, view2d.worked_step(self))
            if hasattr(view2d, "render_line_search"):
                view2d.render_line_search(self)
            self.draw_chart()
            return
        if isinstance(r.request, NewtonRequest):
            self.step_label.set(f"i = {p.iteration:,} / {r.final.iteration:,}")
            self.app._set_text(self.worked_text, newton_view.worked_step(self))
            self.draw_chart()
            return
        passed = abs(p.gradient) <= r.request.tolerance
        lines = [
            f"ITERATION k = {p.iteration}\nf(x) = {r.request.expression}",
            f"Current estimate xₖ = {f(p.x)}\nf(xₖ) = {f(p.value)}\nf′(xₖ) = {f(p.gradient)}  (automatic derivative)",
            f"Stopping test: |f′(xₖ)| ≤ ε\n{f(abs(p.gradient))} ≤ {f(r.request.tolerance)} → {str(passed).upper()}",
        ]
        if self.step_index + 1 < len(r.points):
            nxt = r.points[self.step_index + 1]
            lines.extend(
                [
                    f"Δx = −α f′(xₖ)\n   = −({f(r.request.alpha)}) × ({f(p.gradient)})\n   = {f(-r.request.alpha * p.gradient)}",
                    f"xₖ₊₁ = xₖ − α f′(xₖ)\n     = ({f(p.x)}) − ({f(r.request.alpha)}) × ({f(p.gradient)})\n     = {f(nxt.x)}",
                    f"f(xₖ₊₁) = {f(nxt.value)}\nUse this new estimate in iteration {nxt.iteration}.",
                ]
            )
        else:
            lines.append(f"Stop: {STATUS[r.status]}. No further update was committed.")
        self.app._set_text(self.worked_text, "\n\n".join(lines))
        self.draw_chart()

    def _render_trace(self):
        r = self.result
        view2d = self._two_dimensional_view()
        newton = isinstance(r.request, NewtonRequest)
        columns = (
            [
                "i",
                "xᵢ",
                "f(xᵢ)",
                "f′(xᵢ)",
                "f″(xᵢ)",
                "Δx = −f′/f″",
                "xᵢ₊₁",
                "|xᵢ₊₁ − xᵢ|",
                "Step < ε",
            ]
            if newton
            else ["k", "xₖ", "f(xₖ)", "f′(xₖ)", "Δx = −α f′(xₖ)", "xₖ₊₁"]
        )
        self.app._columns(self.trace_tree, view2d.TRACE_COLUMNS if view2d else columns)
        start = self.trace_page * PAGE_SIZE
        for i in range(start, min(start + PAGE_SIZE, len(r.points))):
            if view2d:
                self.trace_tree.insert(
                    "", "end", iid=str(i), values=view2d.trace_values(self, i)
                )
                continue
            p = r.points[i]
            nxt = r.points[i + 1] if i + 1 < len(r.points) else None
            values = [
                p.iteration,
                self.fmt(p.x),
                self.fmt(p.value),
                self.fmt(p.gradient),
            ]
            if newton:
                values.extend(
                    [
                        self.fmt(p.second_derivative),
                        self.fmt(-p.gradient / p.second_derivative) if nxt else "—",
                        self.fmt(nxt.x) if nxt else "—",
                        self.fmt(abs(nxt.x - p.x)) if nxt else "—",
                        str(abs(nxt.x - p.x) < r.request.tolerance) if nxt else "—",
                    ]
                )
            else:
                values.extend(
                    [
                        self.fmt(-r.request.alpha * p.gradient) if nxt else "—",
                        self.fmt(nxt.x) if nxt else "—",
                    ]
                )
            self.trace_tree.insert("", "end", iid=str(i), values=values)
        total = math.ceil(len(r.points) / PAGE_SIZE)
        entries = (
            "simplexes" if isinstance(r.request, NelderMeadRequest) else "estimates"
        )
        self.page_text.set(
            f"Page {self.trace_page + 1}/{total} · {len(r.points):,} {entries}"
        )
        self.page_prev.configure(state="normal" if self.trace_page else "disabled")
        self.page_next.configure(
            state="normal" if self.trace_page + 1 < total else "disabled"
        )

    def change_page(self, delta):
        if self.result:
            self.trace_page = max(
                0,
                min(
                    self.trace_page + delta, (len(self.result.points) - 1) // PAGE_SIZE
                ),
            )
            self._render_trace()

    def _trace_selected(self, _=None):
        selected = self.trace_tree.selection()
        if selected:
            self.go_step(int(selected[0]))

    def _queue_chart(self, _=None):
        if self.chart_job:
            self.root.after_cancel(self.chart_job)
        self.chart_job = self.root.after(80, self._draw_queued_chart)

    def _draw_queued_chart(self):
        self.chart_job = None
        self.draw_chart()

    def draw_chart(self):
        c = self.chart
        c.delete("all")
        width, height = c.winfo_width(), c.winfo_height()
        if not self.result or width < 250 or height < 100:
            return
        view2d = self._two_dimensional_view()
        if view2d:
            getattr(view2d, "draw_chart", steepest_view.draw_chart)(self)
            return
        r = self.result
        for panel, title in enumerate(
            ("FUNCTION & ESTIMATES", "OBJECTIVE BY ITERATION")
        ):
            left, right = panel * width / 2 + 65, (panel + 1) * width / 2 - 22
            top, bottom = 35, height - 40
            xlow, xhigh = (
                (r.request.plot_min, r.request.plot_max)
                if panel == 0
                else (0, max(1, r.final.iteration))
            )
            curve = (
                r.curve
                if panel == 0
                else tuple((p.iteration, p.value) for p in r.points)
            )
            visible = [p.value for p in r.points if panel or xlow <= p.x <= xhigh]
            values = [y for _, y in curve if y is not None] + visible
            c.create_text(
                left, 12, text=title, anchor="w", fill=MUTED, font=("Arial", 9, "bold")
            )
            if not values:
                c.create_text(
                    (left + right) / 2,
                    (top + bottom) / 2,
                    text="No valid samples in this view",
                    fill=MUTED,
                )
                continue
            low, high = min(values), max(values)
            pad = (
                max(abs(low), abs(high), 1) * 0.05
                if low == high
                else (high - low) * 0.08
            )
            low, high = low - pad, high + pad

            def xy(x, y):
                return (
                    left + (x - xlow) / (xhigh - xlow) * (right - left),
                    bottom - (y - low) / (high - low) * (bottom - top),
                )

            for j in range(3):
                y = high - (high - low) * j / 2
                _, py = xy(xlow, y)
                c.create_line(left, py, right, py, fill="#e3eaee")
                c.create_text(
                    left - 7,
                    py,
                    text=f"{y:.3g}",
                    anchor="e",
                    fill=MUTED,
                    font=("Arial", 9),
                )
            c.create_text(left, bottom + 18, text=f"{xlow:.3g}", anchor="w", fill=MUTED)
            c.create_text(
                right,
                bottom + 18,
                text=f"{'x' if panel == 0 else 'k'} = {xhigh:.4g}",
                anchor="e",
                fill=MUTED,
            )
            stride = max(1, math.ceil(len(curve) / 1000))
            indices = list(range(0, len(curve), stride))
            if indices[-1] != len(curve) - 1:
                indices.append(len(curve) - 1)
            previous = None
            for i in indices:
                x, y = curve[i]
                if y is None:
                    previous = None
                    continue
                point = xy(x, y)
                if previous:
                    c.create_line(*previous, *point, fill=INK, width=2)
                previous = point
            marks = (
                r.points[:: max(1, math.ceil(len(r.points) / 150))]
                if panel == 0
                else ()
            )
            for p in (*marks, r.points[self.step_index], r.final):
                x = p.x if panel == 0 else p.iteration
                if xlow <= x <= xhigh:
                    px, py = xy(x, p.value)
                    radius = 5 if p is r.points[self.step_index] else 3
                    c.create_oval(
                        px - radius,
                        py - radius,
                        px + radius,
                        py + radius,
                        fill=TEAL,
                        outline="white",
                    )

    def save_csv(self):
        if not self.result:
            return
        newton = isinstance(self.result.request, NewtonRequest)
        normalized = isinstance(self.result.request, NormalizedRequest)
        powell = isinstance(self.result.request, PowellRequest)
        nelder_mead = isinstance(self.result.request, NelderMeadRequest)
        newton2d = isinstance(self.result.request, Newton2DRequest)
        steepest = isinstance(self.result.request, SteepestRequest)
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export every Powell move and cycle"
            if powell
            else "Export normalized-gradient outer history"
            if normalized
            else "Export every Nelder–Mead simplex and trial"
            if nelder_mead
            else "Export every 2D Newton estimate"
            if newton2d
            else "Export every 2D gradient estimate"
            if steepest
            else "Export every Newton estimate"
            if newton
            else "Export every gradient estimate",
            initialfile="powell_2d.csv"
            if powell
            else "normalized_gradient_2d.csv"
            if normalized
            else "nelder_mead_2d.csv"
            if nelder_mead
            else "newton_2d.csv"
            if newton2d
            else "steepest_2d.csv"
            if steepest
            else "newton_1d.csv"
            if newton
            else "gradient_1d.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if path:
            try:
                export_trace(path, self.result)
                self.status.set(f"Saved {Path(path).name} · full run precision")
            except OSError as exc:
                messagebox.showerror("Could not save CSV", str(exc), parent=self.root)

    def close(self):
        self.closed = True
        self.cancel_event.set()
        self.root.after_cancel(self.poll_job)
        if self.chart_job:
            self.root.after_cancel(self.chart_job)
