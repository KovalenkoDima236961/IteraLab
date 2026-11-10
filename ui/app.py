"""Tk worksheet. Only the main thread touches Tk; a worker owns Rust processes."""

import math
from pathlib import Path
import queue
import random
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..backend import (
    Cancelled,
    InputError,
    Request,
    STATUS,
    export_epochs,
    export_trace,
    number,
    parse_samples,
    preset_samples,
    samples_text,
    train,
    vector,
)
from ..algorithms.madaline import MAX_HIDDEN_UNITS, MadalineRequest, MadalineResult
from . import madaline_view
from ..algorithms.hebbian import HebbianRequest, HebbianResult
from . import hebbian_view
from ..algorithms.perceptron import (
    BINARY,
    BIPOLAR,
    BIPOLAR_ERROR,
    BAND,
    PerceptronRequest,
    PerceptronResult,
    perceptron_samples,
)
from . import perceptron_view
from ..algorithms.lecture08 import OPTIONS as LECTURE08_OPTIONS
from ..examples.learning import LEARNING_EXAMPLES

BG = "#eef2f5"
INK = "#182d40"
MUTED = "#617284"
TEAL = "#087e76"
MONO = "Menlo" if sys.platform == "darwin" else "Consolas"
PAGE_SIZE = 200

ALGORITHM = """ADALINE · supplied lecture algorithm

Activation function
    y = +1  if y_in ≥ 0
    y = −1  if y_in < 0

Step 0 · Initialize
Set initial weights w₁ … wₙ, bias b, and learning rate α.
Small random values are usual; explicit values and zero initialization are also available.

Steps 1–2 · Repeat complete epochs
While the stopping condition is false, visit every training pair s : t in the displayed row order. No shuffling is performed.

Step 3 · Set inputs
    xᵢ = sᵢ

Step 4 · Compute the linear response
    y_in = b + Σᵢ xᵢwᵢ

Step 5 · Update after EVERY pattern
    e = t − y_in
    b(new)  = b(old)  + αe
    wᵢ(new) = wᵢ(old) + αe xᵢ

All weights and bias use the same pre-update y_in. Their new values are used by the next pattern. The thresholded class y does not enter the update rule, even when y already equals t.

Step 6 · Test only after a complete epoch
    max over all patterns and coordinates of |Δwᵢ|, |Δb| < tolerance

The comparison is strictly <. Bias is included as a weight on constant input 1. The largest individual update is different from the net change between the start and end of an epoch: updates can cancel.

The epoch limit is a user-specified calculation budget. Reaching it is not tolerance convergence. A numerical guard stops an unstable run before committing an update with a non-finite intermediate value or magnitude above 1e100. The last valid model and the incomplete epoch are identified.

Diagnostics
SSE = Σ(t − y_in)², evaluated on the whole training set with the fixed model at the end of each epoch. Accuracy uses the bipolar activation function above. Neither metric changes the stopping condition.

Training data
The default follows the supplied bipolar algorithm: inputs and targets use −1 and +1. Binary-input presets and custom finite real inputs are also accepted by the same update equations. Targets and predictions always remain bipolar. Presets visit (1,1), (1,−1), (−1,1), (−1,−1); NOT visits +1 then −1. XOR is included for exploration; one affine separator cannot classify its full truth table.

Numbers and exports
Calculations use Rust f64. Display precision affects only formatting. CSV exports preserve full floating-point precision. Every displayed result belongs to the immutable inputs captured when Run was pressed.
"""


class ExamApp:
    def __init__(self, root):
        self.root = root
        self.result = None
        self.busy = self.closed = self.loading = False
        self.events = queue.Queue()
        self.cancel_event = threading.Event()
        self.step_index = 0
        self.trace_page = self.epoch_page = 0
        self.chart_job = None
        self._known_data = ""
        self.status = tk.StringVar(
            value="Ready · load a preset or enter your training pairs"
        )
        self.preset = tk.StringVar(value="AND")
        self.exercise_choice = tk.StringVar(value=next(iter(LEARNING_EXAMPLES)))
        self.encoding = tk.StringVar(value="Bipolar inputs")
        self.precision = tk.StringVar(value="10")
        self.algorithm = tk.StringVar(value="ADALINE")
        self.workspace = tk.StringVar(value="Learning")
        self._active_algorithm = "ADALINE"
        self.perceptron_variant = tk.StringVar(value=BINARY)
        self.target_hint = tk.StringVar(
            value="One row: x1, x2, …, target\nTargets are always −1 or +1."
        )
        self.hidden_units = tk.StringVar(value="2")
        self.configured_hidden_units = 2
        self.header_text = tk.StringVar(value="ADALINE  /  STEP-BY-STEP WORKSHEET")
        self.fields = {
            key: tk.StringVar(value=value)
            for key, value in {
                "weights": "0, 0",
                "bias": "0",
                "alpha": "0.2",
                "tolerance": "0.001",
                "max_epochs": "100",
                "weights2": "0, 0",
                "bias2": "0",
                "output_weights": "0.5, 0.5",
                "output_bias": "0.5",
                "theta": "0",
            }.items()
        }
        self.hidden_weight_vars = [self.fields["weights"], self.fields["weights2"]]
        self.hidden_bias_vars = [self.fields["bias"], self.fields["bias2"]]
        root.title("Exam lab · ADALINE")
        width = min(1450, max(1100, root.winfo_screenwidth() - 80))
        height = min(960, max(740, root.winfo_screenheight() - 100))
        x = max(0, (root.winfo_screenwidth() - width) // 2)
        y = max(30, (root.winfo_screenheight() - height) // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")
        root.minsize(1100, 740)
        root.configure(bg=BG)
        self._style()
        self._layout()
        self.load_preset()
        for variable in self.fields.values():
            variable.trace_add("write", self._dirty)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Control-Return>", lambda _: self.start())
        if sys.platform == "darwin":
            root.bind("<Command-Return>", lambda _: self.start())
        root.bind("<Alt-Left>", lambda _: self.navigate_step(-1))
        root.bind("<Alt-Right>", lambda _: self.navigate_step(1))
        self.poll_job = root.after(60, self._poll)

    def _style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Arial", 11), foreground=INK)
        style.configure("TFrame", background=BG)
        style.configure("White.TFrame", background="white")
        style.configure("TLabel", background=BG)
        style.configure("Muted.TLabel", foreground=MUTED, font=("Arial", 10))
        style.configure(
            "Section.TLabel", font=("Arial", 11, "bold"), padding=(0, 12, 0, 6)
        )
        style.configure("TButton", padding=(10, 7), background="#dfe7ec", borderwidth=0)
        style.map("TButton", background=[("active", "#cfdce4")])
        style.configure(
            "Run.TButton",
            background=TEAL,
            foreground="white",
            font=("Arial", 12, "bold"),
        )
        style.map(
            "Run.TButton", background=[("disabled", "#95b6b2"), ("active", "#06665f")]
        )
        style.configure("TEntry", padding=5)
        style.configure("TCombobox", padding=5)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(13, 11), background="#dfe7ec")
        style.map("TNotebook.Tab", background=[("selected", "white")])
        style.configure(
            "Treeview",
            font=(MONO, 10),
            rowheight=29,
            background="white",
            fieldbackground="white",
            borderwidth=0,
        )
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"), padding=7)

    def _layout(self):
        header = tk.Frame(self.root, bg=INK, padx=24, pady=17)
        header.pack(fill="x")
        tk.Label(
            header, text="Exam lab", bg=INK, fg="white", font=("Arial", 25, "bold")
        ).pack(side="left")
        tk.Label(
            header,
            textvariable=self.header_text,
            bg=INK,
            fg="#c0d4df",
            font=("Arial", 11),
        ).pack(side="left", padx=25)
        navigation = ttk.Frame(self.root, padding=(24, 10))
        navigation.pack(fill="x")
        ttk.Label(navigation, text="WORKSPACE", style="Muted.TLabel").pack(
            side="left", padx=(0, 14)
        )
        for label in ("Learning", "Optimization"):
            ttk.Radiobutton(
                navigation,
                text=label,
                value=label,
                variable=self.workspace,
                command=self.switch_workspace,
            ).pack(side="left", padx=(0, 20))
        ttk.Label(
            navigation,
            text="Switch at any time · inputs and completed results are retained",
            style="Muted.TLabel",
        ).pack(side="right")
        body = ttk.Frame(self.root, padding=(18, 16, 18, 8))
        self.body = body
        body.pack(fill="both", expand=True)
        sidebar = ttk.Frame(body, width=325)
        sidebar.pack(side="left", fill="y", padx=(0, 18))
        sidebar.pack_propagate(False)
        actions = ttk.Frame(sidebar, padding=(0, 10, 0, 0))
        actions.pack(side="bottom", fill="x")
        self.run_button = ttk.Button(
            actions, text="Run ADALINE", style="Run.TButton", command=self.start
        )
        self.run_button.pack(fill="x")
        self.cancel_button = ttk.Button(
            actions, text="Cancel calculation", command=self.cancel, state="disabled"
        )
        self.cancel_button.pack(fill="x", pady=(6, 0))
        self.progress = ttk.Progressbar(actions, mode="indeterminate")
        self.progress.pack(fill="x", pady=(8, 0))
        sidebar_canvas = tk.Canvas(sidebar, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            sidebar, orient="vertical", command=sidebar_canvas.yview
        )
        sidebar_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        sidebar_canvas.pack(fill="both", expand=True)
        form = ttk.Frame(sidebar_canvas, padding=(0, 0, 10, 8))
        form_window = sidebar_canvas.create_window((0, 0), window=form, anchor="nw")
        form.bind(
            "<Configure>",
            lambda _: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")),
        )
        sidebar_canvas.bind(
            "<Configure>",
            lambda e: sidebar_canvas.itemconfigure(form_window, width=e.width),
        )
        self._form(form)

        def wheel(event):
            sidebar_canvas.yview_scroll(
                -1 if event.delta > 0 or event.num == 4 else 1, "units"
            )

        def bind_wheel(widget):
            if widget is not self.data_input:
                for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    widget.bind(event, wheel, add="+")
                for child in widget.winfo_children():
                    bind_wheel(child)

        bind_wheel(form)
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        cards = ttk.Frame(right)
        cards.pack(fill="x", pady=(0, 12))
        self.metrics = {}
        for i, (key, label) in enumerate(
            (
                ("status", "STOP CONDITION"),
                ("epochs", "COMPLETE PASSES"),
                ("change", "LAST PASS MAX |Δ|"),
                ("accuracy", "FINAL ACCURACY"),
            )
        ):
            card = tk.Frame(cards, bg="white", padx=14, pady=12)
            card.grid(row=0, column=i, sticky="nsew", padx=(0, 8 if i < 3 else 0))
            cards.columnconfigure(i, weight=1, uniform="metric")
            tk.Label(
                card, text=label, bg="white", fg=MUTED, font=("Arial", 9, "bold")
            ).pack(anchor="w")
            var = tk.StringVar(value="—")
            self.metrics[key] = var
            tk.Label(
                card, textvariable=var, bg="white", fg=TEAL, font=("Arial", 14, "bold")
            ).pack(anchor="w", pady=(8, 0))
        self.snapshot = tk.StringVar(
            value="Your completed run will appear here. Default values are ready to calculate."
        )
        ttk.Label(
            right, textvariable=self.snapshot, style="Muted.TLabel", wraplength=900
        ).pack(fill="x", pady=(0, 10))
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)
        self.walk_tab = ttk.Frame(self.notebook, padding=16, style="White.TFrame")
        self.notebook.add(self.walk_tab, text="Worked steps")
        self._walkthrough()
        self._trace_tab()
        self._epochs_tab()
        self._final_tab()
        algorithm_tab = ttk.Frame(self.notebook, padding=16, style="White.TFrame")
        self.notebook.add(algorithm_tab, text="Algorithm")
        self.algorithm_text = self._text(algorithm_tab)
        self._set_text(self.algorithm_text, ALGORITHM)
        footer = ttk.Frame(self.root, padding=(20, 4, 20, 10))
        self.footer = footer
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel").pack(
            side="left"
        )
        ttk.Label(
            footer,
            text="Rust calculations · local desktop · no network",
            style="Muted.TLabel",
        ).pack(side="right")
        from .lecture08_gui import Lecture08Panel

        self.lecture08 = Lecture08Panel(self, self.root)
        from .optimization_gui import OptimizationPanel

        self.optimization = OptimizationPanel(self, self.root)

    def _form(self, parent):
        ttk.Label(parent, text="ALGORITHM", style="Section.TLabel").pack(anchor="w")
        self.algorithm_box = ttk.Combobox(
            parent,
            textvariable=self.algorithm,
            state="readonly",
            values=("ADALINE", "MADALINE (MRI)", "Hebbian Learning", "Perceptron")
            + LECTURE08_OPTIONS,
        )
        self.algorithm_box.pack(fill="x")
        self.algorithm_box.bind(
            "<<ComboboxSelected>>", lambda _: self.switch_algorithm()
        )
        self.perceptron_fields = ttk.Frame(parent)
        ttk.Label(
            self.perceptron_fields, text="Perceptron variant", style="Muted.TLabel"
        ).pack(anchor="w", pady=(8, 3))
        variant = ttk.Combobox(
            self.perceptron_fields,
            textvariable=self.perceptron_variant,
            state="readonly",
            values=(BINARY, BIPOLAR, BIPOLAR_ERROR, BAND),
        )
        variant.pack(fill="x")
        variant.bind("<<ComboboxSelected>>", lambda _: self.change_perceptron_mode())
        self.theta_row = ttk.Frame(self.perceptron_fields)
        ttk.Label(
            self.theta_row, text="Threshold band θ ≥ 0", style="Muted.TLabel"
        ).pack(anchor="w", pady=(6, 3))
        ttk.Entry(self.theta_row, textvariable=self.fields["theta"]).pack(fill="x")
        self.training_section = ttk.Label(
            parent, text="01  TRAINING PAIRS", style="Section.TLabel"
        )
        self.training_section.pack(anchor="w")
        ttk.Combobox(
            parent,
            textvariable=self.exercise_choice,
            values=tuple(LEARNING_EXAMPLES),
            state="readonly",
        ).pack(fill="x", pady=(0, 5))
        ttk.Button(
            parent, text="Load practical exercise", command=self.load_practical_example
        ).pack(fill="x", pady=(0, 8))
        box = ttk.Combobox(
            parent,
            textvariable=self.preset,
            state="readonly",
            values=("AND", "OR", "NAND", "NOR", "XOR", "NOT", "Custom"),
        )
        box.pack(fill="x")
        box.bind("<<ComboboxSelected>>", lambda _: self.load_preset())
        encoding = ttk.Combobox(
            parent,
            textvariable=self.encoding,
            state="readonly",
            values=("Bipolar inputs", "Binary inputs"),
        )
        encoding.pack(fill="x", pady=(6, 0))
        encoding.bind("<<ComboboxSelected>>", lambda _: self.load_preset())
        ttk.Label(
            parent,
            textvariable=self.target_hint,
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(8, 6))
        self.data_input = tk.Text(
            parent,
            height=7,
            width=26,
            wrap="none",
            font=(MONO, 11),
            bg="white",
            fg=INK,
            relief="flat",
            padx=9,
            pady=9,
            undo=True,
            highlightthickness=1,
            highlightbackground="#cbd6df",
        )
        self.data_input.pack(fill="x")
        self.data_input.bind("<<Modified>>", self._data_modified)
        ttk.Button(parent, text="Import training CSV…", command=self.import_data).pack(
            fill="x", pady=(7, 0)
        )
        ttk.Label(parent, text="02  INITIAL MODEL", style="Section.TLabel").pack(
            anchor="w"
        )
        self.weights_label = ttk.Label(
            parent, text="Weights w1, w2, …", style="Muted.TLabel"
        )
        self.weights_label.pack(anchor="w")
        self.weights_entry = ttk.Entry(parent, textvariable=self.fields["weights"])
        self.weights_entry.pack(fill="x", pady=(3, 6))
        self.bias_label = ttk.Label(parent, text="Bias b", style="Muted.TLabel")
        self.bias_label.pack(anchor="w")
        self.bias_entry = ttk.Entry(parent, textvariable=self.fields["bias"])
        self.bias_entry.pack(fill="x", pady=(3, 6))
        self.hebbian_note = ttk.Label(
            parent,
            text="All weights and bias start at zero.\nExactly one pass, then verification.",
            style="Muted.TLabel",
        )
        self.madaline_fields = ttk.Frame(parent)
        ttk.Label(
            self.madaline_fields, text="Hidden ADALINE units", style="Muted.TLabel"
        ).pack(anchor="w")
        count = ttk.Combobox(
            self.madaline_fields,
            textvariable=self.hidden_units,
            state="readonly",
            values=tuple(map(str, range(1, MAX_HIDDEN_UNITS + 1))),
            width=6,
        )
        count.pack(anchor="w", pady=(3, 8))
        count.bind("<<ComboboxSelected>>", lambda _: self.resize_hidden_units())
        self.extra_hidden_fields = ttk.Frame(self.madaline_fields)
        self.extra_hidden_fields.pack(fill="x")
        self._render_hidden_inputs()
        for key, label in (
            ("output_weights", "Fixed output weights · one per hidden unit"),
            ("output_bias", "Fixed output bias b_out"),
        ):
            ttk.Label(self.madaline_fields, text=label, style="Muted.TLabel").pack(
                anchor="w"
            )
            ttk.Entry(self.madaline_fields, textvariable=self.fields[key]).pack(
                fill="x", pady=(3, 6)
            )
        ttk.Button(
            self.madaline_fields,
            text="Use OR output defaults",
            command=self.reset_output_defaults,
        ).pack(fill="x", pady=(0, 6))
        ttk.Label(
            self.madaline_fields,
            text="OR defaults: vⱼ = 0.5, b_out = (H−1)/2.\nFor t = −1, only z_in > 0 is updated;\nzero initialization can stop with errors.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 8))
        init = self.init_frame = ttk.Frame(parent)
        init.pack(fill="x")
        ttk.Button(init, text="Use zeros", command=lambda: self.initialize(False)).pack(
            side="left", fill="x", expand=True, padx=(0, 5)
        )
        ttk.Button(
            init, text="Random · seed 42", command=lambda: self.initialize(True)
        ).pack(side="left", fill="x", expand=True)
        self.random_note = ttk.Label(
            parent, text="Random values are in [−0.1, 0.1].", style="Muted.TLabel"
        )
        self.random_note.pack(anchor="w", pady=(5, 0))
        self.learning_section = ttk.Label(
            parent, text="03  LEARNING SETTINGS", style="Section.TLabel"
        )
        self.learning_section.pack(anchor="w")
        self.learning_rows = {}
        for key, label in (
            ("alpha", "Learning rate α"),
            ("tolerance", "Tolerance · strict max |Δ| < ε"),
            ("max_epochs", "Maximum epochs"),
        ):
            row = ttk.Frame(parent)
            row.pack(fill="x")
            self.learning_rows[key] = row
            label_widget = ttk.Label(row, text=label, style="Muted.TLabel")
            label_widget.pack(anchor="w", pady=(0, 3))
            if key == "alpha":
                self.rate_label = label_widget
            ttk.Entry(row, textvariable=self.fields[key]).pack(fill="x", pady=(0, 7))
        self.display_section = ttk.Label(
            parent, text="04  DISPLAY", style="Section.TLabel"
        )
        self.display_section.pack(anchor="w")
        precision = ttk.Combobox(
            parent,
            textvariable=self.precision,
            values=("6", "10", "15"),
            state="readonly",
            width=6,
        )
        precision.pack(anchor="w")
        precision.bind("<<ComboboxSelected>>", lambda _: self._refresh())
        ttk.Label(
            parent,
            text="Significant digits; exports keep full precision.\nRun: Ctrl/⌘ + Enter · Steps: Alt + ←/→",
            style="Muted.TLabel",
            wraplength=290,
        ).pack(anchor="w", pady=(6, 0))

    def _text(self, parent, height=None):
        frame = tk.Frame(parent, bg="white")
        frame.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(frame)
        scroll.pack(side="right", fill="y")
        text = tk.Text(
            frame,
            wrap="word",
            bg="white",
            fg=INK,
            relief="flat",
            highlightthickness=0,
            borderwidth=0,
            font=(MONO, 12),
            padx=12,
            pady=10,
            spacing1=4,
            spacing3=5,
            yscrollcommand=scroll.set,
            height=height or 12,
        )
        text.pack(fill="both", expand=True)
        scroll.configure(command=text.yview)
        text.configure(state="disabled")
        return text

    @staticmethod
    def _set_text(widget, content):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _walkthrough(self):
        nav = tk.Frame(self.walk_tab, bg="white")
        nav.pack(fill="x")
        self.nav_buttons = []
        for label, action in (
            ("← Previous", lambda: self.go_step(self.step_index - 1)),
            ("Next pattern →", lambda: self.go_step(self.step_index + 1)),
            ("Next epoch →", self.next_epoch),
        ):
            button = ttk.Button(nav, text=label, command=action, state="disabled")
            button.pack(side="left", padx=(0, 7))
            self.nav_buttons.append(button)
        self.step_label = tk.StringVar(value="Run a worksheet to inspect each update")
        tk.Label(
            self.walk_tab,
            textvariable=self.step_label,
            bg="white",
            fg=TEAL,
            font=("Arial", 13, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(13, 0))
        self.slider = tk.Scale(
            self.walk_tab,
            from_=0,
            to=1,
            orient="horizontal",
            showvalue=False,
            highlightthickness=0,
            bg="white",
            troughcolor="#dbe9e7",
            activebackground=TEAL,
            command=self._slider_changed,
            state="disabled",
        )
        self.slider.pack(fill="x", pady=(4, 10))
        self.worked_text = self._text(self.walk_tab)
        self._set_text(
            self.worked_text,
            "Follow every calculation from your lecture.\n\n1. Edit the training pairs and initial model.\n2. Set α, tolerance and the epoch budget.\n3. Press Run ADALINE.\n\nEach pattern will show the substituted net input, prediction, error, every weight update, and the end-of-epoch stopping test.",
        )

    def _table(self, parent):
        frame = tk.Frame(parent, bg="white")
        frame.pack(fill="both", expand=True)
        y = ttk.Scrollbar(frame, orient="vertical")
        x = ttk.Scrollbar(frame, orient="horizontal")
        tree = ttk.Treeview(
            frame,
            show="headings",
            yscrollcommand=y.set,
            xscrollcommand=x.set,
            selectmode="browse",
        )
        y.configure(command=tree.yview)
        x.configure(command=tree.xview)
        tree.grid(row=0, column=0, sticky="nsew")
        y.grid(row=0, column=1, sticky="ns")
        x.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    @staticmethod
    def _columns(tree, labels):
        tree.delete(*tree.get_children())
        columns = tuple(f"c{i}" for i in range(len(labels)))
        tree.configure(columns=columns)
        for column, label in zip(columns, labels):
            tree.heading(column, text=label)
            tree.column(
                column,
                width=max(100, len(label) * 8 + 18),
                minwidth=75,
                stretch=False,
                anchor="e",
            )

    def _trace_tab(self):
        self.trace_tab = ttk.Frame(self.notebook, padding=12, style="White.TFrame")
        self.notebook.add(self.trace_tab, text="Full training trace")
        toolbar = tk.Frame(self.trace_tab, bg="white")
        toolbar.pack(fill="x", pady=(0, 10))
        self.trace_prev = ttk.Button(
            toolbar,
            text="←",
            command=lambda: self.change_page("trace", -1),
            state="disabled",
        )
        self.trace_prev.pack(side="left")
        self.trace_page_text = tk.StringVar(value="No updates yet")
        tk.Label(toolbar, textvariable=self.trace_page_text, bg="white", fg=MUTED).pack(
            side="left", padx=10
        )
        self.trace_next = ttk.Button(
            toolbar,
            text="→",
            command=lambda: self.change_page("trace", 1),
            state="disabled",
        )
        self.trace_next.pack(side="left")
        self.jump_label = ttk.Label(toolbar, text="  Epoch", background="white")
        self.jump_label.pack(side="left")
        self.jump_value = tk.StringVar(value="1")
        ttk.Entry(toolbar, textvariable=self.jump_value, width=6).pack(
            side="left", padx=5
        )
        ttk.Button(toolbar, text="Go", command=self.jump_epoch).pack(side="left")
        self.export_button = ttk.Button(
            toolbar, text="Export all CSV…", command=self.save_trace, state="disabled"
        )
        self.export_button.pack(side="right")
        tk.Label(
            self.trace_tab,
            text="Select a row to update Worked steps. Scroll horizontally to see every coordinate.",
            bg="white",
            fg=MUTED,
            anchor="w",
        ).pack(fill="x", pady=(0, 8))
        self.trace_tree = self._table(self.trace_tab)
        self.trace_tree.bind("<<TreeviewSelect>>", self._trace_selected)
        self.trace_tree.bind(
            "<Double-1>", lambda _: self.notebook.select(self.walk_tab)
        )

    def _epochs_tab(self):
        tab = ttk.Frame(self.notebook, padding=12, style="White.TFrame")
        self.epoch_tab = tab
        self.notebook.add(tab, text="Epochs & learning")
        self.chart = tk.Canvas(tab, bg="white", height=230, highlightthickness=0)
        self.chart.pack(fill="x", pady=(0, 12))
        self.chart.bind("<Configure>", self._queue_chart)
        toolbar = tk.Frame(tab, bg="white")
        toolbar.pack(fill="x", pady=(0, 10))
        self.epoch_prev = ttk.Button(
            toolbar,
            text="←",
            command=lambda: self.change_page("epoch", -1),
            state="disabled",
        )
        self.epoch_prev.pack(side="left")
        self.epoch_page_text = tk.StringVar(value="No complete epochs yet")
        tk.Label(toolbar, textvariable=self.epoch_page_text, bg="white", fg=MUTED).pack(
            side="left", padx=10
        )
        self.epoch_next = ttk.Button(
            toolbar,
            text="→",
            command=lambda: self.change_page("epoch", 1),
            state="disabled",
        )
        self.epoch_next.pack(side="left")
        self.export_epoch_button = ttk.Button(
            toolbar,
            text="Export epochs CSV…",
            command=self.save_epochs,
            state="disabled",
        )
        self.export_epoch_button.pack(side="right")
        self.epoch_tree = self._table(tab)
        self.epoch_tree.bind("<Double-1>", self._epoch_selected)

    def _final_tab(self):
        tab = ttk.Frame(self.notebook, padding=16, style="White.TFrame")
        self.notebook.add(tab, text="Final model")
        self.final_text = self._text(tab, height=7)
        self._set_text(
            self.final_text,
            "The final separating equation, weights, bias, SSE, and stopping outcome will appear here.",
        )
        self.prediction_tree = self._table(tab)

    def fmt(self, value):
        return f"{value:.{int(self.precision.get())}g}"

    def _hidden_count(self):
        try:
            units = int(self.hidden_units.get())
        except ValueError as exc:
            raise InputError("Hidden unit count must be an integer.") from exc
        if not 1 <= units <= MAX_HIDDEN_UNITS:
            raise InputError(f"Use 1 to {MAX_HIDDEN_UNITS} hidden ADALINE units.")
        return units

    def _render_hidden_inputs(self):
        for child in self.extra_hidden_fields.winfo_children():
            child.destroy()
        for j in range(1, self.configured_hidden_units):
            for variable, label in (
                (self.hidden_weight_vars[j], f"Z{j + 1} input weights"),
                (self.hidden_bias_vars[j], f"Z{j + 1} bias b{j + 1}"),
            ):
                ttk.Label(
                    self.extra_hidden_fields, text=label, style="Muted.TLabel"
                ).pack(anchor="w")
                ttk.Entry(self.extra_hidden_fields, textvariable=variable).pack(
                    fill="x", pady=(3, 6)
                )

    def reset_output_defaults(self):
        units = self._hidden_count()
        self.fields["output_weights"].set(", ".join("0.5" for _ in range(units)))
        self.fields["output_bias"].set(f"{(units - 1) * 0.5:g}")

    def resize_hidden_units(self):
        try:
            units = self._hidden_count()
        except InputError as exc:
            self.hidden_units.set(str(self.configured_hidden_units))
            messagebox.showerror("Hidden units", str(exc), parent=self.root)
            return
        if units == self.configured_hidden_units:
            return
        try:
            n = len(parse_samples(self.data_input.get("1.0", "end"))[0].inputs)
        except InputError:
            n = 2
        n = min(n, 32)
        rng = random.Random(42)
        for _ in range((n + 1) * len(self.hidden_weight_vars)):
            rng.uniform(-0.1, 0.1)
        while len(self.hidden_weight_vars) < units:
            j = len(self.hidden_weight_vars)
            values = [f"{rng.uniform(-0.1, 0.1):.6f}" for _ in range(n + 1)]
            weights = tk.StringVar(value=", ".join(values[:n]))
            bias = tk.StringVar(value=values[n])
            self.fields[f"weights{j + 1}"] = weights
            self.fields[f"bias{j + 1}"] = bias
            self.hidden_weight_vars.append(weights)
            self.hidden_bias_vars.append(bias)
            weights.trace_add("write", self._dirty)
            bias.trace_add("write", self._dirty)
        self.configured_hidden_units = units
        self._render_hidden_inputs()
        self.reset_output_defaults()
        self._dirty()

    def switch_workspace(self):
        if self.workspace.get() == "Optimization":
            self.body.pack_forget()
            self.footer.pack_forget()
            self.lecture08.frame.pack_forget()
            self.optimization.frame.pack(fill="both", expand=True)
            self.optimization.update_header()
        else:
            self.switch_algorithm(restore=True)

    def navigate_step(self, delta):
        panel = (
            self.optimization
            if self.workspace.get() == "Optimization"
            else self.lecture08
            if self.algorithm.get() in LECTURE08_OPTIONS
            else self
        )
        panel.go_step(panel.step_index + delta)

    def switch_algorithm(self, *, restore=False):
        self.workspace.set("Learning")
        self.optimization.frame.pack_forget()
        if self.algorithm.get() in LECTURE08_OPTIONS:
            self.body.pack_forget()
            self.footer.pack_forget()
            self.lecture08.frame.pack(fill="both", expand=True)
            self.lecture08.select(self.algorithm.get())
            self.header_text.set(self.algorithm.get())
            self.root.title(f"Exam lab · {self.algorithm.get()}")
            return
        if not self.body.winfo_manager():
            self.lecture08.frame.pack_forget()
            self.body.pack(fill="both", expand=True)
            self.footer.pack(fill="x")
            restore = True
        if restore and self.algorithm.get() == self._active_algorithm:
            self.header_text.set(f"{self.algorithm.get()}  /  STEP-BY-STEP WORKSHEET")
            self.root.title(f"Exam lab · {self.algorithm.get()}")
            return
        madaline = self.algorithm.get() == "MADALINE (MRI)"
        hebbian = self.algorithm.get() == "Hebbian Learning"
        perceptron = self.algorithm.get() == "Perceptron"
        was_perceptron = self._active_algorithm == "Perceptron"
        self._active_algorithm = self.algorithm.get()
        self.header_text.set(f"{self.algorithm.get()}  /  STEP-BY-STEP WORKSHEET")
        self.root.title(f"Exam lab · {self.algorithm.get()}")
        self.run_button.configure(
            text=f"Run {'Perceptron' if perceptron else 'Hebbian Learning' if hebbian else 'MADALINE' if madaline else 'ADALINE'}"
        )
        self.weights_label.configure(
            text="Z1 weights w11, w21, …" if madaline else "Weights w1, w2, …"
        )
        self.bias_label.configure(text="Z1 bias b1" if madaline else "Bias b")
        self.weights_entry.configure(state="readonly" if hebbian else "normal")
        self.bias_entry.configure(state="readonly" if hebbian else "normal")
        self.rate_label.configure(
            text="Learning rate η" if hebbian else "Learning rate α"
        )
        if hebbian:
            self.madaline_fields.pack_forget()
            self.init_frame.pack_forget()
            self.random_note.pack_forget()
            self.hebbian_note.pack(
                anchor="w", pady=(5, 0), before=self.learning_section
            )
            for key in ("tolerance", "max_epochs"):
                self.learning_rows[key].pack_forget()
        else:
            self.hebbian_note.pack_forget()
            self.random_note.pack(anchor="w", pady=(5, 0), before=self.learning_section)
            self.init_frame.pack(fill="x", before=self.random_note)
            for key in ("tolerance", "max_epochs"):
                self.learning_rows[key].pack(fill="x", before=self.display_section)
            if perceptron:
                self.learning_rows["tolerance"].pack_forget()
            if madaline:
                self.madaline_fields.pack(fill="x", before=self.init_frame)
            else:
                self.madaline_fields.pack_forget()
        self._set_text(
            self.algorithm_text,
            perceptron_view.ALGORITHM
            if perceptron
            else hebbian_view.ALGORITHM
            if hebbian
            else madaline_view.ALGORITHM
            if madaline
            else ALGORITHM,
        )
        if perceptron:
            self.perceptron_fields.pack(fill="x", before=self.training_section)
            self.change_perceptron_mode(reload=False)
            if not was_perceptron:
                self.fields["alpha"].set("1")
                self.encoding.set(
                    "Binary inputs"
                    if self._perceptron_mode() == "binary"
                    else "Bipolar inputs"
                )
                self.load_preset()
        else:
            self.perceptron_fields.pack_forget()
            self.target_hint.set(
                "One row: x1, x2, …, target\nTargets are always −1 or +1."
            )
            if was_perceptron:
                self.load_preset()
        self.initialize(madaline)
        self._dirty()

    def _perceptron_mode(self):
        if self.perceptron_variant.get() == BINARY:
            return "binary"
        if self.perceptron_variant.get() == BIPOLAR:
            return "bipolar"
        if self.perceptron_variant.get() == BIPOLAR_ERROR:
            return "bipolar_error"
        if self.perceptron_variant.get() == BAND:
            return "band"
        raise InputError(
            "Choose binary, bipolar sign, bipolar error-update or threshold-band mode."
        )

    def change_perceptron_mode(self, *, reload=True):
        mode = self._perceptron_mode()
        binary = mode == "binary"
        if mode == "band":
            self.theta_row.pack(fill="x")
        else:
            self.theta_row.pack_forget()
        self.rate_label.configure(
            text="Learning rate η · 0 < η ≤ 1"
            if binary
            else "Learning rate α · 0 < α ≤ 1"
        )
        self.target_hint.set(
            "One row: x1, x2, …, target\nTargets: 0 or 1."
            if binary
            else "One row: x1, x2, …, target\nTargets and outputs: −1 or +1."
            if mode in ("bipolar", "bipolar_error")
            else "One row: x1, x2, …, target\nTargets: −1 or +1; output may be 0."
        )
        if reload:
            self.encoding.set("Binary inputs" if binary else "Bipolar inputs")
            self.load_preset()
        self._dirty()

    def _dirty(self, *_):
        if not self.loading and self.result:
            self.status.set(
                "Inputs edited · displayed results use the last run's input snapshot"
            )

    def _data_modified(self, _=None):
        if self.data_input.edit_modified():
            current = self.data_input.get("1.0", "end-1c")
            if not self.loading and current != self._known_data:
                self.preset.set("Custom")
                self._dirty()
            self._known_data = current
            self.data_input.edit_modified(False)
            if self.algorithm.get() == "Hebbian Learning":
                try:
                    n = len(parse_samples(current)[0].inputs)
                except InputError:
                    return
                if 1 <= n <= 32:
                    self.fields["weights"].set(", ".join("0" for _ in range(n)))
                    self.fields["bias"].set("0")

    def load_preset(self):
        if self.preset.get() == "Custom":
            return
        samples = (
            perceptron_samples(
                self.preset.get(),
                self.encoding.get() == "Bipolar inputs",
                self._perceptron_mode(),
            )
            if self.algorithm.get() == "Perceptron"
            else preset_samples(
                self.preset.get(), self.encoding.get() == "Bipolar inputs"
            )
        )
        self.loading = True
        try:
            self._known_data = samples_text(samples)
            self.data_input.delete("1.0", "end")
            self.data_input.insert("1.0", self._known_data)
            self.data_input.edit_modified(False)
            self.fields["weights"].set(", ".join("0" for _ in samples[0].inputs))
            self.fields["bias"].set("0")
            self.fields["weights2"].set(", ".join("0" for _ in samples[0].inputs))
            self.fields["bias2"].set("0")
            if self.algorithm.get() == "MADALINE (MRI)":
                self.initialize(True)
        finally:
            self.loading = False
        self._dirty()

    def load_practical_example(self):
        values = LEARNING_EXAMPLES[self.exercise_choice.get()]
        self.algorithm.set(values["algorithm"])
        self.switch_algorithm()
        if values["algorithm"] == "Perceptron":
            self.perceptron_variant.set(
                {"binary": BINARY, "bipolar_error": BIPOLAR_ERROR, "band": BAND}[
                    values["mode"]
                ]
            )
            self.switch_algorithm()
        self.loading = True
        try:
            self.preset.set("Custom")
            self._known_data = values["data"]
            self.data_input.delete("1.0", "end")
            self.data_input.insert("1.0", self._known_data)
            self.data_input.edit_modified(False)
            self.encoding.set(
                "Binary inputs"
                if values["algorithm"] == "Perceptron"
                else "Bipolar inputs"
            )
            if values["algorithm"] == "MADALINE (MRI)":
                self.hidden_units.set("2")
                self.resize_hidden_units()
            for key, default in {
                "weights": "0,0",
                "bias": "0",
                "weights2": "0,0",
                "bias2": "0",
                "alpha": "1",
                "theta": "0",
                "tolerance": "0.001",
                "max_epochs": "100",
                "output_weights": "0.5,0.5",
                "output_bias": "0.5",
            }.items():
                self.fields[key].set(values.get(key, default))
            if "weights" not in values:
                self.fields["weights"].set(
                    ",".join("0" for _ in parse_samples(values["data"])[0].inputs)
                )
        finally:
            self.loading = False
        self._dirty()

    def initialize(self, randomize):
        try:
            if self.algorithm.get() == "Hebbian Learning":
                randomize = False
            n = len(parse_samples(self.data_input.get("1.0", "end"))[0].inputs)
            if n > 32:
                raise InputError("Use at most 32 inputs.")
            rng = random.Random(42)
            units = (
                self._hidden_count() if self.algorithm.get() == "MADALINE (MRI)" else 1
            )
            values = [
                f"{rng.uniform(-0.1, 0.1):.6f}" if randomize else "0"
                for _ in range((n + 1) * units)
            ]
            for j in range(units):
                start = j * (n + 1)
                self.hidden_weight_vars[j].set(", ".join(values[start : start + n]))
                self.hidden_bias_vars[j].set(values[start + n])
        except InputError as exc:
            messagebox.showerror("Training data", str(exc), parent=self.root)

    def import_data(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Import training pairs",
            filetypes=[("Training data", "*.csv *.txt *.tsv"), ("All files", "*")],
        )
        if not path:
            return
        try:
            if Path(path).stat().st_size > 1_048_576:
                raise InputError("Training files must be smaller than 1 MiB.")
            text = Path(path).read_text(encoding="utf-8-sig")
            samples = parse_samples(text)
            n = len(samples[0].inputs)
            if self.algorithm.get() == "Perceptron":
                PerceptronRequest(
                    samples, (0.0,) * n, mode=self._perceptron_mode(), max_epochs=1
                ).validate()
            else:
                Request(samples, (0.0,) * n, max_epochs=1).validate()
            self.loading = True
            self._known_data = samples_text(samples)
            self.data_input.delete("1.0", "end")
            self.data_input.insert("1.0", self._known_data)
            self.data_input.edit_modified(False)
            self.preset.set("Custom")
            self.fields["weights"].set(", ".join("0" for _ in range(n)))
            self.fields["bias"].set("0")
            self.fields["weights2"].set(", ".join("0" for _ in range(n)))
            self.fields["bias2"].set("0")
            if self.algorithm.get() == "MADALINE (MRI)":
                self.initialize(True)
        except (OSError, UnicodeError, InputError) as exc:
            messagebox.showerror("Could not import data", str(exc), parent=self.root)
        finally:
            self.loading = False
        self._dirty()

    def read_request(self):
        if self.algorithm.get() == "Hebbian Learning":
            request = HebbianRequest(
                parse_samples(self.data_input.get("1.0", "end")),
                number(self.fields["alpha"].get(), "Learning rate η"),
            )
            request.validate()
            return request
        try:
            epochs = int(self.fields["max_epochs"].get())
        except ValueError as exc:
            raise InputError("Maximum epochs must be an integer.") from exc
        if self.algorithm.get() == "Perceptron":
            mode = self._perceptron_mode()
            request = PerceptronRequest(
                samples=parse_samples(self.data_input.get("1.0", "end")),
                weights=vector(self.fields["weights"].get()),
                bias=number(self.fields["bias"].get(), "Bias"),
                alpha=number(self.fields["alpha"].get(), "Learning rate"),
                max_epochs=epochs,
                mode=mode,
                theta=number(self.fields["theta"].get(), "Theta")
                if mode == "band"
                else 0.0,
            )
            request.validate()
            return request
        request = Request(
            samples=parse_samples(self.data_input.get("1.0", "end")),
            weights=vector(self.fields["weights"].get()),
            bias=number(self.fields["bias"].get(), "Bias"),
            alpha=number(self.fields["alpha"].get(), "Learning rate"),
            tolerance=number(self.fields["tolerance"].get(), "Tolerance"),
            max_epochs=epochs,
        )
        if self.algorithm.get() == "MADALINE (MRI)":
            units = self._hidden_count()
            if units != self.configured_hidden_units:
                raise InputError(
                    "Select the hidden unit count to create its input fields."
                )
            request = MadalineRequest(
                samples=request.samples,
                hidden_weights=tuple(
                    vector(v.get(), f"Z{j + 1} weights")
                    for j, v in enumerate(self.hidden_weight_vars[:units])
                ),
                hidden_biases=tuple(
                    number(v.get(), f"Z{j + 1} bias")
                    for j, v in enumerate(self.hidden_bias_vars[:units])
                ),
                output_weights=vector(
                    self.fields["output_weights"].get(), "Output weights"
                ),
                output_bias=number(self.fields["output_bias"].get(), "Output bias"),
                alpha=request.alpha,
                tolerance=request.tolerance,
                max_epochs=request.max_epochs,
            )
        request.validate()
        return request

    def start(self):
        if self.workspace.get() == "Optimization":
            return self.optimization.start()
        if self.algorithm.get() in LECTURE08_OPTIONS:
            return self.lecture08.start()
        if self.busy or self.closed:
            return
        try:
            request = self.read_request()
        except InputError as exc:
            messagebox.showerror("Check worksheet inputs", str(exc), parent=self.root)
            return
        self.busy = True
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
            self.status.set("Cancelling calculation…")
            self.cancel_button.configure(state="disabled")

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
                if kind == "done" and not self.cancel_event.is_set():
                    self.result = data
                    self.step_index = self.trace_page = self.epoch_page = 0
                    self._refresh()
                    self.status.set(
                        f"{STATUS[data.status]} · {len(data.steps):,} pattern updates recorded"
                    )
                    if self.read_request_safely() != data.request:
                        self._dirty()
                elif kind == "cancelled" or self.cancel_event.is_set():
                    self.status.set(
                        "Calculation cancelled · previous completed results retained"
                    )
                elif kind == "error":
                    self.status.set(
                        "Calculation failed · previous completed results retained"
                    )
                    messagebox.showerror("Calculation failed", data, parent=self.root)
        except queue.Empty:
            pass
        self.poll_job = self.root.after(60, self._poll)

    def read_request_safely(self):
        try:
            return self.read_request()
        except InputError:
            return None

    def _refresh(self):
        if not self.result:
            return
        r = self.result
        self.metrics["status"].set(
            {
                "pass_complete": "One pass complete",
                "weights_stable": "Changes stopped",
                "tolerance_met": "Tolerance met",
                "epoch_limit": "Epoch limit",
                "numerical_limit": "Numerical limit",
            }[r.status]
        )
        self.metrics["epochs"].set(f"{r.completed_epochs:,} / {r.request.max_epochs:,}")
        self.metrics["change"].set(
            self.fmt(r.epochs[-1].max_change) if r.epochs else "—"
        )
        self.metrics["accuracy"].set(f"{r.correct} / {len(r.request.samples)} correct")
        self.snapshot.set(
            perceptron_view.snapshot(self)
            if isinstance(r, PerceptronResult)
            else hebbian_view.snapshot(self)
            if isinstance(r, HebbianResult)
            else madaline_view.snapshot(self)
            if isinstance(r, MadalineResult)
            else f"Run snapshot: ADALINE · α = {self.fmt(r.request.alpha)}   ·   ε = {self.fmt(r.request.tolerance)}   ·   {len(r.weights)} inputs   ·   {len(r.request.samples)} patterns   ·   initial w = ({', '.join(self.fmt(v) for v in r.request.weights)}), b = {self.fmt(r.request.bias)}"
        )
        hebbian = isinstance(r, HebbianResult)
        self.notebook.tab(
            self.epoch_tab, text="Pass summary" if hebbian else "Epochs & learning"
        )
        self.nav_buttons[2].configure(
            text="End of pass →" if hebbian else "Next epoch →"
        )
        self.jump_label.configure(text="  Pass" if hebbian else "  Epoch")
        self.slider.configure(
            state="normal" if r.steps else "disabled", to=max(1, len(r.steps) - 1)
        )
        self.export_button.configure(state="normal")
        self.export_epoch_button.configure(state="normal")
        self.export_epoch_button.configure(
            text="Export pass CSV…" if hebbian else "Export epochs CSV…"
        )
        self._render_trace()
        self._render_epochs()
        self._render_final()
        self.go_step(self.step_index)
        self.draw_chart()

    def _slider_changed(self, value):
        index = int(float(value))
        if self.result and index != self.step_index:
            self.go_step(index)

    def go_step(self, index):
        if not self.result:
            return
        r = self.result
        if not r.steps:
            self.step_label.set("No completed updates · numerical guard reached")
            self._set_text(
                self.worked_text,
                f"The first attempted update reached the numerical guard at {'pass' if isinstance(r, HebbianResult) else 'epoch'} {r.failed_at[0]}, pattern {r.failed_at[1]}.\n\nNo update was committed. Final model shows the initial weights and bias. Try a smaller learning rate or smaller inputs.",
            )
            for b in self.nav_buttons:
                b.configure(state="disabled")
            return
        self.step_index = max(0, min(index, len(r.steps) - 1))
        index = self.step_index
        s = r.steps[index]
        sample = r.request.samples[s.pattern - 1]
        f = self.fmt
        self.slider.set(index)
        self.nav_buttons[0].configure(state="normal" if index > 0 else "disabled")
        for button in self.nav_buttons[1:]:
            button.configure(state="normal" if index < len(r.steps) - 1 else "disabled")
        self.step_label.set(
            f"{'Pass' if isinstance(r, HebbianResult) else 'Epoch'} {s.epoch}  ·  Pattern {s.pattern}/{len(r.request.samples)}  ·  Update {index + 1:,}/{len(r.steps):,}"
        )
        if isinstance(r, HebbianResult):
            hebbian_view.worked_step(self, s)
            return
        if isinstance(r, PerceptronResult):
            perceptron_view.worked_step(self, s)
            return
        if isinstance(r, MadalineResult):
            madaline_view.worked_step(self, s)
            return
        products = " + ".join(
            f"({f(x)})×({f(w)})" for x, w in zip(sample.inputs, s.before_weights)
        )
        lines = [
            f"STEP 3 · Set input activations\nx = s = ({', '.join(f(x) for x in sample.inputs)})    target t = {f(sample.target)}",
            f"STEP 4 · Compute the net input from the OLD model\ny_in = b + Σ xᵢwᵢ\n     = ({f(s.before_bias)}) + {products}\n     = {f(s.net)}",
            f"Prediction: y = {'+1' if s.output == 1 else '−1'} because y_in {'≥' if s.net >= 0 else '<'} 0.\nThe update uses y_in = {f(s.net)}.",
            f"STEP 5 · Compute the shared error and update\ne = t − y_in = ({f(sample.target)}) − ({f(s.net)}) = {f(s.error)}\nαe = ({f(r.request.alpha)}) × ({f(s.error)}) = {f(s.delta_bias)}",
            f"b(new) = b(old) + αe\n       = ({f(s.before_bias)}) + ({f(s.delta_bias)}) = {f(s.after_bias)}",
        ]
        for i, (x, before, delta, after) in enumerate(
            zip(sample.inputs, s.before_weights, s.delta_weights, s.after_weights), 1
        ):
            lines.append(
                f"w{i}(new) = w{i}(old) + αe × x{i}\n        = ({f(before)}) + ({f(s.delta_bias)}) × ({f(x)})\n        = ({f(before)}) + ({f(delta)}) = {f(after)}"
            )
        lines.append(f"Largest absolute update for THIS pattern: {f(s.max_change)}")
        if s.pattern == len(r.request.samples):
            e = r.epochs[s.epoch - 1]
            passed = e.max_change < r.request.tolerance
            lines.append(
                f"STEP 6 · Complete epoch {e.number}: test stopping condition\nmax(|Δwᵢ|, |Δb|) over ALL patterns = {f(e.max_change)}\n{f(e.max_change)} < {f(r.request.tolerance)}  →  {'TRUE: tolerance met; stop.' if passed else 'FALSE: tolerance not met.'}\nNet change from epoch start to end = {f(e.net_change)}\nFixed-model SSE = {f(e.sse)}; correct = {e.correct}/{len(r.request.samples)}"
            )
            if not passed:
                lines.append(
                    "Epoch budget exhausted; stop with status: epoch limit."
                    if e.number == r.request.max_epochs
                    else "Continue with the next epoch."
                )
        else:
            lines.append(
                "STEP 6 · Wait until every pattern in this epoch is processed."
            )
        if index == len(r.steps) - 1 and r.failed_at:
            lines.append(
                f"Numerical guard reached on the NEXT attempted update: epoch {r.failed_at[0]}, pattern {r.failed_at[1]}. That update was not committed."
            )
        self._set_text(self.worked_text, "\n\n".join(lines))

    def next_epoch(self):
        if self.result and self.result.steps:
            epoch = self.result.steps[self.step_index].epoch
            self.go_step(epoch * len(self.result.request.samples))

    def _render_trace(self):
        r = self.result
        if isinstance(r, PerceptronResult):
            return perceptron_view.render_trace(self, PAGE_SIZE)
        if isinstance(r, HebbianResult):
            return hebbian_view.render_trace(self, PAGE_SIZE)
        if isinstance(r, MadalineResult):
            return madaline_view.render_trace(self, PAGE_SIZE)
        n = len(r.weights)
        labels = [
            "Epoch",
            "Pattern",
            *[f"x{i + 1}" for i in range(n)],
            "Target",
            *[f"w{i + 1} before" for i in range(n)],
            "b before",
            "y_in",
            "y",
            "Error",
            *[f"Δw{i + 1}" for i in range(n)],
            "Δb",
            *[f"w{i + 1} after" for i in range(n)],
            "b after",
            "Max |Δ|",
        ]
        self._columns(self.trace_tree, labels)
        start = self.trace_page * PAGE_SIZE
        for index in range(start, min(start + PAGE_SIZE, len(r.steps))):
            s = r.steps[index]
            sample = r.request.samples[s.pattern - 1]
            values = [
                s.epoch,
                s.pattern,
                *map(self.fmt, sample.inputs),
                self.fmt(sample.target),
                *map(self.fmt, s.before_weights),
                self.fmt(s.before_bias),
                self.fmt(s.net),
                s.output,
                self.fmt(s.error),
                *map(self.fmt, s.delta_weights),
                self.fmt(s.delta_bias),
                *map(self.fmt, s.after_weights),
                self.fmt(s.after_bias),
                self.fmt(s.max_change),
            ]
            self.trace_tree.insert("", "end", iid=str(index), values=values)
        self._page_status("trace", len(r.steps))

    def _render_epochs(self):
        r = self.result
        if isinstance(r, PerceptronResult):
            return perceptron_view.render_epochs(self, PAGE_SIZE)
        if isinstance(r, HebbianResult):
            return hebbian_view.render_epochs(self, PAGE_SIZE)
        if isinstance(r, MadalineResult):
            return madaline_view.render_epochs(self, PAGE_SIZE)
        self._columns(
            self.epoch_tree,
            [
                "Epoch",
                "Max |Δ|",
                "Net epoch Δ",
                "SSE",
                "Correct",
                *[f"w{i + 1}" for i in range(len(r.weights))],
                "Bias",
            ],
        )
        start = self.epoch_page * PAGE_SIZE
        for e in r.epochs[start : start + PAGE_SIZE]:
            self.epoch_tree.insert(
                "",
                "end",
                iid=str(e.number),
                values=[
                    e.number,
                    self.fmt(e.max_change),
                    self.fmt(e.net_change),
                    self.fmt(e.sse),
                    f"{e.correct}/{len(r.request.samples)}",
                    *map(self.fmt, e.weights),
                    self.fmt(e.bias),
                ],
            )
        self._page_status("epoch", len(r.epochs))

    def _page_status(self, kind, count):
        page = getattr(self, kind + "_page")
        total = max(1, math.ceil(count / PAGE_SIZE))
        getattr(self, kind + "_page_text").set(
            f"Page {page + 1}/{total} · {count:,} {'updates' if kind == 'trace' else 'passes' if isinstance(self.result, HebbianResult) else 'epochs'}"
        )
        getattr(self, kind + "_prev").configure(
            state="normal" if page > 0 else "disabled"
        )
        getattr(self, kind + "_next").configure(
            state="normal" if page + 1 < total else "disabled"
        )

    def change_page(self, kind, delta):
        if not self.result:
            return
        count = len(self.result.steps if kind == "trace" else self.result.epochs)
        page = max(
            0,
            min(
                getattr(self, kind + "_page") + delta,
                max(0, math.ceil(count / PAGE_SIZE) - 1),
            ),
        )
        setattr(self, kind + "_page", page)
        (self._render_trace if kind == "trace" else self._render_epochs)()

    def jump_epoch(self):
        if not self.result:
            return
        try:
            epoch = int(self.jump_value.get())
            last = self.result.steps[-1].epoch if self.result.steps else 0
            if not 1 <= epoch <= last:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Choose an epoch",
                "Enter an epoch present in this run's trace.",
                parent=self.root,
            )
            return
        index = (epoch - 1) * len(self.result.request.samples)
        self.trace_page = index // PAGE_SIZE
        self._render_trace()
        self.trace_tree.selection_set(str(index))
        self.trace_tree.see(str(index))
        self.go_step(index)

    def _trace_selected(self, _=None):
        selected = self.trace_tree.selection()
        if selected:
            self.go_step(int(selected[0]))

    def _epoch_selected(self, _=None):
        selected = self.epoch_tree.selection()
        if selected and self.result:
            self.go_step(int(selected[0]) * len(self.result.request.samples) - 1)
            self.notebook.select(self.walk_tab)

    def _render_final(self):
        r = self.result
        if isinstance(r, PerceptronResult):
            return perceptron_view.render_final(self)
        if isinstance(r, HebbianResult):
            return hebbian_view.render_final(self)
        if isinstance(r, MadalineResult):
            return madaline_view.render_final(self)
        f = self.fmt
        equation = (
            " + ".join(f"({f(w)})x{i + 1}" for i, w in enumerate(r.weights))
            + f" + ({f(r.bias)})"
        )
        explanation = {
            "tolerance_met": "The largest individual update in the final complete epoch is strictly below ε.",
            "epoch_limit": "The epoch budget was exhausted before the strict tolerance test passed.",
            "numerical_limit": f"The update at epoch {r.failed_at[0]}, pattern {r.failed_at[1]} was rejected by the numerical guard."
            if r.failed_at
            else "Numerical guard reached.",
        }[r.status]
        content = f"{STATUS[r.status]}\n{explanation}\n\ny_in = {equation}\nBoundary: y_in = 0\ny = +1 when y_in ≥ 0; otherwise y = −1.\n\nSSE = {f(r.sse)}    Accuracy = {r.correct}/{len(r.request.samples)}\nCompleted epochs: {r.completed_epochs}; committed pattern updates: {len(r.steps)}."
        if r.status == "numerical_limit":
            content += "\nThis is the last valid model; the final attempted epoch is incomplete."
        self._set_text(self.final_text, content)
        self._columns(
            self.prediction_tree,
            [
                "Pattern",
                *[f"x{i + 1}" for i in range(len(r.weights))],
                "Target",
                "Final y_in",
                "Final y",
                "Correct",
            ],
        )
        for p in r.predictions:
            s = r.request.samples[p.pattern - 1]
            self.prediction_tree.insert(
                "",
                "end",
                values=[
                    p.pattern,
                    *map(f, s.inputs),
                    f(s.target),
                    f(p.net),
                    p.output,
                    "Yes" if p.output == s.target else "No",
                ],
            )

    def _queue_chart(self, _=None):
        if self.chart_job:
            self.root.after_cancel(self.chart_job)
        self.chart_job = self.root.after(80, self.draw_chart)

    def draw_chart(self):
        self.chart_job = None
        canvas = self.chart
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 150:
            return
        if not self.result or not self.result.epochs:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Run a worksheet to see learning across complete epochs",
                fill=MUTED,
            )
            return
        epochs = self.result.epochs
        # Downsample only the drawing; the tables and CSV keep every epoch.
        stride = max(1, math.ceil(len(epochs) / 800))
        drawn = list(epochs[::stride])
        if drawn[-1] is not epochs[-1]:
            drawn.append(epochs[-1])
        for panel, (title, key, color, log) in enumerate(
            (
                (
                    "CLASSIFICATION SSE"
                    if isinstance(
                        self.result, (MadalineResult, HebbianResult, PerceptronResult)
                    )
                    else "FIXED-MODEL SSE",
                    "sse",
                    INK,
                    False,
                ),
                ("LARGEST UPDATE · LOG (0 AT BASE)", "max_change", TEAL, True),
            )
        ):
            left = panel * width / 2 + 77
            right = (panel + 1) * width / 2 - 28
            top, bottom = 45, height - 38
            values = [getattr(e, key) for e in epochs]
            tolerance = getattr(self.result.request, "tolerance", None)
            all_values = values + (
                [tolerance] if log and tolerance is not None else [] if log else [0.0]
            )
            zero_floor = (
                min((math.log10(v) for v in all_values if v > 0), default=0.0) - 1
                if log
                else 0
            )
            transform = (
                (lambda v: math.log10(v) if v > 0 else zero_floor)
                if log
                else (lambda v: v)
            )
            low, high = min(map(transform, all_values)), max(map(transform, all_values))
            if high == low:
                high = low + 1
            span = high - low

            def y(value):
                return bottom - (transform(value) - low) / span * (bottom - top)

            def x(epoch):
                return left + (epoch - 1) / max(1, len(epochs) - 1) * (right - left)

            canvas.create_text(
                left, 16, text=title, anchor="w", fill=MUTED, font=("Arial", 9, "bold")
            )
            for j in range(3):
                ty = top + (bottom - top) * j / 2
                value = high - span * j / 2
                label = (
                    ("0" if 0 in values and j == 2 else f"1e{value:.1f}")
                    if log
                    else f"{value:.3g}"
                )
                canvas.create_line(left, ty, right, ty, fill="#e3eaee")
                canvas.create_text(
                    left - 8, ty, text=label, anchor="e", fill=MUTED, font=("Arial", 9)
                )
            canvas.create_text(
                left, bottom + 16, text="1", fill=MUTED, anchor="w", font=("Arial", 9)
            )
            canvas.create_text(
                right,
                bottom + 16,
                text=f"{'pass' if isinstance(self.result, HebbianResult) else 'epoch'} {len(epochs)}",
                fill=MUTED,
                anchor="e",
                font=("Arial", 9),
            )
            points = [(x(e.number), y(getattr(e, key))) for e in drawn]
            if len(points) == 1:
                px, py = points[0]
                canvas.create_oval(
                    px - 3, py - 3, px + 3, py + 3, fill=color, outline=color
                )
            else:
                canvas.create_line(
                    *[v for point in points for v in point], fill=color, width=2
                )
            if log and tolerance is not None:
                ty = y(tolerance)
                canvas.create_line(left, ty, right, ty, fill="#b47b18", dash=(5, 4))
                canvas.create_text(
                    right,
                    ty - 9,
                    text=f"ε = {self.fmt(tolerance)}",
                    anchor="e",
                    fill="#94600b",
                    font=("Arial", 9),
                )

    def _save(self, exporter, title, name):
        if not self.result:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title=title,
            initialfile=name,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if path:
            try:
                exporter(path, self.result)
                self.status.set(f"Saved {Path(path).name} · full run precision")
            except OSError as exc:
                messagebox.showerror("Could not save CSV", str(exc), parent=self.root)

    def save_trace(self):
        name = (
            "perceptron"
            if isinstance(self.result, PerceptronResult)
            else "hebbian"
            if isinstance(self.result, HebbianResult)
            else "madaline"
            if isinstance(self.result, MadalineResult)
            else "adaline"
        )
        self._save(export_trace, "Export every pattern update", f"{name}_trace.csv")

    def save_epochs(self):
        hebbian = isinstance(self.result, HebbianResult)
        name = (
            "perceptron"
            if isinstance(self.result, PerceptronResult)
            else "hebbian"
            if hebbian
            else "madaline"
            if isinstance(self.result, MadalineResult)
            else "adaline"
        )
        self._save(
            export_epochs,
            "Export pass summary" if hebbian else "Export every complete epoch",
            f"{name}_{'pass' if hebbian else 'epochs'}.csv",
        )

    def close(self):
        self.optimization.close()
        self.lecture08.close()
        self.closed = True
        self.cancel_event.set()
        self.root.after_cancel(self.poll_job)
        if self.chart_job:
            self.root.after_cancel(self.chart_job)
        self.root.destroy()


def run():
    root = tk.Tk()
    ExamApp(root)
    root.mainloop()
