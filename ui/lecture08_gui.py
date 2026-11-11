from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .app import BG, INK, MONO, MUTED
from ..backend import Cancelled, InputError, number, train
from ..algorithms.lecture08 import (
    KINDS,
    LABELS,
    Lecture08Request,
    export_matrices,
    export_report,
    matrix_text,
    parse_matrix,
    exercise_presets,
)

PAGE_SIZE = 200

NOTES = """Lecture 08 matrix and associative-memory tools

These are separate calculations. Existing ADALINE, MADALINE, one-output Hebbian and Perceptron algorithms keep their own input conventions and activation functions.

Rows are patterns; columns are coordinates.
X: P x n training inputs.
S: P x n associative-memory input patterns.
T: P x m target patterns.
W: n x m input-by-output weights.
Recall X: Q x n query patterns; results: Q x m outputs.

Correlation and learning rate (PDF pages 1-2)
R = X^T X / P = mean_p(x_p^T x_p). This is an uncentered correlation matrix. No bias column is automatically added.
Eigenvalues solve det(R-lambda I)=0. For 1 or 2 coordinates, Worked steps expand that equation directly. For larger symmetric R, converged Jacobi diagonalization finds the roots numerically. Eigenvectors are columns. Eigen checks contain lambda, normalized eigenvector residual and det((R-lambda I)/scale). Scaling prevents determinant overflow; sweep history records the numerical iterations.
The handwritten lecture uses 0 < alpha < (1/2) max_i |lambda_i|, which is not a general stability bound. The tool preserves this formula for comparison and also shows the corrected mean LMS bound 0<alpha<2/lambda_max, derived from error_next=(I-alpha*R)error. For gradient descent on mean squared error without a 1/2 factor, use 0<alpha<1/lambda_max. These batch/mean bounds do not guarantee stable constant-rate sequential updates. Add a constant input coordinate to analyze a trainable bias. For R=0, the data impose no curvature bound and do not identify the weights.

Hebbian associative memory (PDF pages 2-6)
Initialize W=0. For each pair (s,t), delta W[i,j]=s[i]t[j], then W+=delta W. If enabled, update a separate bias vector b+=t. There is no learning-rate factor and no repeated-epoch loop in this lecture rule. It supports multiple output neurons and general finite numerical training values.

Outer-product construction
For every pattern, s^T (n x 1) times t (1 x m) gives an n x m outer product. W=sum_p(s_p^T t_p)=S^T T. The outer-product and Hebbian tools share this accumulation engine. An independently calculated direct S^T T is compared with the result; the difference is shown. Bias is disabled in the outer-product tool and is never appended as a hidden row of W.

Orthogonality and cross-talk (PDF pages 3-5)
G=S S^T. Every distinct pair k<p is tested; exact orthogonality means G[k,p]=0. The optional tolerance is an explicitly chosen absolute tolerance, default 0. The diagonal contains squared norms and is not a distinct-pair test.
For bias-free Hebbian W: s_k W = ||s_k||^2 t_k + sum_{p!=k}(s_k dot s_p)t_p. Nonzero off-diagonal dots can introduce cross-talk. They do not necessarily produce an incorrect binary prediction: inspect actual recall. The memory tools show the cross-talk contribution matrix. Zero vectors are orthogonal to all patterns but cannot store a distinct nonzero target without bias.

Recall (PDF pages 5-7)
Weighted sums = XW. A separate bias is added only when enabled.
Choose the activation required by the source. binary: f(z)=1 if z>0, otherwise 0. sign: -1 for z<0, 0 for z=0, +1 for z>0 (08_cvicenie). bipolar: +1 for z>=0, otherwise -1 (CV8Siete). The selected activation applies to training-pattern verification and recall, and is copied by Use trained W for recall.
An optional expected matrix is compared literally with the selected activation outputs. Training T can contain real values; such values may not be reproduced by a binary activation.
Use trained W for recall copies the last calculated memory/outer-product model, separate bias and training patterns into the Recall form. Replace the query rows for new inputs, then Calculate.

Inputs and results
Paste comma-, semicolon- or whitespace-separated numeric rows, or a JSON array of row arrays. Inputs support 1-128 patterns and 1-32 coordinates/output neurons; W has 1-32 input rows. Values must be finite. Inputs/training targets are bounded at 1e50, and supplied recall weights/bias at 1e150. Every intermediate matrix remains available in the matrix selector and exports.
Matrix product shows AB, both transposes and every row-column dot product. A columns must equal B rows.
McCulloch–Pitts network evaluates fixed threshold neurons in order: output=1 if sum>=threshold, else 0. W has one row per neuron, with columns for external inputs followed by all neuron outputs. Entries referring to the same or a later neuron must be zero. Thresholds are one row, one value per neuron. The last output column is the final neuron. Presets include gates, the 06 page 6 network and the CV6 XOR circuit. No learning occurs.

Each tool retains its form when switching to another matrix/network tool. Load lecture example restores that tool's worked example; Clear inputs clears its editor and displayed matrix results. These actions never modify the older worksheets.
All results and exports use the immutable request captured when Calculate/Train was pressed. Editing a form or switching tools does not alter that snapshot. Cancellation and invalid input retain the previous completed result. Export matrices CSV includes all intermediate/final matrices and scalar results; JSON includes the input snapshot, all matrices, notes and worked steps. Display precision affects formatting only; calculations and exports use full Rust f64 values.
"""


class Lecture08Panel:
    def __init__(self, app, parent):
        self.app = app
        self.root = app.root
        self.frame = ttk.Frame(parent, padding=(18, 16, 18, 8))
        self.result = None
        self.active_kind = None
        self.states = {}
        self.inputs = {}
        self.groups = {}
        self.input_labels = {}
        self.events = queue.Queue()
        self.cancel_event = threading.Event()
        self.busy = self.closed = self.loading = False
        self.step_index = self.matrix_page = 0
        self.status = tk.StringVar(
            value="Ready · choose a Lecture 08 matrix calculation"
        )
        self.snapshot = tk.StringVar(
            value="Lecture 08 · rows are patterns; all calculations run in Rust"
        )
        self.precision = tk.StringVar(value="10")
        self.bias_enabled = tk.BooleanVar(value=False)
        self.tolerance = tk.StringVar(value="0")
        self.activation = tk.StringVar(value="binary")
        self.example_choice = tk.StringVar()
        self.step_choice = tk.StringVar()
        self.matrix_choice = tk.StringVar()
        self._build()
        self.bias_enabled.trace_add("write", self._dirty)
        self.tolerance.trace_add("write", self._dirty)
        self.activation.trace_add("write", self._dirty)
        self.select(LABELS["correlation"])
        self.poll_job = self.root.after(60, self._poll)

    def _build(self):
        footer = ttk.Frame(self.frame, padding=(0, 6, 0, 0))
        footer.pack(side="bottom", fill="x")
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel").pack(
            side="left"
        )
        ttk.Label(
            footer, text="Lecture 08 · Rust matrix calculations", style="Muted.TLabel"
        ).pack(side="right")
        sidebar = ttk.Frame(self.frame, width=325)
        sidebar.pack(side="left", fill="y", padx=(0, 18))
        sidebar.pack_propagate(False)
        actions = ttk.Frame(sidebar, padding=(0, 10, 0, 0))
        actions.pack(side="bottom", fill="x")
        self.run_button = ttk.Button(
            actions, text="Calculate", command=self.start, style="Run.TButton"
        )
        self.run_button.pack(fill="x")
        self.cancel_button = ttk.Button(
            actions, text="Cancel calculation", command=self.cancel, state="disabled"
        )
        self.cancel_button.pack(fill="x", pady=(6, 0))
        self.progress = ttk.Progressbar(actions, mode="indeterminate")
        self.progress.pack(fill="x", pady=(7, 0))
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
        ttk.Label(form, text="ALGORITHM", style="Section.TLabel").pack(anchor="w")
        selector = ttk.Combobox(
            form,
            textvariable=self.app.algorithm,
            state="readonly",
            values=self.app.algorithm_box["values"],
        )
        selector.pack(fill="x")
        selector.bind("<<ComboboxSelected>>", lambda _: self.app.switch_algorithm())
        ttk.Label(
            form,
            text="One pattern per row · numeric values\nCommas, spaces, semicolons or JSON rows.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(9, 4))
        utility = ttk.Frame(form)
        utility.pack(fill="x", pady=(0, 8))
        self.example_box = ttk.Combobox(
            utility, textvariable=self.example_choice, state="readonly"
        )
        self.example_box.pack(fill="x", pady=(0, 5))
        self.example_button = ttk.Button(
            utility, text="Load lecture example", command=self.load_example
        )
        self.example_button.pack(fill="x")
        self.clear_button = ttk.Button(utility, text="Clear inputs", command=self.clear)
        self.clear_button.pack(fill="x", pady=(5, 0))
        self.form_fields = ttk.Frame(form)
        self.form_fields.pack(fill="x")
        for key, label, height in (
            ("s", "Training matrix S", 5),
            ("t", "Targets T · one output vector per row", 5),
            ("w", "Weights W · inputs x outputs", 5),
            ("x", "Recall inputs X", 4),
            ("bias", "Separate bias · one row", 2),
            ("expected", "Expected outputs · optional", 3),
        ):
            group = ttk.Frame(self.form_fields)
            self.groups[key] = group
            label_var = tk.StringVar(value=label)
            self.input_labels[key] = label_var
            ttk.Label(
                group, textvariable=label_var, style="Section.TLabel", wraplength=285
            ).pack(anchor="w")
            widget = tk.Text(
                group,
                height=height,
                width=26,
                wrap="none",
                font=(MONO, 10),
                bg="white",
                fg=INK,
                relief="flat",
                padx=8,
                pady=7,
                undo=True,
                highlightthickness=1,
                highlightbackground="#cbd6df",
            )
            widget.pack(fill="x")
            self.inputs[key] = widget
            xscroll = ttk.Scrollbar(group, orient="horizontal", command=widget.xview)
            xscroll.pack(fill="x")
            widget.configure(xscrollcommand=xscroll.set)
            widget.bind("<<Modified>>", lambda _, key=key: self._modified(key))
            ttk.Button(
                group,
                text="Import CSV…",
                command=lambda key=key: self.import_matrix(key),
            ).pack(fill="x", pady=(5, 0))
        self.bias_controls = ttk.Frame(self.form_fields)
        ttk.Checkbutton(
            self.bias_controls,
            text="Enable separate bias",
            variable=self.bias_enabled,
            command=self._show_fields,
        ).pack(anchor="w", pady=(8, 2))
        self.tolerance_controls = ttk.Frame(self.form_fields)
        ttk.Label(
            self.tolerance_controls,
            text="Absolute orthogonality tolerance",
            style="Section.TLabel",
        ).pack(anchor="w")
        ttk.Entry(self.tolerance_controls, textvariable=self.tolerance).pack(fill="x")
        ttk.Label(
            self.tolerance_controls,
            text="Default 0: exact zero dot product.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 0))
        self.activation_controls = ttk.Frame(self.form_fields)
        ttk.Label(
            self.activation_controls, text="Recall activation", style="Section.TLabel"
        ).pack(anchor="w")
        ttk.Combobox(
            self.activation_controls,
            textvariable=self.activation,
            values=("binary", "sign", "bipolar"),
            state="readonly",
        ).pack(fill="x")
        ttk.Label(
            self.activation_controls,
            text="binary: z>0 → 1; else 0\nsign: −1 / 0 / +1 (zero → 0)\nbipolar: z≥0 → +1; else −1",
            style="Muted.TLabel",
        ).pack(anchor="w")
        ttk.Label(form, text="DISPLAY", style="Section.TLabel").pack(anchor="w")
        precision = ttk.Combobox(
            form,
            textvariable=self.precision,
            values=("6", "10", "15"),
            state="readonly",
            width=6,
        )
        precision.pack(anchor="w")
        precision.bind("<<ComboboxSelected>>", lambda _: self._render())
        right = ttk.Frame(self.frame)
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(
            right, textvariable=self.snapshot, style="Muted.TLabel", wraplength=900
        ).pack(fill="x", pady=(0, 10))
        toolbar = ttk.Frame(right)
        toolbar.pack(fill="x", pady=(0, 9))
        self.csv_button = ttk.Button(
            toolbar,
            text="Export matrices CSV…",
            command=self.save_csv,
            state="disabled",
        )
        self.csv_button.pack(side="left", padx=(0, 6))
        self.json_button = ttk.Button(
            toolbar,
            text="Export full report JSON…",
            command=self.save_json,
            state="disabled",
        )
        self.json_button.pack(side="left")
        self.recall_button = ttk.Button(
            toolbar,
            text="Use trained W for recall",
            command=self.use_trained_weights,
            state="disabled",
        )
        self.recall_button.pack(side="right")
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)
        result_tab = ttk.Frame(self.notebook, padding=14, style="White.TFrame")
        self.notebook.add(result_tab, text="Results")
        self.summary_text = self._text(result_tab)
        self.app._set_text(
            self.summary_text,
            "Choose a Lecture 08 tool, enter matrices, and press Calculate / Train.",
        )
        step_tab = ttk.Frame(self.notebook, padding=14, style="White.TFrame")
        self.notebook.add(step_tab, text="Worked steps")
        step_nav = ttk.Frame(step_tab, style="White.TFrame")
        step_nav.pack(fill="x", pady=(0, 10))
        self.step_prev = ttk.Button(
            step_nav,
            text="← Previous",
            command=lambda: self.go_step(self.step_index - 1),
            state="disabled",
        )
        self.step_prev.pack(side="left", padx=(0, 5))
        self.step_next = ttk.Button(
            step_nav,
            text="Next →",
            command=lambda: self.go_step(self.step_index + 1),
            state="disabled",
        )
        self.step_next.pack(side="left", padx=(0, 10))
        self.step_box = ttk.Combobox(
            step_nav, textvariable=self.step_choice, state="readonly"
        )
        self.step_box.pack(side="left", fill="x", expand=True)
        self.step_box.bind(
            "<<ComboboxSelected>>", lambda _: self.go_step(self.step_box.current())
        )
        self.worked_text = self._text(step_tab)
        matrix_tab = ttk.Frame(self.notebook, padding=14, style="White.TFrame")
        self.notebook.add(matrix_tab, text="All matrices")
        matrix_nav = ttk.Frame(matrix_tab, style="White.TFrame")
        matrix_nav.pack(fill="x", pady=(0, 8))
        self.matrix_box = ttk.Combobox(
            matrix_nav, textvariable=self.matrix_choice, state="readonly"
        )
        self.matrix_box.pack(fill="x")
        self.matrix_box.bind("<<ComboboxSelected>>", lambda _: self.choose_matrix())
        page_bar = ttk.Frame(matrix_tab, style="White.TFrame")
        page_bar.pack(fill="x", pady=(0, 7))
        self.matrix_prev = ttk.Button(
            page_bar, text="←", command=lambda: self.page_matrix(-1), state="disabled"
        )
        self.matrix_prev.pack(side="left")
        self.matrix_page_text = tk.StringVar(value="No matrices yet")
        ttk.Label(
            page_bar, textvariable=self.matrix_page_text, background="white"
        ).pack(side="left", padx=10)
        self.matrix_next = ttk.Button(
            page_bar, text="→", command=lambda: self.page_matrix(1), state="disabled"
        )
        self.matrix_next.pack(side="left")
        self.matrix_note = tk.StringVar()
        ttk.Label(
            matrix_tab,
            textvariable=self.matrix_note,
            background="white",
            foreground=MUTED,
            wraplength=850,
        ).pack(fill="x", pady=(0, 8))
        self.matrix_tree = self.app._table(matrix_tab)
        notes_tab = ttk.Frame(self.notebook, padding=14, style="White.TFrame")
        self.notebook.add(notes_tab, text="Lecture notes")
        notes = self.app._text(notes_tab)
        self.app._set_text(notes, NOTES)

    def _text(self, parent):
        widget = self.app._text(parent)
        widget.configure(wrap="none")
        scroll = ttk.Scrollbar(widget.master, orient="horizontal", command=widget.xview)
        scroll.pack(side="bottom", fill="x")
        widget.configure(xscrollcommand=scroll.set)
        return widget

    def fmt(self, value):
        return f"{value:.{int(self.precision.get())}g}"

    def _formatted_matrix(self, name):
        record = next(m for m in self.result.matrices if m.name == name)
        rows = [[self.fmt(v) for v in row] for row in record.data]
        width = max((len(v) for row in rows for v in row), default=1)
        body = "\n".join(
            "[ " + "  ".join(v.rjust(width) for v in row) + " ]" for row in rows
        )
        return f"{name}  ({record.rows} x {record.columns})\n{body if rows else '(no distinct pairs)'}"

    def _set_input(self, key, text):
        widget = self.inputs[key]
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.edit_modified(False)

    def _modified(self, key):
        if self.inputs[key].edit_modified():
            self.inputs[key].edit_modified(False)
            self._dirty()

    def _dirty(self, *_):
        if not self.loading and self.result:
            self.status.set(
                "Inputs edited · displayed matrices belong to the last completed calculation"
            )

    def _save_form(self):
        return (
            {key: widget.get("1.0", "end-1c") for key, widget in self.inputs.items()},
            self.bias_enabled.get(),
            self.tolerance.get(),
            self.activation.get(),
            self.example_choice.get(),
        )

    def select(self, label):
        kind = KINDS[label]
        if kind != self.active_kind:
            if self.active_kind:
                self.states[self.active_kind] = self._save_form()
            self.active_kind = kind
            self.loading = True
            try:
                if kind in self.states:
                    values, bias, tolerance, activation, example = self.states[kind]
                    for key, value in values.items():
                        self._set_input(key, value)
                    self.bias_enabled.set(bias)
                    self.tolerance.set(tolerance)
                    self.activation.set(activation)
                    self.example_choice.set(example)
                else:
                    self.example_choice.set(next(iter(exercise_presets(kind))))
                    self._example(kind)
            finally:
                self.loading = False
        self._show_fields()
        self.run_button.configure(
            text="Train memory" if kind == "hebbian_memory" else "Calculate"
        )
        self._dirty()

    def _show_fields(self):
        for group in (
            *self.groups.values(),
            self.bias_controls,
            self.tolerance_controls,
            self.activation_controls,
        ):
            group.pack_forget()
        kind = self.active_kind
        self.example_box.configure(values=tuple(exercise_presets(kind)))
        self.input_labels["t"].set(
            "Thresholds · one value per neuron"
            if kind == "mcp"
            else "Matrix B"
            if kind == "matrix_product"
            else "Targets T · one output vector per row"
        )
        self.input_labels["w"].set(
            "Neuron weights · inputs, then neurons\nSelf/later-neuron weights must be zero"
            if kind == "mcp"
            else "Weights W · inputs × outputs"
        )
        self.input_labels["s"].set(
            "Binary inputs · one pattern per row"
            if kind == "mcp"
            else "Matrix A"
            if kind == "matrix_product"
            else "Training inputs X · rows are patterns"
            if kind == "correlation"
            else "Training inputs S · rows are patterns"
        )
        if kind in (
            "correlation",
            "orthogonality",
            "hebbian_memory",
            "outer_product",
            "matrix_product",
            "mcp",
        ):
            self.groups["s"].pack(fill="x")
        if kind in ("hebbian_memory", "outer_product", "matrix_product", "mcp"):
            self.groups["t"].pack(fill="x")
        if kind == "recall":
            self.groups["w"].pack(fill="x")
            self.groups["x"].pack(fill="x")
        if kind == "mcp":
            self.groups["w"].pack(fill="x")
            self.groups["expected"].pack(fill="x")
        if kind in ("hebbian_memory", "outer_product", "recall"):
            self.activation_controls.pack(fill="x")
        if kind in ("hebbian_memory", "recall"):
            self.bias_controls.pack(fill="x")
        if kind == "recall" and self.bias_enabled.get():
            self.groups["bias"].pack(fill="x")
        if kind == "recall":
            self.groups["expected"].pack(fill="x")
        if kind in ("orthogonality", "hebbian_memory", "outer_product"):
            self.tolerance_controls.pack(fill="x")

    def _example(self, kind):
        for key in self.inputs:
            self._set_input(key, "")
        self.bias_enabled.set(False)
        self.tolerance.set("0")
        self.activation.set("binary")
        presets = exercise_presets(kind)
        if self.example_choice.get() not in presets:
            self.example_choice.set(next(iter(presets)))
        for key, value in presets[self.example_choice.get()].items():
            if key == "activation":
                self.activation.set(value)
            else:
                self._set_input(key, matrix_text(value))
        if kind == "recall":
            self._set_input("bias", "0, 0")

    def load_example(self):
        self.loading = True
        try:
            self._example(self.active_kind)
        finally:
            self.loading = False
        self._show_fields()
        self._dirty()

    def clear(self):
        if self.busy:
            return
        self.loading = True
        for key in self.inputs:
            self._set_input(key, "")
        self.bias_enabled.set(False)
        self.tolerance.set("0")
        self.loading = False
        self.result = None
        self._show_fields()
        for widget in (self.summary_text, self.worked_text):
            self.app._set_text(widget, "")
        self.matrix_tree.delete(*self.matrix_tree.get_children())
        self.matrix_box.configure(values=())
        self.matrix_choice.set("")
        self.step_box.configure(values=())
        self.step_choice.set("")
        for button in (
            self.csv_button,
            self.json_button,
            self.recall_button,
            self.step_prev,
            self.step_next,
            self.matrix_prev,
            self.matrix_next,
        ):
            button.configure(state="disabled")
        self.snapshot.set("Lecture 08 · no completed calculation")
        self.matrix_note.set("")
        self.matrix_page_text.set("No matrices yet")
        self.status.set("Cleared · enter matrices or load the lecture example")

    def import_matrix(self, key):
        path = filedialog.askopenfilename(
            parent=self.root,
            title=f"Import {key.upper()} matrix",
            filetypes=[
                ("Numeric matrix", "*.csv *.tsv *.txt *.json"),
                ("All files", "*"),
            ],
        )
        if not path:
            return
        try:
            if Path(path).stat().st_size > 1_048_576:
                raise InputError("Matrix files must be smaller than 1 MiB.")
            data = parse_matrix(Path(path).read_text(encoding="utf-8-sig"), key.upper())
            self._set_input(key, matrix_text(data))
            self._dirty()
        except (OSError, UnicodeError, InputError) as exc:
            messagebox.showerror("Could not import matrix", str(exc), parent=self.root)

    def read_request(self):
        kind = self.active_kind
        kwargs = {}
        if kind in (
            "correlation",
            "orthogonality",
            "hebbian_memory",
            "outer_product",
            "matrix_product",
            "mcp",
        ):
            kwargs["s"] = parse_matrix(
                self.inputs["s"].get("1.0", "end"),
                "X" if kind == "correlation" else "S",
            )
        if kind in ("hebbian_memory", "outer_product", "matrix_product", "mcp"):
            kwargs["t"] = parse_matrix(self.inputs["t"].get("1.0", "end"), "T")
        if kind in ("hebbian_memory", "recall"):
            kwargs["bias_enabled"] = self.bias_enabled.get()
        if kind in ("hebbian_memory", "outer_product", "recall"):
            kwargs["activation"] = self.activation.get()
        if kind == "mcp":
            kwargs["w"] = parse_matrix(
                self.inputs["w"].get("1.0", "end"), "Neuron weights"
            )
            kwargs["expected"] = parse_matrix(
                self.inputs["expected"].get("1.0", "end"),
                "Expected neuron outputs",
                optional=True,
            )
        if kind == "recall":
            kwargs["w"] = parse_matrix(self.inputs["w"].get("1.0", "end"), "W")
            kwargs["x"] = parse_matrix(
                self.inputs["x"].get("1.0", "end"), "Recall inputs X"
            )
            kwargs["expected"] = parse_matrix(
                self.inputs["expected"].get("1.0", "end"),
                "Expected targets",
                optional=True,
            )
            if self.bias_enabled.get():
                kwargs["bias"] = parse_matrix(
                    self.inputs["bias"].get("1.0", "end"), "Bias"
                )
        if kind in ("orthogonality", "hebbian_memory", "outer_product"):
            kwargs["tolerance"] = number(
                self.tolerance.get(), "Orthogonality tolerance"
            )
        request = Lecture08Request(kind, **kwargs)
        request.validate()
        return request

    def start(self):
        if self.busy or self.closed:
            return
        try:
            request = self.read_request()
        except InputError as exc:
            messagebox.showerror("Check matrix inputs", str(exc), parent=self.root)
            return
        self.busy = True
        self.cancel_event = threading.Event()
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.clear_button.configure(state="disabled")
        self.example_button.configure(state="disabled")
        self.progress.start(12)
        self.status.set("Preparing Rust matrix calculation…")

        def worker():
            try:
                result = train(
                    request,
                    cancel=self.cancel_event,
                    progress=lambda message: self.events.put(("progress", message)),
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
                self.clear_button.configure(state="normal")
                self.example_button.configure(state="normal")
                if self.cancel_event.is_set() or kind == "cancelled":
                    self.status.set(
                        "Calculation cancelled · previous completed result retained"
                    )
                elif kind == "error":
                    self.status.set(
                        "Calculation failed · previous completed result retained"
                    )
                    messagebox.showerror(
                        "Matrix calculation failed", data, parent=self.root
                    )
                else:
                    self.result = data
                    self.step_index = self.matrix_page = 0
                    self._render()
                    self.status.set(
                        f"Calculated {LABELS[data.request.kind]} · {len(data.steps)} worked steps"
                    )
                    try:
                        current = self.read_request()
                    except InputError:
                        current = None
                    if current != data.request:
                        self._dirty()
        except queue.Empty:
            pass
        self.poll_job = self.root.after(60, self._poll)

    def _render(self):
        if not self.result:
            return
        r = self.result
        kind = r.request.kind
        f = self.fmt
        self.snapshot.set(
            f"Run snapshot: {LABELS[kind]} · {len(r.matrices)} matrices · {len(r.steps)} worked steps · separate bias {'enabled' if r.request.bias_enabled else 'disabled'}"
        )
        lines = [LABELS[kind]]
        if kind == "correlation":
            lines.extend(
                (
                    self._formatted_matrix("R"),
                    self._formatted_matrix("eigenvalues"),
                    f"Maximum eigenvalue = {f(r.scalars['lambda_max'])}",
                    "CORRECTED MEAN LMS BOUND: 0 < α < 2/λmax = "
                    + f(r.scalars["mean_lms_alpha_upper"])
                    if "mean_lms_alpha_upper" in r.scalars
                    else "R=0: no data-curvature bound; weights are not identified by these inputs.",
                    r.notes["stability_rule"],
                    f"LECTURE 08 RULE (strict): 0 < α < ½ max|λᵢ| = {f(r.scalars['lecture_alpha_upper'])}",
                    "No positive α satisfies this bound for the zero correlation matrix."
                    if r.scalars["lecture_alpha_upper"] == 0
                    else "This literal note formula is not a general convergence bound. See the corrected mean/batch bounds above.",
                    f"Largest normalized eigenvector residual = {f(r.scalars['eigen_residual_max'])}",
                )
            )
        elif kind in ("hebbian_memory", "outer_product"):
            lines.extend(
                (
                    self._formatted_matrix("W"),
                    self._formatted_matrix("bias"),
                    f"Independent check W = SᵀT: maximum difference = {f(r.scalars['outer_product_difference'])}",
                    "Hebbian accumulation and outer-product construction agree."
                    if r.scalars["outer_product_agreement"]
                    else "Inspect the nonzero agreement difference.",
                    self._formatted_matrix("training_net"),
                    self._formatted_matrix("training_output"),
                    f"Exact training-pattern recall: {int(r.scalars['training_correct_patterns'])}/{len(r.request.s)} correct",
                    f"The separate bias is outside W. Recall activation: {r.request.activation}.",
                )
            )
        elif kind == "orthogonality":
            lines.extend(
                (
                    self._formatted_matrix("gram"),
                    self._formatted_matrix("pairwise_dots"),
                    f"All distinct pairs orthogonal? {'Yes' if r.scalars['all_orthogonal'] else 'No'} · absolute tolerance = {f(r.request.tolerance)}",
                    f"Zero input rows = {int(r.scalars['zero_vectors'])}",
                    r.notes["cross_talk"],
                )
            )
        elif kind == "matrix_product":
            lines.extend(
                (
                    self._formatted_matrix("A"),
                    self._formatted_matrix("B"),
                    self._formatted_matrix("output"),
                    "C = AB. Each element is the dot product of one A row with one B column.",
                )
            )
        elif kind == "mcp":
            lines.extend(
                (
                    self._formatted_matrix("net"),
                    self._formatted_matrix("output"),
                    "Outputs are in neuron order. Each neuron emits 1 when its weighted sum ≥ its threshold, otherwise 0. The last column is the final neuron.",
                )
            )
            if r.request.expected:
                lines.append(
                    f"Exact expected-row matches: {int(r.scalars['correct_patterns'])}/{len(r.request.s)}"
                )
        else:
            lines.extend(
                (
                    self._formatted_matrix("net"),
                    self._formatted_matrix("output"),
                    f"Recall activation: {r.request.activation}. binary: 1 if z>0, else 0; sign: −1/0/+1; bipolar: +1 if z≥0, else −1.",
                )
            )
            if r.request.expected:
                lines.extend(
                    (
                        self._formatted_matrix("expected"),
                        f"Exact expected-row matches: {int(r.scalars['correct_patterns'])}/{len(r.request.x)}",
                    )
                )
            else:
                lines.append(
                    "No expected targets were supplied; no accuracy claim is made."
                )
        self.app._set_text(self.summary_text, "\n\n".join(lines))
        self.step_box.configure(
            values=tuple(f"{i + 1}. {s.title}" for i, s in enumerate(r.steps))
        )
        self.go_step(self.step_index)
        selected = self.matrix_choice.get()
        self.matrix_box.configure(values=tuple(m.name for m in r.matrices))
        if selected not in tuple(m.name for m in r.matrices):
            self.matrix_choice.set(
                "R"
                if kind == "correlation"
                else "W"
                if kind in ("hebbian_memory", "outer_product")
                else "gram"
                if kind == "orthogonality"
                else "output"
            )
        self.choose_matrix(reset=False)
        self.csv_button.configure(state="normal")
        self.json_button.configure(state="normal")
        self.recall_button.configure(
            state="normal"
            if kind in ("hebbian_memory", "outer_product")
            else "disabled"
        )

    def go_step(self, index):
        if not self.result:
            return
        self.step_index = max(0, min(index, len(self.result.steps) - 1))
        step = self.result.steps[self.step_index]
        self.step_box.current(self.step_index)
        content = [step.title, step.detail]
        for name in step.matrices:
            content.append(self._formatted_matrix(name))
            if name in self.result.notes:
                content.append(self.result.notes[name])
        self.app._set_text(self.worked_text, "\n\n".join(content))
        self.step_prev.configure(state="normal" if self.step_index else "disabled")
        self.step_next.configure(
            state="normal"
            if self.step_index + 1 < len(self.result.steps)
            else "disabled"
        )

    def choose_matrix(self, *, reset=True):
        if not self.result:
            return
        if reset:
            self.matrix_page = 0
        record = next(
            (m for m in self.result.matrices if m.name == self.matrix_choice.get()),
            None,
        )
        if record is None:
            return
        pages = max(1, (record.rows + PAGE_SIZE - 1) // PAGE_SIZE)
        self.matrix_page = min(self.matrix_page, pages - 1)
        self.app._columns(
            self.matrix_tree, ["Row", *[f"c{j + 1}" for j in range(record.columns)]]
        )
        start = self.matrix_page * PAGE_SIZE
        for i, row in enumerate(record.data[start : start + PAGE_SIZE], start + 1):
            self.matrix_tree.insert("", "end", values=[i, *map(self.fmt, row)])
        self.matrix_page_text.set(
            f"{record.rows} × {record.columns} · page {self.matrix_page + 1}/{pages}"
        )
        self.matrix_note.set(
            self.result.notes.get(
                record.name,
                "Rows/columns use the orientation shown in the Lecture notes tab.",
            )
        )
        self.matrix_prev.configure(state="normal" if self.matrix_page else "disabled")
        self.matrix_next.configure(
            state="normal" if self.matrix_page + 1 < pages else "disabled"
        )

    def page_matrix(self, delta):
        self.matrix_page = max(0, self.matrix_page + delta)
        self.choose_matrix(reset=False)

    def use_trained_weights(self):
        if not self.result or self.result.request.kind not in (
            "hebbian_memory",
            "outer_product",
        ):
            return
        trained = self.result
        self.app.algorithm.set(LABELS["recall"])
        self.app.switch_algorithm()
        self.loading = True
        try:
            self._set_input("w", matrix_text(trained.matrix("W")))
            self._set_input("bias", matrix_text(trained.matrix("bias")))
            self._set_input("x", matrix_text(trained.request.s))
            self._set_input("expected", matrix_text(trained.request.t))
            self.bias_enabled.set(trained.request.bias_enabled)
            self.activation.set(trained.request.activation)
        finally:
            self.loading = False
        self._show_fields()
        self.status.set(
            "Loaded the calculated W and separate bias · edit query rows, then Calculate"
        )

    def _save(self, exporter, title, name):
        if not self.result:
            return
        suffix = Path(name).suffix
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title=title,
            initialfile=name,
            defaultextension=suffix,
            filetypes=[(suffix[1:].upper(), "*" + suffix)],
        )
        if path:
            try:
                exporter(path, self.result)
                self.status.set(f"Saved {Path(path).name} · full calculation precision")
            except OSError as exc:
                messagebox.showerror(
                    "Could not export results", str(exc), parent=self.root
                )

    def save_csv(self):
        self._save(
            export_matrices,
            "Export all matrices and scalar results",
            "lecture08_matrices.csv",
        )

    def save_json(self):
        self._save(
            export_report, "Export full calculation report", "lecture08_report.json"
        )

    def close(self):
        self.closed = True
        self.cancel_event.set()
        self.root.after_cancel(self.poll_job)
