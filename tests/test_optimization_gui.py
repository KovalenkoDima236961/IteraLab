"""Opt-in tests of workspace switching, keyboard routing and result isolation."""

import csv
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    os.environ.get("EXAM_GUI_TESTS") == "1", "Requires a desktop display"
)
class OptimizationDesktopTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from every_example_gui.ui.app import ExamApp

        self.root = tk.Tk()
        self.callback_errors = []
        self.root.report_callback_exception = lambda *args: self.callback_errors.append(
            args
        )
        self.app = ExamApp(self.root)
        self.panel = self.app.optimization
        self.root.update()

    def tearDown(self):
        self.app.close()
        self.assertEqual(self.callback_errors, [])

    def wait(self, panel):
        deadline = time.monotonic() + 30
        while panel.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(panel.busy)
        self.assertIsNotNone(panel.result)
        self.root.update()

    def select(self, workspace):
        self.app.workspace.set(workspace)
        self.app.switch_workspace()
        self.root.update()

    def test_single_window_switches_preserve_both_results_and_forms(self):
        a, p = self.app, self.panel
        a.start()
        self.wait(a)
        learning_result = a.result
        fields = {key: value.get() for key, value in a.fields.items()}
        self.select("Optimization")
        self.assertFalse(a.body.winfo_ismapped())
        self.assertTrue(p.frame.winfo_ismapped())
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "gradient_tolerance")
        self.assertIn("xₖ₊₁ = xₖ − α f′(xₖ)", p.worked_text.get("1.0", "end"))
        a.navigate_step(1)
        self.assertEqual(p.step_index, 1)
        self.assertEqual(a.step_index, 0)
        p.go_step(len(p.result.points) - 1)
        self.assertIn("No further update", p.worked_text.get("1.0", "end"))
        p.draw_chart()
        self.assertGreater(len(p.chart.find_all()), 20)
        optimization_result = p.result
        p.fields["expression"].set("x^2")
        self.select("Learning")
        self.assertIs(a.result, learning_result)
        self.assertEqual({key: value.get() for key, value in a.fields.items()}, fields)
        a.fields["alpha"].set("0.123")
        self.select("Optimization")
        self.assertIs(p.result, optimization_result)
        self.assertEqual(p.fields["expression"].get(), "x^2")
        self.select("Learning")
        self.assertEqual(a.fields["alpha"].get(), "0.123")
        self.assertEqual(
            len(
                [w for w in self.root.winfo_children() if w.winfo_class() == "Toplevel"]
            ),
            0,
        )

    def test_lecture_selection_and_background_calculation_survive_switch(self):
        from every_example_gui.algorithms.lecture08 import LABELS

        a, p = self.app, self.panel
        a.algorithm.set(LABELS["correlation"])
        a.switch_algorithm()
        a.start()
        self.wait(a.lecture08)
        matrix_result = a.lecture08.result
        self.select("Optimization")
        a.start()
        self.select("Learning")  # Let optimization finish while using matrix tools.
        self.wait(p)
        self.assertTrue(a.lecture08.frame.winfo_ismapped())
        self.assertIs(a.lecture08.result, matrix_result)
        self.assertEqual(a.algorithm.get(), LABELS["correlation"])
        self.select("Optimization")
        self.assertTrue(p.frame.winfo_ismapped())
        self.assertFalse(a.lecture08.frame.winfo_ismapped())

    def test_trace_pages_exports_validation_cancellation_and_zero_updates(self):
        a, p = self.app, self.panel
        self.select("Optimization")
        p.fields["expression"].set("x^2")
        p.fields["x0"].set("1")
        p.fields["alpha"].set("1")
        p.fields["max_iterations"].set("250")
        a.start()
        self.wait(p)
        self.assertEqual(len(p.trace_tree.get_children()), 200)
        p.change_page(1)
        self.assertEqual(len(p.trace_tree.get_children()), 51)
        p.trace_tree.selection_set("225")
        self.root.update()
        self.assertEqual(p.step_index, 225)
        p.precision.set("15")
        p._render()
        original = p.result
        p.fields["alpha"].set("0.01")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gradient.csv"
            with patch(
                "every_example_gui.ui.optimization_gui.filedialog.asksaveasfilename",
                return_value=str(path),
            ):
                p.save_csv()
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 251)
            self.assertEqual(float(rows[0]["alpha"]), 1)
        p.fields["alpha"].set("0")
        with patch(
            "every_example_gui.ui.optimization_gui.messagebox.showerror"
        ) as error:
            a.start()
            error.assert_called_once()
        self.assertIs(p.result, original)
        p.fields["alpha"].set("0.1")
        p.start()
        p.cancel()
        self.wait(p)
        self.assertIs(p.result, original)
        self.assertIn("cancelled", p.status.get())
        p.fields["x0"].set("0")
        a.start()
        self.wait(p)
        self.assertEqual(p.result.final.iteration, 0)
        self.assertTrue(p.next_button.instate(["disabled"]))
        self.assertEqual(len(p.trace_tree.get_children()), 1)
        self.assertIn("TRUE", p.worked_text.get("1.0", "end"))

    def choose_method(self, method):
        self.panel.method.set(method)
        self.panel.select_method()
        self.root.update()

    def test_newton_workflow_derivatives_exports_and_saved_method_states(self):
        from every_example_gui.algorithms.optimization import (
            GRADIENT,
            NEWTON,
            NewtonResult,
        )

        a, p = self.app, self.panel
        self.select("Optimization")
        a.start()
        self.wait(p)
        gradient_result = p.result
        p.fields["alpha"].set("0.125")
        p.go_step(3)
        self.choose_method(NEWTON)
        self.assertIsNone(p.result)
        self.assertFalse(p.field_rows["alpha"].winfo_ismapped())
        self.assertTrue(p.field_rows["first_derivative"].winfo_ismapped())
        self.assertTrue(p.field_rows["second_derivative"].winfo_ismapped())
        self.assertIn("Newton", self.root.title())
        self.assertIn("|xᵢ₊₁ − xᵢ| < ε", p.field_labels["tolerance"].get())
        p.fields["alpha"].set("unused")
        p.fields["first_derivative"].set("2*(x-2)")
        p.fields["second_derivative"].set("2")
        a.start()
        self.wait(p)
        self.assertIsInstance(p.result, NewtonResult)
        self.assertEqual([point.x for point in p.result.points], [6, 2, 2])
        self.assertIn("xᵢ₊₁ = xᵢ − f′(xᵢ) / f″(xᵢ)", p.worked_text.get("1.0", "end"))
        self.assertIn("FALSE", p.worked_text.get("1.0", "end"))
        a.navigate_step(1)
        self.assertIn("TRUE", p.worked_text.get("1.0", "end"))
        self.assertIn("strict step test passed", p.summary_text.get("1.0", "end"))
        self.assertIn(
            "f″(xᵢ)", [p.trace_tree.heading(c, "text") for c in p.trace_tree["columns"]]
        )
        original = p.result
        p.fields["expression"].set("x^4")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "newton.csv"
            with patch(
                "every_example_gui.ui.optimization_gui.filedialog.asksaveasfilename",
                return_value=str(path),
            ):
                p.save_csv()
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["function"], "(x - 2)^2 + 1")
            self.assertEqual(rows[0]["first_derivative"], "2*(x-2)")
        self.choose_method(GRADIENT)
        self.assertIs(p.result, gradient_result)
        self.assertEqual(p.fields["alpha"].get(), "0.125")
        self.assertEqual(p.step_index, 3)
        self.assertTrue(p.field_rows["alpha"].winfo_ismapped())
        self.choose_method(NEWTON)
        self.assertIs(p.result, original)
        self.assertEqual(p.fields["expression"].get(), "x^4")
        self.assertEqual(p.step_index, 1)
        self.select("Learning")
        self.select("Optimization")
        self.assertIn("Newton", self.root.title())
        self.assertIs(p.result, original)

    def test_newton_equality_zero_curvature_errors_and_cancellation_keep_results(self):
        from every_example_gui.algorithms.optimization import NEWTON

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(NEWTON)
        for key, value in {
            "expression": "x^2",
            "x0": "1",
            "tolerance": "1",
            "max_iterations": "0",
        }.items():
            p.fields[key].set(value)
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "iteration_limit")
        self.assertIn("1 < 1 → FALSE", p.worked_text.get("1.0", "end"))
        p.fields["expression"].set("x^3+x")
        p.fields["x0"].set("0")
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "zero_curvature")
        self.assertIn("Cannot divide", p.worked_text.get("1.0", "end"))
        self.assertIn("No update was committed", p.summary_text.get("1.0", "end"))
        original = p.result
        p.fields["second_derivative"].set("x.real")
        with patch(
            "every_example_gui.ui.optimization_gui.messagebox.showerror"
        ) as error:
            a.start()
            error.assert_called_once()
        self.assertIs(p.result, original)
        p.fields["second_derivative"].set("")
        a.start()
        p.cancel()
        self.wait(p)
        self.assertIs(p.result, original)
        self.assertIn("cancelled", p.status.get())
        # Runtime domain errors also preserve the previous completed run.
        p.fields["expression"].set("sqrt(x)")
        p.fields["x0"].set("-1")
        with patch(
            "every_example_gui.ui.optimization_gui.messagebox.showerror"
        ) as error:
            a.start()
            self.wait(p)
            error.assert_called_once()
        self.assertIs(p.result, original)

    def test_newton_background_result_stays_with_its_method_and_trace_pages(self):
        from every_example_gui.backend import train
        from every_example_gui.algorithms.optimization import GRADIENT, NEWTON

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(NEWTON)
        for key, value in {
            "expression": "x^4/4-x^2+2*x",
            "x0": "0",
            "max_iterations": "250",
        }.items():
            p.fields[key].set(value)
        release = threading.Event()

        def gated_train(*args, **kwargs):
            release.wait()  # The main-thread finally block always releases this gate.
            return train(*args, **kwargs)

        with patch(
            "every_example_gui.ui.optimization_gui.train", side_effect=gated_train
        ):
            try:
                a.start()
                self.choose_method(GRADIENT)
                self.assertTrue(p.run_button.instate(["disabled"]))
            finally:
                release.set()
            # There was no gradient result; wait without assuming the visible method ran.
            deadline = time.monotonic() + 30
            while p.busy and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
        self.assertFalse(p.busy)
        self.assertIsNone(p.result)
        self.choose_method(NEWTON)
        self.assertEqual(p.result.status, "iteration_limit")
        self.assertEqual(len(p.result.points), 252)
        self.assertEqual(len(p.trace_tree.get_children()), 200)
        p.change_page(1)
        self.assertEqual(len(p.trace_tree.get_children()), 52)
        p.trace_tree.selection_set("225")
        self.root.update()
        self.assertEqual(p.step_index, 225)
        p.precision.set("15")
        p._render()
        p.draw_chart()
        self.assertGreater(len(p.chart.find_all()), 20)

    def test_steepest_workflow_heatmap_exports_and_all_method_states(self):
        from every_example_gui.algorithms.optimization import GRADIENT, NEWTON, STEEPEST
        from every_example_gui.algorithms.steepest import GRID_SIZE, SteepestRequest

        a, p = self.app, self.panel
        self.select("Optimization")
        a.start()
        self.wait(p)
        gradient_result = p.result
        p.fields["alpha"].set("0.123")
        p.go_step(3)
        self.choose_method(STEEPEST)
        self.assertTrue(p.field_rows["y0"].winfo_ismapped())
        self.assertTrue(p.field_rows["plot_y_min"].winfo_ismapped())
        self.assertFalse(p.field_rows["first_derivative"].winfo_ismapped())
        self.assertIn("2D", self.root.title())
        self.assertIn("||∇f||₂", p.field_labels["tolerance"].get())
        p.fields["first_derivative"].set("unused")
        p.fields["second_derivative"].set("unused")
        a.start()
        self.wait(p)
        self.assertIsInstance(p.result.request, SteepestRequest)
        self.assertEqual(p.result.status, "gradient_tolerance")
        self.assertAlmostEqual(p.result.final.x, 2, places=6)
        self.assertAlmostEqual(p.result.final.y, -1, places=6)
        self.assertIn("SAME OLD pair", p.worked_text.get("1.0", "end"))
        self.assertIn("yₖ₊₁ = yₖ − α ∂f/∂y", p.worked_text.get("1.0", "end"))
        a.navigate_step(1)
        self.assertEqual(p.step_index, 1)
        p.draw_chart()
        self.assertEqual(len(p.chart.find_withtag("heatmap")), GRID_SIZE**2)
        self.assertGreater(len(p.chart.find_withtag("path")), 0)
        self.assertGreater(len(p.chart.find_withtag("selected")), 0)
        self.assertIn(
            "yₖ", [p.trace_tree.heading(c, "text") for c in p.trace_tree["columns"]]
        )
        result = p.result
        p.fields["y0"].set("9")
        p.fields["expression"].set("x+y")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steepest.csv"
            with patch(
                "every_example_gui.ui.optimization_gui.filedialog.asksaveasfilename",
                return_value=str(path),
            ):
                p.save_csv()
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]["initial_y"]), 3)
            self.assertEqual(rows[0]["function"], result.request.expression)
        self.choose_method(NEWTON)
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "step_tolerance")
        self.assertFalse(p.field_rows["y0"].winfo_ismapped())
        self.choose_method(STEEPEST)
        self.assertIs(p.result, result)
        self.assertEqual(p.fields["y0"].get(), "9")
        self.assertEqual(p.step_index, 1)
        self.choose_method(GRADIENT)
        self.assertIs(p.result, gradient_result)
        self.assertEqual(p.fields["alpha"].get(), "0.123")
        self.assertEqual(p.step_index, 3)
        self.assertEqual(p.field_labels["expression"].get(), "Function f(x)")
        self.assertEqual(
            len(
                [w for w in self.root.winfo_children() if w.winfo_class() == "Toplevel"]
            ),
            0,
        )

    def test_steepest_trace_pages_validation_cancel_domain_error_and_stationary_start(
        self,
    ):
        from every_example_gui.algorithms.optimization import STEEPEST
        from every_example_gui.algorithms.steepest import PRESETS

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(STEEPEST)
        for key, value in {
            "expression": "x^2+y^2",
            "x0": "1",
            "y0": "2",
            "alpha": "1",
            "max_iterations": "250",
        }.items():
            p.fields[key].set(value)
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "iteration_limit")
        self.assertEqual(len(p.trace_tree.get_children()), 200)
        p.change_page(1)
        self.assertEqual(len(p.trace_tree.get_children()), 51)
        p.trace_tree.selection_set("225")
        self.root.update()
        self.assertEqual(p.step_index, 225)
        original = p.result
        p.fields["y0"].set("nan")
        with patch(
            "every_example_gui.ui.optimization_gui.messagebox.showerror"
        ) as error:
            a.start()
            error.assert_called_once()
        self.assertIs(p.result, original)
        p.fields["y0"].set("2")
        a.start()
        p.cancel()
        self.wait(p)
        self.assertIs(p.result, original)
        self.assertIn("cancelled", p.status.get())
        p.fields["expression"].set("ln(y)")
        p.fields["y0"].set("-1")
        with patch(
            "every_example_gui.ui.optimization_gui.messagebox.showerror"
        ) as error:
            a.start()
            self.wait(p)
            error.assert_called_once()
        self.assertIs(p.result, original)
        p.preset.set(next(name for name in PRESETS if name.startswith("Rosenbrock")))
        p.load_example()
        self.assertIn("y-x^2", p.fields["expression"].get())
        self.assertEqual(p.fields["alpha"].get(), "0.001")
        for key, value in {"expression": "x^2+y^2", "x0": "0", "y0": "0"}.items():
            p.fields[key].set(value)
        a.start()
        self.wait(p)
        self.assertEqual(p.result.final.iteration, 0)
        self.assertEqual(len(p.trace_tree.get_children()), 1)
        self.assertTrue(p.next_button.instate(["disabled"]))
        self.assertIn("TRUE", p.worked_text.get("1.0", "end"))

    def test_steepest_background_completion_preserves_other_method_and_learning_view(
        self,
    ):
        from every_example_gui.backend import train
        from every_example_gui.algorithms.optimization import GRADIENT, STEEPEST

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(STEEPEST)
        for key, value in {
            "expression": "x^2+x*y+y^2",
            "x0": "2",
            "y0": "1",
            "max_iterations": "2",
        }.items():
            p.fields[key].set(value)
        release = threading.Event()

        def gated_train(*args, **kwargs):
            release.wait()  # The main-thread finally block always releases this gate.
            return train(*args, **kwargs)

        with patch(
            "every_example_gui.ui.optimization_gui.train", side_effect=gated_train
        ):
            try:
                a.start()
                self.choose_method(GRADIENT)
                self.assertTrue(p.run_button.instate(["disabled"]))
                self.select("Learning")
            finally:
                release.set()
            deadline = time.monotonic() + 30
            while p.busy and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
        self.assertFalse(p.busy)
        self.assertIsNone(p.result)
        self.assertTrue(a.body.winfo_ismapped())
        self.select("Optimization")
        self.choose_method(STEEPEST)
        self.assertEqual(len(p.result.points), 3)
        self.assertAlmostEqual(p.result.final.x, 1.14)
        self.assertAlmostEqual(p.result.final.y, 0.33)
        self.assertIn("2D", self.root.title())

    def test_newton2d_full_hessian_workflow_exports_and_method_preservation(self):
        from every_example_gui.algorithms.optimization import NEWTON, NEWTON2D, STEEPEST
        from every_example_gui.algorithms.newton2d import Newton2DRequest
        from every_example_gui.algorithms.steepest import GRID_SIZE

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(STEEPEST)
        a.start()
        self.wait(p)
        steepest_result = p.result
        p.fields["alpha"].set("0.125")
        p.go_step(3)
        self.choose_method(NEWTON2D)
        self.assertIsNone(p.result)
        self.assertTrue(p.field_rows["y0"].winfo_ismapped())
        self.assertTrue(p.field_rows["plot_y_min"].winfo_ismapped())
        self.assertFalse(p.field_rows["alpha"].winfo_ismapped())
        self.assertFalse(p.field_rows["first_derivative"].winfo_ismapped())
        self.assertIn("Newton 2D", self.root.title())
        self.assertEqual(p.field_labels["max_iterations"].get(), "Maximum iterations")
        p.fields["alpha"].set("unused")
        p.fields["first_derivative"].set("unused")
        p.fields["second_derivative"].set("unused")
        self.assertIsInstance(p.read_request(), Newton2DRequest)
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "gradient_tolerance")
        self.assertEqual([(q.x, q.y) for q in p.result.points], [(6, 3), (2, -1)])
        self.assertIn("FULL HESSIAN", p.worked_text.get("1.0", "end"))
        self.assertIn("fₓᵧ = fᵧₓ = 1", p.worked_text.get("1.0", "end"))
        self.assertIn("s = H⁻¹∇f = (4, 4)", p.worked_text.get("1.0", "end"))
        headings = [p.trace_tree.heading(c, "text") for c in p.trace_tree["columns"]]
        self.assertIn("Hₓᵧ", headings)
        self.assertIn("Hᵧₓ", headings)
        p.draw_chart()
        self.assertEqual(len(p.chart.find_withtag("heatmap")), GRID_SIZE**2)
        self.assertGreater(len(p.chart.find_withtag("path")), 0)
        a.navigate_step(1)
        self.assertIn("TRUE", p.worked_text.get("1.0", "end"))
        self.assertIn("positive definite", p.summary_text.get("1.0", "end"))
        result = p.result
        p.fields["expression"].set("x+y")
        p.fields["x0"].set("9")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "newton2d.csv"
            with patch(
                "every_example_gui.ui.optimization_gui.filedialog.asksaveasfilename",
                return_value=str(path),
            ):
                p.save_csv()
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["function"], result.request.expression)
            self.assertEqual(float(rows[0]["initial_x"]), 6)
            self.assertEqual(float(rows[0]["hessian_xy"]), 1)
            self.assertEqual(float(rows[0]["delta_y"]), -4)
        self.choose_method(NEWTON)
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "step_tolerance")
        self.choose_method(STEEPEST)
        self.assertIs(p.result, steepest_result)
        self.assertEqual(p.fields["alpha"].get(), "0.125")
        self.assertEqual(p.step_index, 3)
        self.choose_method(NEWTON2D)
        self.assertIs(p.result, result)
        self.assertEqual(p.fields["x0"].get(), "9")
        self.assertEqual(p.fields["expression"].get(), "x+y")
        self.assertEqual(p.step_index, 1)
        self.select("Learning")
        self.select("Optimization")
        self.assertIn("Newton 2D", self.root.title())
        self.assertEqual(
            len(
                [w for w in self.root.winfo_children() if w.winfo_class() == "Toplevel"]
            ),
            0,
        )

    def test_newton2d_pagination_singular_hessian_errors_cancel_and_negative_curvature(
        self,
    ):
        from every_example_gui.algorithms.optimization import NEWTON2D
        from every_example_gui.algorithms.newton2d import PRESETS

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(NEWTON2D)
        for key, value in {
            "expression": "x^4/4-x^2+2*x+y^2",
            "x0": "0",
            "y0": "1",
            "max_iterations": "250",
        }.items():
            p.fields[key].set(value)
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "iteration_limit")
        self.assertEqual(len(p.result.points), 251)
        self.assertEqual(len(p.trace_tree.get_children()), 200)
        p.change_page(1)
        self.assertEqual(len(p.trace_tree.get_children()), 51)
        p.trace_tree.selection_set("225")
        self.root.update()
        self.assertEqual(p.step_index, 225)
        p.precision.set("15")
        p._render()
        for key, value in {"expression": "x^2+y", "x0": "1", "y0": "1"}.items():
            p.fields[key].set(value)
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "singular_hessian")
        self.assertEqual(p.result.final.iteration, 0)
        self.assertIn(
            "Cannot invert the singular Hessian", p.worked_text.get("1.0", "end")
        )
        original = p.result
        p.fields["y0"].set("nan")
        with patch(
            "every_example_gui.ui.optimization_gui.messagebox.showerror"
        ) as error:
            a.start()
            error.assert_called_once()
        self.assertIs(p.result, original)
        p.fields["y0"].set("1")
        a.start()
        p.cancel()
        self.wait(p)
        self.assertIs(p.result, original)
        self.assertIn("cancelled", p.status.get())
        p.fields["expression"].set("sqrt(x)+y^2")
        p.fields["x0"].set("-1")
        with patch(
            "every_example_gui.ui.optimization_gui.messagebox.showerror"
        ) as error:
            a.start()
            self.wait(p)
            error.assert_called_once()
        self.assertIs(p.result, original)
        for key, value in {"expression": "-x^2-y^2", "x0": "1", "y0": "2"}.items():
            p.fields[key].set(value)
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "gradient_tolerance")
        self.assertIn("negative definite", p.summary_text.get("1.0", "end"))
        p.preset.set(next(name for name in PRESETS if name.startswith("Rosenbrock")))
        p.load_example()
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "gradient_tolerance")
        self.assertAlmostEqual(p.result.final.x, 1, places=6)
        self.assertAlmostEqual(p.result.final.y, 1, places=6)

    def test_newton2d_background_completion_keeps_its_snapshot_and_learning_view(self):
        from every_example_gui.backend import train
        from every_example_gui.algorithms.optimization import NEWTON2D, STEEPEST

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(NEWTON2D)
        release = threading.Event()

        def gated_train(*args, **kwargs):
            release.wait()  # Always released by the main-thread finally block.
            return train(*args, **kwargs)

        with patch(
            "every_example_gui.ui.optimization_gui.train", side_effect=gated_train
        ):
            try:
                a.start()
                p.fields["x0"].set("10")
                self.choose_method(STEEPEST)
                self.assertTrue(p.run_button.instate(["disabled"]))
                self.select("Learning")
            finally:
                release.set()
            deadline = time.monotonic() + 30
            while p.busy and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
        self.assertFalse(p.busy)
        self.assertIsNone(p.result)
        self.assertTrue(a.body.winfo_ismapped())
        self.select("Optimization")
        self.choose_method(NEWTON2D)
        self.assertEqual(p.result.request.x0, 6)
        self.assertEqual(p.fields["x0"].get(), "10")
        self.assertEqual(p.result.status, "gradient_tolerance")
        self.assertEqual((p.result.final.x, p.result.final.y), (2, -1))
        self.assertIn("Inputs edited", p.status.get())

    def test_nelder_mead_workflow_simplex_plot_exports_and_cached_methods(self):
        from every_example_gui.algorithms.optimization import NEWTON2D, NELDER_MEAD
        from every_example_gui.algorithms.nelder_mead import NelderMeadRequest
        from every_example_gui.algorithms.steepest import GRID_SIZE

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(NEWTON2D)
        a.start()
        self.wait(p)
        newton_result = p.result
        p.fields["x0"].set("9")
        self.choose_method(NELDER_MEAD)
        self.assertIsNone(p.result)
        self.assertTrue(p.field_rows["b_x"].winfo_ismapped())
        self.assertTrue(p.field_rows["c_y"].winfo_ismapped())
        self.assertFalse(p.field_rows["alpha"].winfo_ismapped())
        self.assertFalse(p.field_rows["first_derivative"].winfo_ismapped())
        self.assertIn("Nelder", self.root.title())
        self.assertIn("f(C)−f(A)", p.field_labels["tolerance"].get())
        self.assertIn("Initial vertex A", p.field_labels["x0"].get())
        p.fields["alpha"].set("unused")
        p.fields["first_derivative"].set("unused")
        p.fields["second_derivative"].set("unused")
        self.assertIsInstance(p.read_request(), NelderMeadRequest)
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "function_tolerance")
        self.assertLess(p.result.final.value, 1e-6)
        self.assertIn("CENTROID", p.worked_text.get("1.0", "end"))
        self.assertIn("REFLECTION", p.worked_text.get("1.0", "end"))
        p.draw_chart()
        self.assertEqual(len(p.chart.find_withtag("heatmap")), GRID_SIZE**2)
        self.assertEqual(len(p.chart.find_withtag("simplex")), 3)
        self.assertEqual(len(p.chart.find_withtag("next_simplex")), 3)
        self.assertGreater(len(p.chart.find_withtag("trial_point")), 0)
        a.navigate_step(1)
        self.assertEqual(p.step_index, 1)
        headings = [p.trace_tree.heading(c, "text") for c in p.trace_tree["columns"]]
        self.assertIn("C f", headings)
        self.assertIn("Action", headings)
        original = p.result
        p.fields["b_x"].set("99")
        p.fields["expression"].set("x+y")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nelder.csv"
            with patch(
                "every_example_gui.ui.optimization_gui.filedialog.asksaveasfilename",
                return_value=str(path),
            ):
                p.save_csv()
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["function"], original.request.expression)
            self.assertEqual(float(rows[0]["initial_B_x"]), 7)
            self.assertEqual(len(rows), len(original.points))
            self.assertEqual(rows[-1]["action"], "")
        self.choose_method(NEWTON2D)
        self.assertIs(p.result, newton_result)
        self.assertEqual(p.fields["x0"].get(), "9")
        self.assertFalse(p.field_rows["b_x"].winfo_ismapped())
        self.choose_method(NELDER_MEAD)
        self.assertIs(p.result, original)
        self.assertEqual(p.fields["b_x"].get(), "99")
        self.assertEqual(p.step_index, 1)
        p.go_step(len(p.result.points) - 1)
        self.assertIn("No further simplex update", p.worked_text.get("1.0", "end"))
        self.assertTrue(p.next_button.instate(["disabled"]))
        self.select("Learning")
        self.select("Optimization")
        self.assertIn("Nelder", self.root.title())
        self.assertEqual(
            len(
                [w for w in self.root.winfo_children() if w.winfo_class() == "Toplevel"]
            ),
            0,
        )

    def test_nelder_mead_shrink_failures_validation_cancellation_and_pages(self):
        from every_example_gui.algorithms.optimization import NELDER_MEAD

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(NELDER_MEAD)

        def set_fields(values):
            for key, value in values.items():
                p.fields[key].set(value)

        set_fields(
            {
                "expression": "3*x^2+y^2+5*y*(2-y)*(x-1)^2",
                "x0": "0",
                "y0": "0",
                "b_x": "1",
                "b_y": "0",
                "c_x": "0",
                "c_y": "2",
                "max_iterations": "1",
            }
        )
        a.start()
        self.wait(p)
        self.assertEqual(p.result.steps[0].action, "shrink")
        self.assertIn("SHRINK", p.worked_text.get("1.0", "end"))
        self.assertIn("4 < 4 → FALSE", p.worked_text.get("1.0", "end"))
        original = p.result
        p.fields["b_x"].set("nan")
        with patch(
            "every_example_gui.ui.optimization_gui.messagebox.showerror"
        ) as error:
            a.start()
            error.assert_called_once()
        self.assertIs(p.result, original)
        set_fields({"b_x": "1", "b_y": "1", "c_x": "2", "c_y": "2"})
        with patch(
            "every_example_gui.ui.optimization_gui.messagebox.showerror"
        ) as error:
            a.start()
            self.wait(p)
            error.assert_called_once()
        self.assertIs(p.result, original)
        set_fields({"b_y": "0", "c_x": "0"})
        a.start()
        p.cancel()
        self.wait(p)
        self.assertIs(p.result, original)
        self.assertIn("cancelled", p.status.get())
        set_fields(
            {
                "expression": "x^2+sqrt(y)",
                "x0": "0",
                "y0": "3",
                "b_x": "1",
                "b_y": "3",
                "c_x": "1",
                "c_y": "5",
            }
        )
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "numerical_limit")
        self.assertEqual(p.result.failure.stage, "expansion")
        self.assertIn(
            "entire attempted iteration was rejected", p.worked_text.get("1.0", "end")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.csv"
            with patch(
                "every_example_gui.ui.optimization_gui.filedialog.asksaveasfilename",
                return_value=str(path),
            ):
                p.save_csv()
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[-1]["failed_stage"], "expansion")
        set_fields(
            {
                "expression": "x^2+2*y^2",
                "x0": "1.1",
                "y0": "0.9",
                "b_x": "2.3",
                "b_y": "1.2",
                "c_x": "0.7",
                "c_y": "2.4",
                "max_iterations": "250",
                "tolerance": "1e-300",
            }
        )
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "iteration_limit")
        self.assertEqual(len(p.result.points), 251)
        self.assertEqual(len(p.trace_tree.get_children()), 200)
        p.change_page(1)
        self.assertEqual(len(p.trace_tree.get_children()), 51)
        p.trace_tree.selection_set("225")
        self.root.update()
        self.assertEqual(p.step_index, 225)
        p.precision.set("15")
        p._render()
        self.assertIn("simplexes", p.page_text.get())
        set_fields({"expression": "7", "tolerance": "1e-6"})
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "function_tolerance")
        self.assertEqual(p.result.final.iteration, 0)
        self.assertEqual(len(p.trace_tree.get_children()), 1)
        self.assertTrue(p.next_button.instate(["disabled"]))

    def test_nelder_mead_background_snapshot_survives_switching_to_learning(self):
        from every_example_gui.backend import train
        from every_example_gui.algorithms.optimization import NELDER_MEAD, NEWTON2D

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(NELDER_MEAD)
        release = threading.Event()

        def gated_train(*args, **kwargs):
            release.wait()  # The main-thread finally block always releases this gate.
            return train(*args, **kwargs)

        with patch(
            "every_example_gui.ui.optimization_gui.train", side_effect=gated_train
        ):
            try:
                a.start()
                p.fields["b_x"].set("9")
                self.choose_method(NEWTON2D)
                self.assertTrue(p.run_button.instate(["disabled"]))
                self.select("Learning")
            finally:
                release.set()
            deadline = time.monotonic() + 30
            while p.busy and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
        self.assertFalse(p.busy)
        self.assertIsNone(p.result)
        self.assertTrue(a.body.winfo_ismapped())
        self.select("Optimization")
        self.choose_method(NELDER_MEAD)
        self.assertEqual(p.result.request.simplex[1][0], 7)
        self.assertEqual(p.fields["b_x"].get(), "9")
        self.assertEqual(p.result.status, "function_tolerance")
        self.assertIn("Inputs edited", p.status.get())

    def test_normalized_gradient_separate_method_line_trace_exports_and_state(self):
        from every_example_gui.algorithms.optimization import NORMALIZED, STEEPEST
        from every_example_gui.algorithms.normalized_gradient import NormalizedRequest
        from every_example_gui.ui import normalized_gradient_view

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(STEEPEST)
        a.start()
        self.wait(p)
        old = p.result
        p.fields["alpha"].set("0.125")
        p.go_step(3)
        self.choose_method(NORMALIZED)
        self.assertTrue(p.field_rows["initial_gamma"].winfo_ismapped())
        self.assertTrue(p.field_rows["line_tolerance"].winfo_ismapped())
        self.assertFalse(p.field_rows["alpha"].winfo_ismapped())
        self.assertFalse(p.field_rows["b_x"].winfo_ismapped())
        self.assertEqual(p.notebook.tab(p.line_tab, "state"), "normal")
        p.fields["alpha"].set("unused")
        p.fields["first_derivative"].set("unused")
        self.assertIsInstance(p.read_request(), NormalizedRequest)
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "gradient_tolerance")
        self.assertEqual([(q.x, q.y) for q in p.result.points], [(3, 4), (0, 0)])
        self.assertIn("NORMALIZED DIRECTION", p.worked_text.get("1.0", "end"))
        self.assertIn("γₖ=5", p.worked_text.get("1.0", "end"))
        self.assertEqual(len(p.line_tree.get_children()), 7)
        p.draw_chart()
        self.assertGreater(len(p.chart.find_withtag("normalized_direction")), 0)
        self.assertGreater(len(p.chart.find_withtag("accepted_gamma")), 0)
        p.notebook.select(p.line_tab)
        p.line_tree.selection_set("2")
        self.root.update()
        self.assertEqual(p.line_sample_index, 2)
        self.assertGreater(len(p.chart.find_withtag("selected_probe")), 0)
        a.navigate_step(1)
        self.assertEqual(len(p.line_tree.get_children()), 0)
        self.assertIn("BEFORE normalization", p.worked_text.get("1.0", "end"))
        p.go_step(0)
        result = p.result
        p.fields["initial_gamma"].set("2")
        p.fields["expression"].set("x+y")
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory) / "outer.csv"
            lines = Path(directory) / "lines.csv"
            with patch(
                "every_example_gui.ui.optimization_gui.filedialog.asksaveasfilename",
                return_value=str(outer),
            ):
                p.save_csv()
            with patch(
                "every_example_gui.ui.normalized_gradient_view.filedialog.asksaveasfilename",
                return_value=str(lines),
            ):
                normalized_gradient_view.save_lines(p)
            with outer.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            with lines.open(newline="") as stream:
                samples = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]["gamma_k"]), 5)
            self.assertEqual(rows[0]["function"], "x^2+y^2")
            self.assertEqual(float(samples[0]["initial_gamma"]), 1)
            self.assertEqual(len(samples), 7)
        self.choose_method(STEEPEST)
        self.assertIs(p.result, old)
        self.assertEqual(p.fields["alpha"].get(), "0.125")
        self.assertEqual(p.step_index, 3)
        self.assertEqual(p.notebook.tab(p.line_tab, "state"), "hidden")
        self.assertFalse(p.field_rows["initial_gamma"].winfo_ismapped())
        self.choose_method(NORMALIZED)
        self.assertIs(p.result, result)
        self.assertEqual(p.fields["initial_gamma"].get(), "2")
        self.select("Learning")
        self.select("Optimization")
        self.assertIn("Normalized", self.root.title())
        self.assertEqual(
            len(
                [w for w in self.root.winfo_children() if w.winfo_class() == "Toplevel"]
            ),
            0,
        )

    def test_normalized_line_pagination_limits_validation_cancel_and_zero_gradient(
        self,
    ):
        from every_example_gui.algorithms.optimization import NORMALIZED
        from every_example_gui.ui import normalized_gradient_view

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(NORMALIZED)
        p.fields["initial_gamma"].set("1e-100")
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "gradient_tolerance")
        self.assertEqual(len(p.line_tree.get_children()), 200)
        p.notebook.select(p.line_tab)
        normalized_gradient_view.change_line_page(p, 1)
        self.assertEqual(
            len(p.line_tree.get_children()), len(p.result.searches[0].samples) - 200
        )
        p.line_tree.selection_set("225")
        self.root.update()
        self.assertEqual(p.line_sample_index, 225)
        p.precision.set("15")
        p._render()
        self.assertEqual(p.line_page, 1)
        original = p.result
        p.fields["line_tolerance"].set("0")
        with patch(
            "every_example_gui.ui.optimization_gui.messagebox.showerror"
        ) as error:
            a.start()
            error.assert_called_once()
        self.assertIs(p.result, original)
        p.fields["line_tolerance"].set("1e-10")
        p.fields["initial_gamma"].set("1")
        p.fields["max_line_evaluations"].set("1")
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "line_search_limit")
        self.assertEqual(p.result.final.iteration, 0)
        self.assertEqual(len(p.line_tree.get_children()), 2)
        self.assertIn("old point is retained", p.worked_text.get("1.0", "end"))
        original = p.result
        p.fields["max_line_evaluations"].set("512")
        a.start()
        p.cancel()
        self.wait(p)
        self.assertIs(p.result, original)
        self.assertIn("cancelled", p.status.get())
        p.fields["expression"].set("sqrt(x)+y^2")
        p.fields["x0"].set("-1")
        with patch(
            "every_example_gui.ui.optimization_gui.messagebox.showerror"
        ) as error:
            a.start()
            self.wait(p)
            error.assert_called_once()
        self.assertIs(p.result, original)
        p.fields["expression"].set("x^2+y^2")
        p.fields["x0"].set("0")
        p.fields["y0"].set("0")
        a.start()
        self.wait(p)
        self.assertEqual(p.result.searches, ())
        self.assertEqual(len(p.line_tree.get_children()), 0)
        self.assertTrue(p.line_export_button.instate(["disabled"]))
        self.assertTrue(p.next_button.instate(["disabled"]))

    def test_normalized_background_completion_keeps_original_line_settings(self):
        from every_example_gui.backend import train
        from every_example_gui.algorithms.optimization import NORMALIZED, NEWTON2D

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(NORMALIZED)
        release = threading.Event()

        def gated_train(*args, **kwargs):
            release.wait()  # Released by the main-thread finally block.
            return train(*args, **kwargs)

        with patch(
            "every_example_gui.ui.optimization_gui.train", side_effect=gated_train
        ):
            try:
                a.start()
                p.fields["initial_gamma"].set("2")
                self.choose_method(NEWTON2D)
                self.assertTrue(p.run_button.instate(["disabled"]))
                self.select("Learning")
            finally:
                release.set()
            deadline = time.monotonic() + 30
            while p.busy and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
        self.assertFalse(p.busy)
        self.assertIsNone(p.result)
        self.assertTrue(a.body.winfo_ismapped())
        self.select("Optimization")
        self.choose_method(NORMALIZED)
        self.assertEqual(p.result.request.initial_gamma, 1)
        self.assertEqual(p.fields["initial_gamma"].get(), "2")
        self.assertEqual(p.result.searches[0].accepted.gamma, 5)
        self.assertEqual(len(p.line_tree.get_children()), 7)
        self.assertIn("Inputs edited", p.status.get())

    def test_powell_exercise_views_exports_and_independent_method_state(self):
        from every_example_gui.algorithms.optimization import POWELL, NORMALIZED
        from every_example_gui.algorithms.powell import PowellRequest
        from every_example_gui.ui import powell_view

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(NORMALIZED)
        a.start()
        self.wait(p)
        normalized = p.result
        p.fields["initial_gamma"].set("2")
        self.choose_method(POWELL)
        self.assertIsInstance(p.read_request(), PowellRequest)
        self.assertEqual(p.notebook.tab(p.line_tab, "state"), "hidden")
        self.assertEqual(p.notebook.tab(p.powell_line_tab, "state"), "normal")
        self.assertTrue(p.field_rows["initial_step"].winfo_ismapped())
        self.assertTrue(p.field_rows["y0"].winfo_ismapped())
        self.assertFalse(p.field_rows["alpha"].winfo_ismapped())
        self.assertFalse(p.field_rows["initial_gamma"].winfo_ismapped())
        self.assertFalse(p.field_rows["first_derivative"].winfo_ismapped())
        p.fields["alpha"].set("unused")
        p.fields["first_derivative"].set("unused")
        p.fields["max_iterations"].set("2")
        a.start()
        self.wait(p)
        r = p.result
        self.assertEqual(len(r.cycles), 2)
        self.assertEqual(r.final.iteration, 6)
        self.assertAlmostEqual(r.final.x, 1, places=10)
        self.assertAlmostEqual(r.final.y, 1, places=10)
        self.assertIn("Complete cycles: 2", p.summary_text.get("1.0", "end"))
        self.assertIn("SUPPLIED EXERCISE", p.notes_text.get("1.0", "end"))
        self.assertEqual(len(p.trace_tree.get_children()), 7)
        a.navigate_step(2)
        self.assertEqual(p.step_index, 2)
        self.assertIn("NEW DIRECTION", p.worked_text.get("1.0", "end"))
        self.assertIn("POST-CYCLE STOPPING TESTS", p.worked_text.get("1.0", "end"))
        p.draw_chart()
        self.assertGreater(len(p.chart.find_withtag("powell_move")), 0)
        self.assertGreater(len(p.chart.find_withtag("powell_probe")), 0)
        p.notebook.select(p.powell_line_tab)
        p.powell_line_tree.selection_set("2")
        self.root.update()
        self.assertEqual(p.powell_sample_index, 2)
        self.assertGreater(len(p.chart.find_withtag("powell_selected_probe")), 0)
        p.go_step(6)
        self.assertEqual(len(p.powell_line_tree.get_children()), 0)
        self.assertTrue(p.next_button.instate(["disabled"]))
        p.fields["initial_step"].set("2")
        p.fields["expression"].set("x+y")
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory) / "outer.csv"
            lines = Path(directory) / "lines.csv"
            with patch(
                "every_example_gui.ui.optimization_gui.filedialog.asksaveasfilename",
                return_value=str(outer),
            ):
                p.save_csv()
            with patch(
                "every_example_gui.ui.powell_view.filedialog.asksaveasfilename",
                return_value=str(lines),
            ):
                powell_view.save_lines(p)
            with outer.open(newline="") as f:
                rows = list(csv.DictReader(f))
            with lines.open(newline="") as f:
                samples = list(csv.DictReader(f))
            self.assertEqual(len(rows), 7)
            self.assertEqual(rows[0]["function"], r.request.expression)
            self.assertEqual(float(rows[0]["initial_alpha_probe"]), 1)
            self.assertEqual(len(samples), sum(len(s.samples) for s in r.searches))
        self.choose_method(NORMALIZED)
        self.assertIs(p.result, normalized)
        self.assertEqual(p.fields["initial_gamma"].get(), "2")
        self.assertEqual(p.notebook.tab(p.line_tab, "state"), "normal")
        self.assertEqual(p.notebook.tab(p.powell_line_tab, "state"), "hidden")
        self.choose_method(POWELL)
        self.assertIs(p.result, r)
        self.assertEqual(p.step_index, 6)
        self.assertEqual(p.fields["initial_step"].get(), "2")
        self.select("Learning")
        self.select("Optimization")
        self.assertIn("Powell", self.root.title())
        self.assertEqual(
            len(
                [w for w in self.root.winfo_children() if w.winfo_class() == "Toplevel"]
            ),
            0,
        )

    def test_powell_line_pagination_validation_failures_and_cancellation(self):
        from every_example_gui.algorithms.optimization import POWELL
        from every_example_gui.ui import powell_view

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(POWELL)
        p.fields["max_iterations"].set("1")
        p.fields["initial_step"].set("1e-80")
        p.fields["max_line_evaluations"].set("250")
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "line_search_limit")
        self.assertEqual(p.result.final.iteration, 0)
        self.assertEqual(len(p.powell_line_tree.get_children()), 200)
        powell_view.change_line_page(p, 1)
        self.assertEqual(len(p.powell_line_tree.get_children()), 51)
        p.powell_line_tree.selection_set("225")
        self.root.update()
        self.assertEqual(p.powell_sample_index, 225)
        p.precision.set("15")
        p._render()
        self.assertEqual(p.powell_line_page, 1)
        self.assertEqual(p.powell_sample_index, 225)
        original = p.result
        p.fields["line_tolerance"].set("0")
        with patch(
            "every_example_gui.ui.optimization_gui.messagebox.showerror"
        ) as error:
            a.start()
            error.assert_called_once()
        self.assertFalse(p.busy)
        self.assertIs(p.result, original)
        p.load_example()
        a.start()
        p.cancel()
        self.wait(p)
        self.assertIs(p.result, original)
        p.fields["expression"].set("ln(x)+y^2")
        p.fields["x0"].set("-1")
        with patch(
            "every_example_gui.ui.optimization_gui.messagebox.showerror"
        ) as error:
            a.start()
            self.wait(p)
            error.assert_called_once()
        self.assertIs(p.result, original)
        p.fields["expression"].set("7")
        p.fields["x0"].set("0")
        a.start()
        self.wait(p)
        self.assertEqual(p.result.status, "powell_function_tolerance")
        p.go_step(2)
        self.assertEqual(p.result.searches[2].status, "zero_direction")
        self.assertEqual(len(p.powell_line_tree.get_children()), 1)
        self.assertIn("exactly zero", p.worked_text.get("1.0", "end"))

    def test_powell_background_completion_keeps_original_inputs(self):
        from every_example_gui.backend import train
        from every_example_gui.algorithms.optimization import POWELL, NEWTON2D

        a, p = self.app, self.panel
        self.select("Optimization")
        self.choose_method(POWELL)
        p.fields["max_iterations"].set("2")
        release = threading.Event()

        def gated_train(*args, **kwargs):
            release.wait()  # The main-thread finally block releases the worker.
            return train(*args, **kwargs)

        with patch(
            "every_example_gui.ui.optimization_gui.train", side_effect=gated_train
        ):
            try:
                a.start()
                p.fields["initial_step"].set("3")
                self.choose_method(NEWTON2D)
                self.assertTrue(p.run_button.instate(["disabled"]))
                self.select("Learning")
            finally:
                release.set()
            deadline = time.monotonic() + 30
            while p.busy and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
        self.assertFalse(p.busy)
        self.assertIsNone(p.result)
        self.select("Optimization")
        self.choose_method(POWELL)
        self.assertEqual(p.result.request.initial_step, 1)
        self.assertEqual(p.fields["initial_step"].get(), "3")
        self.assertEqual(p.result.final.iteration, 6)
        self.assertIn("Inputs edited", p.status.get())


if __name__ == "__main__":
    unittest.main()
