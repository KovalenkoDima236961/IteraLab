"""Desktop exercise coverage: load, calculate, render, switch and export."""

import csv
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    os.environ.get("EXAM_GUI_TESTS") == "1", "Requires a desktop display"
)
class PracticalDesktopTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from every_example_gui.ui.app import ExamApp

        self.root = tk.Tk()
        self.callbacks = []
        self.root.report_callback_exception = lambda *args: self.callbacks.append(args)
        self.app = ExamApp(self.root)
        self.errors = patch("tkinter.messagebox.showerror").start()
        self.root.update()

    def tearDown(self):
        self.app.close()
        patch.stopall()
        self.assertEqual(self.callbacks, [])
        self.errors.assert_not_called()

    def wait(self, panel):
        deadline = time.monotonic() + 30
        while panel.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(panel.busy)
        self.assertIsNotNone(panel.result)
        self.errors.assert_not_called()
        self.root.update()

    def test_every_learning_exercise_preset_runs_and_preserves_its_request(self):
        from every_example_gui.examples.learning import LEARNING_EXAMPLES
        from every_example_gui.backend import parse_samples

        a = self.app
        for title, values in LEARNING_EXAMPLES.items():
            with self.subTest(title=title):
                a.exercise_choice.set(title)
                a.load_practical_example()
                self.root.update()
                a.start()
                self.wait(a)
                self.assertEqual(
                    a.result.request.samples, parse_samples(values["data"])
                )
                self.assertEqual(
                    a.result.request.alpha
                    if hasattr(a.result.request, "alpha")
                    else a.result.request.eta,
                    float(values["alpha"]),
                )
                self.assertTrue(a.worked_text.get("1.0", "end").strip())
                if "MADALINE AND" in title:
                    self.assertEqual(a.result.steps[1].selected, (True, True))
                    self.assertEqual(len(a.result.steps), 8)

    def test_every_matrix_and_mcp_source_preset_runs_and_renders(self):
        from every_example_gui.algorithms.lecture08 import LABELS, exercise_presets

        a, p = self.app, self.app.lecture08
        for kind, label in LABELS.items():
            a.algorithm.set(label)
            a.switch_algorithm()
            self.root.update()
            for title in exercise_presets(kind):
                with self.subTest(kind=kind, title=title):
                    p.example_choice.set(title)
                    p.load_example()
                    request = p.read_request()
                    a.start()
                    self.wait(p)
                    self.assertEqual(p.result.request, request)
                    self.assertTrue(p.summary_text.get("1.0", "end").strip())
                    p.go_step(len(p.result.steps) - 1)
                    self.assertTrue(p.worked_text.get("1.0", "end").strip())
                    self.assertIn(
                        p.matrix_choice.get(), [m.name for m in p.result.matrices]
                    )

    def test_bipolar_memory_to_recall_copies_activation_and_query_inputs(self):
        from every_example_gui.algorithms.lecture08 import LABELS, IL_T

        a, p = self.app, self.app.lecture08
        a.algorithm.set(LABELS["hebbian_memory"])
        a.switch_algorithm()
        p.example_choice.set("08 pp5–6 · I and L (2×2)")
        p.load_example()
        a.start()
        self.wait(p)
        p.use_trained_weights()
        self.assertEqual(p.activation.get(), "sign")
        a.start()
        self.wait(p)
        self.assertEqual(p.result.matrix("output"), IL_T)
        self.assertIn("sign", p.summary_text.get("1.0", "end"))

    def test_optimized_steepest_direction_and_trace_export(self):
        from every_example_gui.algorithms.optimization import LINE_STEEPEST, GRADIENT

        a, p = self.app, self.app.optimization
        a.workspace.set("Optimization")
        a.switch_workspace()
        p.method.set(LINE_STEEPEST)
        p.select_method()
        p.preset.set("CV5 · optimized steepest step from (0,0)")
        p.load_example()
        p.fields["max_iterations"].set("1")
        self.root.update()
        self.assertFalse(p.field_rows["alpha"].winfo_ismapped())
        self.assertTrue(p.field_rows["initial_gamma"].winfo_ismapped())
        a.start()
        self.wait(p)
        result = p.result
        self.assertAlmostEqual(result.searches[0].accepted.gamma, 5 / 18, places=9)
        self.assertIn("unnormalized", p.worked_text.get("1.0", "end"))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.csv"
            with patch("tkinter.filedialog.asksaveasfilename", return_value=str(path)):
                p.save_csv()
            with path.open() as stream:
                self.assertEqual(
                    next(csv.DictReader(stream))["algorithm"], "LINE_STEEPEST2D"
                )
        p.method.set(GRADIENT)
        p.select_method()
        p.method.set(LINE_STEEPEST)
        p.select_method()
        self.assertIs(p.result, result)

    def test_nelder_source_presets_variants_and_position_stopping(self):
        from every_example_gui.algorithms.optimization import NELDER_MEAD
        from every_example_gui.algorithms.nelder_mead import PRESETS

        a, p = self.app, self.app.optimization
        a.workspace.set("Optimization")
        a.switch_workspace()
        p.method.set(NELDER_MEAD)
        p.select_method()
        for title in PRESETS:
            if not title.startswith(("04", "CV4", "Slides")):
                continue
            with self.subTest(title=title):
                p.preset.set(title)
                p.load_example()
                a.start()
                self.wait(p)
                self.assertEqual(p.result.request.variant, p.fields["variant"].get())
                self.assertEqual(p.result.request.stopping, p.fields["stopping"].get())
                p.go_step(len(p.result.points) - 1)
                p.draw_chart()
                self.assertGreater(len(p.chart.find_all()), 10)
        self.assertTrue(p.simplex_options.winfo_ismapped())
