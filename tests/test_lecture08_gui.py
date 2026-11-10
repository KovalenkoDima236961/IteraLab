"""Opt-in desktop tests of the Lecture 08 workspace and existing-view isolation."""

import csv
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    os.environ.get("EXAM_GUI_TESTS") == "1", "Requires a desktop display"
)
class Lecture08DesktopTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from every_example_gui.ui.app import ExamApp

        self.root = tk.Tk()
        self.callback_errors = []
        self.root.report_callback_exception = lambda *args: self.callback_errors.append(
            args
        )
        self.app = ExamApp(self.root)
        self.panel = self.app.lecture08
        self.root.update()
        self.errors = patch(
            "every_example_gui.ui.lecture08_gui.messagebox.showerror"
        ).start()

    def tearDown(self):
        self.app.close()
        patch.stopall()
        self.assertEqual(self.callback_errors, [])

    def select(self, kind):
        from every_example_gui.algorithms.lecture08 import LABELS

        self.app.algorithm.set(LABELS[kind])
        self.app.switch_algorithm()
        self.root.update()
        self.assertTrue(self.panel.frame.winfo_ismapped())
        self.assertFalse(self.app.body.winfo_ismapped())

    def set_input(self, key, text):
        self.panel._set_input(key, text)
        self.root.update()

    def wait(self, workspace):
        deadline = time.monotonic() + 30
        while workspace.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(workspace.busy)
        self.root.update()
        self.assertIsNotNone(workspace.result)
        self.assertEqual(self.callback_errors, [])
        self.errors.assert_not_called()

    def run_matrix(self):
        self.app.start()  # Exercise the existing Run/keyboard entry point, too.
        self.wait(self.panel)

    def test_correlation_navigation_and_old_worksheet_preservation(self):
        a, p = self.app, self.panel
        a.start()
        self.wait(a)
        original = a.result
        original_fields = {key: value.get() for key, value in a.fields.items()}
        original_data = a.data_input.get("1.0", "end")
        self.select("correlation")
        self.run_matrix()
        self.assertAlmostEqual(p.result.scalars["lecture_alpha_upper"], 0.375)
        self.assertIn("0.375", p.summary_text.get("1.0", "end"))
        p.go_step(len(p.result.steps) - 2)
        self.assertIn("lambda^2", p.worked_text.get("1.0", "end"))
        p.matrix_choice.set("eigen_checks")
        p.choose_matrix()
        self.assertEqual(len(p.matrix_tree.get_children()), 2)
        self.assertIn("det", p.matrix_note.get())
        matrix_result = p.result
        self.set_input("s", "2\n4")
        p.precision.set("15")
        p._render()
        self.assertIs(p.result, matrix_result)
        self.run_matrix()
        self.assertEqual(p.result.matrix("R"), ((10,),))
        a.algorithm.set("ADALINE")
        a.switch_algorithm()
        self.root.update()
        self.assertTrue(a.body.winfo_ismapped())
        self.assertFalse(p.frame.winfo_ismapped())
        self.assertIs(a.result, original)
        self.assertEqual(
            {key: value.get() for key, value in a.fields.items()}, original_fields
        )
        self.assertEqual(a.data_input.get("1.0", "end"), original_data)
        self.select("correlation")
        self.assertEqual(p.inputs["s"].get("1.0", "end-1c"), "2\n4")
        self.assertEqual(p.result.matrix("R"), ((10,),))

    def test_memory_outer_products_exports_and_recall_of_trained_model(self):
        from every_example_gui.algorithms.lecture08 import (
            EXAMPLE_T,
            EXAMPLE_W,
            parse_matrix,
        )

        p = self.panel
        self.select("hebbian_memory")
        self.run_matrix()
        memory_result = p.result
        self.assertEqual(memory_result.matrix("W"), EXAMPLE_W)
        p.go_step(2)
        self.assertIn("weights_after_2", p.worked_text.get("1.0", "end"))
        self.assertFalse(p.recall_button.instate(["disabled"]))
        self.set_input("t", "9,9")  # Exports/recall must use the calculated snapshot.
        with tempfile.TemporaryDirectory() as directory:
            for name, method in (
                ("memory.csv", p.save_csv),
                ("memory.json", p.save_json),
            ):
                path = Path(directory) / name
                with patch(
                    "every_example_gui.ui.lecture08_gui.filedialog.asksaveasfilename",
                    return_value=str(path),
                ):
                    method()
            saved = json.loads((Path(directory) / "memory.json").read_text())
            self.assertEqual(saved["request"]["t"], [list(row) for row in EXAMPLE_T])
            with (Path(directory) / "memory.csv").open(newline="") as stream:
                self.assertTrue(
                    any(
                        row["name"] == "weights_after_4"
                        for row in csv.DictReader(stream)
                    )
                )
        p.use_trained_weights()
        self.root.update()
        self.assertEqual(p.active_kind, "recall")
        self.assertEqual(parse_matrix(p.inputs["w"].get("1.0", "end")), EXAMPLE_W)
        self.assertEqual(
            parse_matrix(p.inputs["expected"].get("1.0", "end")), EXAMPLE_T
        )
        self.run_matrix()
        self.assertEqual(p.result.scalars["correct_patterns"], 4)
        self.set_input("x", "0,0,0,0")
        self.set_input("expected", "")
        self.run_matrix()
        self.assertEqual(p.result.matrix("output"), ((0, 0),))
        self.assertIn("No expected targets", p.summary_text.get("1.0", "end"))
        self.select("outer_product")
        self.run_matrix()
        self.assertEqual(p.result.matrix("W"), memory_result.matrix("W"))
        self.assertEqual(p.result.scalars["outer_product_difference"], 0)
        self.assertFalse(p.bias_controls.winfo_manager())
        self.select("hebbian_memory")
        self.assertEqual(p.inputs["t"].get("1.0", "end-1c"), "9,9")

    def test_custom_dimensions_separate_bias_and_matrix_import(self):
        from every_example_gui.algorithms.lecture08 import parse_matrix

        p = self.panel
        self.select("hebbian_memory")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.json"
            path.write_text("[[1,0],[0,1]]")
            with patch(
                "every_example_gui.ui.lecture08_gui.filedialog.askopenfilename",
                return_value=str(path),
            ):
                p.import_matrix("s")
        self.set_input("t", "1,0,1\n0,1,1")
        p.bias_enabled.set(True)
        self.run_matrix()
        self.assertEqual(p.result.matrix("W"), ((1, 0, 1), (0, 1, 1)))
        self.assertEqual(p.result.matrix("bias"), ((1, 1, 2),))
        p.use_trained_weights()
        self.root.update()
        self.assertTrue(p.bias_enabled.get())
        self.assertTrue(p.groups["bias"].winfo_manager())
        self.assertEqual(parse_matrix(p.inputs["bias"].get("1.0", "end")), ((1, 1, 2),))
        self.set_input("x", "0,0")
        self.set_input("expected", "1,1,1")
        self.run_matrix()
        self.assertEqual(p.result.matrix("linear"), ((0, 0, 0),))
        self.assertEqual(p.result.matrix("output"), ((1, 1, 1),))
        self.assertEqual(p.result.scalars["correct_patterns"], 1)

    def test_pair_pagination_validation_cancellation_and_clear(self):
        p = self.panel
        self.select("orthogonality")
        self.set_input("s", "\n".join("1,0" for _ in range(24)))
        self.run_matrix()
        p.matrix_choice.set("pairwise_dots")
        p.choose_matrix()
        self.assertEqual(len(p.matrix_tree.get_children()), 200)
        self.assertIn("276 × 4", p.matrix_page_text.get())
        p.page_matrix(1)
        self.assertEqual(len(p.matrix_tree.get_children()), 76)
        original = p.result
        self.set_input("s", "1,2\n3")
        p.start()
        self.errors.assert_called_once()
        self.assertFalse(p.busy)
        self.assertIs(p.result, original)
        self.errors.reset_mock()
        self.set_input("s", "1,0\n0,1")
        p.start()
        p.cancel()
        self.wait(p)
        self.assertIs(p.result, original)
        self.assertIn("cancelled", p.status.get())
        p.clear()
        self.assertIsNone(p.result)
        self.assertEqual(p.inputs["s"].get("1.0", "end-1c"), "")
        self.assertEqual(len(p.matrix_tree.get_children()), 0)
        self.assertTrue(p.csv_button.instate(["disabled"]))
        p.load_example()
        self.run_matrix()
        self.assertEqual(len(p.result.matrix("pairwise_dots")), 6)


if __name__ == "__main__":
    unittest.main()
