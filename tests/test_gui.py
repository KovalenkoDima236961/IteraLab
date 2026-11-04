"""EXAM_GUI_TESTS=1 python3 -m unittest every_example_gui.tests.test_gui -v"""

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
class DesktopTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from every_example_gui.app import ExamApp

        self.root = tk.Tk()
        self.app = ExamApp(self.root)
        self.root.update()
        self.assertEqual(self.app.preset.get(), "AND")

    def tearDown(self):
        self.app.close()

    def wait_for_result(self):
        deadline = time.monotonic() + 30
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(self.app.busy)
        self.assertIsNotNone(self.app.result)
        self.root.update()

    def test_complete_workflow_navigation_exports_and_immutable_result(self):
        a = self.app
        a.start()
        self.wait_for_result()
        self.assertEqual(a.result.status, "epoch_limit")
        self.assertEqual(len(a.trace_tree.get_children()), 200)
        self.assertEqual(a.metrics["accuracy"].get(), "4 / 4 correct")
        self.assertIn("STEP 3", a.worked_text.get("1.0", "end"))
        a.go_step(3)
        self.assertIn("STEP 6 · Complete epoch 1", a.worked_text.get("1.0", "end"))
        a.next_epoch()
        self.assertEqual(a.step_index, 4)
        a.change_page("trace", 1)
        self.assertIn("200", a.trace_tree.get_children())
        a.jump_value.set("75")
        a.jump_epoch()
        self.assertEqual(a.step_index, 296)
        a.notebook.select(2)
        self.root.update()
        a.draw_chart()
        self.assertGreater(len(a.chart.find_all()), 10)
        a.precision.set("15")
        a._refresh()
        self.assertEqual(a.step_index, 296)
        original = a.result
        a.fields["alpha"].set("0.1")
        self.assertIn("Inputs edited", a.status.get())
        with tempfile.TemporaryDirectory() as directory:
            for method, name in (
                (a.save_trace, "trace.csv"),
                (a.save_epochs, "epochs.csv"),
            ):
                path = Path(directory) / name
                with patch(
                    "every_example_gui.app.filedialog.asksaveasfilename",
                    return_value=str(path),
                ):
                    method()
                with path.open() as stream:
                    row = next(csv.DictReader(stream))
                    self.assertEqual(float(row["alpha"]), 0.2)
        self.assertIs(a.result, original)
        a.fields["tolerance"].set("-1")
        with patch("every_example_gui.app.messagebox.showerror") as error:
            a.start()
            error.assert_called_once()
        self.assertFalse(a.busy)
        self.assertIs(a.result, original)

    def test_unary_custom_csv_and_numerical_limit_without_updates(self):
        a = self.app
        a.preset.set("NOT")
        a.load_preset()
        a.start()
        self.wait_for_result()
        self.assertEqual(a.result.status, "tolerance_met")
        self.assertEqual(len(a.result.weights), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            path.write_text("x1,x2,x3,target\n1,-1,1,-1\n", encoding="utf-8")
            with patch(
                "every_example_gui.app.filedialog.askopenfilename",
                return_value=str(path),
            ):
                a.import_data()
            a.fields["max_epochs"].set("1")
            a.start()
            self.wait_for_result()
            self.assertEqual(len(a.result.weights), 3)
        a.data_input.delete("1.0", "end")
        a.data_input.insert("1.0", "1e50,1")
        a.fields["weights"].set("1e50")
        a.fields["bias"].set("1e50")
        a.fields["alpha"].set("1e50")
        a.start()
        self.wait_for_result()
        self.assertEqual(a.result.status, "numerical_limit")
        self.assertEqual(len(a.result.steps), 0)
        self.assertIn("No update was committed", a.worked_text.get("1.0", "end"))

    def test_cancel_retains_previous_result(self):
        a = self.app
        a.preset.set("NOT")
        a.load_preset()
        a.start()
        self.wait_for_result()
        original = a.result
        a.start()
        a.cancel()
        self.wait_for_result()
        self.assertIs(a.result, original)
        self.assertIn("cancelled", a.status.get())

    def test_madaline_switch_xor_worked_cases_exports_and_switch_back(self):
        from every_example_gui.madaline import MadalineResult

        a = self.app
        a.algorithm.set("MADALINE (MRI)")
        a.switch_algorithm()
        a.preset.set("XOR")
        a.load_preset()
        self.root.update()
        self.assertTrue(a.madaline_fields.winfo_ismapped())
        self.assertIn("MADALINE", a.algorithm_text.get("1.0", "end"))
        for key, value in {
            "weights": "0.125,-0.25",
            "weights2": "-0.375,0.125",
            "bias": "0.0625",
            "bias2": "0.1875",
            "alpha": "0.25",
        }.items():
            a.fields[key].set(value)
        a.start()
        self.wait_for_result()
        self.assertIsInstance(a.result, MadalineResult)
        self.assertEqual(a.result.status, "weights_stable")
        self.assertEqual(a.metrics["accuracy"].get(), "4 / 4 correct")
        self.assertEqual(len(a.trace_tree.get_children()), 12)
        a.go_step(0)
        self.assertIn(
            "No weights or biases are updated", a.worked_text.get("1.0", "end")
        )
        a.go_step(3)
        self.assertIn("Selected units: Z1, Z2", a.worked_text.get("1.0", "end"))
        self.assertIn("STEP 8 · Complete epoch 1", a.worked_text.get("1.0", "end"))
        a.go_step(5)
        self.assertIn("Select Z1", a.worked_text.get("1.0", "end"))
        self.assertIn("Fixed output", a.final_text.get("1.0", "end"))
        self.assertEqual(len(a.prediction_tree.get_children()), 4)
        a.notebook.select(2)
        self.root.update()
        a.draw_chart()
        self.assertGreater(len(a.chart.find_all()), 10)
        a.precision.set("15")
        a._refresh()
        original = a.result
        a.algorithm.set("ADALINE")
        a.switch_algorithm()
        self.root.update()
        self.assertFalse(a.madaline_fields.winfo_ismapped())
        a._refresh()  # The old MADALINE run must still render after the form switches.
        self.assertIs(a.result, original)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            with patch(
                "every_example_gui.app.filedialog.asksaveasfilename",
                return_value=str(path),
            ):
                a.save_trace()
            with path.open() as stream:
                row = next(csv.DictReader(stream))
                self.assertEqual(row["algorithm"], "MADALINE_MRI")
                self.assertEqual(float(row["alpha"]), 0.25)
        a.preset.set("AND")
        a.load_preset()
        a.start()
        self.wait_for_result()
        self.assertNotIsInstance(a.result, MadalineResult)
        self.assertIn("STEP 4", a.worked_text.get("1.0", "end"))

    def test_madaline_variable_hidden_units_resize_run_export_and_preserve_snapshot(
        self,
    ):
        a = self.app
        a.algorithm.set("MADALINE (MRI)")
        a.switch_algorithm()
        original_weights = a.fields["weights"].get()
        a.hidden_units.set("3")
        a.resize_hidden_units()
        self.assertEqual(a.fields["weights"].get(), original_weights)
        self.assertEqual(a.fields["output_weights"].get(), "0.5, 0.5, 0.5")
        self.assertEqual(a.fields["output_bias"].get(), "1")
        a.data_input.delete("1.0", "end")
        a.data_input.insert("1.0", "1,-1,1")
        a.initialize(False)
        for variable, bias in zip(a.hidden_bias_vars, (-0.8, -0.5, -0.1)):
            variable.set(str(bias))
        a.fields["max_epochs"].set("1")
        a.start()
        self.wait_for_result()
        self.assertEqual(a.result.steps[0].selected, (False, False, True))
        self.assertIn("Select Z3", a.worked_text.get("1.0", "end"))
        self.assertIn("z_in3", a.final_text.get("1.0", "end"))
        headings = [a.trace_tree.heading(c, "text") for c in a.trace_tree["columns"]]
        self.assertIn("Z3 selected", headings)
        self.assertIn("b3 after", headings)
        original = a.result
        a.hidden_units.set("5")
        a.resize_hidden_units()
        a._refresh()
        self.assertIs(a.result, original)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "three.csv"
            with patch(
                "every_example_gui.app.filedialog.asksaveasfilename",
                return_value=str(path),
            ):
                a.save_trace()
            with path.open() as stream:
                row = next(csv.DictReader(stream))
                self.assertEqual(int(row["hidden_units"]), 3)
                self.assertIn("v3_fixed", row)
                self.assertNotIn("v4_fixed", row)
        a.data_input.delete("1.0", "end")
        a.data_input.insert("1.0", "1,-1,-1")
        a.initialize(False)
        for variable, bias in zip(a.hidden_bias_vars, (0.25, -0.5, 0.75, 0.0, 0.125)):
            variable.set(str(bias))
        a.start()
        self.wait_for_result()
        self.assertEqual(a.result.steps[0].selected, (True, False, True, False, True))
        self.assertIn("Selected units: Z1, Z3, Z5", a.worked_text.get("1.0", "end"))
        for units in (1, 32):
            a.hidden_units.set(str(units))
            a.resize_hidden_units()
            a.initialize(False)
            for variable in a.hidden_bias_vars[:units]:
                variable.set("-0.5")
            a.start()
            self.wait_for_result()
            self.assertEqual(len(a.result.biases), units)
            self.assertEqual(a.result.correct, 1)
        a.hidden_units.set("0")
        with patch("every_example_gui.app.messagebox.showerror") as error:
            a.resize_hidden_units()
            error.assert_called_once()
        self.assertEqual(a.hidden_units.get(), "32")

    def test_hebbian_one_pass_worked_steps_verification_exports_and_switch_back(self):
        from every_example_gui.hebbian import HebbianResult

        a = self.app
        a.algorithm.set("Hebbian Learning")
        a.switch_algorithm()
        a.fields["alpha"].set("1")
        self.root.update()
        self.assertEqual(str(a.weights_entry.cget("state")), "readonly")
        self.assertFalse(a.learning_rows["tolerance"].winfo_ismapped())
        self.assertFalse(a.learning_rows["max_epochs"].winfo_ismapped())
        self.assertFalse(a.init_frame.winfo_ismapped())
        self.assertFalse(a.madaline_fields.winfo_ismapped())
        self.assertIn("η", a.rate_label.cget("text"))
        # Other algorithms' settings cannot alter Hebb's zero start or pass count.
        a.fields["weights"].set("99, 99")
        a.fields["bias"].set("99")
        a.fields["tolerance"].set("not used")
        a.fields["max_epochs"].set("not used")
        a.start()
        self.wait_for_result()
        self.assertIsInstance(a.result, HebbianResult)
        self.assertEqual(a.result.status, "pass_complete")
        self.assertEqual(a.result.weights, (2.0, 2.0))
        self.assertEqual(a.result.bias, -2.0)
        self.assertEqual(len(a.result.steps), 4)
        self.assertEqual(a.metrics["epochs"].get(), "1 / 1")
        self.assertEqual(a.metrics["accuracy"].get(), "4 / 4 correct")
        self.assertEqual(a.notebook.tab(a.epoch_tab, "text"), "Pass summary")
        self.assertIn("Δw1 = ηx1t", a.worked_text.get("1.0", "end"))
        a.next_epoch()
        self.assertEqual(a.step_index, 3)
        self.assertIn("STEP 5 · One complete pass", a.worked_text.get("1.0", "end"))
        self.assertIn("Pattern 4", a.worked_text.get("1.0", "end"))
        self.assertEqual(len(a.prediction_tree.get_children()), 4)
        a.notebook.select(a.epoch_tab)
        self.root.update()
        a.draw_chart()
        self.assertGreater(len(a.chart.find_all()), 10)
        a.precision.set("15")
        a._refresh()
        original = a.result
        a.fields["alpha"].set("0.25")
        with tempfile.TemporaryDirectory() as directory:
            for method, name in (
                (a.save_trace, "trace.csv"),
                (a.save_epochs, "pass.csv"),
            ):
                path = Path(directory) / name
                with patch(
                    "every_example_gui.app.filedialog.asksaveasfilename",
                    return_value=str(path),
                ):
                    method()
                with path.open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(float(rows[0]["eta"]), 1.0)
                self.assertEqual(rows[0]["algorithm"], "HEBBIAN")
        a.algorithm.set("ADALINE")
        a.switch_algorithm()
        self.root.update()
        self.assertTrue(a.learning_rows["max_epochs"].winfo_ismapped())
        self.assertTrue(a.init_frame.winfo_ismapped())
        self.assertEqual(str(a.weights_entry.cget("state")), "normal")
        a._refresh()
        self.assertIs(a.result, original)
        a.fields["tolerance"].set("0.001")
        a.fields["max_epochs"].set("2")
        a.start()
        self.wait_for_result()
        self.assertNotIsInstance(a.result, HebbianResult)
        self.assertEqual(a.notebook.tab(a.epoch_tab, "text"), "Epochs & learning")

    def test_hebbian_xor_custom_dimensions_numerical_limit_and_reset(self):
        a = self.app
        a.algorithm.set("Hebbian Learning")
        a.switch_algorithm()
        a.preset.set("XOR")
        a.load_preset()
        a.fields["alpha"].set("1")
        a.start()
        self.wait_for_result()
        self.assertEqual(a.result.status, "pass_complete")
        self.assertEqual(a.result.correct, 2)
        self.assertEqual(a.result.completed_epochs, 1)
        a.data_input.delete("1.0", "end")
        a.data_input.insert("1.0", "2,-3,0.5,-1")
        a.fields["alpha"].set("0.25")
        self.root.update()
        self.assertEqual(a.fields["weights"].get(), "0, 0, 0")
        a.initialize(True)
        self.assertEqual(a.fields["weights"].get(), "0, 0, 0")
        a.start()
        self.wait_for_result()
        self.assertEqual(a.result.weights, (-0.5, 0.75, -0.125))
        self.assertIn("Δw3", a.worked_text.get("1.0", "end"))
        a.start()
        self.wait_for_result()
        self.assertEqual(a.result.steps[0].before_weights, (0.0, 0.0, 0.0))
        a.data_input.delete("1.0", "end")
        a.data_input.insert("1.0", "1e50,1")
        a.fields["alpha"].set("1e50")
        a.start()
        self.wait_for_result()
        self.assertEqual(a.result.status, "numerical_limit")
        self.assertEqual(a.result.completed_epochs, 0)
        self.assertEqual(len(a.result.steps), 0)
        self.assertIn("No update was committed", a.worked_text.get("1.0", "end"))
        self.assertIn("incomplete", a.final_text.get("1.0", "end"))


if __name__ == "__main__":
    unittest.main()
