import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest

from every_example_gui.backend import (
    Cancelled,
    InputError,
    Sample,
    export_epochs,
    export_trace,
    preset_samples,
    train,
)
from every_example_gui.algorithms.madaline import (
    MadalineRequest,
    MadalineResult,
    weight_names,
)


def example(max_epochs=2):
    # Binary fractions make all eight hand-worked pattern responses exact in f64.
    return MadalineRequest(
        preset_samples("XOR"),
        ((0.125, -0.25), (-0.375, 0.125)),
        hidden_biases=(0.0625, 0.1875),
        alpha=0.25,
        max_epochs=max_epochs,
    )


class MadalineTests(unittest.TestCase):
    def test_positive_update_selects_third_unit_and_later_ties_use_lowest_index(self):
        request = MadalineRequest(
            (Sample((1.0, -1.0), 1.0),),
            ((0.0, 0.0),) * 3,
            hidden_biases=(-0.8, -0.5, -0.1),
            max_epochs=1,
        )
        r = train(request)
        s = r.steps[0]
        self.assertEqual(s.selected, (False, False, True))
        self.assertEqual(s.hidden_outputs, (-1, -1, -1))
        self.assertEqual(s.net, -0.5)
        self.assertAlmostEqual(s.errors[2], 1.1)
        self.assertAlmostEqual(s.after_weights[4], 0.22)
        self.assertAlmostEqual(s.after_weights[5], -0.22)
        self.assertAlmostEqual(s.after_biases[2], 0.12)
        tied = train(replace(request, hidden_biases=(-0.8, -0.1, -0.1)))
        self.assertEqual(tied.steps[0].selected, (False, True, False))

    def test_negative_update_selects_every_positive_unit_in_five_unit_network(self):
        request = MadalineRequest(
            (Sample((1.0, -1.0), -1.0),),
            ((0.0, 0.0),) * 5,
            hidden_biases=(0.25, -0.5, 0.75, 0.0, 0.125),
            max_epochs=1,
        )
        r = train(request)
        s = r.steps[0]
        self.assertEqual(s.selected, (True, False, True, True, True))
        self.assertAlmostEqual(s.errors[4], -1.125)
        self.assertAlmostEqual(s.delta_biases[4], -0.225)
        self.assertAlmostEqual(r.epochs[0].max_change, 0.35)
        self.assertEqual(s.after_weights[2:4], (0.0, 0.0))
        self.assertEqual(s.after_weights[6:8], (-0.2, 0.2))
        self.assertEqual(request.output_weights, (0.5,) * 5)
        self.assertEqual(request.output_bias, 2.0)

    def test_single_and_32_hidden_units_and_or_output_defaults(self):
        r = train(
            MadalineRequest((Sample((0.0,), 1.0),), ((0.0,),), hidden_biases=(-0.25,))
        )
        self.assertEqual(r.status, "weights_stable")
        self.assertEqual(r.completed_epochs, 2)
        self.assertEqual(r.request.output_bias, 0.0)
        self.assertEqual(r.predictions[0].hidden_outputs, (1,))
        for units in (1, 3, 4, 12, 32):
            with self.subTest(units=units):
                request = MadalineRequest(
                    (Sample((0.0,), 1.0),),
                    ((0.0,),) * units,
                    hidden_biases=(-0.5,) * (units - 1) + (0.25,),
                )
                r = train(request)
                self.assertEqual(r.correct, 1)
                self.assertEqual(r.steps[0].rule, "correct")
                self.assertEqual(len(r.steps[0].nets), units)
                self.assertEqual(len(r.biases), units)
                self.assertEqual(r.predictions[0].net, 0.5)

    def test_variable_unit_csv_shapes_and_unambiguous_large_weight_indices(self):
        request = MadalineRequest(
            (Sample((1.0, -1.0), -1.0),),
            ((0.0, 0.0),) * 5,
            hidden_biases=(0.25, -0.5, 0.75, 0.0, 0.125),
            max_epochs=1,
        )
        r = train(request)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "five.csv"
            for exporter in (export_trace, export_epochs):
                exporter(path, r)
                with path.open() as stream:
                    reader = csv.DictReader(stream)
                    rows = list(reader)
                    self.assertEqual(
                        len(reader.fieldnames), len(set(reader.fieldnames))
                    )
                self.assertTrue(
                    all(None not in row and None not in row.values() for row in rows)
                )
                self.assertEqual(int(rows[0]["hidden_units"]), 5)
                self.assertEqual(float(rows[0]["v5_fixed"]), 0.5)
                self.assertEqual(float(rows[0]["b_out_fixed"]), 2.0)
                self.assertAlmostEqual(
                    float(
                        rows[0]["b5_after"]
                        if exporter is export_trace
                        else rows[0]["b5"]
                    ),
                    r.biases[4],
                )
        names = weight_names(12, 12)
        self.assertEqual(len(names), 144)
        self.assertEqual(len(set(names)), 144)
        self.assertIn("w1_11", names)
        self.assertIn("w11_1", names)

    def test_dynamic_count_validation_and_all_unit_trace_budget(self):
        for request in (
            MadalineRequest((Sample((1.0,), 1.0),), ()),
            MadalineRequest((Sample((1.0,), 1.0),), ((0.0,),) * 33),
            MadalineRequest(
                (Sample((1.0,), 1.0),), ((0.0,),) * 3, hidden_biases=(0.0, 0.0)
            ),
            MadalineRequest(
                (Sample((1.0,), 1.0),), ((0.0,),) * 3, output_weights=(0.5, 0.5)
            ),
            MadalineRequest(
                (Sample((1.0,) * 32, 1.0),), ((0.0,) * 32,) * 32, max_epochs=1000
            ),
        ):
            with self.subTest(request=request), self.assertRaises(InputError):
                request.validate()

    def test_two_hand_worked_xor_epochs_and_fixed_output_layer(self):
        r = train(example())
        self.assertIsInstance(r, MadalineResult)
        self.assertEqual(r.status, "epoch_limit")
        self.assertEqual(r.completed_epochs, 2)
        self.assertEqual(len(r.steps), 8)
        expected = [
            ((-0.0625, -0.0625), (-1, -1), -1, (False, False)),
            ((0.4375, -0.3125), (1, -1), 1, (False, False)),
            ((-0.3125, 0.6875), (-1, 1), 1, (False, False)),
            ((0.1875, 0.4375), (1, 1), 1, (True, True)),
            ((0.234375, 0.296875), (1, 1), 1, (True, True)),
            ((-0.16796875, -0.99609375), (-1, -1), -1, (True, False)),
            ((-1.2099609375, 0.00390625), (-1, 1), 1, (False, False)),
            ((-0.1025390625, -0.31640625), (-1, -1), -1, (False, False)),
        ]
        for step, (nets, hidden_outputs, output, selected) in zip(r.steps, expected):
            self.assertEqual(step.nets, nets)
            self.assertEqual(step.hidden_outputs, hidden_outputs)
            self.assertEqual(step.output, output)
            self.assertEqual(step.selected, selected)
        self.assertEqual(r.steps[3].delta_biases, (-0.296875, -0.359375))
        self.assertEqual(
            r.steps[3].after_weights, (0.421875, 0.046875, -0.015625, 0.484375)
        )
        self.assertEqual(
            r.weights, (0.4052734375, -0.5537109375, -0.33984375, 0.16015625)
        )
        self.assertEqual(r.biases, (-0.2509765625, -0.49609375))
        self.assertEqual(r.correct, 4)
        self.assertEqual(r.sse, 0.0)
        for step in r.steps:
            self.assertEqual(
                step.net,
                0.5 + step.hidden_outputs[0] * 0.5 + step.hidden_outputs[1] * 0.5,
            )
        stable = train(example(10))
        self.assertEqual(stable.status, "weights_stable")
        self.assertEqual(stable.completed_epochs, 3)
        self.assertEqual(stable.weights, r.weights)

    def test_closest_unit_tie_zero_boundary_and_correct_skip(self):
        samples = (Sample((1.0, -1.0), 1.0),)
        for biases, selected in (
            ((-0.4, -0.1), (False, True)),
            ((-0.2, -0.2), (True, False)),
        ):
            r = train(
                MadalineRequest(
                    samples,
                    ((0.0, 0.0), (0.0, 0.0)),
                    hidden_biases=biases,
                    max_epochs=1,
                )
            )
            self.assertEqual(r.steps[0].selected, selected)
        r = train(MadalineRequest((Sample((1.0,), -1.0),), ((0.0,), (0.0,))))
        self.assertEqual(r.steps[0].hidden_outputs, (1, 1))
        self.assertEqual(r.steps[0].selected, (True, True))
        self.assertEqual(r.status, "weights_stable")
        self.assertEqual(r.correct, 1)
        self.assertEqual(r.sse, 0.0)
        correct = train(
            MadalineRequest(
                samples, ((0.0, 0.0), (0.0, 0.0)), hidden_biases=(0.2, -0.3)
            )
        )
        self.assertEqual(correct.steps[0].rule, "correct")
        self.assertEqual(
            correct.steps[0].after_weights, correct.steps[0].before_weights
        )

    def test_nonstandard_fixed_output_and_three_input_coordinates(self):
        request = MadalineRequest(
            (Sample((1.0, -1.0, 1.0), 1.0),),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            hidden_biases=(0.1, -0.9),
            output_weights=(1.0, -1.0),
            output_bias=-3.0,
            max_epochs=1,
        )
        r = train(request)
        self.assertEqual(r.steps[0].output, -1)
        self.assertEqual(r.steps[0].selected, (True, False))
        self.assertAlmostEqual(r.steps[0].delta_weights[0], 0.18)
        self.assertAlmostEqual(r.steps[0].delta_weights[1], -0.18)
        self.assertAlmostEqual(r.steps[0].delta_weights[2], 0.18)
        self.assertEqual(r.weights[3:], request.hidden_weights[1])
        self.assertEqual(r.predictions[0].net, -1.0)

    def test_tolerance_is_strict_and_includes_hidden_biases(self):
        request = MadalineRequest(
            (Sample((0.0,), -1.0),),
            ((0.0,), (0.0,)),
            hidden_biases=(0.25, -0.5),
            tolerance=0.25,
            max_epochs=1,
        )
        r = train(request)
        self.assertEqual(r.epochs[0].max_change, 0.25)
        self.assertEqual(r.status, "epoch_limit")
        self.assertEqual(
            train(replace(request, tolerance=0.26)).status, "tolerance_met"
        )

    def test_invalid_inputs_and_numerical_limit_are_explicit(self):
        valid = example()
        for request in (
            replace(valid, hidden_weights=((0.0,), (0.0, 0.0))),
            replace(valid, hidden_biases=(0.0,)),
            replace(valid, output_weights=(0.5,)),
            replace(valid, output_bias=float("nan")),
            replace(valid, max_epochs=100_000),
            replace(valid, samples=(Sample((1.0, 1.0), 0.0),)),
        ):
            with self.subTest(request=request), self.assertRaises(InputError):
                request.validate()
        request = MadalineRequest(
            (Sample((1e50,), -1.0),), ((1e50,), (1e50,)), alpha=1e50
        )
        r = train(request)
        self.assertEqual(r.status, "numerical_limit")
        self.assertEqual(r.failed_at, (1, 1))
        self.assertEqual(r.completed_epochs, 0)
        self.assertEqual(len(r.steps), 0)
        self.assertEqual(r.weights, (1e50, 1e50))

    def test_csv_exports_capture_both_hidden_units_and_the_run_snapshot(self):
        r = train(example())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "madaline.csv"
            export_trace(path, r)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertNotIn(None, rows[0])
            self.assertEqual(rows[3]["selected_z1"], "True")
            self.assertEqual(rows[3]["selected_z2"], "True")
            self.assertEqual(float(rows[-1]["w11_after"]), r.weights[0])
            self.assertEqual(float(rows[-1]["w22_after"]), r.weights[3])
            self.assertEqual(float(rows[-1]["b2_after"]), r.biases[1])
            self.assertTrue(all(float(row["v1_fixed"]) == 0.5 for row in rows))
            self.assertEqual(float(rows[0]["initial_w12"]), -0.375)
            export_epochs(path, r)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[-1]["classification_sse"]), 0.0)
            self.assertEqual(float(rows[-1]["w22"]), r.weights[-1])

    def test_cancellation_is_shared_with_adaline(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            train(example(), cancel=cancel)


if __name__ == "__main__":
    unittest.main()
