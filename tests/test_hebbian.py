import csv
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
from every_example_gui.hebbian import HebbianRequest, HebbianResult


class HebbianTests(unittest.TestCase):
    def test_hand_worked_bipolar_and_trace_and_final_verification(self):
        r = train(HebbianRequest(preset_samples("AND"), eta=1.0))
        self.assertIsInstance(r, HebbianResult)
        self.assertEqual(r.status, "pass_complete")
        self.assertEqual(r.completed_epochs, 1)
        self.assertEqual(len(r.steps), 4)
        expected = [
            ((1.0, 1.0), 1.0),
            ((0.0, 2.0), 0.0),
            ((1.0, 1.0), -1.0),
            ((2.0, 2.0), -2.0),
        ]
        for step, (weights, bias) in zip(r.steps, expected):
            self.assertEqual(step.after_weights, weights)
            self.assertEqual(step.after_bias, bias)
        self.assertEqual(r.steps[0].before_weights, (0.0, 0.0))
        self.assertEqual(r.steps[0].before_bias, 0.0)
        self.assertEqual(r.steps[0].delta_weights, (1.0, 1.0))
        self.assertEqual([p.net for p in r.predictions], [2.0, -2.0, -2.0, -6.0])
        self.assertEqual([p.output for p in r.predictions], [1, -1, -1, -1])
        self.assertEqual(r.correct, 4)
        self.assertEqual(r.sse, 0.0)
        self.assertEqual(len(r.epochs), 1)

    def test_learning_rate_scaling_binary_inputs_and_three_coordinates(self):
        samples = (Sample((2.0, -3.0, 0.5), -1.0), Sample((1.0, 0.0, 2.0), 1.0))
        r = train(HebbianRequest(samples, eta=0.25))
        self.assertEqual(r.steps[0].delta_weights, (-0.5, 0.75, -0.125))
        self.assertEqual(r.steps[0].delta_bias, -0.25)
        self.assertEqual(r.weights, (-0.25, 0.75, 0.375))
        self.assertEqual(r.bias, 0.0)
        binary = train(HebbianRequest(preset_samples("AND", False), eta=1.0))
        self.assertEqual(binary.weights, (0.0, 0.0))
        self.assertEqual(binary.bias, -2.0)
        self.assertEqual(binary.correct, 3)
        self.assertEqual(binary.status, "pass_complete")

    def test_xor_completes_once_with_errors_and_activation_tie_is_positive(self):
        r = train(HebbianRequest(preset_samples("XOR"), eta=1.0))
        self.assertEqual(r.status, "pass_complete")
        self.assertEqual(len(r.steps), 4)
        self.assertEqual(r.weights, (0.0, 0.0))
        self.assertEqual(r.bias, 0.0)
        self.assertEqual(r.correct, 2)
        self.assertEqual(r.sse, 8.0)
        self.assertTrue(all(p.net == 0.0 and p.output == 1 for p in r.predictions))

    def test_given_row_order_and_each_run_starts_again_from_zero(self):
        samples = tuple(reversed(preset_samples("AND")))
        r = train(HebbianRequest(samples, eta=1.0))
        self.assertEqual(r.steps[0].delta_weights, (1.0, 1.0))
        self.assertEqual(r.steps[0].delta_bias, -1.0)
        self.assertEqual(r.steps[1].before_weights, r.steps[0].after_weights)
        again = train(r.request)
        self.assertEqual(again.steps, r.steps)
        self.assertEqual(again.steps[0].before_weights, (0.0, 0.0))

    def test_invalid_eta_shapes_targets_and_workload(self):
        for request in (
            HebbianRequest(()),
            *[
                HebbianRequest(preset_samples("AND"), eta)
                for eta in (0.0, -1.0, float("nan"), float("inf"), 1e51)
            ],
            HebbianRequest((Sample((1.0,), 0.0),)),
            HebbianRequest((Sample((1.0,), 1.0), Sample((1.0, 2.0), -1.0))),
            HebbianRequest((Sample((float("nan"),), 1.0),)),
            HebbianRequest((Sample((1.0,) * 33, 1.0),)),
            HebbianRequest((Sample((1.0,), 1.0),) * 129),
        ):
            with self.subTest(request=request), self.assertRaises(InputError):
                request.validate()

    def test_numerical_limit_keeps_prefix_and_has_no_completed_pass(self):
        large = 2.0**165
        r = train(HebbianRequest((Sample((large,), 1.0),) * 5, eta=large))
        self.assertEqual(r.status, "numerical_limit")
        self.assertEqual(r.failed_at, (1, 5))
        self.assertEqual(len(r.steps), 4)
        self.assertEqual(r.completed_epochs, 0)
        self.assertEqual(r.epochs, ())
        self.assertEqual(r.weights, (2.0**332,))
        self.assertEqual(len(r.predictions), 5)
        self.assertTrue(r.sse < float("inf"))
        first_failure = train(HebbianRequest((Sample((1e50,), 1.0),), eta=1e50))
        self.assertEqual(first_failure.failed_at, (1, 1))
        self.assertEqual(first_failure.weights, (0.0,))

    def test_trace_and_pass_exports_have_full_precision_and_original_eta(self):
        r = train(HebbianRequest(preset_samples("NOT"), eta=0.123456789012345))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hebbian.csv"
            export_trace(path, r)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertNotIn(None, rows[0])
            self.assertNotIn(None, rows[0].values())
            self.assertEqual(rows[0]["algorithm"], "HEBBIAN")
            self.assertEqual(float(rows[0]["eta"]), r.request.eta)
            self.assertEqual(float(rows[0]["initial_w1"]), 0.0)
            self.assertEqual(float(rows[-1]["w1_after"]), r.weights[0])
            self.assertEqual(rows[-1]["pass_complete"], "True")
            export_epochs(path, r)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["classification_sse"]), r.sse)
            self.assertNotIn("tolerance", rows[0])

    def test_cancellation_and_progress_identify_hebbian(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            train(HebbianRequest(preset_samples("NOT")), cancel=cancel)
        messages = []
        train(HebbianRequest(preset_samples("NOT")), progress=messages.append)
        self.assertTrue(any("Hebbian Learning" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
