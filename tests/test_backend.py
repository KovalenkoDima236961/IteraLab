import csv
from pathlib import Path
import tempfile
import threading
import unittest

from every_example_gui.backend import (
    Cancelled,
    InputError,
    Request,
    Sample,
    export_epochs,
    export_trace,
    parse_samples,
    preset_samples,
    samples_text,
    train,
    vector,
)


class InputTests(unittest.TestCase):
    def test_training_text_headers_comments_and_scientific_notation(self):
        samples = parse_samples("# my example\nx1,x2,target\n1, -1, 1\n-1e0; 1; -1\n")
        self.assertEqual(samples, (Sample((1.0, -1.0), 1.0), Sample((-1.0, 1.0), -1.0)))
        self.assertEqual(parse_samples(samples_text(samples)), samples)
        self.assertEqual(vector("0.1, -0.2 3e-2"), (0.1, -0.2, 0.03))

    def test_validation_rejects_bad_values_shapes_and_workload(self):
        for text in ("", "1", "NaN, 1", "text, -1"):
            with self.subTest(text=text), self.assertRaises(InputError):
                parse_samples(text)
        for request in (
            Request(preset_samples("AND"), (0.0,)),
            Request((Sample((1.0,), 0.0),), (0.0,)),
            Request(preset_samples("NOT"), (0.0,), alpha=0),
            Request(preset_samples("NOT"), (float("inf"),)),
            Request(preset_samples("NOT"), (0.0,), tolerance=float("nan")),
            Request(preset_samples("AND"), (0.0, 0.0), max_epochs=100_000),
            Request(preset_samples("NOT"), (0.0,), max_epochs=1.5),
        ):
            with self.subTest(request=request), self.assertRaises(InputError):
                request.validate()

    def test_presets_have_bipolar_targets_and_lecture_order(self):
        self.assertEqual(preset_samples("AND")[0], Sample((1.0, 1.0), 1.0))
        self.assertEqual(preset_samples("AND")[-1], Sample((-1.0, -1.0), -1.0))
        self.assertEqual(
            preset_samples("NOT"), (Sample((1.0,), -1.0), Sample((-1.0,), 1.0))
        )
        self.assertEqual(preset_samples("AND", False)[-1], Sample((0.0, 0.0), -1.0))
        self.assertEqual(
            [s.target for s in preset_samples("XOR")], [-1.0, 1.0, 1.0, -1.0]
        )


class RustIntegrationTests(unittest.TestCase):
    def test_every_value_in_existing_first_two_epochs_for_bipolar_targets(self):
        """Use the previous exercise's saved results as an independent oracle."""
        fixture = (
            Path(__file__).resolve().parents[2]
            / "du_7_prednaska_2/results/first_two_epochs.csv"
        )
        if not fixture.exists():
            self.skipTest("Previous exercise's fixture is not present")
        with fixture.open() as stream:
            rows = list(csv.DictReader(stream))
        for gate in ("AND", "OR", "NAND", "NOT"):
            for bipolar in (False, True):
                with self.subTest(gate=gate, bipolar=bipolar):
                    samples = preset_samples(gate, bipolar)
                    r = train(
                        Request(samples, (0.0,) * len(samples[0].inputs), max_epochs=2)
                    )
                    expected = [
                        row
                        for row in rows
                        if row["gate"] == gate
                        and row["input_encoding"]
                        == ("bipolar" if bipolar else "binary")
                        and row["target_encoding"] == "bipolar"
                    ]
                    self.assertEqual(len(r.steps), len(expected))
                    for step, row in zip(r.steps, expected):
                        for actual, key in (
                            (step.net, "y_in"),
                            (step.error, "error"),
                            (step.delta_bias, "alpha_error"),
                            (step.before_bias, "b_before"),
                            (step.after_bias, "b_after"),
                        ):
                            self.assertAlmostEqual(actual, float(row[key]), places=12)
                        for i in range(len(r.weights)):
                            self.assertAlmostEqual(
                                step.before_weights[i],
                                float(row[f"w{i + 1}_before"]),
                                places=12,
                            )
                            self.assertAlmostEqual(
                                step.after_weights[i],
                                float(row[f"w{i + 1}_after"]),
                                places=12,
                            )

    def test_matches_existing_final_status_weights_and_diagnostics(self):
        fixture = (
            Path(__file__).resolve().parents[2] / "du_7_prednaska_2/results/summary.csv"
        )
        if not fixture.exists():
            self.skipTest("Previous exercise's fixture is not present")
        with fixture.open() as stream:
            rows = [
                row
                for row in csv.DictReader(stream)
                if row["target_encoding"] == "bipolar"
            ]
        for row in rows:
            with self.subTest(gate=row["gate"], inputs=row["input_encoding"]):
                samples = preset_samples(
                    row["gate"], row["input_encoding"] == "bipolar"
                )
                r = train(
                    Request(samples, (0.0,) * len(samples[0].inputs), max_epochs=1000)
                )
                self.assertEqual(r.status, row["status"])
                self.assertEqual(r.completed_epochs, int(row["epochs"]))
                self.assertEqual(r.correct, int(row["correct"]))
                self.assertAlmostEqual(r.sse, float(row["sse"]), places=12)
                self.assertAlmostEqual(r.bias, float(row["b"]), places=12)
                self.assertAlmostEqual(
                    r.epochs[-1].max_change, float(row["max_step_change"]), places=12
                )
                for i, weight in enumerate(r.weights, 1):
                    self.assertAlmostEqual(weight, float(row[f"w{i}"]), places=12)

    def test_custom_three_input_sample_updates_every_coordinate(self):
        r = train(
            Request(
                (Sample((1.0, -1.0, 1.0), -1.0),),
                (0.1, 0.2, 0.3),
                bias=-0.1,
                max_epochs=1,
            )
        )
        s = r.steps[0]
        self.assertAlmostEqual(s.net, 0.1)
        for actual, expected in zip(s.delta_weights, (-0.22, 0.22, -0.22)):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(r.bias, -0.32)

    def test_strict_tolerance_includes_bias_and_activation_tie(self):
        request = Request((Sample((0.0,), 1.0),), (0.0,), tolerance=0.2, max_epochs=1)
        r = train(request)
        self.assertEqual(r.steps[0].output, 1)
        self.assertEqual(r.steps[0].after_weights, (0.0,))
        self.assertEqual(r.status, "epoch_limit")
        self.assertEqual(r.epochs[0].max_change, 0.2)
        self.assertEqual(
            train(Request(request.samples, request.weights, tolerance=0.21)).status,
            "tolerance_met",
        )

    def test_divergence_and_failure_before_the_first_update(self):
        r = train(
            Request(preset_samples("AND"), (0.0, 0.0), alpha=1.0, max_epochs=1000)
        )
        self.assertEqual(r.status, "numerical_limit")
        self.assertIsNotNone(r.failed_at)
        self.assertTrue(r.sse < float("inf"))
        request = Request((Sample((1e50,), 1.0),), (1e50,), bias=1e50, alpha=1e50)
        r = train(request)
        self.assertEqual(r.status, "numerical_limit")
        self.assertEqual(r.failed_at, (1, 1))
        self.assertEqual(r.completed_epochs, 0)
        self.assertEqual(len(r.steps), 0)
        self.assertEqual(r.weights, request.weights)

    def test_exports_contain_every_step_and_original_run_inputs(self):
        r = train(Request(preset_samples("NOT"), (0.1,), bias=-0.05, max_epochs=3))
        with tempfile.TemporaryDirectory() as directory:
            trace, epochs = (
                Path(directory) / "trace.csv",
                Path(directory) / "epochs.csv",
            )
            export_trace(trace, r)
            export_epochs(epochs, r)
            with trace.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(r.steps))
            self.assertEqual(float(rows[0]["initial_w1"]), 0.1)
            self.assertEqual(float(rows[-1]["w1_after"]), r.weights[0])
            self.assertEqual(float(rows[-1]["bias_after"]), r.bias)
            with epochs.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), r.completed_epochs)
            self.assertEqual(float(rows[-1]["sse"]), r.sse)

    def test_cancellation_before_start(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            train(Request(preset_samples("NOT"), (0.0,)), cancel=cancel)


if __name__ == "__main__":
    unittest.main()
