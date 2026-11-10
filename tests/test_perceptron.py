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
    train,
)
from every_example_gui.algorithms.perceptron import (
    PerceptronRequest,
    PerceptronResult,
    perceptron_samples,
)


class PerceptronTests(unittest.TestCase):
    def test_hand_worked_bipolar_and_never_predicts_neutral_and_keeps_verification_pass(
        self,
    ):
        r = train(
            PerceptronRequest(
                perceptron_samples("AND", True, "bipolar"), (0.0, 0.0), mode="bipolar"
            )
        )
        expected = [
            (0.0, 1, (0.0, 0.0), 0.0),
            (0.0, 1, (-1.0, 1.0), -1.0),
            (1.0, 1, (0.0, 0.0), -2.0),
            (-2.0, -1, (0.0, 0.0), -2.0),
        ]
        for step, (net, y, weights, bias) in zip(r.steps, expected):
            self.assertEqual(
                (step.net, step.output, step.after_weights, step.after_bias),
                (net, y, weights, bias),
            )
        self.assertEqual(r.steps[1].error, -2.0)
        self.assertEqual(r.steps[1].delta_bias, -1.0)
        self.assertEqual(r.status, "weights_stable")
        self.assertEqual(r.completed_epochs, 3)
        self.assertEqual((r.weights, r.bias), ((1.0, 1.0), -1.0))
        self.assertEqual(r.epochs[-2].correct, 4)
        self.assertEqual(r.epochs[-2].updates, 1)
        self.assertEqual(r.epochs[-1].updates, 0)
        self.assertTrue(all(s.output in (-1, 1) for s in r.steps))
        self.assertEqual([p.output for p in r.predictions], [1, -1, -1, -1])

    def test_bipolar_zero_tie_validation_and_classification_sse(self):
        request = PerceptronRequest(
            (Sample((0.0,), -1.0),), (0.0,), mode="bipolar", max_epochs=1
        )
        r = train(request)
        self.assertEqual((r.steps[0].net, r.steps[0].output), (0.0, 1))
        self.assertEqual(r.steps[0].delta_bias, -1.0)
        # Both contradictory labels at zero leave one wrong sign prediction.
        r = train(replace(request, samples=(Sample((0.0,), -1.0), Sample((0.0,), 1.0))))
        self.assertEqual(r.sse, 4.0)
        for bad in (
            replace(request, theta=0.1),
            replace(request, samples=(Sample((0.0,), 0.0),)),
        ):
            with self.assertRaises(InputError):
                bad.validate()

    def test_bipolar_exports_keep_distinct_variant_and_target_based_deltas(self):
        r = train(
            PerceptronRequest(
                (Sample((1.0,), -1.0),),
                (0.0,),
                mode="bipolar",
                alpha=0.25,
                max_epochs=1,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bipolar.csv"
            for exporter in (export_trace, export_epochs):
                exporter(path, r)
                with path.open() as stream:
                    row = next(csv.DictReader(stream))
                self.assertEqual(row["mode"], "bipolar")
                self.assertEqual(float(row["theta"]), 0.0)
                self.assertEqual(float(row["learning_rate"]), 0.25)
                if exporter is export_trace:
                    self.assertEqual(float(row["error"]), -2.0)
                    self.assertEqual(float(row["delta_bias"]), -0.25)

    def test_hand_worked_binary_and_and_required_unchanged_epoch(self):
        r = train(PerceptronRequest(perceptron_samples("AND"), (0.0, 0.0)))
        self.assertIsInstance(r, PerceptronResult)
        self.assertEqual(r.status, "weights_stable")
        self.assertEqual(r.completed_epochs, 8)
        expected = [
            (0.0, 1, 0.0, (0.0, 0.0), 0.0),
            (0.0, 1, -1.0, (-1.0, 0.0), -1.0),
            (-1.0, 0, 0.0, (-1.0, 0.0), -1.0),
            (-1.0, 0, 0.0, (-1.0, 0.0), -1.0),
            (-2.0, 0, 1.0, (0.0, 1.0), 0.0),
            (0.0, 1, -1.0, (-1.0, 1.0), -1.0),
            (0.0, 1, -1.0, (-1.0, 0.0), -2.0),
            (-2.0, 0, 0.0, (-1.0, 0.0), -2.0),
        ]
        for s, (net, output, error, weights, bias) in zip(r.steps, expected):
            self.assertEqual(
                (s.net, s.output, s.error, s.after_weights, s.after_bias),
                (net, output, error, weights, bias),
            )
        self.assertFalse(r.steps[0].changed)
        self.assertEqual((r.weights, r.bias), ((1.0, 2.0), -3.0))
        self.assertEqual(r.epochs[-2].correct, 4)
        self.assertGreater(r.epochs[-2].updates, 0)
        self.assertEqual(r.epochs[-1].updates, 0)
        self.assertEqual(r.epochs[-1].mistakes, 0)
        self.assertEqual(r.correct, 4)
        self.assertEqual(r.sse, 0.0)

    def test_band_matches_all_existing_lecture_traces_with_their_original_row_order(
        self,
    ):
        directory = Path(__file__).resolve().parents[2] / "du_7_prednaska_1/results"
        if not (directory / "summary.csv").exists():
            self.skipTest("Previous lecture's independent traces are not present")
        with (directory / "summary.csv").open() as stream:
            summaries = list(csv.DictReader(stream))
        for row in summaries:
            with self.subTest(gate=row["gate"], encoding=row["encoding"]):
                path = directory / f"{row['gate'].lower()}_{row['encoding']}_trace.csv"
                with path.open() as stream:
                    expected = list(csv.DictReader(stream))
                n = 1 if row["gate"] == "NOT" else 2
                samples = tuple(
                    Sample(
                        tuple(float(e[f"x{i + 1}"]) for i in range(n)),
                        float(e["target"]),
                    )
                    for e in expected
                    if int(e["epoch"]) == 1
                )
                r = train(
                    PerceptronRequest(
                        samples,
                        (0.0,) * n,
                        alpha=float(row["alpha"]),
                        mode="band",
                        theta=float(row["theta"]),
                        max_epochs=1000,
                    )
                )
                self.assertEqual(r.completed_epochs, int(row["total_epochs"]))
                self.assertEqual(len(r.steps), len(expected))
                for s, e in zip(r.steps, expected):
                    self.assertEqual(
                        (s.net, s.output, s.before_bias, s.after_bias),
                        tuple(
                            float(e[key])
                            for key in ("y_in", "y", "bias_before", "bias_after")
                        ),
                    )
                    self.assertEqual(
                        s.before_weights,
                        tuple(float(e[f"w{i + 1}_before"]) for i in range(n)),
                    )
                    self.assertEqual(
                        s.after_weights,
                        tuple(float(e[f"w{i + 1}_after"]) for i in range(n)),
                    )
                    self.assertEqual(s.changed, e["updated"] == "true")

    def test_band_update_uses_target_instead_of_error_and_correct_patterns_skip(self):
        r = train(
            PerceptronRequest(
                (Sample((1.0,), 1.0),),
                (0.0,),
                bias=-0.5,
                alpha=0.25,
                mode="band",
                max_epochs=1,
            )
        )
        s = r.steps[0]
        self.assertEqual((s.output, s.error, s.delta_bias), (-1, 2.0, 0.25))
        self.assertEqual(s.after_weights, (0.25,))
        self.assertEqual(s.after_bias, -0.25)
        correct = train(
            PerceptronRequest((Sample((1.0,), 1.0),), (1.0,), alpha=0.5, mode="band")
        )
        self.assertFalse(correct.steps[0].changed)
        self.assertEqual(correct.steps[0].delta_bias, 0.0)
        self.assertEqual(correct.completed_epochs, 1)

    def test_activation_boundary_inclusion_and_binary_zero_are_exact(self):
        for mode, theta, net, expected in (
            ("binary", 0.0, 0.0, 1),
            ("binary", 0.0, -0.125, 0),
            ("band", 0.25, -0.25, 0),
            ("band", 0.25, 0.25, 0),
            ("band", 0.0, 0.0, 0),
            ("band", 0.25, -0.5, -1),
            ("band", 0.25, 0.5, 1),
        ):
            with self.subTest(mode=mode, theta=theta, net=net):
                r = train(
                    PerceptronRequest(
                        (Sample((0.0,), 1.0),),
                        (0.0,),
                        bias=net,
                        mode=mode,
                        theta=theta,
                        max_epochs=1,
                    )
                )
                self.assertEqual(r.steps[0].output, expected)

    def test_cancelling_updates_and_xor_reach_epoch_budget(self):
        for mode, targets in (
            ("binary", (0.0, 1.0)),
            ("bipolar", (-1.0, 1.0)),
            ("band", (1.0, -1.0)),
        ):
            r = train(
                PerceptronRequest(
                    tuple(Sample((0.0,), t) for t in targets),
                    (0.0,),
                    mode=mode,
                    max_epochs=3,
                )
            )
            self.assertEqual(r.status, "epoch_limit")
            self.assertTrue(all(e.net_change == 0 and e.updates == 2 for e in r.epochs))
        r = train(
            PerceptronRequest(perceptron_samples("XOR"), (0.0, 0.0), max_epochs=20)
        )
        self.assertEqual(r.status, "epoch_limit")
        self.assertEqual(r.completed_epochs, 20)
        self.assertLess(r.correct, 4)

    def test_custom_three_input_initialization_and_single_already_correct_pattern(self):
        r = train(
            PerceptronRequest(
                (Sample((1.0, -1.0, 2.0), 0.0),),
                (0.25, -0.25, 0.5),
                bias=0.5,
                alpha=0.5,
                max_epochs=1,
            )
        )
        self.assertEqual(r.steps[0].net, 2.0)
        self.assertEqual(r.steps[0].delta_weights, (-0.5, 0.5, -1.0))
        self.assertEqual(r.weights, (-0.25, 0.25, -0.5))
        self.assertEqual(r.bias, 0.0)
        correct = train(PerceptronRequest((Sample((0.0,), 1.0),), (0.0,)))
        self.assertEqual(correct.status, "weights_stable")
        self.assertEqual(correct.completed_epochs, 1)
        self.assertEqual(correct.epochs[0].max_change, 0.0)

    def test_invalid_targets_rates_thresholds_shapes_and_trace_budget(self):
        valid = PerceptronRequest(perceptron_samples("AND"), (0.0, 0.0))
        for request in (
            replace(valid, mode="unknown"),
            replace(valid, alpha=0.0),
            replace(valid, alpha=1.1),
            replace(valid, alpha=float("nan")),
            replace(valid, theta=0.1),
            replace(valid, mode="band", theta=-1.0),
            replace(valid, mode="band", theta=float("inf")),
            replace(valid, samples=(Sample((1.0, 1.0), -1.0),)),
            replace(valid, mode="band"),
            replace(valid, weights=(0.0,)),
            replace(valid, max_epochs=100_000),
            replace(valid, max_epochs=0),
        ):
            with self.subTest(request=request), self.assertRaises(InputError):
                request.validate()

    def test_numerical_limit_preserves_initial_model_and_incomplete_epoch(self):
        r = train(PerceptronRequest((Sample((1e50, 1e50), 1.0),), (1e50, 1e50)))
        self.assertEqual(r.status, "numerical_limit")
        self.assertEqual(r.failed_at, (1, 1))
        self.assertEqual(r.completed_epochs, 0)
        self.assertEqual(r.steps, ())
        self.assertEqual(r.weights, (1e50, 1e50))
        self.assertTrue(r.sse < float("inf"))

    def test_exports_preserve_mode_theta_rate_and_all_steps(self):
        r = train(
            PerceptronRequest(
                (Sample((1.0,), 1.0),),
                (0.0,),
                bias=-0.5,
                alpha=0.25,
                mode="band",
                theta=0.125,
                max_epochs=2,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "perceptron.csv"
            export_trace(path, r)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(r.steps))
            self.assertEqual(rows[0]["mode"], "band")
            self.assertEqual(float(rows[0]["theta"]), 0.125)
            self.assertEqual(float(rows[0]["error"]), 2.0)
            self.assertEqual(float(rows[0]["delta_bias"]), 0.25)
            self.assertEqual(float(rows[0]["initial_bias"]), -0.5)
            self.assertNotIn(None, rows[0])
            export_epochs(path, r)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), r.completed_epochs)
            self.assertEqual(int(rows[0]["updates"]), r.epochs[0].updates)
            self.assertEqual(float(rows[-1]["classification_sse"]), r.sse)

    def test_preset_encodings_cancellation_and_progress(self):
        binary = perceptron_samples("AND")
        self.assertEqual(binary[-1], Sample((0.0, 0.0), 0.0))
        band = perceptron_samples("AND", True, "band")
        self.assertEqual(band[-1], Sample((-1.0, -1.0), -1.0))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            train(PerceptronRequest(binary, (0.0, 0.0)), cancel=cancel)
        messages = []
        train(PerceptronRequest(binary, (0.0, 0.0)), progress=messages.append)
        self.assertTrue(any("Perceptron" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
