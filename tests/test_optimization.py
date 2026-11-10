"""Independent numerical fixtures for the lecture recurrence and its boundaries."""

import csv
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import threading
import unittest

from every_example_gui.backend import Cancelled, InputError, train
from every_example_gui.algorithms.optimization import (
    GradientRequest,
    compile_expression,
    export_trace,
)


class GradientTests(unittest.TestCase):
    def test_hand_worked_quadratic_updates_and_final_row(self):
        r = train(GradientRequest("x^2", x0=4, alpha=0.1, max_iterations=3))
        self.assertEqual(r.status, "iteration_limit")
        for p, x in zip(r.points, (4, 3.2, 2.56, 2.048)):
            self.assertAlmostEqual(p.x, x)
            self.assertAlmostEqual(p.value, x * x)
            self.assertAlmostEqual(p.gradient, 2 * x)
        self.assertEqual(len(r.points), 4)

    def test_convergence_and_gradient_test_at_initial_and_last_point(self):
        r = train(GradientRequest())
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertAlmostEqual(r.final.x, 2, places=6)
        self.assertAlmostEqual(r.final.value, 1, places=10)
        self.assertLessEqual(abs(r.final.gradient), r.request.tolerance)
        self.assertGreater(abs(r.points[-2].gradient), r.request.tolerance)
        # Equality passes, including an initial point and a final budgeted update.
        r = train(GradientRequest("x^2", x0=0.5, tolerance=1))
        self.assertEqual(r.final.iteration, 0)
        self.assertEqual(r.status, "gradient_tolerance")
        r = train(GradientRequest("x^2", x0=4, alpha=0.5, max_iterations=1))
        self.assertEqual(r.final.x, 0)
        self.assertEqual(r.status, "gradient_tolerance")

    def test_automatic_derivatives_against_analytic_values(self):
        for expression, x, value, derivative in (
            ("sin(x) + cos(x)", 0, 1, 1),
            ("exp(x) + ln(x)", 1, math.e, math.e + 1),
            ("sqrt(x) + log(x)", 4, 2 + math.log(4), 0.5),
            ("x^x", 2, 4, 4 * (math.log(2) + 1)),
            ("x^3", -2, -8, 12),
            ("-x^2", -2, -4, 4),
            ("x^0", 0, 1, 0),
            ("(x^2 + 1)/(x + 2)", 1, 2 / 3, 4 / 9),
            ("tan(x)", 0, 0, 1),
            ("pi + e + 7", 3, math.pi + math.e + 7, 0),
        ):
            with self.subTest(expression=expression):
                r = train(GradientRequest(expression, x0=x, max_iterations=1))
                self.assertAlmostEqual(r.points[0].value, value)
                self.assertAlmostEqual(r.points[0].gradient, derivative)

    def test_fixed_rate_can_oscillate_or_increase_without_rate_changes(self):
        r = train(GradientRequest("x^2", x0=1, alpha=1, max_iterations=4))
        self.assertEqual([p.x for p in r.points], [1, -1, 1, -1, 1])
        self.assertEqual(r.status, "iteration_limit")
        r = train(GradientRequest("x^2", x0=1, alpha=2, max_iterations=2))
        self.assertEqual([p.x for p in r.points], [1, -3, 9])
        self.assertEqual(r.status, "iteration_limit")
        # View bounds never project iterates or alter the method.
        self.assertGreater(r.final.x, r.request.plot_max)

    def test_tiny_steps_are_not_gradient_convergence_and_stagnation_is_explicit(self):
        r = train(GradientRequest("x^2", x0=1, alpha=1e-10, max_iterations=2))
        self.assertEqual(r.status, "iteration_limit")
        r = train(GradientRequest("x^2", x0=1, alpha=1e-30))
        self.assertEqual(r.status, "stagnation")
        self.assertEqual(r.final.iteration, 0)

    def test_stationary_maximum_is_only_reported_as_gradient_tolerance(self):
        r = train(GradientRequest("-x^2", x0=0))
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertEqual(r.final.iteration, 0)

    def test_domain_failure_keeps_last_valid_point_and_plot_has_gaps(self):
        r = train(GradientRequest("ln(x)", x0=1, alpha=2))
        self.assertEqual(r.status, "numerical_limit")
        self.assertEqual(r.final.x, 1)
        self.assertEqual(len(r.points), 1)
        self.assertTrue(any(y is None for _, y in r.curve))
        with self.assertRaisesRegex(RuntimeError, "starting x0"):
            train(GradientRequest("sqrt(x)", x0=-1))
        r = train(GradientRequest("x^2", x0=1, alpha=1e50))
        self.assertEqual(r.status, "numerical_limit")
        self.assertTrue(all(math.isfinite(p.value) for p in r.points))

    def test_rejects_code_and_invalid_parameters(self):
        for expression in (
            "",
            "__import__('os').getcwd()",
            "x.real",
            "[x]",
            "x[0]",
            "lambda: x",
            "sin(x, 1)",
            "True",
            "1e999",
            "y^2",
            "2x",
        ):
            with self.subTest(expression=expression), self.assertRaises(InputError):
                compile_expression(expression)
        for changes in (
            {"alpha": 0},
            {"alpha": -1},
            {"alpha": math.inf},
            {"x0": math.nan},
            {"max_iterations": 0},
            {"max_iterations": 100001},
            {"max_iterations": 1.5},
            {"tolerance": 0},
            {"plot_min": 7},
            {"plot_max": math.inf},
        ):
            with self.subTest(changes=changes), self.assertRaises(InputError):
                replace(GradientRequest(), **changes).validate()

    def test_csv_exports_snapshot_full_precision_and_cancelled_run(self):
        r = train(GradientRequest("x^2", x0=4, max_iterations=3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gradient.csv"
            export_trace(path, r)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(float(rows[1]["x_k"]), r.points[1].x)
            self.assertEqual(rows[0]["function"], "x^2")
            self.assertEqual(rows[-1]["x_next"], "")
            self.assertEqual(rows[-1]["delta_x"], "")
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(Cancelled):
            train(GradientRequest(), cancel=cancelled)


if __name__ == "__main__":
    unittest.main()
