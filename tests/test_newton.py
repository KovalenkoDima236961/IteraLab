"""Newton fixtures for the literal update, strict post-update test and curvature."""

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
    NewtonRequest,
    export_trace,
)


class NewtonTests(unittest.TestCase):
    def test_quadratic_requires_post_update_test_and_stationary_start_still_updates(
        self,
    ):
        r = train(NewtonRequest())
        self.assertEqual(r.status, "step_tolerance")
        self.assertEqual([p.x for p in r.points], [6, 2, 2])
        self.assertEqual([p.gradient for p in r.points], [8, 0, 0])
        self.assertEqual([p.second_derivative for p in r.points], [2, 2, 2])
        r = train(NewtonRequest(x0=2))
        self.assertEqual([p.x for p in r.points], [2, 2])
        self.assertEqual(r.status, "step_tolerance")

    def test_strict_tolerance_equality_and_inclusive_Nmax(self):
        request = NewtonRequest("x^2", x0=1, tolerance=1, max_iterations=0)
        r = train(request)
        self.assertEqual([p.x for p in r.points], [1, 0])
        self.assertEqual(r.status, "iteration_limit")  # |0-1| == epsilon, not <.
        r = train(replace(request, max_iterations=1))
        self.assertEqual([p.x for p in r.points], [1, 0, 0])
        self.assertEqual(r.status, "step_tolerance")
        r = train(replace(request, tolerance=1.01))
        self.assertEqual(r.status, "step_tolerance")

    def test_hand_worked_quartic_recomputes_both_derivatives_at_old_point(self):
        r = train(NewtonRequest("x^4", x0=3, max_iterations=2))
        self.assertEqual(r.status, "iteration_limit")
        self.assertEqual(len(r.points), 4)  # i=0,1,2 performs three updates.
        for p, x in zip(r.points, (3, 2, 4 / 3, 8 / 9)):
            self.assertAlmostEqual(p.x, x)
            self.assertAlmostEqual(p.value, x**4)
            self.assertAlmostEqual(p.gradient, 4 * x**3)
            self.assertAlmostEqual(p.second_derivative, 12 * x**2)

    def test_second_order_automatic_derivatives_against_analytic_values(self):
        for expression, x, first, second in (
            ("sin(x) + cos(x)", 0, 1, -1),
            ("exp(x) + ln(x)", 1, math.e + 1, math.e - 1),
            ("sqrt(x) + log(x)", 4, 0.5, -0.09375),
            ("x^x", 2, 4 * (math.log(2) + 1), 4 * ((math.log(2) + 1) ** 2 + 0.5)),
            ("x^3", -2, 12, -12),
            ("x^2", 0, 0, 2),
            ("x^1", 0, 1, 0),
            ("x^0", 0, 0, 0),
            ("(x^2+1)/(x+2)", 1, 4 / 9, 10 / 27),
            ("(x^2+1)*(x-3)", 1, -2, 0),
            (
                "tan(x)",
                0.25,
                1 / math.cos(0.25) ** 2,
                2 * math.tan(0.25) / math.cos(0.25) ** 2,
            ),
            ("pi + e + 7", 3, 0, 0),
        ):
            with self.subTest(expression=expression):
                r = train(NewtonRequest(expression, x0=x, max_iterations=0))
                self.assertAlmostEqual(r.points[0].gradient, first)
                self.assertAlmostEqual(r.points[0].second_derivative, second)

    def test_user_supplied_derivatives_and_independent_automatic_defaults(self):
        base = NewtonRequest("(x-2)^2+1", x0=6)
        expected = train(base).points
        for first, second in (("2*(x-2)", "2"), ("2*(x-2)", ""), ("", "2")):
            with self.subTest(first=first, second=second):
                self.assertEqual(
                    train(
                        replace(base, first_derivative=first, second_derivative=second)
                    ).points,
                    expected,
                )
        # Use supplied expressions literally, without a gradient-based stop or damping.
        r = train(
            NewtonRequest(
                "x",
                x0=1,
                first_derivative="1e20",
                second_derivative="1e30",
                tolerance=1e-6,
            )
        )
        self.assertEqual(r.status, "step_tolerance")
        self.assertEqual(r.final.iteration, 1)
        self.assertEqual(r.final.gradient, 1e20)
        # The slide compares actual estimates, not the unrounded quotient step.
        r = train(
            NewtonRequest(
                "x", x0=1e20, first_derivative="1", second_derivative="1", tolerance=0.1
            )
        )
        self.assertEqual(r.status, "step_tolerance")
        self.assertEqual([p.x for p in r.points], [1e20, 1e20])

    def test_zero_curvature_never_divides_and_small_nonzero_curvature_is_used(self):
        for expression in ("x^3", "x^3+x", "7"):
            with self.subTest(expression=expression):
                r = train(NewtonRequest(expression, x0=0))
                self.assertEqual(r.status, "zero_curvature")
                self.assertEqual(r.final.iteration, 0)
        r = train(NewtonRequest("1e-20*x^2 + x", x0=0, max_iterations=0))
        self.assertEqual(r.final.iteration, 1)
        self.assertAlmostEqual(r.final.x / 1e19, -5)

    def test_no_gradient_shortcut_negative_curvature_rejection_or_backtracking(self):
        r = train(NewtonRequest("1e-12*x^2", x0=1, max_iterations=0))
        self.assertEqual(r.status, "iteration_limit")  # Tiny gradient, but step = 1.
        r = train(NewtonRequest("-x^2", x0=1))
        self.assertEqual(r.status, "step_tolerance")
        self.assertEqual(r.final.x, 0)
        self.assertEqual(
            r.final.second_derivative, -2
        )  # A maximum, not a proven minimum.
        r = train(NewtonRequest("ln(x)", x0=1, max_iterations=0))
        self.assertEqual(r.final.x, 2)
        self.assertGreater(r.final.value, r.points[0].value)
        # Newton's known 0 -> 1 -> 0 cycle for f'=x^3-2x+2.
        r = train(NewtonRequest("x^4/4 - x^2 + 2*x", x0=0, max_iterations=3))
        self.assertEqual([p.x for p in r.points], [0, 1, 0, 1, 0])
        self.assertEqual(r.status, "iteration_limit")

    def test_domain_and_numerical_failures_reject_only_the_attempted_update(self):
        r = train(NewtonRequest("-ln(x)+10*x", x0=1))
        self.assertEqual(r.status, "numerical_limit")
        self.assertEqual(r.final.x, 1)
        r = train(
            NewtonRequest("x", x0=0, first_derivative="1", second_derivative="1e-101")
        )
        self.assertEqual(r.status, "numerical_limit")
        self.assertEqual(r.final.iteration, 0)
        with self.assertRaisesRegex(RuntimeError, "starting x0"):
            train(NewtonRequest("x^1.5", x0=0))
        # The existing gradient method needs only f', and still accepts this start.
        self.assertEqual(
            train(GradientRequest("x^1.5", x0=0)).status, "gradient_tolerance"
        )

    def test_validation_and_cancellation(self):
        for changes in (
            {"max_iterations": -1},
            {"max_iterations": 100000},
            {"max_iterations": 1.5},
            {"tolerance": 0},
            {"x0": math.inf},
            {"plot_min": 7},
            {"first_derivative": "x.real"},
            {"second_derivative": "__import__('os')"},
        ):
            with self.subTest(changes=changes), self.assertRaises(InputError):
                replace(NewtonRequest(), **changes).validate()
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(Cancelled):
            train(NewtonRequest(), cancel=cancelled)

    def test_full_precision_export_includes_derivative_sources_and_strict_tests(self):
        r = train(
            NewtonRequest(
                "x^2",
                x0=1,
                tolerance=1,
                max_iterations=1,
                first_derivative="2*x",
                second_derivative="2",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "newton.csv"
            export_trace(path, r)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["algorithm"], "NEWTON1D")
        self.assertEqual(rows[0]["first_derivative"], "2*x")
        self.assertEqual(rows[0]["second_derivative"], "2")
        self.assertEqual(float(rows[0]["second_derivative_x_i"]), 2)
        self.assertEqual(rows[0]["step_tolerance_passed"], "False")
        self.assertEqual(rows[1]["step_tolerance_passed"], "True")
        self.assertEqual(rows[-1]["delta_x"], "")
        self.assertEqual(rows[-1]["x_next"], "")


if __name__ == "__main__":
    unittest.main()
