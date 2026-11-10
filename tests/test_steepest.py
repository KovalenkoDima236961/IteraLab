"""Independent fixtures for simultaneous 2D fixed-rate gradient descent."""

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
    compile_expression,
    export_trace,
)
from every_example_gui.algorithms.steepest import GRID_SIZE, SteepestRequest


class SteepestTests(unittest.TestCase):
    def test_coupled_hand_calculation_uses_same_old_pair_for_both_updates(self):
        r = train(SteepestRequest("x^2 + x*y + y^2", x0=2, y0=1, max_iterations=2))
        self.assertEqual(r.status, "iteration_limit")
        for p, (x, y, value, gx, gy) in zip(
            r.points,
            (
                (2, 1, 7, 5, 4),
                (1.5, 0.6, 3.51, 3.6, 2.7),
                (1.14, 0.33, 1.7847, 2.61, 1.8),
            ),
        ):
            for actual, expected in (
                (p.x, x),
                (p.y, y),
                (p.value, value),
                (p.gradient_x, gx),
                (p.gradient_y, gy),
                (p.gradient_norm, math.hypot(gx, gy)),
            ):
                self.assertAlmostEqual(actual, expected)
        self.assertEqual(len(r.points), 3)

    def test_default_bowl_converges_to_known_minimum(self):
        r = train(SteepestRequest())
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertAlmostEqual(r.final.x, 2, places=6)
        self.assertAlmostEqual(r.final.y, -1, places=6)
        self.assertAlmostEqual(r.final.value, 0, places=10)
        self.assertLessEqual(r.final.gradient_norm, r.request.tolerance)
        self.assertGreater(r.points[-2].gradient_norm, r.request.tolerance)

    def test_euclidean_norm_inclusive_tolerance_and_initial_stationarity(self):
        r = train(SteepestRequest("3*x+4*y", x0=0, y0=0, tolerance=5))
        self.assertEqual(r.final.gradient_norm, 5)
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertEqual(r.final.iteration, 0)
        r = train(
            SteepestRequest("0.8*x+0.8*y", x0=0, y0=0, tolerance=1, max_iterations=1)
        )
        self.assertEqual(r.status, "iteration_limit")  # Both partials < 1, norm > 1.
        for function in ("x^2-y^2", "-x^2-y^2", "7"):
            with self.subTest(function=function):
                r = train(SteepestRequest(function, x0=0, y0=0))
                self.assertEqual(r.status, "gradient_tolerance")
                self.assertEqual(r.final.iteration, 0)
        r = train(SteepestRequest("x^2+y^2", x0=1, y0=2, alpha=0.5, max_iterations=1))
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertEqual((r.final.x, r.final.y), (0, 0))

    def test_automatic_partial_derivatives_against_analytic_values(self):
        for expression, x, y, value, gx, gy in (
            ("sin(x)+cos(y)", 0, 0, 1, 1, 0),
            (
                "exp(x*y)+ln(x)+sqrt(y)",
                1,
                4,
                math.exp(4) + 2,
                4 * math.exp(4) + 1,
                math.exp(4) + 0.25,
            ),
            ("sqrt(x^2+y^2)", 3, 4, 5, 0.6, 0.8),
            ("(x^2+1)/(y+2)", 1, 1, 2 / 3, 2 / 3, -2 / 9),
            ("x^y", 2, 3, 8, 12, 8 * math.log(2)),
            ("y^3", 1, -2, -8, 0, 12),
            ("(x-y)^2", 2, 1, 1, 2, -2),
            ("pi+e+x^0+y^0", 0, 0, math.pi + math.e + 2, 0, 0),
            ("tan(x)+log(y)", 0, 1, 0, 1, 1),
        ):
            with self.subTest(expression=expression):
                p = train(
                    SteepestRequest(expression, x0=x, y0=y, max_iterations=1)
                ).points[0]
                for actual, expected in (
                    (p.value, value),
                    (p.gradient_x, gx),
                    (p.gradient_y, gy),
                ):
                    self.assertAlmostEqual(actual, expected)

    def test_rosenbrock_first_step_matches_independent_fixture(self):
        r = train(
            SteepestRequest(
                "(1-x)^2+100*(y-x^2)^2", x0=-1.2, y0=1, alpha=0.001, max_iterations=1
            )
        )
        self.assertAlmostEqual(r.points[0].value, 24.2)
        self.assertAlmostEqual(r.points[0].gradient_x, -215.6)
        self.assertAlmostEqual(r.points[0].gradient_y, -88)
        self.assertAlmostEqual(r.final.x, -0.9844)
        self.assertAlmostEqual(r.final.y, 1.088)

    def test_fixed_rate_can_oscillate_or_increase_and_gradient_is_not_normalized(self):
        r = train(SteepestRequest("x^2+y^2", x0=1, y0=2, alpha=1, max_iterations=3))
        self.assertEqual(
            [(p.x, p.y) for p in r.points], [(1, 2), (-1, -2), (1, 2), (-1, -2)]
        )
        self.assertEqual(r.status, "iteration_limit")
        r = train(SteepestRequest("x^2+y^2", x0=1, y0=2, alpha=2, max_iterations=1))
        self.assertEqual((r.final.x, r.final.y), (-3, -6))
        self.assertGreater(r.final.value, r.points[0].value)
        self.assertEqual(r.status, "iteration_limit")

    def test_stagnation_requires_both_coordinates_and_tiny_steps_are_not_convergence(
        self,
    ):
        r = train(SteepestRequest("x+y", x0=1e20, y0=1e20, alpha=1e-20))
        self.assertEqual(r.status, "stagnation")
        self.assertEqual(r.final.iteration, 0)
        r = train(
            SteepestRequest("x+y^2", x0=1e20, y0=1, alpha=1e-10, max_iterations=2)
        )
        self.assertEqual(r.status, "iteration_limit")
        self.assertEqual(r.final.x, r.request.x0)
        self.assertLess(r.final.y, r.request.y0)
        self.assertEqual(r.final.iteration, 2)

    def test_domain_or_numerical_failure_rejects_entire_pair(self):
        r = train(SteepestRequest("x+ln(y)", x0=1, y0=1, alpha=2))
        self.assertEqual(r.status, "numerical_limit")
        self.assertEqual([(p.x, p.y) for p in r.points], [(1, 1)])
        self.assertTrue(any(v is None for row in r.grid for v in row))
        r = train(SteepestRequest("x*y", x0=1e49, y0=1e49, alpha=1e50))
        self.assertEqual(r.status, "numerical_limit")
        self.assertEqual(r.final.iteration, 0)
        with self.assertRaisesRegex(RuntimeError, "starting"):
            train(SteepestRequest("ln(y)", y0=-1))

    def test_grid_sampling_and_view_bounds_do_not_change_the_iterations(self):
        request = SteepestRequest("x^2+x*y+y^2", x0=2, y0=1, max_iterations=3)
        r = train(request)
        other = train(
            replace(request, plot_min=10, plot_max=20, plot_y_min=10, plot_y_max=20)
        )
        self.assertEqual(other.points, r.points)
        self.assertEqual(len(r.grid), GRID_SIZE)
        self.assertTrue(all(len(row) == GRID_SIZE for row in r.grid))
        x = request.plot_min + (request.plot_max - request.plot_min) * 0.5 / GRID_SIZE
        y = (
            request.plot_y_min
            + (request.plot_y_max - request.plot_y_min) * 0.5 / GRID_SIZE
        )
        self.assertAlmostEqual(r.grid[0][0], x * x + x * y + y * y)

    def test_validation_safe_two_variable_syntax_and_unchanged_1D_syntax(self):
        for expression in (
            "__import__('os')",
            "x.real+y",
            "x[0]+y",
            "z+y",
            "sin(x,y)",
            "True+y",
        ):
            with self.subTest(expression=expression), self.assertRaises(InputError):
                compile_expression(expression, dimensions=2)
        for request in (
            GradientRequest("y^2"),
            NewtonRequest("x^2", first_derivative="y"),
        ):
            with self.subTest(request=request), self.assertRaises(InputError):
                request.validate()
        for changes in (
            {"alpha": 0},
            {"alpha": math.inf},
            {"y0": math.nan},
            {"y0": 1e51},
            {"tolerance": 0},
            {"max_iterations": 0},
            {"max_iterations": 100001},
            {"max_iterations": 1.5},
            {"plot_y_min": 4},
            {"plot_y_max": math.inf},
        ):
            with self.subTest(changes=changes), self.assertRaises(InputError):
                replace(SteepestRequest(), **changes).validate()

    def test_csv_captures_both_coordinates_and_snapshot_at_full_precision(self):
        r = train(SteepestRequest("x^2+x*y+y^2", x0=2, y0=1, max_iterations=2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steepest.csv"
            export_trace(path, r)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["algorithm"], "STEEPEST2D")
        self.assertEqual(float(rows[0]["initial_y"]), 1)
        self.assertEqual(float(rows[0]["partial_x"]), 5)
        self.assertEqual(float(rows[0]["partial_y"]), 4)
        self.assertEqual(float(rows[1]["y_k"]), r.points[1].y)
        self.assertEqual(float(rows[1]["gradient_norm"]), r.points[1].gradient_norm)
        for key in ("delta_x", "delta_y", "x_next", "y_next"):
            self.assertEqual(rows[-1][key], "")
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(Cancelled):
            train(SteepestRequest(), cancel=cancelled)


if __name__ == "__main__":
    unittest.main()
