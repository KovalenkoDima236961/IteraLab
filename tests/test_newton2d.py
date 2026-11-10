"""Independent full-Hessian Newton fixtures and numerical/domain boundaries."""

import csv
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import threading
import unittest

from every_example_gui.backend import Cancelled, InputError, train
from every_example_gui.algorithms.newton2d import Newton2DRequest
from every_example_gui.algorithms.optimization import export_trace
from every_example_gui.algorithms.steepest import SteepestRequest, GRID_SIZE


class Newton2DTests(unittest.TestCase):
    def test_coupled_bowl_uses_full_inverse_and_reaches_known_minimum_in_one_step(self):
        r = train(Newton2DRequest())
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertEqual([(p.x, p.y) for p in r.points], [(6, 3), (2, -1)])
        p = r.points[0]
        self.assertEqual((p.gradient_x, p.gradient_y), (12, 20))
        self.assertEqual(p.hessian, ((2, 1), (1, 4)))
        self.assertEqual((p.delta_x, p.delta_y), (-4, -4))
        self.assertEqual((r.final.value, r.final.gradient_norm), (0, 0))
        self.assertIsNone(r.final.delta_x)
        self.assertIsNone(r.final.delta_y)

    def test_nonlinear_hand_worked_steps_recompute_gradient_and_full_hessian(self):
        r = train(Newton2DRequest("x^4+y^4+x*y", x0=1, y0=1, max_iterations=2))
        self.assertEqual(r.status, "iteration_limit")
        for p, coordinate in zip(r.points, (1, 8 / 13, 4096 / 12181)):
            self.assertAlmostEqual(p.x, coordinate)
            self.assertAlmostEqual(p.y, coordinate)
            self.assertAlmostEqual(p.gradient_x, 4 * coordinate**3 + coordinate)
            self.assertAlmostEqual(p.hessian_xx, 12 * coordinate**2)
            self.assertEqual(p.hessian_xy, 1)
            self.assertAlmostEqual(p.hessian_yy, 12 * coordinate**2)

    def test_analytic_gradient_and_hessian_values_including_mixed_terms(self):
        e2, log2 = math.exp(2), math.log(2)
        sec2 = 1 / math.cos(0.3) ** 2
        for expr, x, y, gx, gy, hxx, hxy, hyy in (
            ("x^2*y^3", 2, 3, 108, 108, 54, 108, 72),
            ("exp(x*y)", 1, 2, 2 * e2, e2, 4 * e2, 3 * e2, e2),
            ("ln(x)+log(y)", 2, 4, 0.5, 0.25, -0.25, 0, -0.0625),
            ("sqrt(x^2+y^2)", 3, 4, 0.6, 0.8, 16 / 125, -12 / 125, 9 / 125),
            ("sin(x*y)+cos(x)", 0, 2, 2, 0, -1, 1, 0),
            (
                "tan(x+y)",
                0.1,
                0.2,
                sec2,
                sec2,
                2 * math.tan(0.3) * sec2,
                2 * math.tan(0.3) * sec2,
                2 * math.tan(0.3) * sec2,
            ),
            ("x/y", 2, 4, 0.25, -0.125, 0, -0.0625, 0.0625),
            ("x^y", 2, 3, 12, 8 * log2, 12, 4 * (1 + 3 * log2), 8 * log2**2),
            ("(-x)^3+y^2", 2, 1, -12, 2, -12, 0, 2),
            ("x^0+y^0+pi+e", 0, 0, 0, 0, 0, 0, 0),
            ("x^1+y^2", 0, 0, 1, 0, 0, 0, 2),
        ):
            with self.subTest(expression=expr):
                p = train(Newton2DRequest(expr, x0=x, y0=y, max_iterations=1)).points[0]
                for actual, expected in (
                    (p.gradient_x, gx),
                    (p.gradient_y, gy),
                    (p.hessian_xx, hxx),
                    (p.hessian_xy, hxy),
                    (p.hessian_yy, hyy),
                ):
                    self.assertAlmostEqual(actual, expected)

    def test_rosenbrock_gradient_hessian_and_first_step(self):
        r = train(
            Newton2DRequest("(1-x)^2+100*(y-x^2)^2", x0=-1.2, y0=1, max_iterations=1)
        )
        p = r.points[0]
        self.assertAlmostEqual(p.gradient_x, -215.6)
        self.assertAlmostEqual(p.gradient_y, -88)
        self.assertAlmostEqual(p.hessian_xx, 1330)
        self.assertAlmostEqual(p.hessian_xy, 480)
        self.assertAlmostEqual(p.hessian_yy, 200)
        self.assertAlmostEqual(p.delta_x, 11 / 445)
        self.assertAlmostEqual(p.delta_y, 847 / 2225)
        self.assertAlmostEqual(r.final.x, -1.2 + 11 / 445)
        self.assertAlmostEqual(r.final.y, 1 + 847 / 2225)

    def test_mixed_derivative_is_not_lost_beside_a_large_diagonal(self):
        r = train(Newton2DRequest("5e49*x^2+x*y", x0=1, y0=0))
        self.assertEqual(r.points[0].hessian_xy, 1)
        self.assertEqual(r.points[0].hessian_yy, 0)
        self.assertLess(r.points[0].determinant_scaled, 0)
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertAlmostEqual(r.final.x, 0)
        self.assertAlmostEqual(r.final.y, 0)

    def test_scaled_system_handles_underflow_in_unscaled_determinant(self):
        r = train(Newton2DRequest("1e-200*(x^2+y^2)", x0=1, y0=2, tolerance=1e-250))
        p = r.points[0]
        self.assertEqual(p.hessian_xx * p.hessian_yy, 0)  # Raw determinant underflows.
        self.assertEqual(p.determinant_scaled, 1)
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertEqual((r.final.x, r.final.y), (0, 0))

    def test_singular_hessian_at_start_and_after_a_committed_update(self):
        for expr in ("x^2+y", "(x+y)^2"):
            with self.subTest(expression=expr):
                r = train(Newton2DRequest(expr, x0=1, y0=1))
                self.assertEqual(r.status, "singular_hessian")
                self.assertEqual(r.final.iteration, 0)
                self.assertIsNone(r.final.delta_x)
        r = train(Newton2DRequest("x^3/3+x+y^2", x0=1, y0=1))
        self.assertEqual(r.status, "singular_hessian")
        self.assertEqual([(p.x, p.y) for p in r.points], [(1, 1), (0, 0)])
        self.assertEqual(r.final.gradient_norm, 1)

    def test_gradient_norm_tolerance_at_initial_point_and_budget_endpoint(self):
        r = train(Newton2DRequest("3*x+4*y", x0=0, y0=0, tolerance=5))
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertEqual(r.final.iteration, 0)  # No inverse is needed, even with H=0.
        self.assertEqual(r.final.gradient_norm, 5)
        r = train(Newton2DRequest("0.8*x+0.8*y", x0=0, y0=0, tolerance=1))
        self.assertEqual(
            r.status, "singular_hessian"
        )  # Both partials < 1 but norm > 1.
        r = train(Newton2DRequest("x^2+y^2", x0=1, y0=2, max_iterations=1))
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertEqual(r.final.iteration, 1)

    def test_indefinite_and_negative_hessians_use_literal_newton_step(self):
        for expr in ("x^2-y^2", "-x^2-y^2"):
            with self.subTest(expression=expr):
                r = train(Newton2DRequest(expr, x0=1, y0=2))
                self.assertEqual(r.status, "gradient_tolerance")
                self.assertEqual((r.final.x, r.final.y), (0, 0))
                self.assertEqual(r.final.iteration, 1)

    def test_cycles_and_increases_are_recorded_without_line_search(self):
        r = train(Newton2DRequest("x^4/4-x^2+2*x+y^2", x0=0, y0=1, max_iterations=4))
        self.assertEqual(r.status, "iteration_limit")
        self.assertEqual(
            [(p.x, p.y) for p in r.points], [(0, 1), (1, 0), (0, 0), (1, 0), (0, 0)]
        )
        self.assertGreater(r.points[1].value, r.points[0].value)

    def test_domain_failure_rejects_entire_pair_and_hessian_is_required(self):
        r = train(Newton2DRequest("x^2-ln(y)+10*y", x0=1, y0=1))
        self.assertEqual(r.status, "numerical_limit")
        self.assertEqual([(p.x, p.y) for p in r.points], [(1, 1)])
        self.assertIsNone(r.final.delta_x)
        with self.assertRaisesRegex(RuntimeError, "starting"):
            train(Newton2DRequest("x^1.5+y^2", x0=0, y0=1))
        r = train(SteepestRequest("x^1.5+y^2", x0=0, y0=1, max_iterations=1))
        self.assertAlmostEqual(
            r.final.y, 0.8
        )  # The other method still needs only first derivatives.

    def test_both_coordinates_must_stagnate_and_small_step_is_not_convergence(self):
        r = train(Newton2DRequest("x+1e10*(x-1e20)^2+y^2", x0=1e20, y0=1))
        self.assertEqual(r.status, "stagnation")
        self.assertEqual([(p.x, p.y) for p in r.points], [(1e20, 1), (1e20, 0)])
        self.assertEqual(r.final.gradient_norm, 1)

    def test_view_bounds_and_grid_are_independent_of_update(self):
        request = Newton2DRequest("x^2+x*y+2*y^2", x0=2, y0=1)
        r = train(request)
        other = train(
            replace(request, plot_min=10, plot_max=20, plot_y_min=10, plot_y_max=20)
        )
        self.assertEqual(other.points, r.points)
        self.assertEqual((len(r.grid), len(r.grid[0])), (GRID_SIZE, GRID_SIZE))
        gradient_grid = train(
            SteepestRequest(request.expression, x0=2, y0=1, max_iterations=1)
        ).grid
        self.assertEqual(r.grid, gradient_grid)

    def test_validation_cancellation_and_full_precision_exports(self):
        for changes in (
            {"expression": "x.__class__+y"},
            {"y0": math.nan},
            {"tolerance": 0},
            {"max_iterations": 0},
            {"max_iterations": 100001},
            {"max_iterations": 1.5},
            {"plot_y_min": 4},
            {"plot_max": math.inf},
        ):
            with self.subTest(changes=changes), self.assertRaises(InputError):
                replace(Newton2DRequest(), **changes).validate()
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(Cancelled):
            train(Newton2DRequest(), cancel=cancelled)
        r = train(Newton2DRequest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "newton2d.csv"
            export_trace(path, r)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["algorithm"], "NEWTON2D")
        self.assertEqual(rows[0]["function"], r.request.expression)
        self.assertNotIn("alpha", rows[0])
        for key, value in (
            ("hessian_xx", 2),
            ("hessian_xy", 1),
            ("hessian_yx", 1),
            ("hessian_yy", 4),
            ("delta_x", -4),
            ("delta_y", -4),
        ):
            self.assertEqual(float(rows[0][key]), value)
        self.assertEqual(
            float(rows[0]["determinant_scaled"]), r.points[0].determinant_scaled
        )
        for key in ("delta_x", "delta_y", "x_next", "y_next"):
            self.assertEqual(rows[-1][key], "")


if __name__ == "__main__":
    unittest.main()
