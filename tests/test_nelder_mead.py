"""Hand-worked branches for the slide's inside-contraction-only Nelder–Mead."""

import csv
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import threading
import unittest

from every_example_gui.backend import Cancelled, InputError, train
from every_example_gui.algorithms.nelder_mead import NelderMeadRequest
from every_example_gui.algorithms.optimization import export_trace


class NelderMeadTests(unittest.TestCase):
    def run_one(self, expression, simplex):
        return train(NelderMeadRequest(expression, simplex, max_iterations=1))

    def test_five_hand_worked_branch_paths_and_next_simplexes(self):
        fixtures = (
            (
                "reflect",
                "x^2+y^2",
                ((0.0, 0.0), (1.0, 0.0), (1.0, 0.5)),
                (0.5, 0),
                (0, -0.5),
                ((0, 0), (0, -0.5), (1, 0)),
            ),
            (
                "expand",
                "x^2+y^2",
                ((1.0, 1.0), (2.0, 0.0), (2.0, 1.0)),
                (1.5, 0.5),
                (1, 0),
                ((0.5, -0.5), (1, 1), (2, 0)),
            ),
            (
                "reflect_after_expand",
                "x^2+y^2",
                ((0.0, 1.0), (1.0, 1.0), (1.0, 2.0)),
                (0.5, 1),
                (0, 0),
                ((0, 0), (0, 1), (1, 1)),
            ),
            (
                "contract",
                "x^2+y^2",
                ((0.0, 0.0), (1.0, 0.0), (0.0, 2.0)),
                (0.5, 0),
                (1, -2),
                ((0, 0), (1, 0), (0.25, 1)),
            ),
            (
                "shrink",
                "x^2+y^2+8*y*(2-y)*(x-1)^2",
                ((0.0, 0.0), (1.0, 0.0), (0.0, 2.0)),
                (0.5, 0),
                (1, -2),
                ((0, 0), (0.5, 0), (0, 1)),
            ),
        )
        for action, expression, simplex, centroid, reflected, after in fixtures:
            with self.subTest(action=action):
                r = self.run_one(expression, simplex)
                self.assertEqual(len(r.steps), 1)
                s = r.steps[0]
                self.assertEqual(s.action, action)
                self.assertEqual(s.centroid, centroid)
                self.assertEqual((s.reflection.x, s.reflection.y), reflected)
                self.assertEqual(tuple((v.x, v.y) for v in r.final.simplex), after)
        r = self.run_one(fixtures[-1][1], fixtures[-1][2])
        self.assertEqual(
            (r.steps[0].shrunk_b.value, r.steps[0].shrunk_c.value), (0.25, 9)
        )
        self.assertGreater(
            r.final.spread, r.points[0].spread
        )  # Shrink has no extra acceptance test.

    def test_reflection_lower_equality_is_accepted_but_upper_equality_contracts(self):
        r = self.run_one("x^2+y^2", ((1.0, 1.0), (2.0, 0.0), (2.0, 2.0)))
        self.assertEqual(r.steps[0].reflection.value, r.points[0].simplex[0].value)
        self.assertEqual(r.steps[0].action, "reflect")
        self.assertIsNone(r.steps[0].expansion)
        r = self.run_one("x^2+y^2", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)))
        self.assertEqual(r.steps[0].reflection.value, r.points[0].simplex[1].value)
        self.assertEqual(r.steps[0].action, "contract")
        self.assertEqual(
            (r.steps[0].contraction.x, r.steps[0].contraction.y), (0.75, 0.5)
        )

    def test_expansion_equality_selects_reflection(self):
        r = self.run_one("(x+.25)^2+(y+.5)^2", ((0.0, 1.0), (1.0, 1.0), (1.0, 2.0)))
        s = r.steps[0]
        self.assertEqual(s.expansion.value, s.reflection.value)
        self.assertEqual(s.action, "reflect_after_expand")
        self.assertEqual((r.final.x, r.final.y), (0, 0))

    def test_inside_contraction_uses_old_C_even_when_reflection_is_better_than_C(self):
        r = self.run_one("x^2+y^2", ((0.0, 0.0), (1.0, 0.0), (0.75, 1.0)))
        s = r.steps[0]
        self.assertGreater(s.reflection.value, r.points[0].simplex[1].value)
        self.assertLess(s.reflection.value, r.points[0].simplex[2].value)
        self.assertEqual((s.contraction.x, s.contraction.y), (0.625, 0.5))
        self.assertEqual(s.contraction.value, 0.640625)
        self.assertEqual(s.action, "contract")

    def test_contraction_equality_shrinks_both_vertices_toward_same_old_A(self):
        r = self.run_one(
            "3*x^2+y^2+5*y*(2-y)*(x-1)^2", ((0.0, 0.0), (1.0, 0.0), (0.0, 2.0))
        )
        s = r.steps[0]
        self.assertEqual(s.contraction.value, r.points[0].simplex[2].value)
        self.assertEqual(s.action, "shrink")
        self.assertEqual((s.shrunk_b.x, s.shrunk_b.y, s.shrunk_b.value), (0.5, 0, 0.75))
        self.assertEqual((s.shrunk_c.x, s.shrunk_c.y, s.shrunk_c.value), (0, 1, 6))
        self.assertEqual(r.final.simplex[0], r.points[0].simplex[0])

    def test_sorting_excludes_worst_from_centroid_and_breaks_ties_stably(self):
        simplex = ((0.0, 2.0), (0.0, 0.0), (1.0, 0.0))
        r = self.run_one("x^2+y^2", simplex)
        self.assertEqual(
            tuple((v.x, v.y) for v in r.points[0].simplex), ((0, 0), (1, 0), (0, 2))
        )
        self.assertEqual(r.steps[0].centroid, (0.5, 0))
        self.assertEqual(r.request.simplex, simplex)
        r = self.run_one("x^2+y^2", ((-1.0, 0.0), (1.0, 0.0), (0.0, 2.0)))
        self.assertEqual(
            tuple((v.x, v.y) for v in r.points[0].simplex[:2]), ((-1, 0), (1, 0))
        )

    def test_function_spread_tolerance_is_inclusive_and_flat_simplex_can_stop_initially(
        self,
    ):
        r = train(
            NelderMeadRequest(
                "x^2+y^2", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), tolerance=1
            )
        )
        self.assertEqual(r.status, "function_tolerance")
        self.assertEqual(r.final.iteration, 0)
        self.assertEqual(r.final.spread, 1)
        r = train(NelderMeadRequest("7", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))))
        self.assertEqual(r.status, "function_tolerance")
        self.assertEqual(r.final.iteration, 0)
        self.assertGreater(r.final.diameter, 1)

    def test_unchanged_best_value_is_not_a_stopping_rule(self):
        r = train(
            NelderMeadRequest(
                "x^2+y^2",
                ((0.0, 0.0), (1.0, 0.0), (0.0, 2.0)),
                tolerance=1e-20,
                max_iterations=2,
            )
        )
        self.assertEqual([p.value for p in r.points], [0, 0, 0])
        self.assertEqual(r.status, "iteration_limit")
        self.assertEqual(r.final.iteration, 2)
        self.assertGreater(r.final.spread, r.request.tolerance)

    def test_known_minimum_and_sorted_history(self):
        r = train(NelderMeadRequest())
        self.assertEqual(r.status, "function_tolerance")
        self.assertLess(abs(r.final.x - 2), 0.002)
        self.assertLess(abs(r.final.y + 1), 0.002)
        self.assertLess(r.final.value, 1e-6)
        self.assertLessEqual(r.final.spread, r.request.tolerance)
        self.assertGreater(r.points[-2].spread, r.request.tolerance)
        self.assertTrue(
            all(
                p.simplex[0].value <= p.simplex[1].value <= p.simplex[2].value
                for p in r.points
            )
        )
        self.assertTrue(all(b.value <= a.value for a, b in zip(r.points, r.points[1:])))

    def test_nonlinear_rosenbrock_and_nondifferentiable_origin(self):
        r = train(
            NelderMeadRequest(
                "(1-x)^2+100*(y-x^2)^2",
                ((-1.2, 1.0), (-0.2, 1.0), (-1.2, 2.0)),
                max_iterations=500,
            )
        )
        self.assertEqual(r.status, "function_tolerance")
        self.assertLess(r.final.value, 1e-5)
        self.assertLess(abs(r.final.x - 1), 0.01)
        self.assertLess(abs(r.final.y - 1), 0.01)
        r = self.run_one("sqrt(x^2+y^2)", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
        self.assertEqual(
            r.final.value, 0
        )  # No gradient or Hessian is evaluated at the origin.
        self.assertEqual(r.final.iteration, 1)

    def test_required_trial_domain_failure_rejects_entire_iteration(self):
        fixtures = (
            (
                "reflection",
                "x^2+y^2+ln(y)",
                ((0.0, 1.0), (1.0, 1.0), (0.0, 3.0)),
                (1, -1),
            ),
            (
                "expansion",
                "x^2+sqrt(y)",
                ((0.0, 3.0), (1.0, 3.0), (1.0, 5.0)),
                (-0.5, -1),
            ),
            (
                "contraction",
                "x^2+y^2+sqrt((y-1)^2-0.5)",
                ((0.0, 0.0), (1.0, 0.0), (0.0, 2.0)),
                (0.25, 1),
            ),
            (
                "shrink_B",
                "3*x^2+y^2+6*y*(2-y)*(x-1)^2+sqrt((x-0.5)^2-0.01)",
                ((0.0, 0.0), (1.0, 0.0), (0.0, 2.0)),
                (0.5, 0),
            ),
            (
                "shrink_C",
                "3*x^2+y^2+7*y*(2-y)*(x-1)^2+sqrt(x^2+(y-1)^2-0.01)",
                ((0.0, 0.0), (1.0, 0.0), (0.0, 2.0)),
                (0, 1),
            ),
        )
        for stage, expression, simplex, coordinates in fixtures:
            with self.subTest(stage=stage):
                r = self.run_one(expression, simplex)
                self.assertEqual(r.status, "numerical_limit")
                self.assertEqual(r.failure.stage, stage)
                self.assertEqual((r.failure.x, r.failure.y), coordinates)
                self.assertEqual(r.final.iteration, 0)
                self.assertEqual(r.steps, ())
                self.assertEqual(tuple((v.x, v.y) for v in r.final.simplex), simplex)

    def test_scaled_initial_area_accepts_tiny_triangle_and_rejects_collinearity(self):
        r = train(
            NelderMeadRequest(
                "x+y",
                ((0.0, 0.0), (1e-200, 0.0), (0.0, 1e-200)),
                tolerance=1e-250,
                max_iterations=1,
            )
        )
        self.assertEqual(r.final.iteration, 1)
        for simplex in (
            ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)),
            ((0.0, 0.0), (0.0, 0.0), (1.0, 1.0)),
        ):
            with (
                self.subTest(simplex=simplex),
                self.assertRaisesRegex(RuntimeError, "non-collinear"),
            ):
                train(NelderMeadRequest(simplex=simplex))

    def test_stagnation_is_reported_when_shrink_rounds_to_same_triangle(self):
        a = math.nextafter(1e20, math.inf)
        b = math.nextafter(a, math.inf)
        unit = b - a
        expression = f"((x-{a!r})/{unit!r})^2+2*((y-{a!r})/{unit!r})^2"
        r = train(NelderMeadRequest(expression, ((a, a), (b, a), (a, b))))
        self.assertEqual(r.status, "stagnation")
        self.assertEqual(r.final.iteration, 0)
        self.assertGreater(r.final.spread, r.request.tolerance)

    def test_unbounded_expansion_hits_numerical_guard_and_retains_finite_history(self):
        r = train(NelderMeadRequest("x+y", max_iterations=1000))
        self.assertEqual(r.status, "numerical_limit")
        self.assertGreater(r.final.iteration, 0)
        self.assertLess(r.final.iteration, 1000)
        self.assertTrue(
            all(
                math.isfinite(v.value) and abs(v.value) <= 1e100
                for p in r.points
                for v in p.simplex
            )
        )

    def test_plot_bounds_do_not_change_result_and_nonsmooth_grid_center_is_valid(self):
        request = NelderMeadRequest(max_iterations=3)
        r = train(request)
        other = train(
            replace(request, plot_min=10, plot_max=20, plot_y_min=10, plot_y_max=20)
        )
        self.assertEqual(other.points, r.points)
        self.assertEqual(other.steps, r.steps)
        r = train(
            NelderMeadRequest(
                "sqrt(x^2+y^2)",
                ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
                max_iterations=1,
                plot_min=-0.5,
                plot_max=0.5,
                plot_y_min=-0.5,
                plot_y_max=0.5,
            )
        )
        self.assertEqual(r.grid[24][24], 0)

    def test_validation_initial_domain_and_cancellation(self):
        for changes in (
            {"simplex": ((0.0, 0.0), (1.0, 1.0))},
            {"simplex": ((0.0, 0.0), (1.0, 1.0), (math.nan, 2.0))},
            {"expression": "x.__class__+y"},
            {"tolerance": 0},
            {"max_iterations": 0},
            {"max_iterations": 100001},
            {"max_iterations": 1.5},
            {"plot_y_min": 5},
            {"plot_max": math.inf},
        ):
            with self.subTest(changes=changes), self.assertRaises(InputError):
                replace(NelderMeadRequest(), **changes).validate()
        with self.assertRaisesRegex(RuntimeError, "initial vertices"):
            train(NelderMeadRequest("ln(y)", ((0.0, 1.0), (1.0, 1.0), (0.0, -1.0))))
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(Cancelled):
            train(NelderMeadRequest(), cancel=cancelled)

    def test_csv_preserves_initial_vertices_sorted_simplexes_and_all_trial_points(self):
        request = NelderMeadRequest(
            "3*x^2+y^2+5*y*(2-y)*(x-1)^2",
            ((0.0, 2.0), (0.0, 0.0), (1.0, 0.0)),
            max_iterations=1,
        )
        r = train(request)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nelder.csv"
            export_trace(path, r)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertTrue(
            all(None not in row and None not in row.values() for row in rows)
        )
        self.assertEqual(rows[0]["algorithm"], "NELDER_MEAD2D")
        self.assertEqual(float(rows[0]["initial_A_y"]), 2)
        self.assertEqual(float(rows[0]["A_y"]), 0)
        self.assertEqual(rows[0]["action"], "shrink")
        for key, value in (
            ("reflection_alpha", 1),
            ("expansion_gamma", 2),
            ("contraction_rho", 0.5),
            ("shrink_sigma", 0.5),
            ("centroid_x", 0.5),
            ("reflection_f", 7),
            ("contraction_f", 4),
            ("shrunk_B_f", 0.75),
            ("shrunk_C_f", 6),
            ("next_C_f", 6),
        ):
            self.assertEqual(float(rows[0][key]), value)
        self.assertEqual(rows[0]["expansion_f"], "")
        self.assertEqual(float(rows[0]["simplex_diameter"]), r.points[0].diameter)
        for key in ("action", "centroid_x", "reflection_f", "shrunk_B_f", "next_A_x"):
            self.assertEqual(rows[-1][key], "")


if __name__ == "__main__":
    unittest.main()
