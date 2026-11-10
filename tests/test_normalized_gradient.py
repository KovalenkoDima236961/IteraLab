"""Independent normalized directions and stationary line-search fixtures."""

import csv
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import threading
import unittest

from every_example_gui.backend import Cancelled, InputError, train
from every_example_gui.algorithms.normalized_gradient import (
    NormalizedRequest,
    export_line_searches,
)
from every_example_gui.algorithms.optimization import export_trace


class NormalizedGradientTests(unittest.TestCase):
    def test_hand_worked_three_four_five_direction_line_equation_and_update(self):
        r = train(NormalizedRequest())
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertEqual(
            [(p.x, p.y, p.value) for p in r.points], [(3, 4, 25), (0, 0, 0)]
        )
        s = r.searches[0]
        self.assertEqual((s.direction_x, s.direction_y), (-0.6, -0.8))
        self.assertEqual(s.accepted.gamma, 5)
        self.assertEqual([p.gamma for p in s.samples], [0, 1, 2, 4, 8, 6, 5])
        for p, expected in zip(s.samples, (-10, -8, -6, -2, 6, 2, 0)):
            self.assertAlmostEqual(p.derivative, expected)
        self.assertEqual(s.samples[0].phase, "start")
        self.assertEqual(s.accepted.derivative, 0)

    def test_two_hand_worked_elliptic_steps_use_variable_gamma(self):
        r = train(NormalizedRequest("(x-2)^2+2*(y+1)^2", x0=6, y0=3, max_iterations=2))
        self.assertEqual(r.status, "iteration_limit")
        for p, (x, y) in zip(
            r.points, ((6, 3), (34 / 9, -13 / 9), (62 / 27, -19 / 27))
        ):
            self.assertAlmostEqual(p.x, x, places=8)
            self.assertAlmostEqual(p.y, y, places=8)
        self.assertAlmostEqual(
            r.searches[0].accepted.gamma, 20 * math.sqrt(5) / 9, places=8
        )
        self.assertAlmostEqual(
            r.searches[1].accepted.gamma, 20 * math.sqrt(5) / 27, places=8
        )
        self.assertAlmostEqual(r.points[1].value, 32 / 9, places=8)

    def test_coupled_quadratic_line_minimizer_uses_both_old_gradient_components(self):
        r = train(NormalizedRequest("x^2+x*y+y^2", x0=2, y0=1, max_iterations=1))
        s = r.searches[0]
        self.assertAlmostEqual(s.direction_x, -5 / math.sqrt(41))
        self.assertAlmostEqual(s.direction_y, -4 / math.sqrt(41))
        self.assertAlmostEqual(s.accepted.gamma, 41 * math.sqrt(41) / 122, places=8)
        self.assertAlmostEqual(r.final.x, 39 / 122, places=8)
        self.assertAlmostEqual(r.final.y, -21 / 61, places=8)

    def test_convergence_unit_directions_and_perpendicular_new_gradient(self):
        r = train(NormalizedRequest("(x-2)^2+2*(y+1)^2", x0=6, y0=3))
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertAlmostEqual(r.final.x, 2, places=6)
        self.assertAlmostEqual(r.final.y, -1, places=6)
        for i, s in enumerate(r.searches):
            a, b = r.points[i : i + 2]
            self.assertAlmostEqual(math.hypot(s.direction_x, s.direction_y), 1)
            self.assertLessEqual(b.value, a.value)
            self.assertLessEqual(
                abs(s.accepted.derivative / a.gradient_norm), r.request.line_tolerance
            )
            self.assertAlmostEqual(
                b.gradient_x * s.direction_x + b.gradient_y * s.direction_y,
                s.accepted.derivative,
            )
            self.assertAlmostEqual(math.hypot(b.x - a.x, b.y - a.y), s.accepted.gamma)

    def test_relative_line_residual_keeps_step_independent_of_objective_scaling(self):
        results = [
            train(
                NormalizedRequest(
                    f"{scale}*(x^2+2*y^2)",
                    x0=1.2,
                    y0=-0.7,
                    tolerance=1e-25,
                    max_iterations=1,
                )
            )
            for scale in ("1e-12", "1", "1e12")
        ]
        for r in results[1:]:
            self.assertAlmostEqual(r.final.x, results[0].final.x, places=8)
            self.assertAlmostEqual(r.final.y, results[0].final.y, places=8)
            self.assertAlmostEqual(
                r.searches[0].accepted.gamma,
                results[0].searches[0].accepted.gamma,
                places=8,
            )

    def test_gradient_test_precedes_normalization_and_is_inclusive(self):
        for request in (
            NormalizedRequest("7"),
            NormalizedRequest("x^2-y^2", x0=0, y0=0),
            NormalizedRequest("3*x+4*y", x0=0, y0=0, tolerance=5),
        ):
            with self.subTest(request=request):
                r = train(request)
                self.assertEqual(r.status, "gradient_tolerance")
                self.assertEqual(r.final.iteration, 0)
                self.assertEqual(r.searches, ())
        r = train(NormalizedRequest(max_iterations=1))
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertEqual(r.final.iteration, 1)

    def test_only_first_derivatives_are_required(self):
        r = train(NormalizedRequest("x^1.5+(y-2)^2", x0=0, y0=4))
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertEqual((r.final.x, r.final.y), (0, 2))
        self.assertEqual(r.searches[0].accepted.gamma, 2)

    def test_invalid_trial_is_recorded_and_domain_refinement_finds_interior_root(self):
        r = train(NormalizedRequest("-ln(x)+x^2+y^2", x0=2, y0=0))
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertAlmostEqual(r.final.x, 1 / math.sqrt(2), places=8)
        s = r.searches[0]
        self.assertTrue(any(not p.valid for p in s.samples))
        self.assertTrue(any(p.phase == "domain_refine" for p in s.samples))
        invalid = next(p for p in s.samples if not p.valid)
        self.assertEqual(invalid.gamma, 2)
        self.assertEqual(
            (invalid.value, invalid.gradient_x, invalid.gradient_y, invalid.derivative),
            (None, None, None, None),
        )
        self.assertTrue(all(p.x > 0 for p in r.points))

    def test_no_stationary_point_before_domain_boundary_retains_old_pair(self):
        r = train(NormalizedRequest("ln(x)+y^2", x0=1, y0=0))
        self.assertEqual(r.status, "line_search_failed")
        self.assertEqual([(p.x, p.y) for p in r.points], [(1, 0)])
        self.assertEqual(r.searches[0].status, "not_bracketed")
        self.assertIsNone(r.searches[0].accepted)
        self.assertTrue(all(p.derivative < 0 for p in r.searches[0].samples if p.valid))

    def test_exhausted_search_does_not_commit_a_merely_decreasing_probe(self):
        r = train(NormalizedRequest(max_line_evaluations=1))
        self.assertEqual(r.status, "line_search_limit")
        self.assertEqual(r.final.iteration, 0)
        self.assertEqual(len(r.searches[0].samples), 2)
        self.assertLess(r.searches[0].samples[-1].value, r.final.value)
        self.assertIsNone(r.searches[0].accepted)

    def test_precision_limit_and_stationary_objective_increase_are_explicit(self):
        r = train(NormalizedRequest("x+1e10*(x-1e20)^2+y^2", x0=1e20, y0=0))
        self.assertEqual(r.status, "line_search_precision")
        self.assertEqual(r.final.iteration, 0)
        r = train(
            NormalizedRequest("sin(x)+y^2", x0=0, y0=0, initial_gamma=3 * math.pi / 2)
        )
        self.assertEqual(r.status, "line_search_increase")
        self.assertEqual(r.final.iteration, 0)
        self.assertGreater(r.searches[0].samples[-1].value, r.final.value)

    def test_unbounded_forward_search_stops_at_numerical_boundary(self):
        r = train(NormalizedRequest("x+y", x0=0, y0=0))
        self.assertEqual(r.status, "line_search_failed")
        self.assertEqual(r.final.iteration, 0)
        self.assertTrue(any(not p.valid for p in r.searches[0].samples))

    def test_initial_probe_is_not_a_fixed_accepted_step_and_long_search_is_complete(
        self,
    ):
        for gamma in (0.25, 2, 8, 1e-100):
            with self.subTest(initial_gamma=gamma):
                r = train(NormalizedRequest(initial_gamma=gamma))
                self.assertEqual(r.status, "gradient_tolerance")
                self.assertAlmostEqual(r.searches[0].accepted.gamma, 5, places=8)
                if gamma == 1e-100:
                    self.assertGreater(len(r.searches[0].samples), 200)

    def test_plot_bounds_do_not_constrain_the_line_search(self):
        request = NormalizedRequest("(x-2)^2+2*(y+1)^2", x0=6, y0=3, max_iterations=2)
        r = train(request)
        other = train(
            replace(request, plot_min=10, plot_max=20, plot_y_min=10, plot_y_max=20)
        )
        self.assertEqual(other.points, r.points)
        self.assertEqual(other.searches, r.searches)

    def test_validation_bad_start_and_cancellation(self):
        for changes in (
            {"initial_gamma": 0},
            {"initial_gamma": math.inf},
            {"line_tolerance": 0},
            {"line_tolerance": 1},
            {"max_line_evaluations": 0},
            {"max_line_evaluations": 2049},
            {"max_line_evaluations": 1.5},
            {"max_iterations": 2000},
            {"tolerance": 0},
            {"y0": math.nan},
            {"expression": "x.real+y"},
        ):
            with self.subTest(changes=changes), self.assertRaises(InputError):
                replace(NormalizedRequest(), **changes).validate()
        with self.assertRaisesRegex(RuntimeError, "starting"):
            train(NormalizedRequest("sqrt(x)+y^2", x0=-1))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            train(NormalizedRequest(), cancel=cancel)

    def test_outer_and_line_csv_export_full_precision_snapshots_and_invalid_trials(
        self,
    ):
        r = train(NormalizedRequest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outer.csv"
            lines = Path(directory) / "lines.csv"
            export_trace(path, r)
            export_line_searches(lines, r)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            with lines.open(newline="") as stream:
                trials = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(trials), 7)
            self.assertNotIn("alpha", rows[0])
            self.assertEqual(float(rows[0]["gamma_k"]), 5)
            self.assertEqual(float(rows[0]["S_x"]), -0.6)
            self.assertEqual(rows[-1]["gamma_k"], "")
            self.assertEqual(sum(p["accepted"] == "True" for p in trials), 1)
            self.assertEqual(float(trials[-1]["phi_prime_gamma"]), 0)
            self.assertEqual(float(trials[0]["initial_gamma"]), 1)
            failed = train(NormalizedRequest("ln(x)+y^2", x0=1, y0=0))
            export_line_searches(lines, failed)
            with lines.open(newline="") as stream:
                trials = list(csv.DictReader(stream))
            self.assertTrue(
                any(
                    p["valid"] == "False"
                    and p["phi_gamma"] == ""
                    and p["phi_prime_gamma"] == ""
                    for p in trials
                )
            )


if __name__ == "__main__":
    unittest.main()
