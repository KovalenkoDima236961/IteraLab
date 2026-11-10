"""Lecture exercise, direction shifts, derivative-free searches and stopping boundaries."""

import csv
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import threading
import unittest

from every_example_gui.backend import Cancelled, InputError, parse_output, train
from every_example_gui.algorithms.optimization import export_trace
from every_example_gui.algorithms.powell import PowellRequest, export_line_searches


class PowellTests(unittest.TestCase):
    def test_supplied_exercise_all_six_moves_and_function_values(self):
        r = train(PowellRequest(max_iterations=2))
        self.assertEqual(r.status, "iteration_limit")
        expected = [
            (0, 0, 0),
            (2, 0, -4),
            (2, 0.5, -4.5),
            (22 / 13, 11 / 26, -121 / 26),
            (22 / 13, 17 / 26, -1609 / 338),
            (250 / 169, 203 / 338, -21241 / 4394),
            (1, 1, -5),
        ]
        self.assertEqual(len(r.points), 7)
        for p, values in zip(r.points, expected):
            for actual, value in zip((p.x, p.y, p.value), values):
                self.assertAlmostEqual(actual, value, places=11)
        self.assertEqual(len(r.cycles), 2)
        self.assertEqual(len(r.searches), 6)

    def test_raw_displacement_third_search_origin_and_direction_shift(self):
        r = train(PowellRequest(max_iterations=2))
        expected_directions = [
            (1, 0),
            (0, 1),
            (2, 0.5),
            (0, 1),
            (2, 0.5),
            (-36 / 169, 30 / 169),
        ]
        expected_alpha = [2, 0.5, -2 / 13, 3 / 13, -18 / 169, 9 / 4]
        for i, (s, d, alpha) in enumerate(
            zip(r.searches, expected_directions, expected_alpha)
        ):
            self.assertEqual(s.cycle, i // 3 + 1)
            self.assertEqual(s.stage, i % 3 + 1)
            self.assertAlmostEqual(s.direction_x, d[0], places=11)
            self.assertAlmostEqual(s.direction_y, d[1], places=11)
            self.assertAlmostEqual(s.accepted.alpha, alpha, places=11)
            a, b = r.points[i : i + 2]
            self.assertEqual((s.samples[0].x, s.samples[0].y), (a.x, a.y))
            self.assertAlmostEqual(
                b.x, a.x + s.accepted.alpha * s.direction_x, places=13
            )
            self.assertAlmostEqual(
                b.y, a.y + s.accepted.alpha * s.direction_y, places=13
            )
            self.assertLessEqual(b.value, a.value)
        self.assertGreater(
            math.hypot(r.searches[2].direction_x, r.searches[2].direction_y), 2
        )
        for c in r.cycles:
            a, b = r.points[3 * (c.number - 1)], r.points[3 * c.number]
            self.assertAlmostEqual(c.function_change, abs(a.value - b.value))
            self.assertAlmostEqual(c.displacement, math.hypot(a.x - b.x, a.y - b.y))

    def test_complete_cycle_confirmation_after_reaching_exercise_minimum(self):
        r = train(PowellRequest())
        self.assertEqual(r.status, "powell_function_tolerance")
        self.assertEqual(len(r.cycles), 3)
        self.assertAlmostEqual(r.points[6].x, 1, places=11)
        self.assertAlmostEqual(r.points[6].y, 1, places=11)
        self.assertEqual(r.final.iteration, 9)
        self.assertEqual(r.cycles[-1].function_change, 0)
        self.assertEqual(r.searches[-1].status, "zero_direction")
        self.assertEqual(len(r.searches[-1].samples), 1)
        self.assertEqual(r.searches[-1].accepted.alpha, 0)

    def test_strict_function_and_position_tolerance_boundaries(self):
        q = PowellRequest("x^2+y^2", x0=1, y0=0, tolerance=1, max_iterations=1)
        r = train(q)
        self.assertEqual(r.cycles[0].function_change, 1)
        self.assertEqual(r.cycles[0].displacement, 1)
        self.assertEqual(r.status, "iteration_limit")
        r = train(replace(q, tolerance=math.nextafter(1, math.inf)))
        self.assertEqual(r.status, "powell_function_tolerance")
        self.assertEqual(r.final.iteration, 3)

    def test_position_criterion_can_pass_without_function_change_criterion(self):
        r = train(
            PowellRequest(
                "1e12*x^2+y^2", x0=1e-4, y0=0, tolerance=0.001, max_iterations=1
            )
        )
        self.assertEqual(r.status, "powell_position_tolerance")
        self.assertLess(r.cycles[0].displacement, r.request.tolerance)
        self.assertGreater(r.cycles[0].function_change, r.request.tolerance)

    def test_stationary_and_constant_starts_still_complete_one_cycle(self):
        for expr in ("x^2+y^2", "7"):
            with self.subTest(expr=expr):
                r = train(PowellRequest(expr))
                self.assertEqual(r.status, "powell_function_tolerance")
                self.assertEqual(r.final.iteration, 3)
                self.assertEqual(r.cycles[0].function_change, 0)
                self.assertEqual(r.cycles[0].displacement, 0)
                self.assertEqual(r.searches[-1].status, "zero_direction")

    def test_signed_bracketing_expands_to_distant_line_minima(self):
        for start in (-10, 20):
            r = train(
                PowellRequest("(x-7)^2+2*(y+3)^2", x0=start, y0=5, max_iterations=1)
            )
            self.assertAlmostEqual(r.points[1].x, 7, places=8)
            self.assertAlmostEqual(r.points[2].y, -3, places=8)
            self.assertEqual(
                math.copysign(1, r.searches[0].accepted.alpha),
                math.copysign(1, 7 - start),
            )
            self.assertTrue(any(p.phase == "expand" for p in r.searches[0].samples))

    def test_nonsmooth_function_needs_no_derivative_at_minimum(self):
        r = train(PowellRequest("sqrt(x^2)+sqrt(y^2)", x0=2, y0=-3))
        self.assertEqual(r.status, "powell_function_tolerance")
        self.assertAlmostEqual(r.final.x, 0, places=8)
        self.assertAlmostEqual(r.final.y, 0, places=8)
        self.assertLess(r.final.value, 1e-8)
        self.assertFalse(hasattr(r.final, "gradient_x"))

    def test_rosenbrock_converges_with_updated_directions(self):
        r = train(PowellRequest("(1-x)^2+100*(y-x^2)^2", x0=-1.2, y0=1, tolerance=1e-8))
        self.assertIn(
            r.status, {"powell_function_tolerance", "powell_position_tolerance"}
        )
        self.assertLess(r.final.value, 1e-10)
        self.assertAlmostEqual(r.final.x, 1, places=5)
        self.assertAlmostEqual(r.final.y, 1, places=5)
        self.assertTrue(all(b.value <= a.value for a, b in zip(r.points, r.points[1:])))

    def test_failed_search_retains_earlier_moves_in_unfinished_cycle(self):
        r = train(PowellRequest("(x-1)^2+(y-10)^2", max_line_evaluations=8))
        self.assertEqual(r.status, "line_search_limit")
        self.assertEqual(len(r.cycles), 0)
        self.assertEqual(r.final.iteration, 1)
        self.assertEqual(r.final.x, 1)
        self.assertEqual(r.final.y, 0)
        self.assertEqual(len(r.searches), 2)
        self.assertIsNone(r.searches[-1].accepted)
        self.assertTrue(
            any(p.valid and p.value < r.final.value for p in r.searches[-1].samples)
        )

    def test_failed_later_cycle_preserves_previous_cycle_and_direction_shift(self):
        r = train(PowellRequest(max_line_evaluations=10))
        self.assertEqual(r.status, "line_search_limit")
        self.assertEqual(len(r.cycles), 1)
        self.assertEqual(r.final.iteration, 3)
        self.assertEqual(
            (r.searches[-1].direction_x, r.searches[-1].direction_y), (0, 1)
        )
        self.assertEqual(len(r.searches[-1].samples), 11)

    def test_domain_refinement_and_nonexistent_line_minimum(self):
        r = train(PowellRequest("-ln(x)+x^2+y^2", x0=2, y0=1))
        self.assertAlmostEqual(r.final.x, 1 / math.sqrt(2), places=7)
        self.assertAlmostEqual(r.final.y, 0, places=7)
        self.assertTrue(any(not p.valid for s in r.searches for p in s.samples))
        self.assertTrue(
            any(p.phase == "domain_refine" for s in r.searches for p in s.samples)
        )
        r = train(PowellRequest("ln(x)+y^2", x0=1, y0=0))
        self.assertEqual(r.status, "powell_line_failed")
        self.assertEqual(r.final.iteration, 0)
        self.assertIsNone(r.searches[0].accepted)
        r = train(PowellRequest("x+y", max_line_evaluations=512))
        self.assertEqual(r.status, "powell_line_failed")
        self.assertEqual(r.final.iteration, 0)

    def test_plot_bounds_are_only_view_settings_and_grid_is_scalar(self):
        r = train(PowellRequest("sqrt(x^2)+y^2", x0=2, y0=1, plot_min=10, plot_max=11))
        self.assertAlmostEqual(r.final.x, 0, places=8)
        self.assertAlmostEqual(r.final.y, 0, places=8)
        self.assertEqual(len(r.grid), 49)
        self.assertEqual(len(r.grid[0]), 49)
        r = train(
            PowellRequest(
                "sqrt(x^2)+sqrt(y^2)",
                x0=0,
                y0=0,
                plot_min=-1,
                plot_max=1,
                plot_y_min=-1,
                plot_y_max=1,
            )
        )
        self.assertEqual(r.grid[24][24], 0)

    def test_complete_immutable_exports_include_signed_steps_trials_and_cycles(self):
        r = train(PowellRequest(max_iterations=2))
        with tempfile.TemporaryDirectory() as d:
            outer, lines = Path(d) / "outer.csv", Path(d) / "lines.csv"
            export_trace(outer, r)
            export_line_searches(lines, r)
            with outer.open(newline="") as f:
                rows = list(csv.DictReader(f))
            with lines.open(newline="") as f:
                samples = list(csv.DictReader(f))
        self.assertEqual(len(rows), 7)
        self.assertEqual(len(samples), sum(len(s.samples) for s in r.searches))
        self.assertEqual(rows[0]["function"], r.request.expression)
        self.assertAlmostEqual(float(rows[2]["alpha"]), -2 / 13, places=11)
        self.assertEqual(rows[-1]["alpha"], "")
        self.assertEqual(float(rows[0]["cycle_d1_x"]), 1)
        self.assertAlmostEqual(float(rows[3]["cycle_d2_x"]), 2)
        self.assertEqual(sum(p["accepted"] == "True" for p in samples), 6)
        self.assertFalse(any("gradient" in key or "partial" in key for key in rows[0]))

    def test_validation_runtime_domain_errors_cancellation_and_protocol(self):
        q = PowellRequest()
        for changes in (
            {"expression": "__import__('os')"},
            {"expression": "z^2"},
            {"initial_step": 0},
            {"initial_step": math.inf},
            {"line_tolerance": 0},
            {"line_tolerance": 1},
            {"max_line_evaluations": 0},
            {"max_line_evaluations": 2049},
            {"max_line_evaluations": 2.5},
            {"max_iterations": 0},
            {"max_iterations": 1000, "max_line_evaluations": 512},
            {"plot_y_min": 3, "plot_y_max": 2},
        ):
            with self.subTest(changes=changes), self.assertRaises(InputError):
                replace(q, **changes).validate()
        with self.assertRaisesRegex(RuntimeError, "domain"):
            train(replace(q, expression="ln(x)+y^2", x0=-1))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            train(q, cancel=cancel)
        for output in (
            "",
            "PWR\titeration_limit\n",
            "PWR\tbogus\n",
            "PWP\t0\tnan\t0\t0\n",
        ):
            with self.assertRaisesRegex(RuntimeError, "Invalid Powell"):
                parse_output(output, q)
        self.assertTrue(q.serialize().startswith("POWELL2_V1\n"))


if __name__ == "__main__":
    unittest.main()
