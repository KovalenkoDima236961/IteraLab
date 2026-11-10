"""Independent numerical fixtures transcribed from all practical-example families.

Page numbers and corrected source mistakes are documented in
PRACTICAL_EXAMPLES_AUDIT.md. All calculations use the GUI's real Rust process.
"""

import csv
from dataclasses import replace
from fractions import Fraction
import math
from pathlib import Path
import tempfile
import unittest

from every_example_gui.backend import InputError, Request, Sample, preset_samples, train
from every_example_gui.algorithms.lecture08 import (
    IL_S,
    IL_T,
    TL_S,
    TLH_S,
    TLH_T,
    TLH_W,
    TLH_QUERIES,
    Lecture08Request,
    exercise_presets,
)
from every_example_gui.algorithms.madaline import MadalineRequest
from every_example_gui.algorithms.nelder_mead import (
    NelderMeadRequest,
    export_trace as export_simplex,
)
from every_example_gui.algorithms.newton2d import Newton2DRequest
from every_example_gui.algorithms.normalized_gradient import (
    LineSteepestRequest,
    NormalizedRequest,
    export_trace,
    export_line_searches,
)
from every_example_gui.algorithms.optimization import GradientRequest, NewtonRequest
from every_example_gui.algorithms.perceptron import (
    PerceptronRequest,
    perceptron_samples,
)
from every_example_gui.algorithms.hebbian import HebbianRequest
from every_example_gui.algorithms.powell import PowellRequest
from every_example_gui.algorithms.steepest import SteepestRequest


class PracticalExamplesTests(unittest.TestCase):
    def assertVector(self, actual, expected, places=10):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, float(b), places=places)

    def test_04_corrects_quartic_gradient_arithmetic_in_both_note_sets(self):
        r = train(GradientRequest("x^4-2*x^2", x0=2, alpha=0.05, max_iterations=2))
        self.assertVector([p.x for p in r.points], [2, 0.8, 0.8576])
        # The source's .6 and .6768 propagate a first-step subtraction error.
        self.assertNotAlmostEqual(r.points[1].x, 0.6)

    def test_slide_5_exponential_gradient_reaches_stated_accuracy_after_nine_steps(
        self,
    ):
        r = train(
            GradientRequest(
                "exp(x)-2*x", x0=-1, alpha=0.35, tolerance=0.001, max_iterations=9
            )
        )
        self.assertEqual(r.status, "gradient_tolerance")
        self.assertEqual(r.final.iteration, 9)
        self.assertLess(abs(r.final.x - math.log(2)), 0.001)
        self.assertGreater(abs(r.points[-2].gradient), 0.001)

    def test_04_corrects_quartic_newton_arithmetic(self):
        r = train(NewtonRequest("x^4-2*x^2", x0=2, max_iterations=1))
        self.assertVector(
            [p.x for p in r.points], [2, Fraction(16, 11), Fraction(8192, 7117)]
        )
        self.assertNotAlmostEqual(r.points[1].x, 1.36, places=2)

    def test_04_two_dimensional_gradient_and_newton(self):
        r = train(SteepestRequest("x^2+2*y^2", x0=2, y0=2, alpha=0.1, max_iterations=2))
        for p, expected in zip(
            r.points, [(2, 2, 12), (1.6, 1.2, 5.44), (1.28, 0.72, 2.6752)]
        ):
            self.assertVector((p.x, p.y, p.value), expected)
        r = train(Newton2DRequest("x^2+2*y^2", x0=2, y0=2))
        self.assertVector((r.final.x, r.final.y, r.final.value), (0, 0, 0))
        self.assertEqual(r.final.iteration, 1)

    def test_04_simplex_expansion_and_reflection_match_the_handwritten_steps(self):
        r = train(
            NelderMeadRequest("x^2+2*y^2", ((2, 2), (2, 1), (1, 2)), max_iterations=2)
        )
        self.assertEqual([s.action for s in r.steps], ["expand", "reflect"])
        self.assertVector(
            (
                r.steps[0].expansion.x,
                r.steps[0].expansion.y,
                r.steps[0].expansion.value,
            ),
            (0.5, 0.5, 0.75),
        )
        self.assertVector(
            (
                r.steps[1].reflection.x,
                r.steps[1].reflection.y,
                r.steps[1].reflection.value,
            ),
            (1.5, -0.5, 2.75),
        )

    def test_cv4_newton_stationary_point_is_not_necessarily_a_minimum(self):
        r = train(NewtonRequest("x^4-3*x^3+2", x0=1, max_iterations=0))
        self.assertAlmostEqual(r.final.x, 1 / 6)
        r = train(Newton2DRequest("x^4/4+x^2/2+y^2/2", x0=1, y0=1, max_iterations=1))
        self.assertVector((r.final.x, r.final.y), (0.5, 0))

    def test_cv4_standard_outside_contraction_differs_from_slide_variant(self):
        q = NelderMeadRequest(
            "x^2+y^2", ((0, 0), (1, 0), (0.75, 1)), max_iterations=1, variant="standard"
        )
        r = train(q)
        self.assertEqual(r.steps[0].action, "contract_outside")
        self.assertVector(
            (
                r.steps[0].contraction.x,
                r.steps[0].contraction.y,
                r.steps[0].contraction.value,
            ),
            (0.375, -0.5, 0.390625),
        )
        r = train(replace(q, variant="slides"))
        self.assertEqual(r.steps[0].action, "contract")
        self.assertVector(
            (r.steps[0].contraction.x, r.steps[0].contraction.y), (0.625, 0.5)
        )

    def test_cv4_strict_position_stopping_does_not_accept_a_large_flat_simplex(self):
        q = NelderMeadRequest(
            "7",
            ((0, 0), (1, 0), (0, 1)),
            tolerance=0.125,
            stopping="position",
            variant="standard",
        )
        self.assertIn("\tstandard\tposition\n", q.serialize())
        self.assertNotIn(
            "'standard'", q.serialize()
        )  # Reported Rust request regression.
        r = train(q)
        self.assertEqual(r.status, "simplex_tolerance")
        self.assertEqual(r.final.iteration, 4)
        self.assertEqual(r.points[-2].best_radius, 0.125)
        self.assertEqual(r.final.best_radius, 0.0625)

    def test_slides_23_table_uses_minus_4x_despite_printed_minus_4y(self):
        s = ((0, 0), (1.2, 0), (0, 0.8))
        r = train(NelderMeadRequest("x^2-4*x+y^2-y-x*y", s, max_iterations=5))
        self.assertVector([v.value for v in r.points[0].simplex], [-3.36, -0.16, 0])
        self.assertVector(
            [p.value for p in r.points], [-3.36, -5.88, -5.88, -6.24, -6.24, -6.72]
        )
        literal = train(NelderMeadRequest("x^2-4*y+y^2-y-x*y", s, max_iterations=1))
        initial = {(v.x, v.y): v.value for v in literal.points[0].simplex}
        self.assertAlmostEqual(initial[(1.2, 0)], 1.44)
        self.assertAlmostEqual(initial[(0, 0.8)], -3.36)

    def test_slides_25_26_shifted_bowl_and_page_27_three_iterations(self):
        r = train(
            NelderMeadRequest(
                "(x-2)^2+3*(y-1)^2", ((0, 0), (2, 0), (0, 2)), max_iterations=2
            )
        )
        self.assertEqual(
            [s.action for s in r.steps], ["contract", "reflect_after_expand"]
        )
        self.assertVector(
            (
                r.steps[0].contraction.x,
                r.steps[0].contraction.y,
                r.steps[0].contraction.value,
            ),
            (0.5, 1, 2.25),
        )
        self.assertVector((r.final.x, r.final.y, r.final.value), (2.5, 1, 0.25))
        r = train(
            NelderMeadRequest(
                "x^2+y^2+x-2*y-x*y+2", ((1.2, 0), (0, 0.8), (0, 0)), max_iterations=3
            )
        )
        self.assertEqual(
            [s.action for s in r.steps], ["contract", "reflect", "contract"]
        )
        for v, ref in zip(
            r.final.simplex, [(0, 0.8, 1.04), (-0.6, 0.6, 1.28), (-0.15, 0.35, 1.3475)]
        ):
            self.assertVector((v.x, v.y, v.value), ref)

    def test_cv5_fixed_gradient_and_optimized_step_are_separate_algorithms(self):
        f = "2*x^2+x*y+3*y^2-4*x+2*y"
        r = train(SteepestRequest(f, x0=0, y0=0, alpha=0.125, max_iterations=2))
        self.assertVector(
            (r.points[1].x, r.points[1].y, r.points[1].value), (0.5, -0.25, -1.9375)
        )
        self.assertVector(
            (r.final.x, r.final.y, r.final.value), (0.78125, -0.375, -2.525390625)
        )
        q = LineSteepestRequest(f, x0=0, y0=0, max_iterations=1)
        r = train(q)
        self.assertVector(
            (r.searches[0].direction_x, r.searches[0].direction_y), (4, -2)
        )
        self.assertAlmostEqual(r.searches[0].accepted.gamma, 5 / 18, places=9)
        self.assertVector(
            (r.final.x, r.final.y, r.final.value), (10 / 9, -5 / 9, -25 / 9), places=8
        )
        r = train(replace(q, max_iterations=200))
        self.assertVector(
            (r.final.x, r.final.y, r.final.value),
            (26 / 23, -12 / 23, -64 / 23),
            places=5,
        )

    def test_05_normalized_direction_and_line_minimum(self):
        r = train(NormalizedRequest("x^2-4*x+y^2-y-x*y", x0=5, y0=5, max_iterations=1))
        s = r.searches[0]
        self.assertVector(
            (s.direction_x, s.direction_y), (-1 / math.sqrt(17), -4 / math.sqrt(17))
        )
        self.assertAlmostEqual(s.accepted.gamma, 17 * math.sqrt(17) / 26, places=9)
        self.assertVector(
            (r.final.x, r.final.y, r.final.value),
            (113 / 26, 31 / 13, -289 / 52),
            places=8,
        )
        # Notes round ||(30/13,-15/26)|| to 2.3; exact value is about 2.3787.
        self.assertAlmostEqual(r.final.gradient_norm, 15 * math.sqrt(17) / 26, places=8)

    def test_05_powell_starting_at_sweep_end_is_the_same_line_as_notes(self):
        r = train(PowellRequest("x^2-4*x+y^2-y-x*y", x0=5, y0=5, max_iterations=2))
        for p, ref in zip(
            r.points[:4],
            [
                (5, 5, 0),
                (4.5, 5, -0.25),
                (4.5, 2.75, -85 / 16),
                (297 / 67, 164 / 67, -361 / 67),
            ],
        ):
            self.assertVector((p.x, p.y, p.value), ref, places=7)
        # Notes use gamma=76/67 from X0; GUI uses alpha=9/67 from P2.
        self.assertAlmostEqual(r.searches[2].accepted.alpha, 9 / 67, places=8)
        self.assertVector((r.final.x, r.final.y, r.final.value), (3, 2, -7), places=6)

    def test_06_hebbian_and_perceptron_complete_epoch_confirmation(self):
        samples = (
            Sample((1, 1), 1),
            Sample((-1, 1), -1),
            Sample((1, -1), -1),
            Sample((-1, -1), -1),
        )
        r = train(HebbianRequest(samples, eta=1))
        self.assertVector((*r.weights, r.bias), (2, 2, -2))
        self.assertEqual(r.correct, 4)
        for name, expected, epochs in [("AND", (3, 2, -4), 10), ("OR", (2, 2, -1), 5)]:
            ss = perceptron_samples(name, False, "band")
            ss = (ss[0], ss[2], ss[1], ss[3])
            r = train(PerceptronRequest(ss, (0, 0), mode="band", theta=0.2))
            self.assertVector((*r.weights, r.bias), expected)
            self.assertEqual(r.completed_epochs, epochs)
            self.assertEqual(r.correct, 4)

    def test_06_mcp_gates_and_three_neuron_network(self):
        for title, expected in [
            ("06 · AND", (1, 0, 0, 0)),
            ("06 · OR", (1, 1, 1, 0)),
            ("06 · NAND", (0, 1, 1, 1)),
            ("NOT", (0, 1)),
            ("CV6 · XOR network (OR, NAND, AND)", (0, 1, 1, 0)),
        ]:
            with self.subTest(title=title):
                r = train(Lecture08Request("mcp", **exercise_presets("mcp")[title]))
                self.assertVector([row[-1] for row in r.matrix("output")], expected)
        r = train(
            Lecture08Request(
                "mcp", **exercise_presets("mcp")["06 p6 · three-neuron network"]
            )
        )
        self.assertEqual(
            r.matrix("output"), ((1, 0, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0))
        )
        self.assertEqual(
            r.matrix("net"), ((2, -3, 2), (1, -2, 0), (1, -1, 0), (0, 0, 0))
        )

    def test_07_adaline_and_exact_two_epoch_fixture(self):
        r = train(Request(preset_samples("AND"), (0, 0), alpha=0.2, max_epochs=2))
        expected = [
            (0, 0.2, 0.2, 0.2),
            (0.2, -0.04, 0.44, -0.04),
            (0.44, 0.248, 0.152, -0.328),
            (-0.728, 0.3024, 0.2064, -0.3824),
            (0.1264, 0.47712, 0.38112, -0.20768),
            (-0.11168, 0.299456, 0.558784, -0.385344),
            (-0.126016, 0.4742528, 0.3839872, -0.5601408),
            (-1.4183808, 0.39057664, 0.30031104, -0.47646464),
        ]
        for s, ref in zip(r.steps, expected):
            self.assertVector((s.net, *s.after_weights, s.after_bias), ref)

    def test_07_housing_two_epoch_fixture(self):
        r = train(
            Request(
                (Sample((2, 1), -1), Sample((3, 2), 1), Sample((4, 3), 1)),
                (0, 0),
                alpha=0.1,
                max_epochs=2,
            )
        )
        expected = [
            (0, -0.2, -0.1, -0.1),
            (-0.9, 0.37, 0.28, 0.09),
            (2.41, -0.194, -0.143, -0.051),
            (-0.582, -0.2776, -0.1848, -0.0928),
            (-1.2952, 0.41096, 0.27424, 0.13672),
            (2.60328, -0.230352, -0.206744, -0.023608),
        ]
        for s, ref in zip(r.steps, expected):
            self.assertVector((s.net, *s.after_weights, s.after_bias), ref)

    def test_cv7_adaline_and_or_final_two_epoch_values(self):
        for name, expected in [
            ("AND", (0.28852339, 0.25711319, -0.31780539)),
            ("OR", (0.32433359, 0.35108739, 0.29562441)),
        ]:
            r = train(Request(preset_samples(name), (0, 0), alpha=0.1, max_epochs=2))
            self.assertVector((*r.weights, r.bias), expected)

    def test_cv6_bipolar_error_updates_reproduce_twice_sized_handwritten_steps(self):
        for name, expected in [
            ("AND", (-2, 0, -4)),
            ("OR", (2, 2, -2)),
            ("XOR", (0, 0, 0)),
        ]:
            r = train(
                PerceptronRequest(
                    perceptron_samples(name, False, "bipolar_error"),
                    (0, 0),
                    mode="bipolar_error",
                    max_epochs=2,
                )
            )
            self.assertVector((*r.weights, r.bias), expected)
            self.assertEqual(
                r.steps[1 if name == "AND" else 0].delta_bias,
                -2 if name in ("AND", "XOR") else 0,
            )
        from every_example_gui.examples.learning import LEARNING_EXAMPLES
        from every_example_gui.backend import parse_samples

        values = LEARNING_EXAMPLES["CV6 · majority vote, two epochs"]
        r = train(
            PerceptronRequest(
                parse_samples(values["data"]),
                (0, 0, 0),
                mode="bipolar_error",
                max_epochs=2,
            )
        )
        self.assertVector((*r.weights, r.bias), (-2, 0, 2, -4))

    def test_07_madaline_zero_start_matches_both_worked_epochs(self):
        r = train(
            MadalineRequest(
                preset_samples("AND"), ((0, 0), (0, 0)), alpha=0.2, max_epochs=2
            )
        )
        expected = [
            ((0, 0, 0, 0), (0, 0)),
            ((-0.2, 0.2, -0.2, 0.2), (-0.2, -0.2)),
            ((0.04, -0.04, 0.04, -0.04), (-0.44, -0.44)),
            ((0.04, -0.04, 0.04, -0.04), (-0.44, -0.44)),
            ((0.328, 0.248, 0.04, -0.04), (-0.152, -0.44)),
        ]
        for s, (w, b) in zip(r.steps, expected):
            self.assertVector(s.after_weights, w)
            self.assertVector(s.after_biases, b)
        self.assertEqual(r.steps[1].selected, (True, True))
        for s in r.steps[5:]:
            self.assertEqual(s.rule, "correct")

    def test_08_matrix_products_dot_products_and_transposes(self):
        for title, expected in [
            ("08 p1 · matrix multiplication", ((4, 13), (11, 27))),
            ("08 p1 · dot product", ((10,),)),
            (
                "CV8 · rectangular matrix product",
                ((9, 12, 15), (19, 26, 33), (29, 40, 51)),
            ),
            ("CV8 · eigenvector transformation", ((4,), (3,))),
        ]:
            r = train(
                Lecture08Request(
                    "matrix_product", **exercise_presets("matrix_product")[title]
                )
            )
            self.assertEqual(r.matrix("output"), expected)
            self.assertEqual(len(r.matrix("A_transpose")), len(r.request.s[0]))

    def test_08_and_cv8_correlation_eigenvalues_and_corrected_mean_bounds(self):
        fixtures = [
            (((1, 0), (0, 1), (1, 1)), (1, 1 / 3), 2),
            (((1, 1), (1, -1), (-1, 1), (-1, -1)), (1, 1), 2),
            (((1, 1, 0), (1, 0, 1), (0, 1, 1)), (4 / 3, 1 / 3, 1 / 3), 1.5),
        ]
        for s, eigen, upper in fixtures:
            r = train(Lecture08Request("correlation", s=s))
            self.assertVector(r.matrix("eigenvalues")[0], eigen)
            self.assertAlmostEqual(r.scalars["mean_lms_alpha_upper"], upper)
        r = train(Lecture08Request("correlation", s=((2, 0), (0, 2), (2, 2))))
        self.assertAlmostEqual(r.scalars["mean_lms_alpha_upper"], 0.5)
        zero = train(Lecture08Request("correlation", s=((0, 0),)))
        self.assertNotIn("mean_lms_alpha_upper", zero.scalars)

    def test_cv8_rate_example_retains_full_precision_instead_of_rounding_each_row(self):
        r = train(
            Request(
                (Sample((1, 0), -1), Sample((0, 1), 1), Sample((1, 1), 1)),
                (0, 0),
                alpha=0.45,
                max_epochs=1,
            )
        )
        self.assertVector((*r.weights, r.bias), (-0.18225, 0.92025, 0.47025))

    def test_08_il_tl_memories_and_damaged_recall_use_bipolar_outputs(self):
        for s, net in [(IL_S, ((2, -2), (-2, 2))), (TL_S, ((12, -12), (-12, 12)))]:
            r = train(
                Lecture08Request("hebbian_memory", s=s, t=IL_T, activation="sign")
            )
            self.assertEqual(r.matrix("training_output"), IL_T)
            self.assertEqual(r.matrix("training_net"), net)
        r = train(
            Lecture08Request(
                "recall",
                w=((0, 0), (0, 0), (0, 0), (-2, 2)),
                x=((0, -1, 1, 1),),
                activation="sign",
            )
        )
        self.assertEqual(r.matrix("output"), ((-1, 1),))

    def test_cv8_letter_weights_queries_and_inconsistent_first_query(self):
        r = train(
            Lecture08Request("outer_product", s=TLH_S, t=TLH_T, activation="bipolar")
        )
        self.assertEqual(r.matrix("W"), TLH_W)
        self.assertEqual(r.matrix("training_output"), TLH_T)
        r = train(
            Lecture08Request("recall", w=TLH_W, x=TLH_QUERIES, activation="bipolar")
        )
        self.assertEqual(
            r.matrix("net"), ((23, -1, -17), (-21, 15, -5), (-15, -7, 17), (-27, 5, 13))
        )
        # The diagram changes only pixels 1 and 8; numeric rows also change 11.
        diagram = list(TLH_QUERIES[0])
        diagram[10] = 1
        r = train(
            Lecture08Request(
                "recall", w=TLH_W, x=(tuple(diagram),), activation="bipolar"
            )
        )
        self.assertEqual(r.matrix("net"), ((29, -3, -19),))

    def test_cv8_animal_ambiguous_query_and_zero_activation_choices(self):
        q = Lecture08Request(
            "recall", w=((2, -2), (-2, 2), (0, 0)), x=((1, 1, 1),), activation="sign"
        )
        self.assertEqual(train(q).matrix("output"), ((0, 0),))
        self.assertEqual(
            train(replace(q, activation="bipolar")).matrix("output"), ((1, 1),)
        )
        q = Lecture08Request("recall", w=((1,),), x=((-1,), (0,), (1,)))
        for activation, expected in [
            ("binary", ((0,), (0,), (1,))),
            ("sign", ((-1,), (0,), (1,))),
            ("bipolar", ((-1,), (1,), (1,))),
        ]:
            self.assertEqual(
                train(replace(q, activation=activation)).matrix("output"), expected
            )

    def test_added_options_validate_and_exports_preserve_conventions(self):
        for q in [
            Lecture08Request("mcp", s=((1,),), w=((1, 1),), t=((1,),)),
            Lecture08Request("mcp", s=((0.5,),), w=((1, 0),), t=((1,),)),
            Lecture08Request("matrix_product", s=((1, 2),), t=((1,),)),
            Lecture08Request("recall", w=((1,),), x=((1,),), activation="unknown"),
        ]:
            with self.assertRaises(InputError):
                q.validate()
        r = train(LineSteepestRequest("x^2+y^2", x0=3, y0=4))
        n = train(
            NelderMeadRequest(
                "x^2+y^2",
                ((0, 0), (1, 0), (0.75, 1)),
                variant="standard",
                stopping="position",
                max_iterations=1,
            )
        )
        with tempfile.TemporaryDirectory() as d:
            for exporter, result, name in [
                (export_trace, r, "trace.csv"),
                (export_line_searches, r, "lines.csv"),
                (export_simplex, n, "simplex.csv"),
            ]:
                p = Path(d) / name
                exporter(p, result)
                with p.open() as stream:
                    reader = csv.DictReader(stream)
                    rows = list(reader)
                    self.assertEqual(
                        len(reader.fieldnames), len(set(reader.fieldnames))
                    )
                    self.assertTrue(
                        all(
                            None not in row and None not in row.values() for row in rows
                        )
                    )
                if result is r:
                    self.assertEqual(rows[0]["algorithm"], "LINE_STEEPEST2D")
                else:
                    self.assertEqual(rows[0]["branch_variant"], "standard")


if __name__ == "__main__":
    unittest.main()
