"""Lecture 08 calculations through the same Rust process used by the GUI."""

import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest

from every_example_gui.backend import Cancelled, InputError, train
from every_example_gui.algorithms.lecture08 import (
    EXAMPLE_S,
    EXAMPLE_T,
    EXAMPLE_W,
    EXAMPLE_X,
    Lecture08Request,
    export_matrices,
    export_report,
    matrix_text,
    parse_matrix,
)


class Lecture08Tests(unittest.TestCase):
    def assertMatrixClose(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for row, reference in zip(actual, expected):
            self.assertEqual(len(row), len(reference))
            for value, wanted in zip(row, reference):
                self.assertTrue(
                    math.isclose(value, wanted, rel_tol=1e-11, abs_tol=1e-12),
                    (value, wanted),
                )

    def test_lecture_correlation_and_literal_rate(self):
        messages = []
        result = train(
            Lecture08Request("correlation", s=EXAMPLE_X), progress=messages.append
        )
        self.assertEqual(result.matrix("R"), ((0.5, 0.25), (0.25, 0.5)))
        self.assertMatrixClose(result.matrix("eigenvalues"), ((0.75, 0.25),))
        self.assertAlmostEqual(result.scalars["lambda_max"], 0.75)
        self.assertAlmostEqual(result.scalars["lecture_alpha_upper"], 0.375)
        self.assertLess(result.scalars["eigen_residual_max"], 1e-12)
        self.assertIn("lambda^2", result.steps[-2].detail)
        self.assertIn("strict", result.steps[-1].detail)
        self.assertEqual(result.matrix("outer_4"), ((0, 0), (0, 0)))
        self.assertEqual(result.matrix("sum_outer"), ((2, 1), (1, 2)))
        self.assertTrue(any("Lecture 08" in message for message in messages))

    def test_zero_scalar_and_rank_deficient_correlation(self):
        zero = train(Lecture08Request("correlation", s=((0, 0),)))
        self.assertEqual(zero.matrix("eigenvalues"), ((0, 0),))
        self.assertEqual(zero.scalars["lecture_alpha_upper"], 0)
        self.assertIn("No positive alpha", zero.steps[-1].detail)
        scalar = train(Lecture08Request("correlation", s=((2,), (-4,))))
        self.assertEqual(scalar.matrix("R"), ((10,),))
        self.assertEqual(scalar.matrix("eigenvalues"), ((10,),))
        self.assertEqual(scalar.scalars["lecture_alpha_upper"], 5)
        rank_one = train(Lecture08Request("correlation", s=((1, 2, 3),)))
        self.assertMatrixClose(rank_one.matrix("eigenvalues"), ((14, 0, 0),))

    def test_general_symmetric_spectrum_invariants_and_scaling(self):
        s = ((1, 2, -3), (4, -1, 2), (0, 3, 1), (-2, 0.5, 4))
        result = train(Lecture08Request("correlation", s=s))
        r = result.matrix("R")
        eigenvalues = result.matrix("eigenvalues")[0]
        vectors = result.matrix("eigenvectors")
        self.assertAlmostEqual(sum(eigenvalues), sum(r[i][i] for i in range(3)))
        for j, eigenvalue in enumerate(eigenvalues):
            for i in range(3):
                self.assertAlmostEqual(
                    sum(r[i][k] * vectors[k][j] for k in range(3)),
                    eigenvalue * vectors[i][j],
                    places=10,
                )
            for k in range(3):
                self.assertAlmostEqual(
                    sum(vectors[i][j] * vectors[i][k] for i in range(3)),
                    float(j == k),
                )
        for factor in (1e-40, 1e40):
            scaled = train(
                Lecture08Request(
                    "correlation", s=tuple(tuple(v * factor for v in row) for row in s)
                )
            )
            for value, reference in zip(scaled.matrix("eigenvalues")[0], eigenvalues):
                self.assertTrue(
                    math.isclose(value / factor**2, reference, rel_tol=1e-11)
                )
            self.assertLess(scaled.scalars["eigen_residual_max"], 1e-12)

    def test_lecture_memory_updates_outer_products_and_recall(self):
        results = [
            train(Lecture08Request(kind, s=EXAMPLE_S, t=EXAMPLE_T))
            for kind in ("hebbian_memory", "outer_product")
        ]
        for result in results:
            self.assertEqual(result.matrix("W"), EXAMPLE_W)
            self.assertEqual(result.matrix("W_from_matrix_product"), EXAMPLE_W)
            self.assertEqual(result.matrix("initial_W"), ((0, 0),) * 4)
            self.assertEqual(result.matrix("bias"), ((0, 0),))
            self.assertEqual(result.scalars["outer_product_difference"], 0)
            self.assertEqual(result.scalars["outer_product_agreement"], 1)
            self.assertEqual(result.scalars["training_correct_patterns"], 4)
            self.assertEqual(result.scalars["all_orthogonal"], 0)
            self.assertEqual(result.matrix("training_output"), EXAMPLE_T)
            self.assertEqual(result.matrix("cross_talk"), EXAMPLE_T)
            self.assertEqual(
                result.matrix("weights_after_2"), ((2, 0), (1, 0), (0, 0), (0, 0))
            )
            self.assertEqual(result.matrix("outer_4"), ((0, 0), (0, 0), (0, 1), (0, 1)))
        recall = train(
            Lecture08Request(
                "recall",
                w=results[0].matrix("W"),
                x=EXAMPLE_S[:2],
                expected=EXAMPLE_T[:2],
            )
        )
        self.assertEqual(recall.matrix("linear"), ((2, 0), (3, 0)))
        self.assertEqual(recall.matrix("output"), EXAMPLE_T[:2])
        self.assertEqual(recall.scalars["correct_patterns"], 2)

    def test_real_inputs_three_outputs_and_separate_bias(self):
        s, t = ((1.5, -2), (0.5, 3)), ((1, 2, -1), (-2, 0.5, 4))
        request = Lecture08Request("hebbian_memory", s=s, t=t, bias_enabled=True)
        biased = train(request)
        plain = train(replace(request, bias_enabled=False))
        self.assertEqual(biased.matrix("W"), ((0.5, 3.25, 0.5), (-8, -2.5, 14)))
        self.assertEqual(biased.matrix("W"), plain.matrix("W"))
        self.assertEqual(biased.matrix("bias_after_1"), (t[0],))
        self.assertEqual(biased.matrix("bias"), ((-1, 2.5, 3),))
        self.assertEqual(plain.matrix("bias"), ((0, 0, 0),))
        for linear, net in zip(
            biased.matrix("training_linear"), biased.matrix("training_net")
        ):
            self.assertEqual(net, tuple(v + b for v, b in zip(linear, (-1, 2.5, 3))))
        # Enabling a bias changes recall, rather than changing the dimensions of W.
        lecture = train(
            Lecture08Request(
                "hebbian_memory", s=EXAMPLE_S, t=EXAMPLE_T, bias_enabled=True
            )
        )
        self.assertEqual(lecture.matrix("bias"), ((2, 2),))
        self.assertEqual(lecture.matrix("training_output"), ((1, 1),) * 4)
        self.assertEqual(lecture.scalars["training_correct_patterns"], 0)

    def test_all_pairs_tolerance_zero_vectors_and_singleton(self):
        result = train(Lecture08Request("orthogonality", s=((1, 1), (1, -1), (0, 0))))
        self.assertEqual(result.scalars["all_orthogonal"], 1)
        self.assertEqual(result.scalars["zero_vectors"], 1)
        self.assertEqual(
            result.matrix("pairwise_dots"), ((1, 2, 0, 1), (1, 3, 0, 1), (2, 3, 0, 1))
        )
        request = Lecture08Request("orthogonality", s=((1, 0), (1e-9, 1)))
        self.assertEqual(train(request).scalars["all_orthogonal"], 0)
        self.assertEqual(
            train(replace(request, tolerance=1e-8)).scalars["all_orthogonal"], 1
        )
        one = train(Lecture08Request("orthogonality", s=((1, 2, 3),)))
        pairs = next(m for m in one.matrices if m.name == "pairwise_dots")
        self.assertEqual((pairs.rows, pairs.columns), (0, 4))
        self.assertEqual(one.scalars["all_orthogonal"], 1)

    def test_recall_zero_negative_values_bias_and_optional_expectations(self):
        request = Lecture08Request("recall", w=((1, -1), (-1, 1)), x=((0, 0), (1, 0)))
        plain = train(request)
        self.assertEqual(plain.matrix("net"), ((0, 0), (1, -1)))
        self.assertEqual(plain.matrix("output"), ((0, 0), (1, 0)))
        self.assertNotIn("correct_patterns", plain.scalars)
        biased = train(
            replace(
                request, bias_enabled=True, bias=((0, 1),), expected=((0, 1), (1, 0))
            )
        )
        self.assertEqual(biased.matrix("linear"), plain.matrix("linear"))
        self.assertEqual(biased.matrix("net"), ((0, 1), (1, 0)))
        self.assertEqual(biased.matrix("matches"), ((1, 1), (1, 1)))
        self.assertEqual(biased.scalars["correct_patterns"], 2)
        self.assertEqual(biased.matrix("terms_2"), ((1, -1), (0, 0)))

    def test_dimension_and_numerical_validation(self):
        invalid = [
            Lecture08Request("unknown", s=((1,),)),
            Lecture08Request("correlation"),
            Lecture08Request("correlation", s=((1, 2), (3,))),
            Lecture08Request("correlation", s=((float("nan"),),)),
            Lecture08Request("correlation", s=((1e51,),)),
            Lecture08Request("correlation", s=((1,) * 33,)),
            Lecture08Request("orthogonality", s=((1,),) * 129),
            Lecture08Request("orthogonality", s=((1,),), tolerance=-1),
            Lecture08Request("orthogonality", s=((1,),), tolerance=float("inf")),
            Lecture08Request("hebbian_memory", s=((1,),), t=((1,), (2,))),
            Lecture08Request("outer_product", s=((1,),), t=((1,),), bias_enabled=True),
            Lecture08Request("recall", w=((1, 2),), x=((1, 2),)),
            Lecture08Request("recall", w=((1, 2),), x=((1,),), expected=((1,),)),
            Lecture08Request("recall", w=((1,),), x=((1,),), expected=((1,), (0,))),
            Lecture08Request("recall", w=((1,),), x=((1,),), bias_enabled=True),
            Lecture08Request(
                "recall", w=((1,),), x=((1,),), bias_enabled=True, bias=((1, 2),)
            ),
        ]
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(InputError):
                train(request)

    def test_matrix_input_formats_limits_and_lossless_copying(self):
        reference = ((1, 0.25), (-2, 3))
        for text in (
            "1, .25\n-2,3",
            "# pattern rows\n1;.25\n\n-2 3",
            "[[1,.25],[-2,3]]".replace(".25", "0.25"),
        ):
            self.assertEqual(parse_matrix(text), reference)
        original = ((1.2345678901234567, 1e-40), (-0.0, 1e40))
        self.assertEqual(parse_matrix(matrix_text(original)), original)
        self.assertEqual(parse_matrix("", optional=True), ())
        for text in (
            "",
            "# no patterns",
            "1,2\n3",
            "[1,2]",
            "[[]]",
            "[[1],[]]",
            "NaN",
            "inf",
            "1e999",
            "[[" + "9" * 400 + "]]",
            "1," * 33,
            "1\n" * 129,
        ):
            with self.subTest(text=text[:50]), self.assertRaises(InputError):
                parse_matrix(text)

    def test_full_precision_exports_and_cancel_before_start(self):
        request = Lecture08Request(
            "hebbian_memory", s=((0.12345678901234567, 2),), t=((3, 4),)
        )
        result = train(request)
        with tempfile.TemporaryDirectory() as directory:
            json_path, csv_path = (
                Path(directory) / "result.json",
                Path(directory) / "result.csv",
            )
            export_report(json_path, result)
            export_matrices(csv_path, result)
            saved = json.loads(json_path.read_text())
            self.assertEqual(saved["request"]["s"][0][0], request.s[0][0])
            self.assertEqual(saved["format"], "lecture08-v1")
            self.assertEqual(len(saved["steps"]), len(result.steps))
            with csv_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            value = next(
                r["value"]
                for r in rows
                if r["name"] == "W" and r["row"] == "1" and r["column"] == "1"
            )
            self.assertEqual(float(value), result.matrix("W")[0][0])
            self.assertTrue(any(r["name"] == "outer_1" for r in rows))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            train(request, cancel=cancel)


if __name__ == "__main__":
    unittest.main()
