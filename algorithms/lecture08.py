"""Lecture 08 matrix requests/reports. Every numerical operation runs in Rust."""

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re

from ..backend import InputError, number

Matrix = tuple[tuple[float, ...], ...]
LABELS = {
    "correlation": "Lecture 08 · Correlation",
    "hebbian_memory": "Lecture 08 · Hebbian memory",
    "outer_product": "Lecture 08 · Outer products",
    "orthogonality": "Lecture 08 · Orthogonality",
    "recall": "Lecture 08 · Recall",
    "matrix_product": "Lecture 08 · Matrix product",
    "mcp": "McCulloch–Pitts network",
}
OPTIONS = tuple(LABELS.values())
KINDS = {label: kind for kind, label in LABELS.items()}

EXAMPLE_X = ((1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
EXAMPLE_S = (
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 1.0),
)
EXAMPLE_T = ((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0))
EXAMPLE_W = ((2.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 2.0))

IL_S = ((1, -1, 1, -1), (1, -1, 1, 1))
IL_T = ((1, -1), (-1, 1))
TL_S = ((1, 1, 1, -1, 1, -1, -1, 1, -1), (1, -1, -1, 1, -1, -1, 1, 1, 1))
TLH_S = (
    (1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1),
    (1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1),
    (1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1),
)
TLH_T = ((1, -1, -1), (-1, 1, -1), (-1, -1, 1))
TLH_W = (
    (-1, -1, -1),
    (3, -1, -1),
    (1, -3, 1),
    (-3, 1, 1),
    (3, -1, -1),
    (-1, -1, 3),
    (-3, 1, 1),
    (1, -3, 1),
    (-1, -1, 3),
    (-3, 1, 1),
    (3, -1, -1),
    (-1, -1, 3),
    (-3, 1, 1),
    (1, 1, -3),
    (-3, 1, 1),
)
TLH_QUERIES = (
    (-1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1),
    (1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 0, 0),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1),
    (1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1),
)


def exercise_presets(kind):
    """Transcribed inputs, never calculated answers, from the supplied PDFs."""
    binary = ((1, 1), (1, 0), (0, 1), (0, 0))
    if kind == "mcp":
        return {
            "06 · AND": dict(s=binary, w=((1, 1, 0),), t=((2,),)),
            "06 · OR": dict(s=binary, w=((1, 1, 0),), t=((1,),)),
            "06 · NAND": dict(s=binary, w=((-1, -1, 0),), t=((-1,),)),
            "NOT": dict(s=((1,), (0,)), w=((-1, 0),), t=((0,),)),
            "06 p6 · three-neuron network": dict(
                s=binary,
                w=((1, 1, 0, 0, 0), (-2, -1, 0, 0, 0), (0, 0, 2, 2, 0)),
                t=((2, 2, 2),),
            ),
            "CV6 · XOR network (OR, NAND, AND)": dict(
                s=binary,
                w=((1, 1, 0, 0, 0), (-1, -1, 0, 0, 0), (0, 0, 1, 1, 0)),
                t=((1, -1, 2),),
            ),
        }
    if kind == "matrix_product":
        return {
            "08 p1 · matrix multiplication": dict(
                s=((2, 1), (3, 4)), t=((1, 5), (2, 3))
            ),
            "08 p1 · dot product": dict(s=((2, 3, 4),), t=((1,), (0,), (2,))),
            "CV8 · rectangular matrix product": dict(
                s=((1, 2), (3, 4), (5, 6)), t=((1, 2, 3), (4, 5, 6))
            ),
            "CV8 · eigenvector transformation": dict(
                s=((2, 0), (0, 3)), t=((2,), (1,))
            ),
        }
    if kind == "correlation":
        return {
            "Lecture · four binary patterns": dict(s=EXAMPLE_X),
            "08 pp2–3 · three binary patterns": dict(s=((1, 0), (0, 1), (1, 1))),
            "08 pp3–4 · four bipolar patterns": dict(
                s=((1, 1), (1, -1), (-1, 1), (-1, -1))
            ),
            "CV8 · three-coordinate correlation": dict(
                s=((1, 1, 0), (1, 0, 1), (0, 1, 1))
            ),
        }
    if kind in ("hebbian_memory", "outer_product", "orthogonality"):
        pairs = {
            "Lecture · binary memory": dict(s=EXAMPLE_S, t=EXAMPLE_T),
            "08 pp5–6 · I and L (2×2)": dict(s=IL_S, t=IL_T, activation="sign"),
            "08 pp7–8 · T and L (3×3)": dict(s=TL_S, t=IL_T, activation="sign"),
            "CV8 · animal associations": dict(
                s=((1, -1, 1), (-1, 1, 1)), t=IL_T, activation="sign"
            ),
            "CV8 · T, L, H (5×3)": dict(s=TLH_S, t=TLH_T, activation="bipolar"),
        }
        return (
            {
                name: {k: v for k, v in fields.items() if k != "t"}
                for name, fields in pairs.items()
            }
            if kind == "orthogonality"
            else pairs
        )
    return {
        "Lecture · binary recall": dict(
            w=EXAMPLE_W, x=EXAMPLE_S[:2], expected=EXAMPLE_T[:2]
        ),
        "08 p6 · I, L and damaged input": dict(
            w=((0, 0), (0, 0), (0, 0), (-2, 2)),
            x=(*IL_S, (0, -1, 1, 1)),
            expected=(*IL_T, (-1, 1)),
            activation="sign",
        ),
        "CV8 · four letter queries, literal numeric rows": dict(
            w=TLH_W, x=TLH_QUERIES, activation="bipolar"
        ),
        "CV8 · letter queries, diagram flips only 1 and 8": dict(
            w=TLH_W,
            x=((*TLH_QUERIES[0][:10], 1, *TLH_QUERIES[0][11:]), *TLH_QUERIES[1:]),
            activation="bipolar",
        ),
        "08 p8 · damaged T (3×3)": dict(
            w=(
                (0, 0),
                (2, -2),
                (2, -2),
                (-2, 2),
                (2, -2),
                (0, 0),
                (-2, 2),
                (0, 0),
                (-2, 2),
            ),
            x=((1, 0, 1, -1, -1, -1, -1, 1, -1),),
            expected=((1, -1),),
            activation="sign",
        ),
        "CV8 · ambiguous animal query": dict(
            w=((2, -2), (-2, 2), (0, 0)),
            x=((1, 1, 1),),
            expected=((0, 0),),
            activation="sign",
        ),
    }


def parse_matrix(text, label="Matrix", *, optional=False):
    text = text.strip()
    if not text:
        if optional:
            return ()
        raise InputError(f"{label}: enter at least one row.")
    if len(text) > 1_048_576:
        raise InputError(f"{label}: input exceeds 1 MiB.")
    if text.startswith("["):
        try:
            rows = json.loads(text)
        except (ValueError, RecursionError) as exc:
            raise InputError(
                f"{label}: use a JSON array of rows or one numeric row per line."
            ) from exc
        if (
            not isinstance(rows, list)
            or not rows
            or any(not isinstance(row, list) for row in rows)
        ):
            raise InputError(f"{label}: expected an array of nonempty row arrays.")
    else:
        rows = [
            re.split(r"[,;\s]+", line.strip())
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not rows or not rows[0]:
        raise InputError(f"{label}: enter at least one nonempty row.")
    if len(rows) > 128 or len(rows[0]) > 32:
        raise InputError(f"{label}: use at most 128 rows and 32 columns.")
    width = len(rows[0])
    result = []
    for i, row in enumerate(rows, 1):
        if len(row) != width:
            raise InputError(
                f"{label}, row {i}: expected {width} values, received {len(row)}."
            )
        try:
            result.append(tuple(number(v, f"{label}, row {i}") for v in row))
        except OverflowError as exc:
            raise InputError(
                f"{label}, row {i}: a number is too large for finite arithmetic."
            ) from exc
    return tuple(result)


def matrix_text(matrix):
    return "\n".join(", ".join(repr(v) for v in row) for row in matrix)


def _validate_matrix(matrix, label, *, rows=128, limit=1e50):
    width = len(matrix[0]) if matrix else 0
    if not 1 <= len(matrix) <= rows or not 1 <= width <= 32:
        raise InputError(f"{label}: use 1–{rows} rows and 1–32 columns.")
    if any(len(row) != width for row in matrix):
        raise InputError(f"{label}: every row must have {width} values.")
    if any(not math.isfinite(v) or abs(v) > limit for row in matrix for v in row):
        raise InputError(
            f"{label}: values must be finite and at most {limit:g} in magnitude."
        )


@dataclass(frozen=True)
class Lecture08Request:
    kind: str
    s: Matrix = ()
    t: Matrix = ()
    w: Matrix = ()
    x: Matrix = ()
    bias: Matrix = ()
    expected: Matrix = ()
    bias_enabled: bool = False
    tolerance: float = 0.0
    activation: str = "binary"

    @property
    def algorithm_name(self):
        return LABELS[self.kind]

    def validate(self):
        if self.kind not in LABELS:
            raise InputError("Choose a matrix or McCulloch–Pitts tool.")
        if self.activation not in ("binary", "sign", "bipolar"):
            raise InputError("Recall activation must be binary, sign, or bipolar.")
        if type(self.bias_enabled) is not bool:
            raise InputError("Bias enabled must be a boolean.")
        if not math.isfinite(self.tolerance) or not 0 <= self.tolerance <= 1e50:
            raise InputError(
                "Orthogonality tolerance must be finite and between 0 and 1e50."
            )
        if self.kind == "matrix_product":
            _validate_matrix(self.s, "A")
            _validate_matrix(self.t, "B", rows=32)
            if len(self.s[0]) != len(self.t):
                raise InputError("Matrix product: A columns must equal B rows.")
        elif self.kind == "mcp":
            _validate_matrix(self.s, "Binary input patterns")
            _validate_matrix(self.w, "Neuron weights", rows=32)
            _validate_matrix(self.t, "Neuron thresholds", rows=1)
            n, m = len(self.s[0]), len(self.w)
            if len(self.w[0]) != n + m or len(self.t[0]) != m:
                raise InputError(
                    "For n inputs and m neurons, W needs m rows and n+m columns; thresholds need one row of m values."
                )
            if any(v not in (0, 1) for row in self.s for v in row):
                raise InputError("McCulloch–Pitts inputs must be binary: 0 or 1.")
            if any(v != 0 for j, row in enumerate(self.w) for v in row[n + j :]):
                raise InputError(
                    "Neurons must be in feed-forward order. Weights on the same or later neuron must be zero."
                )
            if self.expected:
                _validate_matrix(self.expected, "Expected neuron outputs")
                if len(self.expected) != len(self.s) or len(self.expected[0]) != m:
                    raise InputError(
                        "Expected outputs need one row per pattern and one column per neuron."
                    )
        elif self.kind in ("correlation", "orthogonality"):
            _validate_matrix(self.s, "X / S")
        elif self.kind in ("hebbian_memory", "outer_product"):
            _validate_matrix(self.s, "S")
            _validate_matrix(self.t, "T")
            if len(self.s) != len(self.t):
                raise InputError("S and T must have the same number of training rows.")
            if self.kind == "outer_product" and self.bias_enabled:
                raise InputError(
                    "Outer products have no bias. Enable separate bias in Hebbian memory instead."
                )
        else:
            _validate_matrix(self.w, "W", rows=32, limit=1e150)
            _validate_matrix(self.x, "Recall inputs X")
            if len(self.x[0]) != len(self.w):
                raise InputError(
                    "Each recall input must have as many coordinates as W has rows."
                )
            if self.bias_enabled:
                _validate_matrix(self.bias, "Bias", rows=1, limit=1e150)
                if len(self.bias[0]) != len(self.w[0]):
                    raise InputError(
                        "Bias must contain one value per output column of W."
                    )
            if self.expected:
                _validate_matrix(self.expected, "Expected targets")
                if len(self.expected) != len(self.x) or len(self.expected[0]) != len(
                    self.w[0]
                ):
                    raise InputError(
                        "Expected targets must have one row per recall input and one column per output."
                    )

    def serialize(self):
        self.validate()
        lines = [
            "LECTURE08_V1",
            self.kind,
            f"{int(self.bias_enabled)}\t{self.tolerance!r}\t{self.activation}",
        ]
        for matrix in (self.s, self.t, self.w, self.x, self.bias, self.expected):
            lines.extend(
                (
                    f"{len(matrix)}\t{len(matrix[0]) if matrix else 0}",
                    ",".join(repr(v) for row in matrix for v in row),
                )
            )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class NamedMatrix:
    name: str
    data: Matrix
    columns: int

    @property
    def rows(self):
        return len(self.data)


@dataclass(frozen=True)
class MatrixStep:
    title: str
    detail: str
    matrices: tuple[str, ...]


@dataclass(frozen=True)
class Lecture08Result:
    request: Lecture08Request
    matrices: tuple[NamedMatrix, ...]
    scalars: dict[str, float]
    notes: dict[str, str]
    steps: tuple[MatrixStep, ...]

    def matrix(self, name):
        return next(m.data for m in self.matrices if m.name == name)


def _decode(text):
    return json.loads('"' + text + '"')


def parse_output(output, request):
    lines = output.splitlines()
    if not lines or lines[0] != f"L08\t{request.kind}" or lines[-1] != "END":
        raise RuntimeError("Incomplete Lecture 08 response from Rust.")
    matrices = []
    scalars = {}
    notes = {}
    steps = []
    total = 0
    for line in lines[1:-1]:
        f = line.split("\t")
        if f[0] == "M" and len(f) == 5:
            rows, cols = int(f[2]), int(f[3])
            flat = tuple(map(float, f[4].split(","))) if f[4] else ()
            total += len(flat)
            if (
                rows < 0
                or cols < 0
                or rows * cols != len(flat)
                or total > 1_000_000
                or (rows and not cols)
            ):
                raise RuntimeError("Invalid matrix dimensions in the Rust report.")
            data = tuple(flat[i * cols : (i + 1) * cols] for i in range(rows))
            matrices.append(NamedMatrix(f[1], data, cols))
        elif f[0] == "D" and len(f) == 3:
            scalars[f[1]] = float(f[2])
        elif f[0] == "N" and len(f) == 3:
            notes[f[1]] = _decode(f[2])
        elif f[0] == "S" and len(f) == 4:
            steps.append(
                MatrixStep(
                    _decode(f[1]), _decode(f[2]), tuple(f[3].split(",")) if f[3] else ()
                )
            )
        else:
            raise RuntimeError("Unexpected Lecture 08 response from Rust.")
    names = {m.name for m in matrices}
    if (
        len(names) != len(matrices)
        or not matrices
        or not steps
        or any(name not in names for s in steps for name in s.matrices)
    ):
        raise RuntimeError(
            "The Rust report has duplicate/missing matrices or missing worked steps."
        )
    if not all(
        math.isfinite(v) for m in matrices for row in m.data for v in row
    ) or not all(map(math.isfinite, scalars.values())):
        raise RuntimeError("The Rust report contains non-finite results.")
    return Lecture08Result(request, tuple(matrices), scalars, notes, tuple(steps))


def export_matrices(path, result):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["operation", "type", "name", "row", "column", "value"])
        for m in result.matrices:
            for i, row in enumerate(m.data, 1):
                for j, value in enumerate(row, 1):
                    writer.writerow(
                        [result.request.kind, "matrix", m.name, i, j, value]
                    )
        for name, value in result.scalars.items():
            writer.writerow([result.request.kind, "scalar", name, "", "", value])


def export_report(path, result):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(
            {"format": "lecture08-v1", **asdict(result)},
            stream,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")
