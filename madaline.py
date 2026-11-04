"""MADALINE Rule I run data and protocol. Numerical operations execute in Rust."""

import csv
from dataclasses import dataclass
import math
from pathlib import Path

from .backend import InputError, Request, Sample, STATUS, _vec

MAX_HIDDEN_UNITS = 32


@dataclass(frozen=True)
class MadalineRequest:
    samples: tuple[Sample, ...]
    hidden_weights: tuple[tuple[float, ...], ...]
    hidden_biases: tuple[float, ...] | None = None
    output_weights: tuple[float, ...] | None = None
    output_bias: float | None = None
    alpha: float = 0.2
    tolerance: float = 0.001
    max_epochs: int = 100

    def __post_init__(self):
        units = len(self.hidden_weights)
        if self.hidden_biases is None:
            object.__setattr__(self, "hidden_biases", (0.0,) * units)
        if self.output_weights is None:
            object.__setattr__(self, "output_weights", (0.5,) * units)
        if self.output_bias is None:
            object.__setattr__(self, "output_bias", (units - 1) * 0.5)

    def validate(self):
        units = len(self.hidden_weights)
        if not 1 <= units <= MAX_HIDDEN_UNITS:
            raise InputError(f"Use 1 to {MAX_HIDDEN_UNITS} hidden ADALINE units.")
        if len(self.hidden_biases) != units:
            raise InputError("Provide one bias per hidden unit.")
        for weights, bias in zip(self.hidden_weights, self.hidden_biases):
            Request(
                self.samples, weights, bias, self.alpha, self.tolerance, self.max_epochs
            ).validate()
        if len(self.output_weights) != units:
            raise InputError("Enter one fixed output weight per hidden unit.")
        if any(
            not math.isfinite(v) or abs(v) > 1e50
            for v in (*self.output_weights, self.output_bias)
        ):
            raise InputError(
                "Fixed output weights and bias must be finite and at most 1e50 in magnitude."
            )
        if (
            len(self.samples) * self.max_epochs * len(self.hidden_weights[0]) * units
            > 1_000_000
        ):
            raise InputError(
                "Reduce maximum epochs: the full hidden-unit trace supports 1,000,000 input coordinates."
            )

    def serialize(self):
        self.validate()
        lines = [
            "MADALINE2",
            f"{self.alpha!r}\t{self.tolerance!r}\t{self.max_epochs}",
            str(len(self.hidden_weights)),
        ]
        for weights, bias in zip(self.hidden_weights, self.hidden_biases):
            lines.extend((",".join(map(repr, weights)), repr(bias)))
        lines.extend((",".join(map(repr, self.output_weights)), repr(self.output_bias)))
        lines.extend(",".join(map(repr, (*s.inputs, s.target))) for s in self.samples)
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class MadalineStep:
    epoch: int
    pattern: int
    nets: tuple[float, ...]
    hidden_outputs: tuple[int, ...]
    net: float
    output: int
    rule: str
    selected: tuple[bool, ...]
    errors: tuple[float, ...]
    before_weights: tuple[float, ...]
    before_biases: tuple[float, ...]
    delta_weights: tuple[float, ...]
    delta_biases: tuple[float, ...]
    after_weights: tuple[float, ...]
    after_biases: tuple[float, ...]
    max_change: float


@dataclass(frozen=True)
class MadalineEpoch:
    number: int
    max_change: float
    net_change: float
    sse: float
    correct: int
    weights: tuple[float, ...]
    biases: tuple[float, ...]


@dataclass(frozen=True)
class MadalinePrediction:
    pattern: int
    nets: tuple[float, ...]
    hidden_outputs: tuple[int, ...]
    net: float
    output: int


@dataclass(frozen=True)
class MadalineResult:
    request: MadalineRequest
    status: str
    completed_epochs: int
    failed_at: tuple[int, int] | None
    steps: tuple[MadalineStep, ...]
    epochs: tuple[MadalineEpoch, ...]
    predictions: tuple[MadalinePrediction, ...]
    weights: tuple[float, ...]
    biases: tuple[float, ...]
    sse: float
    correct: int


def parse_output(output, request):
    steps, epochs, predictions = [], [], []
    meta = model = None
    for line in output.splitlines():
        f = line.split("\t")
        if f[0] == "R" and len(f) == 5:
            meta = f
        elif f[0] == "MS" and len(f) == 17:
            steps.append(
                MadalineStep(
                    int(f[1]),
                    int(f[2]),
                    _vec(f[3]),
                    tuple(map(int, f[4].split(","))),
                    float(f[5]),
                    int(f[6]),
                    f[7],
                    tuple(bool(int(v)) for v in f[8].split(",")),
                    *map(_vec, f[9:16]),
                    float(f[16]),
                )
            )
        elif f[0] == "ME" and len(f) == 8:
            epochs.append(
                MadalineEpoch(
                    int(f[1]), *map(float, f[2:5]), int(f[5]), _vec(f[6]), _vec(f[7])
                )
            )
        elif f[0] == "MP" and len(f) == 6:
            predictions.append(
                MadalinePrediction(
                    int(f[1]),
                    _vec(f[2]),
                    tuple(map(int, f[3].split(","))),
                    float(f[4]),
                    int(f[5]),
                )
            )
        elif f[0] == "MM" and len(f) == 5:
            model = f
        else:
            raise RuntimeError("Unexpected MADALINE response from Rust.")
    if (
        meta is None
        or model is None
        or meta[1] not in STATUS
        or len(predictions) != len(request.samples)
    ):
        raise RuntimeError("Incomplete MADALINE response from Rust.")
    return MadalineResult(
        request,
        meta[1],
        int(meta[2]),
        (int(meta[3]), int(meta[4])) if int(meta[3]) else None,
        tuple(steps),
        tuple(epochs),
        tuple(predictions),
        _vec(model[1]),
        _vec(model[2]),
        float(model[3]),
        int(model[4]),
    )


def weight_names(n, units=2):
    """w_ij: i is the input coordinate, j is the destination hidden unit."""
    return [
        f"w{i + 1}{j + 1}" if n < 10 and units < 10 else f"w{i + 1}_{j + 1}"
        for j in range(units)
        for i in range(n)
    ]


def export_trace(path, result):
    r = result.request
    n = len(r.hidden_weights[0])
    units = len(r.hidden_weights)
    names = weight_names(n, units)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "algorithm",
                "status",
                "alpha",
                "tolerance",
                "epoch_limit",
                "epoch",
                "pattern",
                "epoch_complete",
                "hidden_units",
                *[f"x{i + 1}" for i in range(n)],
                "target",
                *[f"v{j + 1}_fixed" for j in range(units)],
                "b_out_fixed",
                *[f"initial_{name}" for name in names],
                *[f"initial_b{j + 1}" for j in range(units)],
                *[f"z_in{j + 1}" for j in range(units)],
                *[f"z{j + 1}" for j in range(units)],
                "y_in",
                "y",
                "rule",
                *[f"selected_z{j + 1}" for j in range(units)],
                *[f"error_z{j + 1}" for j in range(units)],
                *[name + "_before" for name in names],
                *[f"b{j + 1}_before" for j in range(units)],
                *["delta_" + name for name in names],
                *[f"delta_b{j + 1}" for j in range(units)],
                *[name + "_after" for name in names],
                *[f"b{j + 1}_after" for j in range(units)],
                "max_individual_change",
            ]
        )
        for s in result.steps:
            sample = r.samples[s.pattern - 1]
            writer.writerow(
                [
                    "MADALINE_MRI",
                    result.status,
                    r.alpha,
                    r.tolerance,
                    r.max_epochs,
                    s.epoch,
                    s.pattern,
                    s.epoch <= result.completed_epochs,
                    units,
                    *sample.inputs,
                    sample.target,
                    *r.output_weights,
                    r.output_bias,
                    *(w for unit in r.hidden_weights for w in unit),
                    *r.hidden_biases,
                    *s.nets,
                    *s.hidden_outputs,
                    s.net,
                    s.output,
                    s.rule,
                    *s.selected,
                    *s.errors,
                    *s.before_weights,
                    *s.before_biases,
                    *s.delta_weights,
                    *s.delta_biases,
                    *s.after_weights,
                    *s.after_biases,
                    s.max_change,
                ]
            )


def export_epochs(path, result):
    r = result.request
    units = len(r.hidden_weights)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "algorithm",
                "status",
                "alpha",
                "tolerance",
                "epoch",
                "max_individual_change",
                "net_epoch_change",
                "classification_sse",
                "correct",
                "patterns",
                "hidden_units",
                *weight_names(len(r.hidden_weights[0]), units),
                *[f"b{j + 1}" for j in range(units)],
                *[f"v{j + 1}_fixed" for j in range(units)],
                "b_out_fixed",
            ]
        )
        for e in result.epochs:
            writer.writerow(
                [
                    "MADALINE_MRI",
                    result.status,
                    r.alpha,
                    r.tolerance,
                    e.number,
                    e.max_change,
                    e.net_change,
                    e.sse,
                    e.correct,
                    len(r.samples),
                    units,
                    *e.weights,
                    *e.biases,
                    *r.output_weights,
                    r.output_bias,
                ]
            )
