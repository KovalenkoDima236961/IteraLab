"""Hebbian worksheet data and I/O. The single training pass executes in Rust."""

import csv
from dataclasses import dataclass
from pathlib import Path

from ..backend import Epoch, InputError, Prediction, Request, Sample, _vec


@dataclass(frozen=True)
class HebbianRequest:
    samples: tuple[Sample, ...]
    eta: float = 0.2

    algorithm_name = "Hebbian Learning"

    @property
    def max_epochs(self):
        return 1

    def validate(self):
        if not self.samples:
            raise InputError("Enter at least one training pattern.")
        n = len(self.samples[0].inputs)
        Request(self.samples, (0.0,) * n, alpha=self.eta, max_epochs=1).validate()

    def serialize(self):
        self.validate()
        rows = "\n".join(
            ",".join(map(repr, (*s.inputs, s.target))) for s in self.samples
        )
        return f"HEBBIAN1\n{self.eta!r}\n{rows}\n"


@dataclass(frozen=True)
class HebbianStep:
    epoch: int
    pattern: int
    delta_bias: float
    before_bias: float
    after_bias: float
    max_change: float
    before_weights: tuple[float, ...]
    delta_weights: tuple[float, ...]
    after_weights: tuple[float, ...]


@dataclass(frozen=True)
class HebbianResult:
    request: HebbianRequest
    status: str
    completed_epochs: int
    failed_at: tuple[int, int] | None
    steps: tuple[HebbianStep, ...]
    epochs: tuple[Epoch, ...]
    predictions: tuple[Prediction, ...]
    weights: tuple[float, ...]
    bias: float
    sse: float
    correct: int


def parse_output(output, request):
    steps, epochs, predictions = [], [], []
    meta = model = None
    for line in output.splitlines():
        f = line.split("\t")
        if f[0] == "R" and len(f) == 5:
            meta = f
        elif f[0] == "HS" and len(f) == 10:
            steps.append(
                HebbianStep(
                    int(f[1]), int(f[2]), *map(float, f[3:7]), *map(_vec, f[7:10])
                )
            )
        elif f[0] == "E" and len(f) == 8:
            epochs.append(
                Epoch(
                    int(f[1]), *map(float, f[2:5]), int(f[5]), _vec(f[6]), float(f[7])
                )
            )
        elif f[0] == "P" and len(f) == 4:
            predictions.append(Prediction(int(f[1]), float(f[2]), int(f[3])))
        elif f[0] == "M" and len(f) == 5:
            model = f
        else:
            raise RuntimeError("Unexpected Hebbian response from Rust.")
    if (
        meta is None
        or model is None
        or meta[1] not in ("pass_complete", "numerical_limit")
        or len(predictions) != len(request.samples)
    ):
        raise RuntimeError("Incomplete Hebbian response from Rust.")
    return HebbianResult(
        request,
        meta[1],
        int(meta[2]),
        (int(meta[3]), int(meta[4])) if int(meta[3]) else None,
        tuple(steps),
        tuple(epochs),
        tuple(predictions),
        _vec(model[1]),
        float(model[2]),
        float(model[3]),
        int(model[4]),
    )


def export_trace(path, result):
    r = result.request
    n = len(result.weights)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "algorithm",
                "status",
                "eta",
                "pass",
                "pattern",
                "pass_complete",
                *[f"x{i + 1}" for i in range(n)],
                "target",
                *[f"initial_w{i + 1}" for i in range(n)],
                "initial_bias",
                *[f"w{i + 1}_before" for i in range(n)],
                "bias_before",
                *[f"delta_w{i + 1}" for i in range(n)],
                "delta_bias",
                *[f"w{i + 1}_after" for i in range(n)],
                "bias_after",
                "max_individual_change",
            ]
        )
        for s in result.steps:
            sample = r.samples[s.pattern - 1]
            writer.writerow(
                [
                    "HEBBIAN",
                    result.status,
                    r.eta,
                    1,
                    s.pattern,
                    result.completed_epochs == 1,
                    *sample.inputs,
                    sample.target,
                    *([0.0] * n),
                    0.0,
                    *s.before_weights,
                    s.before_bias,
                    *s.delta_weights,
                    s.delta_bias,
                    *s.after_weights,
                    s.after_bias,
                    s.max_change,
                ]
            )


def export_epochs(path, result):
    """The one-pass endpoint; an interrupted pass has no completed endpoint row."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "algorithm",
                "status",
                "eta",
                "pass",
                "max_individual_change",
                "net_pass_change",
                "classification_sse",
                "correct",
                "patterns",
                *[f"w{i + 1}" for i in range(len(result.weights))],
                "bias",
            ]
        )
        for e in result.epochs:
            writer.writerow(
                [
                    "HEBBIAN",
                    result.status,
                    result.request.eta,
                    1,
                    e.max_change,
                    e.net_change,
                    e.sse,
                    e.correct,
                    len(result.request.samples),
                    *e.weights,
                    e.bias,
                ]
            )
