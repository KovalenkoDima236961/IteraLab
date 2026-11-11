import csv
from dataclasses import dataclass
import math
from pathlib import Path

from ..backend import (
    Epoch,
    InputError,
    Prediction,
    Request,
    Sample,
    Step,
    _vec,
    preset_samples,
)

BINARY = "Binary (0/1)"
BIPOLAR = "Bipolar (−1/+1)"
BIPOLAR_ERROR = "Bipolar · error update"
BAND = "Threshold band (−1/0/+1)"
MODE_LABELS = {
    "binary": BINARY,
    "bipolar": BIPOLAR,
    "bipolar_error": BIPOLAR_ERROR,
    "band": BAND,
}


def perceptron_samples(name, bipolar_inputs=False, mode="binary"):
    samples = preset_samples(name, bipolar_inputs)
    if mode == "binary":
        return tuple(Sample(s.inputs, 1.0 if s.target == 1.0 else 0.0) for s in samples)
    if mode in ("bipolar", "bipolar_error", "band"):
        return samples
    raise InputError(
        "Choose binary, bipolar sign, bipolar error-update or threshold-band mode."
    )


@dataclass(frozen=True)
class PerceptronRequest:
    samples: tuple[Sample, ...]
    weights: tuple[float, ...]
    bias: float = 0.0
    alpha: float = 1.0
    max_epochs: int = 100
    mode: str = "binary"
    theta: float = 0.0

    algorithm_name = "Perceptron"

    def validate(self):
        if self.mode not in MODE_LABELS:
            raise InputError(
                "Choose binary, bipolar sign, bipolar error-update or threshold-band mode."
            )
        if not math.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise InputError("Perceptron learning rate must satisfy 0 < rate ≤ 1.")
        if not math.isfinite(self.theta) or not 0 <= self.theta <= 1e50:
            raise InputError("Theta must be finite and between 0 and 1e50.")
        if self.mode != "band" and self.theta != 0:
            raise InputError(
                "Binary and bipolar sign modes use threshold zero; set theta to 0."
            )
        targets = (0.0, 1.0) if self.mode == "binary" else (-1.0, 1.0)
        if any(s.target not in targets for s in self.samples):
            raise InputError(
                "Binary perceptron targets must be 0 or 1."
                if self.mode == "binary"
                else "Bipolar perceptron targets must be −1 or +1."
                if self.mode in ("bipolar", "bipolar_error")
                else "Threshold-band targets must be −1 or +1."
            )
        canonical = tuple(
            Sample(s.inputs, -1.0 if s.target == 0.0 else s.target)
            for s in self.samples
        )
        Request(
            canonical, self.weights, self.bias, self.alpha, 1.0, self.max_epochs
        ).validate()

    def serialize(self):
        self.validate()
        rows = "\n".join(
            ",".join(map(repr, (*s.inputs, s.target))) for s in self.samples
        )
        return f"PERCEPTRON1\n{self.mode}\n{self.alpha!r}\t{self.theta!r}\t{self.max_epochs}\n{','.join(map(repr, self.weights))}\n{self.bias!r}\n{rows}\n"


@dataclass(frozen=True)
class PerceptronStep(Step):
    changed: bool


@dataclass(frozen=True)
class PerceptronEpoch(Epoch):
    mistakes: int
    updates: int


@dataclass(frozen=True)
class PerceptronResult:
    request: PerceptronRequest
    status: str
    completed_epochs: int
    failed_at: tuple[int, int] | None
    steps: tuple[PerceptronStep, ...]
    epochs: tuple[PerceptronEpoch, ...]
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
        elif f[0] == "PS" and len(f) == 14:
            steps.append(
                PerceptronStep(
                    int(f[1]),
                    int(f[2]),
                    float(f[3]),
                    int(f[4]),
                    *map(float, f[5:10]),
                    *map(_vec, f[10:13]),
                    bool(int(f[13])),
                )
            )
        elif f[0] == "PE" and len(f) == 10:
            epochs.append(
                PerceptronEpoch(
                    int(f[1]),
                    *map(float, f[2:5]),
                    int(f[5]),
                    _vec(f[6]),
                    float(f[7]),
                    int(f[8]),
                    int(f[9]),
                )
            )
        elif f[0] == "P" and len(f) == 4:
            predictions.append(Prediction(int(f[1]), float(f[2]), int(f[3])))
        elif f[0] == "M" and len(f) == 5:
            model = f
        else:
            raise RuntimeError("Unexpected perceptron response from Rust.")
    if (
        meta is None
        or model is None
        or meta[1] not in ("weights_stable", "epoch_limit", "numerical_limit")
        or len(predictions) != len(request.samples)
    ):
        raise RuntimeError("Incomplete perceptron response from Rust.")
    return PerceptronResult(
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
    n = len(r.weights)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "algorithm",
                "mode",
                "status",
                "learning_rate",
                "theta",
                "epoch_limit",
                "epoch",
                "pattern",
                "epoch_complete",
                *[f"x{i + 1}" for i in range(n)],
                "target",
                *[f"initial_w{i + 1}" for i in range(n)],
                "initial_bias",
                *[f"w{i + 1}_before" for i in range(n)],
                "bias_before",
                "y_in",
                "y",
                "error",
                "mistake",
                "weights_changed",
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
                    "PERCEPTRON",
                    r.mode,
                    result.status,
                    r.alpha,
                    r.theta,
                    r.max_epochs,
                    s.epoch,
                    s.pattern,
                    s.epoch <= result.completed_epochs,
                    *sample.inputs,
                    sample.target,
                    *r.weights,
                    r.bias,
                    *s.before_weights,
                    s.before_bias,
                    s.net,
                    s.output,
                    s.error,
                    s.error != 0,
                    s.changed,
                    *s.delta_weights,
                    s.delta_bias,
                    *s.after_weights,
                    s.after_bias,
                    s.max_change,
                ]
            )


def export_epochs(path, result):
    r = result.request
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "algorithm",
                "mode",
                "status",
                "learning_rate",
                "theta",
                "epoch_limit",
                "epoch",
                "mistakes",
                "updates",
                "max_individual_change",
                "net_epoch_change",
                "classification_sse",
                "correct",
                "patterns",
                *[f"w{i + 1}" for i in range(len(r.weights))],
                "bias",
            ]
        )
        for e in result.epochs:
            writer.writerow(
                [
                    "PERCEPTRON",
                    r.mode,
                    result.status,
                    r.alpha,
                    r.theta,
                    r.max_epochs,
                    e.number,
                    e.mistakes,
                    e.updates,
                    e.max_change,
                    e.net_change,
                    e.sse,
                    e.correct,
                    len(r.samples),
                    *e.weights,
                    e.bias,
                ]
            )
