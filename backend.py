"""Shared process control and ADALINE data; numerical calculations live in Rust."""

from dataclasses import dataclass
import csv
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "rust_backend" / "src"
STATUS = {
    "pass_complete": "One pass complete",
    "weights_stable": "Weight changes stopped",
    "tolerance_met": "Tolerance met",
    "epoch_limit": "Epoch limit reached",
    "numerical_limit": "Numerical limit reached",
}


class InputError(ValueError):
    """An editable worksheet input needs attention."""


class Cancelled(Exception):
    pass


def number(text, label):
    try:
        value = float(text)
    except (ValueError, TypeError) as exc:
        raise InputError(f"{label}: enter a number, such as 0.2 or 1e-3.") from exc
    if not math.isfinite(value):
        raise InputError(f"{label} must be finite.")
    return value


def vector(text, label="Weights"):
    parts = re.split(r"[,;\s]+", text.strip())
    return tuple(number(part, label) for part in parts)


@dataclass(frozen=True)
class Sample:
    inputs: tuple[float, ...]
    target: float


def parse_samples(text):
    samples = []
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[,;\s]+", line)
        if not samples and parts[-1].lower() in ("t", "target"):
            continue
        values = tuple(number(part, f"Training row {line_number}") for part in parts)
        if len(values) < 2:
            raise InputError(
                f"Training row {line_number}: provide inputs followed by a target."
            )
        samples.append(Sample(values[:-1], values[-1]))
    if not samples:
        raise InputError("Enter at least one training pattern: x1, x2, …, target.")
    return tuple(samples)


def preset_samples(name, bipolar=True):
    """Truth tables in lecture order: true/true, true/false, false/true, false/false."""
    if name not in ("AND", "OR", "NAND", "NOR", "XOR", "NOT"):
        raise InputError("Choose a truth-table preset.")
    n = 1 if name == "NOT" else 2
    rows = []
    for pattern in reversed(range(1 << n)):
        bits = tuple(bool(pattern & (1 << i)) for i in reversed(range(n)))
        truth = {
            "AND": all(bits),
            "OR": any(bits),
            "NAND": not all(bits),
            "NOR": not any(bits),
            "XOR": sum(bits) == 1,
            "NOT": not bits[0],
        }[name]
        rows.append(
            Sample(
                tuple(1.0 if b else (-1.0 if bipolar else 0.0) for b in bits),
                1.0 if truth else -1.0,
            )
        )
    return tuple(rows)


def samples_text(samples):
    return "\n".join(
        ", ".join(f"{v:g}" for v in (*s.inputs, s.target)) for s in samples
    )


@dataclass(frozen=True)
class Request:
    samples: tuple[Sample, ...]
    weights: tuple[float, ...]
    bias: float = 0.0
    alpha: float = 0.2
    tolerance: float = 0.001
    max_epochs: int = 100

    def validate(self):
        if not math.isfinite(self.alpha) or not 0 < self.alpha <= 1e50:
            raise InputError(
                "Learning rate α must be positive, finite, and at most 1e50."
            )
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise InputError("Tolerance must be positive and finite.")
        if type(self.max_epochs) is not int or not 1 <= self.max_epochs <= 100_000:
            raise InputError("Maximum epochs must be an integer between 1 and 100,000.")
        n = len(self.weights)
        if not 1 <= n <= 32:
            raise InputError("Enter 1 to 32 initial weights, one per input coordinate.")
        if not 1 <= len(self.samples) <= 128:
            raise InputError("Use 1 to 128 training patterns.")
        for sample in self.samples:
            if len(sample.inputs) != n:
                raise InputError(
                    f"Every row must have {n} input(s) and a target. Enter one initial weight per input."
                )
            if sample.target not in (-1.0, 1.0):
                raise InputError(
                    "Targets must be bipolar: -1 or +1, as in the supplied algorithm."
                )
        values = (
            *self.weights,
            self.bias,
            *(v for s in self.samples for v in s.inputs),
        )
        if any(not math.isfinite(v) or abs(v) > 1e50 for v in values):
            raise InputError(
                "Inputs, initial weights and bias must be finite and at most 1e50 in magnitude."
            )
        steps = len(self.samples) * self.max_epochs
        if steps > 100_000 or steps * n > 1_000_000:
            raise InputError(
                "Reduce maximum epochs: the full trace supports 100,000 updates and 1,000,000 input coordinates."
            )

    def serialize(self):
        self.validate()
        weights = ",".join(repr(v) for v in self.weights)
        rows = "\n".join(
            ",".join(repr(v) for v in (*s.inputs, s.target)) for s in self.samples
        )
        return f"ADALINE1\n{self.alpha!r}\t{self.tolerance!r}\t{self.max_epochs}\n{weights}\n{self.bias!r}\n{rows}\n"


@dataclass(frozen=True)
class Step:
    epoch: int
    pattern: int
    net: float
    output: int
    error: float
    delta_bias: float
    before_bias: float
    after_bias: float
    max_change: float
    before_weights: tuple[float, ...]
    delta_weights: tuple[float, ...]
    after_weights: tuple[float, ...]


@dataclass(frozen=True)
class Epoch:
    number: int
    max_change: float
    net_change: float
    sse: float
    correct: int
    weights: tuple[float, ...]
    bias: float


@dataclass(frozen=True)
class Prediction:
    pattern: int
    net: float
    output: int


@dataclass(frozen=True)
class Result:
    request: Request
    status: str
    completed_epochs: int
    failed_at: tuple[int, int] | None
    steps: tuple[Step, ...]
    epochs: tuple[Epoch, ...]
    predictions: tuple[Prediction, ...]
    weights: tuple[float, ...]
    bias: float
    sse: float
    correct: int


def _vec(text):
    return tuple(float(v) for v in text.split(","))


def parse_output(output, request):
    from .perceptron import PerceptronRequest, parse_output as parse_perceptron

    if isinstance(request, PerceptronRequest):
        return parse_perceptron(output, request)
    from .hebbian import HebbianRequest, parse_output as parse_hebbian

    if isinstance(request, HebbianRequest):
        return parse_hebbian(output, request)
    if not isinstance(request, Request):
        from .madaline import parse_output as parse_madaline

        return parse_madaline(output, request)
    steps, epochs, predictions = [], [], []
    meta = model = None
    for line in output.splitlines():
        f = line.split("\t")
        if f[0] == "R" and len(f) == 5:
            meta = f
        elif f[0] == "S" and len(f) == 13:
            steps.append(
                Step(
                    int(f[1]),
                    int(f[2]),
                    float(f[3]),
                    int(f[4]),
                    *map(float, f[5:10]),
                    *map(_vec, f[10:13]),
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
            raise RuntimeError("Unexpected response from the Rust backend.")
    if (
        meta is None
        or model is None
        or meta[1] not in STATUS
        or len(predictions) != len(request.samples)
    ):
        raise RuntimeError("Incomplete response from the Rust backend.")
    return Result(
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


def _process(command, cancel, payload=None, timeout=120):
    if cancel.is_set():
        raise Cancelled()
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        deadline = time.monotonic() + timeout
        first = True
        try:
            while True:
                if cancel.is_set():
                    raise Cancelled()
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Rust exceeded the two-minute time limit. Reduce the epoch limit or dataset."
                    )
                data = payload if first else None
                first = False
                try:
                    stdout, stderr = process.communicate(input=data, timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
        if process.returncode:
            raise RuntimeError(f"Rust backend failed:\n{stderr.strip()[:3000]}")
        return stdout


def train(request, *, cancel=None, progress=None):
    request.validate()
    cancel = cancel or threading.Event()
    compiler = shutil.which("rustc")
    if compiler is None:
        raise RuntimeError(
            "Rust is required. Install the Rust toolchain and put rustc on PATH."
        )
    version = _process([compiler, "--version"], cancel)
    digest = hashlib.sha256(version.encode())
    for source in sorted(SOURCE.glob("*.rs")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    cache = ROOT / ".cache" / "rust"
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / (digest.hexdigest()[:24] + (".exe" if os.name == "nt" else ""))
    if not binary.is_file():
        if progress:
            progress("Compiling the Rust backend…")
        fd, temporary = tempfile.mkstemp(
            prefix="build_", suffix=".exe" if os.name == "nt" else "", dir=cache
        )
        os.close(fd)
        try:
            _process(
                [
                    compiler,
                    "--edition=2021",
                    "-O",
                    str(SOURCE / "main.rs"),
                    "-o",
                    temporary,
                ],
                cancel,
            )
            Path(temporary).replace(binary)
        finally:
            Path(temporary).unlink(missing_ok=True)
    if progress:
        name = getattr(
            request,
            "algorithm_name",
            "ADALINE" if isinstance(request, Request) else "MADALINE MRI",
        )
        progress(f"Training {name} in Rust…")
    output = _process([str(binary)], cancel, payload=request.serialize())
    if cancel.is_set():
        raise Cancelled()
    return parse_output(output, request)


def export_trace(path, result):
    """Export full precision, with the immutable run inputs in each row."""
    from .perceptron import PerceptronRequest, export_trace as export_perceptron

    if isinstance(result.request, PerceptronRequest):
        return export_perceptron(path, result)
    from .hebbian import HebbianRequest, export_trace as export_hebbian

    if isinstance(result.request, HebbianRequest):
        return export_hebbian(path, result)
    if not isinstance(result.request, Request):
        from .madaline import export_trace as export_madaline

        return export_madaline(path, result)
    r = result.request
    n = len(r.weights)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "status",
                "alpha",
                "tolerance",
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
                    result.status,
                    r.alpha,
                    r.tolerance,
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
                    *s.delta_weights,
                    s.delta_bias,
                    *s.after_weights,
                    s.after_bias,
                    s.max_change,
                ]
            )


def export_epochs(path, result):
    from .perceptron import PerceptronRequest, export_epochs as export_perceptron

    if isinstance(result.request, PerceptronRequest):
        return export_perceptron(path, result)
    from .hebbian import HebbianRequest, export_epochs as export_hebbian

    if isinstance(result.request, HebbianRequest):
        return export_hebbian(path, result)
    if not isinstance(result.request, Request):
        from .madaline import export_epochs as export_madaline

        return export_madaline(path, result)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "status",
                "alpha",
                "tolerance",
                "epoch",
                "max_individual_change",
                "net_epoch_change",
                "sse",
                "correct",
                "patterns",
                *[f"w{i + 1}" for i in range(len(result.weights))],
                "bias",
            ]
        )
        for e in result.epochs:
            writer.writerow(
                [
                    result.status,
                    result.request.alpha,
                    result.request.tolerance,
                    e.number,
                    e.max_change,
                    e.net_change,
                    e.sse,
                    e.correct,
                    len(result.request.samples),
                    *e.weights,
                    e.bias,
                ]
            )
