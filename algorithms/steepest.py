import csv
from dataclasses import dataclass
import math
from pathlib import Path

from ..backend import InputError
from .optimization import GradientRequest, compile_expression

GRID_SIZE = 49
PRESETS = {
    "Elliptic bowl · minimum (2, −1)": {
        "expression": "(x - 2)^2 + 2*(y + 1)^2",
        "x0": "6",
        "y0": "3",
        "alpha": "0.1",
        "plot_min": "-2",
        "plot_max": "7",
        "plot_y_min": "-3",
        "plot_y_max": "4",
    },
    "Circular bowl · minimum (0, 0)": {
        "expression": "x^2 + y^2",
        "x0": "4",
        "y0": "3",
        "alpha": "0.1",
        "plot_min": "-5",
        "plot_max": "5",
        "plot_y_min": "-4",
        "plot_y_max": "4",
    },
    "Coupled quadratic · minimum (0, 0)": {
        "expression": "x^2 + x*y + y^2",
        "x0": "2",
        "y0": "1",
        "alpha": "0.1",
        "plot_min": "-3",
        "plot_max": "3",
        "plot_y_min": "-3",
        "plot_y_max": "3",
    },
    "Rosenbrock · minimum (1, 1)": {
        "expression": "(1-x)^2 + 100*(y-x^2)^2",
        "x0": "-1.2",
        "y0": "1",
        "alpha": "0.001",
        "plot_min": "-2",
        "plot_max": "2",
        "plot_y_min": "-1",
        "plot_y_max": "3",
        "max_iterations": "20000",
        "tolerance": "1e-5",
    },
}


PRESETS.update(
    {
        "04 · x²+2y² from (2,2)": dict(
            expression="x^2+2*y^2",
            x0="2",
            y0="2",
            alpha="0.1",
            plot_min="-1",
            plot_max="3",
            plot_y_min="-1",
            plot_y_max="3",
        ),
        "CV5 · fixed gradient from (0,0)": dict(
            expression="2*x^2+x*y+3*y^2-4*x+2*y",
            x0="0",
            y0="0",
            alpha="0.125",
            plot_min="-0.5",
            plot_max="2",
            plot_y_min="-1",
            plot_y_max="0.5",
        ),
    }
)


@dataclass(frozen=True)
class SteepestRequest:
    expression: str = "(x - 2)^2 + 2*(y + 1)^2"
    x0: float = 6.0
    y0: float = 3.0
    alpha: float = 0.1
    tolerance: float = 1e-6
    max_iterations: int = 200
    plot_min: float = -2.0
    plot_max: float = 7.0
    plot_y_min: float = -3.0
    plot_y_max: float = 4.0

    algorithm_name = "2D Steepest Gradient Descent"

    def validate(self):
        compile_expression(self.expression, dimensions=2)
        # Reuse scalar/range/workload checks, while validating f with both variables.
        GradientRequest(
            "x",
            self.x0,
            self.alpha,
            self.tolerance,
            self.max_iterations,
            self.plot_min,
            self.plot_max,
        ).validate()
        if any(
            not math.isfinite(v) or abs(v) > 1e50
            for v in (self.y0, self.plot_y_min, self.plot_y_max)
        ):
            raise InputError(
                "Starting y and plot y bounds must be finite and at most 1e50 in magnitude."
            )
        if self.plot_y_min >= self.plot_y_max:
            raise InputError("Plot minimum y must be smaller than plot maximum y.")

    def serialize(self):
        self.validate()
        options = (
            self.x0,
            self.y0,
            self.alpha,
            self.tolerance,
            self.max_iterations,
            self.plot_min,
            self.plot_max,
            self.plot_y_min,
            self.plot_y_max,
        )
        return (
            "STEEPEST2_V1\n"
            + "\t".join(map(repr, options))
            + "\n"
            + compile_expression(self.expression, dimensions=2)
            + "\n"
        )


@dataclass(frozen=True)
class SteepestPoint:
    iteration: int
    x: float
    y: float
    value: float
    gradient_x: float
    gradient_y: float
    gradient_norm: float


@dataclass(frozen=True)
class SteepestResult:
    request: SteepestRequest
    status: str
    points: tuple[SteepestPoint, ...]
    grid: tuple[tuple[float | None, ...], ...]

    @property
    def final(self):
        return self.points[-1]


def parse_output(output, request):
    status = None
    points, samples = [], []
    try:
        for line in output.splitlines():
            fields = line.split("\t")
            if fields[0] == "G2R" and len(fields) == 2 and status is None:
                status = fields[1]
            elif fields[0] == "G2P" and len(fields) == 8:
                points.append(SteepestPoint(int(fields[1]), *map(float, fields[2:])))
            elif (
                fields[0] == "G2C"
                and len(fields) == 3
                and int(fields[1]) == len(samples)
            ):
                samples.append(None if fields[2] == "_" else float(fields[2]))
            else:
                raise ValueError
        if (
            status
            not in {
                "gradient_tolerance",
                "iteration_limit",
                "numerical_limit",
                "stagnation",
            }
            or not points
            or len(points) > request.max_iterations + 1
            or len(samples) != GRID_SIZE**2
            or any(
                p.iteration != i
                or p.gradient_norm < 0
                or not all(
                    math.isfinite(v)
                    for v in (
                        p.x,
                        p.y,
                        p.value,
                        p.gradient_x,
                        p.gradient_y,
                        p.gradient_norm,
                    )
                )
                for i, p in enumerate(points)
            )
            or any(v is not None and not math.isfinite(v) for v in samples)
        ):
            raise ValueError
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Invalid 2D steepest-descent response from Rust.") from exc
    grid = tuple(
        tuple(samples[i : i + GRID_SIZE]) for i in range(0, len(samples), GRID_SIZE)
    )
    return SteepestResult(request, status, tuple(points), grid)


def export_trace(path, result):
    r = result.request
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "algorithm",
                "status",
                "function",
                "initial_x",
                "initial_y",
                "alpha",
                "gradient_tolerance",
                "iteration_limit",
                "plot_x_min",
                "plot_x_max",
                "plot_y_min",
                "plot_y_max",
                "k",
                "x_k",
                "y_k",
                "f_x_y",
                "partial_x",
                "partial_y",
                "gradient_norm",
                "delta_x",
                "delta_y",
                "x_next",
                "y_next",
            ]
        )
        for i, p in enumerate(result.points):
            nxt = result.points[i + 1] if i + 1 < len(result.points) else None
            writer.writerow(
                [
                    "STEEPEST2D",
                    result.status,
                    r.expression,
                    r.x0,
                    r.y0,
                    r.alpha,
                    r.tolerance,
                    r.max_iterations,
                    r.plot_min,
                    r.plot_max,
                    r.plot_y_min,
                    r.plot_y_max,
                    p.iteration,
                    p.x,
                    p.y,
                    p.value,
                    p.gradient_x,
                    p.gradient_y,
                    p.gradient_norm,
                    -r.alpha * p.gradient_x if nxt else "",
                    -r.alpha * p.gradient_y if nxt else "",
                    nxt.x if nxt else "",
                    nxt.y if nxt else "",
                ]
            )
