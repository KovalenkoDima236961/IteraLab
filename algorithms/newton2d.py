import csv
from dataclasses import dataclass
import math
from pathlib import Path

from .optimization import compile_expression
from .steepest import GRID_SIZE, SteepestRequest, PRESETS as STEEPEST_PRESETS

PRESETS = {
    "Coupled bowl · minimum (2, −1)": {
        "expression": "(x-2)^2 + (x-2)*(y+1) + 2*(y+1)^2",
        "x0": "6",
        "y0": "3",
        "plot_min": "-2",
        "plot_max": "7",
        "plot_y_min": "-3",
        "plot_y_max": "4",
    },
    **{
        name: {
            key: value
            for key, value in preset.items()
            if key not in ("alpha", "max_iterations", "tolerance")
        }
        for name, preset in STEEPEST_PRESETS.items()
        if not name.startswith("Elliptic")
    },
}


PRESETS.update(
    {
        "04 · Newton x²+2y² from (2,2)": dict(
            expression="x^2+2*y^2",
            x0="2",
            y0="2",
            plot_min="-1",
            plot_max="3",
            plot_y_min="-1",
            plot_y_max="3",
        ),
        "CV4 · quartic bowl from (1,1)": dict(
            expression="x^4/4+x^2/2+y^2/2",
            x0="1",
            y0="1",
            plot_min="-1",
            plot_max="2",
            plot_y_min="-1",
            plot_y_max="2",
        ),
    }
)


@dataclass(frozen=True)
class Newton2DRequest:
    expression: str = "(x-2)^2 + (x-2)*(y+1) + 2*(y+1)^2"
    x0: float = 6.0
    y0: float = 3.0
    tolerance: float = 1e-6
    max_iterations: int = 200
    plot_min: float = -2.0
    plot_max: float = 7.0
    plot_y_min: float = -3.0
    plot_y_max: float = 4.0

    algorithm_name = "2D Newton's Method"

    def validate(self):
        SteepestRequest(
            expression=self.expression,
            x0=self.x0,
            y0=self.y0,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            plot_min=self.plot_min,
            plot_max=self.plot_max,
            plot_y_min=self.plot_y_min,
            plot_y_max=self.plot_y_max,
        ).validate()

    def serialize(self):
        self.validate()
        options = (
            self.x0,
            self.y0,
            self.tolerance,
            self.max_iterations,
            self.plot_min,
            self.plot_max,
            self.plot_y_min,
            self.plot_y_max,
        )
        return (
            "NEWTON2_V1\n"
            + "\t".join(map(repr, options))
            + "\n"
            + compile_expression(self.expression, dimensions=2)
            + "\n"
        )


@dataclass(frozen=True)
class Newton2DPoint:
    iteration: int
    x: float
    y: float
    value: float
    gradient_x: float
    gradient_y: float
    gradient_norm: float
    hessian_xx: float
    hessian_xy: float
    hessian_yy: float
    hessian_scale: float
    determinant_scaled: float
    delta_x: float | None
    delta_y: float | None

    @property
    def hessian(self):
        return ((self.hessian_xx, self.hessian_xy), (self.hessian_xy, self.hessian_yy))


@dataclass(frozen=True)
class Newton2DResult:
    request: Newton2DRequest
    status: str
    points: tuple[Newton2DPoint, ...]
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
            if fields[0] == "N2R" and len(fields) == 2 and status is None:
                status = fields[1]
            elif fields[0] == "N2P" and len(fields) == 15:
                deltas = [
                    None if value == "_" else float(value) for value in fields[13:]
                ]
                points.append(
                    Newton2DPoint(int(fields[1]), *map(float, fields[2:13]), *deltas)
                )
            elif (
                fields[0] == "N2C"
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
                "singular_hessian",
            }
            or not points
            or len(points) > request.max_iterations + 1
            or len(samples) != GRID_SIZE**2
            or any(
                p.iteration != i
                or p.gradient_norm < 0
                or p.hessian_scale < 0
                or not all(
                    math.isfinite(v)
                    for v in (
                        p.x,
                        p.y,
                        p.value,
                        p.gradient_x,
                        p.gradient_y,
                        p.gradient_norm,
                        p.hessian_xx,
                        p.hessian_xy,
                        p.hessian_yy,
                        p.hessian_scale,
                        p.determinant_scaled,
                    )
                )
                or (i < len(points) - 1 and (p.delta_x is None or p.delta_y is None))
                or (
                    i == len(points) - 1
                    and (p.delta_x is not None or p.delta_y is not None)
                )
                or any(
                    v is not None and not math.isfinite(v)
                    for v in (p.delta_x, p.delta_y)
                )
                for i, p in enumerate(points)
            )
            or any(v is not None and not math.isfinite(v) for v in samples)
        ):
            raise ValueError
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Invalid 2D Newton response from Rust.") from exc
    grid = tuple(
        tuple(samples[i : i + GRID_SIZE]) for i in range(0, len(samples), GRID_SIZE)
    )
    return Newton2DResult(request, status, tuple(points), grid)


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
                "hessian_xx",
                "hessian_xy",
                "hessian_yx",
                "hessian_yy",
                "hessian_scale",
                "determinant_scaled",
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
                    "NEWTON2D",
                    result.status,
                    r.expression,
                    r.x0,
                    r.y0,
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
                    p.hessian_xx,
                    p.hessian_xy,
                    p.hessian_xy,
                    p.hessian_yy,
                    p.hessian_scale,
                    p.determinant_scaled,
                    p.delta_x if nxt else "",
                    p.delta_y if nxt else "",
                    nxt.x if nxt else "",
                    nxt.y if nxt else "",
                ]
            )
