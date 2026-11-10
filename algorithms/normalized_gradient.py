"""Normalized 2D gradient descent with a recorded numerical stationary line search."""

import csv
from dataclasses import dataclass
import math
from pathlib import Path

from ..backend import InputError
from .optimization import compile_expression
from .steepest import (
    GRID_SIZE,
    SteepestPoint,
    SteepestRequest,
    PRESETS as STEEPEST_PRESETS,
)

SEARCH_FAILURES = {
    "evaluation_limit": "line_search_limit",
    "not_bracketed": "line_search_failed",
    "precision_limit": "line_search_precision",
    "objective_increase": "line_search_increase",
}
PRESETS = {
    "Circular bowl · minimum (0, 0)": {
        "expression": "x^2+y^2",
        "x0": "3",
        "y0": "4",
        "plot_min": "-1",
        "plot_max": "4",
        "plot_y_min": "-1",
        "plot_y_max": "5",
    },
    **{
        name: {k: v for k, v in values.items() if k not in ("alpha", "max_iterations")}
        for name, values in STEEPEST_PRESETS.items()
        if not name.startswith("Circular")
    },
}
PRESETS["Rosenbrock · minimum (1, 1)"]["max_iterations"] = "1000"


PRESETS.update(
    {
        "05 · normalized gradient from (5,5)": dict(
            expression="x^2-4*x+y^2-y-x*y",
            x0="5",
            y0="5",
            plot_min="1",
            plot_max="6",
            plot_y_min="0",
            plot_y_max="6",
        ),
        "CV5 · optimized steepest step from (0,0)": dict(
            expression="2*x^2+x*y+3*y^2-4*x+2*y",
            x0="0",
            y0="0",
            plot_min="-0.5",
            plot_max="2",
            plot_y_min="-1",
            plot_y_max="0.5",
        ),
    }
)


@dataclass(frozen=True)
class NormalizedRequest:
    expression: str = "x^2+y^2"
    x0: float = 3.0
    y0: float = 4.0
    tolerance: float = 1e-6
    max_iterations: int = 200
    initial_gamma: float = 1.0
    line_tolerance: float = 1e-10
    max_line_evaluations: int = 512
    plot_min: float = -1.0
    plot_max: float = 4.0
    plot_y_min: float = -1.0
    plot_y_max: float = 5.0
    normalize: bool = True

    algorithm_name = "2D Gradient Method with Normalized Vector"

    def validate(self):
        if type(self.normalize) is not bool:
            raise InputError("Normalize direction must be a boolean.")
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
        if not math.isfinite(self.initial_gamma) or not 0 < self.initial_gamma <= 1e50:
            raise InputError(
                "Initial line-search γ probe must be positive, finite and at most 1e50."
            )
        if not math.isfinite(self.line_tolerance) or not 0 < self.line_tolerance < 1:
            raise InputError(
                "Relative line-derivative tolerance must be strictly between 0 and 1."
            )
        if (
            type(self.max_line_evaluations) is not int
            or not 1 <= self.max_line_evaluations <= 2048
        ):
            raise InputError(
                "Maximum line-search trial evaluations must be an integer between 1 and 2,048."
            )
        if self.max_iterations * self.max_line_evaluations > 1_000_000:
            raise InputError(
                "Reduce the iteration or line-search budget: the complete trace supports at most 1,000,000 trial evaluations."
            )

    def serialize(self):
        self.validate()
        options = (
            self.x0,
            self.y0,
            self.tolerance,
            self.max_iterations,
            self.initial_gamma,
            self.line_tolerance,
            self.max_line_evaluations,
            self.plot_min,
            self.plot_max,
            self.plot_y_min,
            self.plot_y_max,
            int(self.normalize),
        )
        return (
            "NORMALIZED2_V1\n"
            + "\t".join(map(repr, options))
            + "\n"
            + compile_expression(self.expression, dimensions=2)
            + "\n"
        )


@dataclass(frozen=True)
class LineSteepestRequest(NormalizedRequest):
    """The CV5 steepest-descent rule d=-gradient with an optimized scalar step."""

    normalize: bool = False
    algorithm_name = "2D Steepest Descent with Line Search"


@dataclass(frozen=True)
class LineSample:
    number: int
    phase: str
    gamma: float
    x: float
    y: float
    value: float | None
    gradient_x: float | None
    gradient_y: float | None
    derivative: float | None

    @property
    def valid(self):
        return self.value is not None


@dataclass(frozen=True)
class LineSearch:
    iteration: int
    direction_x: float
    direction_y: float
    status: str
    accepted_index: int | None
    samples: tuple[LineSample, ...]

    @property
    def accepted(self):
        return (
            self.samples[self.accepted_index]
            if self.accepted_index is not None
            else None
        )


@dataclass(frozen=True)
class NormalizedResult:
    request: NormalizedRequest
    status: str
    points: tuple[SteepestPoint, ...]
    searches: tuple[LineSearch, ...]
    grid: tuple[tuple[float | None, ...], ...]

    @property
    def final(self):
        return self.points[-1]


def parse_output(output, request):
    status = None
    points, headers, probes, grid = [], [], [], []
    try:
        for line in output.splitlines():
            f = line.split("\t")
            if f[0] == "NGR" and len(f) == 2 and status is None:
                status = f[1]
            elif f[0] == "NGP" and len(f) == 8:
                points.append(SteepestPoint(int(f[1]), *map(float, f[2:])))
            elif f[0] == "NGL" and len(f) == 6:
                headers.append(
                    (
                        int(f[1]),
                        float(f[2]),
                        float(f[3]),
                        f[4],
                        None if f[5] == "_" else int(f[5]),
                    )
                )
                probes.append([])
            elif f[0] == "NGS" and len(f) == 11 and int(f[1]) == len(headers) - 1:
                probes[-1].append(
                    LineSample(
                        int(f[2]),
                        f[3],
                        *map(float, f[4:7]),
                        *[None if v == "_" else float(v) for v in f[7:]],
                    )
                )
            elif f[0] == "NGC" and len(f) == 3 and int(f[1]) == len(grid):
                grid.append(None if f[2] == "_" else float(f[2]))
            else:
                raise ValueError
        searches = tuple(
            LineSearch(*h, tuple(samples)) for h, samples in zip(headers, probes)
        )
        failure = status in SEARCH_FAILURES.values()
        if (
            status
            not in {"gradient_tolerance", "iteration_limit", *SEARCH_FAILURES.values()}
            or not points
            or len(points) > request.max_iterations + 1
            or len(grid) != GRID_SIZE**2
            or len(searches) != len(points) - 1 + int(failure)
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
            or any(v is not None and not math.isfinite(v) for v in grid)
        ):
            raise ValueError
        for i, search in enumerate(searches):
            accepted = search.status == "stationary"
            if (
                search.iteration != i
                or not all(
                    math.isfinite(v) for v in (search.direction_x, search.direction_y)
                )
                or search.status not in {"stationary", *SEARCH_FAILURES}
                or not search.samples
                or len(search.samples) > request.max_line_evaluations + 1
                or search.samples[0].gamma != 0
                or search.samples[0].phase != "start"
                or not search.samples[0].valid
                or accepted != (search.accepted_index is not None)
                or (accepted and not 1 <= search.accepted_index < len(search.samples))
                or (i < len(points) - 1 and not accepted)
                or (
                    i == len(points) - 1
                    and SEARCH_FAILURES.get(search.status) != status
                )
            ):
                raise ValueError
            for j, sample in enumerate(search.samples):
                values = (
                    sample.value,
                    sample.gradient_x,
                    sample.gradient_y,
                    sample.derivative,
                )
                if (
                    sample.number != j
                    or sample.phase
                    not in {"start", "bracket", "bisect", "domain_refine"}
                    or not all(
                        math.isfinite(v) for v in (sample.gamma, sample.x, sample.y)
                    )
                    or sample.gamma < 0
                    or (j > 0 and sample.gamma == 0)
                    or (
                        any(v is None for v in values)
                        and not all(v is None for v in values)
                    )
                    or any(v is not None and not math.isfinite(v) for v in values)
                ):
                    raise ValueError
            if accepted and not search.accepted.valid:
                raise ValueError
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Invalid normalized-gradient response from Rust.") from exc
    return NormalizedResult(
        request,
        status,
        tuple(points),
        searches,
        tuple(tuple(grid[i : i + GRID_SIZE]) for i in range(0, len(grid), GRID_SIZE)),
    )


def _snapshot_columns():
    return [
        "function",
        "initial_x",
        "initial_y",
        "gradient_tolerance",
        "iteration_limit",
        "initial_gamma",
        "relative_line_tolerance",
        "line_trial_limit",
    ]


def _snapshot_values(request):
    return [
        request.expression,
        request.x0,
        request.y0,
        request.tolerance,
        request.max_iterations,
        request.initial_gamma,
        request.line_tolerance,
        request.max_line_evaluations,
    ]


def export_trace(path, result):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "algorithm",
                "status",
                *_snapshot_columns(),
                "k",
                "x_k",
                "y_k",
                "f_x_y",
                "partial_x",
                "partial_y",
                "gradient_norm",
                "S_x",
                "S_y",
                "search_status",
                "line_trials",
                "gamma_k",
                "phi_prime_gamma",
                "x_next",
                "y_next",
            ]
        )
        for i, p in enumerate(result.points):
            search = result.searches[i] if i < len(result.searches) else None
            accepted = search.accepted if search else None
            nxt = result.points[i + 1] if i + 1 < len(result.points) else None
            writer.writerow(
                [
                    "NORMALIZED_GRADIENT2D"
                    if result.request.normalize
                    else "LINE_STEEPEST2D",
                    result.status,
                    *_snapshot_values(result.request),
                    p.iteration,
                    p.x,
                    p.y,
                    p.value,
                    p.gradient_x,
                    p.gradient_y,
                    p.gradient_norm,
                    search.direction_x if search else "",
                    search.direction_y if search else "",
                    search.status if search else "",
                    len(search.samples) - 1 if search else "",
                    accepted.gamma if accepted else "",
                    accepted.derivative if accepted else "",
                    nxt.x if nxt else "",
                    nxt.y if nxt else "",
                ]
            )


def export_line_searches(path, result):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "algorithm",
                "status",
                *_snapshot_columns(),
                "k",
                "search_status",
                "S_x",
                "S_y",
                "sample",
                "phase",
                "gamma",
                "x_gamma",
                "y_gamma",
                "valid",
                "phi_gamma",
                "partial_x",
                "partial_y",
                "phi_prime_gamma",
                "accepted",
            ]
        )
        for search in result.searches:
            for s in search.samples:
                writer.writerow(
                    [
                        "NORMALIZED_GRADIENT2D"
                        if result.request.normalize
                        else "LINE_STEEPEST2D",
                        result.status,
                        *_snapshot_values(result.request),
                        search.iteration,
                        search.status,
                        search.direction_x,
                        search.direction_y,
                        s.number,
                        s.phase,
                        s.gamma,
                        s.x,
                        s.y,
                        s.valid,
                        s.value if s.valid else "",
                        s.gradient_x if s.valid else "",
                        s.gradient_y if s.valid else "",
                        s.derivative if s.valid else "",
                        s.number == search.accepted_index,
                    ]
                )
