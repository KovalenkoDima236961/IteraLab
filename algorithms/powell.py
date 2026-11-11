import csv
from dataclasses import dataclass
import math
from pathlib import Path

from ..backend import InputError
from .optimization import compile_expression
from .steepest import GRID_SIZE, SteepestRequest

PRESETS = {
    "Supplied exercise · minimum (1, 1), f = −5": {
        "expression": "x^2+2*y^2+2*x*y-4*x-6*y",
        "x0": "0",
        "y0": "0",
        "plot_min": "-0.2",
        "plot_max": "2.3",
        "plot_y_min": "-0.1",
        "plot_y_max": "1.8",
        "initial_step": "1",
        "line_tolerance": "1e-10",
        "max_line_evaluations": "256",
    },
    "Elliptic bowl · minimum (2, −1)": {
        "expression": "(x-2)^2+2*(y+1)^2",
        "x0": "6",
        "y0": "3",
        "plot_min": "-2",
        "plot_max": "7",
        "plot_y_min": "-3",
        "plot_y_max": "4",
    },
    "Rosenbrock · minimum (1, 1)": {
        "expression": "(1-x)^2+100*(y-x^2)^2",
        "x0": "-1.2",
        "y0": "1",
        "plot_min": "-2",
        "plot_max": "2",
        "plot_y_min": "-1",
        "plot_y_max": "3",
        "tolerance": "1e-8",
    },
    "Nonsmooth bowl · minimum (0, 0)": {
        "expression": "sqrt(x^2)+sqrt(y^2)",
        "x0": "2",
        "y0": "-3",
        "plot_min": "-3",
        "plot_max": "3",
        "plot_y_min": "-4",
        "plot_y_max": "2",
    },
}
FAILURES = {
    "evaluation_limit": "line_search_limit",
    "not_bracketed": "powell_line_failed",
    "precision_limit": "line_search_precision",
}
STATUSES = {
    "powell_function_tolerance",
    "powell_position_tolerance",
    "iteration_limit",
    "numerical_limit",
    *FAILURES.values(),
}


PRESETS.update(
    {
        "05 · Powell from (5,5), minimum (3,2)": dict(
            expression="x^2-4*x+y^2-y-x*y",
            x0="5",
            y0="5",
            plot_min="1",
            plot_max="6",
            plot_y_min="0",
            plot_y_max="6",
        ),
    }
)


@dataclass(frozen=True)
class PowellRequest:
    expression: str = "x^2+2*y^2+2*x*y-4*x-6*y"
    x0: float = 0.0
    y0: float = 0.0
    tolerance: float = 1e-6
    max_iterations: int = 200
    initial_step: float = 1.0
    line_tolerance: float = 1e-10
    max_line_evaluations: int = 256
    plot_min: float = -0.2
    plot_max: float = 2.3
    plot_y_min: float = -0.1
    plot_y_max: float = 1.8

    algorithm_name = "2D Powell's Method"

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
        if not math.isfinite(self.initial_step) or not 0 < self.initial_step <= 1e50:
            raise InputError(
                "Initial line-search α probe must be positive, finite and at most 1e50."
            )
        if not math.isfinite(self.line_tolerance) or not 0 < self.line_tolerance < 1:
            raise InputError(
                "Line-search α tolerance must be strictly between 0 and 1."
            )
        if (
            type(self.max_line_evaluations) is not int
            or not 1 <= self.max_line_evaluations <= 2048
        ):
            raise InputError(
                "Maximum line-search trial evaluations must be an integer between 1 and 2,048."
            )
        if 3 * self.max_iterations * self.max_line_evaluations > 1_000_000:
            raise InputError(
                "Reduce the cycle or line-search budget: 3 × cycles × trials must be at most 1,000,000."
            )

    def serialize(self):
        self.validate()
        values = (
            self.x0,
            self.y0,
            self.tolerance,
            self.max_iterations,
            self.initial_step,
            self.line_tolerance,
            self.max_line_evaluations,
            self.plot_min,
            self.plot_max,
            self.plot_y_min,
            self.plot_y_max,
        )
        return (
            "POWELL2_V1\n"
            + "\t".join(map(repr, values))
            + "\n"
            + compile_expression(self.expression, dimensions=2)
            + "\n"
        )


@dataclass(frozen=True)
class PowellPoint:
    iteration: int  # Global move index: each cycle contributes three moves.
    x: float
    y: float
    value: float


@dataclass(frozen=True)
class PowellSample:
    number: int
    phase: str
    alpha: float
    x: float
    y: float
    value: float | None

    @property
    def valid(self):
        return self.value is not None


@dataclass(frozen=True)
class PowellSearch:
    iteration: int
    direction_x: float
    direction_y: float
    status: str
    accepted_index: int | None
    samples: tuple[PowellSample, ...]

    @property
    def accepted(self):
        return (
            self.samples[self.accepted_index]
            if self.accepted_index is not None
            else None
        )

    @property
    def cycle(self):
        return self.iteration // 3 + 1

    @property
    def stage(self):
        return self.iteration % 3 + 1


@dataclass(frozen=True)
class PowellCycle:
    number: int
    d1_x: float
    d1_y: float
    d2_x: float
    d2_y: float
    d3_x: float
    d3_y: float
    function_change: float
    displacement: float


@dataclass(frozen=True)
class PowellResult:
    request: PowellRequest
    status: str
    points: tuple[PowellPoint, ...]
    cycles: tuple[PowellCycle, ...]
    searches: tuple[PowellSearch, ...]
    grid: tuple[tuple[float | None, ...], ...]

    @property
    def final(self):
        return self.points[-1]


def parse_output(output, request):
    status = None
    points, cycles, headers, probes, grid = [], [], [], [], []
    try:
        for line in output.splitlines():
            f = line.split("\t")
            if f[0] == "PWR" and len(f) == 2 and status is None:
                status = f[1]
            elif f[0] == "PWP" and len(f) == 5:
                points.append(PowellPoint(int(f[1]), *map(float, f[2:])))
            elif f[0] == "PWC" and len(f) == 10:
                cycles.append(PowellCycle(int(f[1]), *map(float, f[2:])))
            elif f[0] == "PWL" and len(f) == 6:
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
            elif f[0] == "PWS" and len(f) == 8 and int(f[1]) == len(headers) - 1:
                probes[-1].append(
                    PowellSample(
                        int(f[2]),
                        f[3],
                        *map(float, f[4:7]),
                        None if f[7] == "_" else float(f[7]),
                    )
                )
            elif f[0] == "PWG" and len(f) == 3 and int(f[1]) == len(grid):
                grid.append(None if f[2] == "_" else float(f[2]))
            else:
                raise ValueError
        searches = tuple(
            PowellSearch(*h, tuple(samples)) for h, samples in zip(headers, probes)
        )
        failure = status in FAILURES.values()
        if (
            status not in STATUSES
            or not points
            or len(points) > 3 * request.max_iterations + 1
            or len(grid) != GRID_SIZE**2
            or len(cycles) != (len(points) - 1) // 3
            or len(searches) != len(points) - 1 + int(failure)
            or any(
                p.iteration != i
                or not all(math.isfinite(v) for v in (p.x, p.y, p.value))
                for i, p in enumerate(points)
            )
            or any(v is not None and not math.isfinite(v) for v in grid)
        ):
            raise ValueError
        for i, c in enumerate(cycles):
            if (
                c.number != i + 1
                or not all(
                    math.isfinite(v)
                    for v in (
                        c.d1_x,
                        c.d1_y,
                        c.d2_x,
                        c.d2_y,
                        c.d3_x,
                        c.d3_y,
                        c.function_change,
                        c.displacement,
                    )
                )
                or c.function_change < 0
                or c.displacement < 0
            ):
                raise ValueError
        for i, s in enumerate(searches):
            accepted = s.status in {"minimum", "zero_direction"}
            if (
                s.iteration != i
                or s.status not in {"minimum", "zero_direction", *FAILURES}
                or not all(math.isfinite(v) for v in (s.direction_x, s.direction_y))
                or not s.samples
                or len(s.samples) > request.max_line_evaluations + 1
                or accepted != (s.accepted_index is not None)
                or (accepted and not 0 <= s.accepted_index < len(s.samples))
                or (i < len(points) - 1 and not accepted)
                or (i == len(points) - 1 and FAILURES.get(s.status) != status)
            ):
                raise ValueError
            for j, p in enumerate(s.samples):
                if (
                    p.number != j
                    or p.phase
                    not in {
                        "start",
                        "bracket",
                        "expand",
                        "domain_refine",
                        "golden",
                        "parabolic",
                    }
                    or not all(math.isfinite(v) for v in (p.alpha, p.x, p.y))
                    or (p.value is not None and not math.isfinite(p.value))
                ):
                    raise ValueError
            p = s.samples[0]
            if (
                p.phase != "start"
                or p.alpha != 0
                or (p.x, p.y, p.value) != (points[i].x, points[i].y, points[i].value)
            ):
                raise ValueError
            if accepted:
                a = s.accepted
                nxt = points[i + 1]
                if (
                    not a.valid
                    or a.value > p.value
                    or (a.x, a.y, a.value) != (nxt.x, nxt.y, nxt.value)
                ):
                    raise ValueError
        if (
            status not in {*FAILURES.values(), "numerical_limit"}
            and (len(points) - 1) % 3
        ):
            raise ValueError
    except (ValueError, IndexError, TypeError) as exc:
        raise RuntimeError("Invalid Powell response from Rust.") from exc
    return PowellResult(
        request,
        status,
        tuple(points),
        tuple(cycles),
        searches,
        tuple(tuple(grid[i : i + GRID_SIZE]) for i in range(0, len(grid), GRID_SIZE)),
    )


def _snapshot(request):
    return [
        request.expression,
        request.x0,
        request.y0,
        request.tolerance,
        request.max_iterations,
        request.initial_step,
        request.line_tolerance,
        request.max_line_evaluations,
    ]


SNAPSHOT_COLUMNS = [
    "function",
    "initial_x",
    "initial_y",
    "cycle_tolerance",
    "cycle_limit",
    "initial_alpha_probe",
    "line_alpha_tolerance",
    "line_trial_limit",
]


def export_trace(path, result):
    """All committed moves, outgoing searches, and full-cycle stopping diagnostics."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(
            [
                "algorithm",
                "status",
                *SNAPSHOT_COLUMNS,
                "move",
                "cycle",
                "stage",
                "x",
                "y",
                "f_x_y",
                "direction_x",
                "direction_y",
                "search_status",
                "alpha",
                "x_next",
                "y_next",
                "cycle_d1_x",
                "cycle_d1_y",
                "cycle_d2_x",
                "cycle_d2_y",
                "cycle_d3_x",
                "cycle_d3_y",
                "cycle_function_change",
                "cycle_displacement",
            ]
        )
        for i, p in enumerate(result.points):
            s = result.searches[i] if i < len(result.searches) else None
            a = s.accepted if s else None
            c = result.cycles[i // 3] if i // 3 < len(result.cycles) else None
            w.writerow(
                [
                    "POWELL2D",
                    result.status,
                    *_snapshot(result.request),
                    p.iteration,
                    i // 3 + 1 if s else "",
                    i % 3 + 1 if s else "",
                    p.x,
                    p.y,
                    p.value,
                    s.direction_x if s else "",
                    s.direction_y if s else "",
                    s.status if s else "",
                    a.alpha if a else "",
                    a.x if a else "",
                    a.y if a else "",
                    *(
                        [
                            c.d1_x,
                            c.d1_y,
                            c.d2_x,
                            c.d2_y,
                            c.d3_x,
                            c.d3_y,
                            c.function_change,
                            c.displacement,
                        ]
                        if c
                        else [""] * 8
                    ),
                ]
            )


def export_line_searches(path, result):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(
            [
                "algorithm",
                "status",
                *SNAPSHOT_COLUMNS,
                "move",
                "cycle",
                "stage",
                "direction_x",
                "direction_y",
                "search_status",
                "sample",
                "phase",
                "alpha",
                "x_alpha",
                "y_alpha",
                "phi_alpha",
                "valid",
                "accepted",
            ]
        )
        for s in result.searches:
            for p in s.samples:
                w.writerow(
                    [
                        "POWELL2D",
                        result.status,
                        *_snapshot(result.request),
                        s.iteration,
                        s.cycle,
                        s.stage,
                        s.direction_x,
                        s.direction_y,
                        s.status,
                        p.number,
                        p.phase,
                        p.alpha,
                        p.x,
                        p.y,
                        p.value if p.valid else "",
                        p.valid,
                        p.number == s.accepted_index,
                    ]
                )
