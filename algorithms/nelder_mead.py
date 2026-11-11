import csv
from dataclasses import dataclass
import math
from pathlib import Path

from ..backend import InputError
from .optimization import compile_expression
from .steepest import GRID_SIZE, SteepestRequest

ACTIONS = {
    "reflect": "Reflection",
    "expand": "Expansion",
    "reflect_after_expand": "Reflection · expansion rejected",
    "contract": "Inside contraction",
    "contract_outside": "Outside contraction",
    "shrink": "Shrink toward A",
}
PRESETS = {
    "Elliptic bowl · minimum (2, −1)": {
        "expression": "(x-2)^2+2*(y+1)^2",
        "x0": "6",
        "y0": "3",
        "b_x": "7",
        "b_y": "3",
        "c_x": "6",
        "c_y": "4",
        "plot_min": "-2",
        "plot_max": "8",
        "plot_y_min": "-3",
        "plot_y_max": "5",
    },
    "Coupled quadratic · minimum (0, 0)": {
        "expression": "x^2+x*y+y^2",
        "x0": "2",
        "y0": "1",
        "b_x": "3",
        "b_y": "1",
        "c_x": "2",
        "c_y": "2",
        "plot_min": "-3",
        "plot_max": "4",
        "plot_y_min": "-3",
        "plot_y_max": "3",
    },
    "Rosenbrock · minimum (1, 1)": {
        "expression": "(1-x)^2+100*(y-x^2)^2",
        "x0": "-1.2",
        "y0": "1",
        "b_x": "-0.2",
        "b_y": "1",
        "c_x": "-1.2",
        "c_y": "2",
        "plot_min": "-2",
        "plot_max": "2",
        "plot_y_min": "-1",
        "plot_y_max": "3",
        "max_iterations": "500",
    },
    "Nonsmooth norm · minimum (0, 0)": {
        "expression": "sqrt(x^2+y^2)",
        "x0": "2",
        "y0": "1",
        "b_x": "3",
        "b_y": "1",
        "c_x": "2",
        "c_y": "2",
        "plot_min": "-4",
        "plot_max": "4",
        "plot_y_min": "-4",
        "plot_y_max": "4",
        "max_iterations": "500",
    },
}


PRESETS.update(
    {
        "04 · simplex for x²+2y²": dict(
            expression="x^2+2*y^2",
            x0="2",
            y0="2",
            b_x="2",
            b_y="1",
            c_x="1",
            c_y="2",
            max_iterations="2",
            variant="slides",
            stopping="function",
        ),
        "CV4 · coupled quadratic, standard branches": dict(
            expression="x^2-4*x+y^2-y-x*y",
            x0="0",
            y0="1",
            b_x="1",
            b_y="1",
            c_x="0",
            c_y="1.5",
            variant="standard",
            stopping="position",
            tolerance="0.001",
        ),
        "Slides p23 · corrected −4x table": dict(
            expression="x^2-4*x+y^2-y-x*y",
            x0="0",
            y0="0",
            b_x="1.2",
            b_y="0",
            c_x="0",
            c_y="0.8",
            max_iterations="9",
            variant="slides",
            stopping="function",
        ),
        "Slides p23 · literal printed −4y function": dict(
            expression="x^2-4*y+y^2-y-x*y",
            x0="0",
            y0="0",
            b_x="1.2",
            b_y="0",
            c_x="0",
            c_y="0.8",
            variant="slides",
            stopping="function",
        ),
        "Slides pp25–26 · shifted bowl": dict(
            expression="(x-2)^2+3*(y-1)^2",
            x0="0",
            y0="0",
            b_x="2",
            b_y="0",
            c_x="0",
            c_y="2",
            max_iterations="2",
            variant="slides",
            stopping="function",
        ),
        "Slides p27 · three iterations": dict(
            expression="x^2+y^2+x-2*y-x*y+2",
            x0="1.2",
            y0="0",
            b_x="0",
            b_y="0.8",
            c_x="0",
            c_y="0",
            max_iterations="3",
            variant="slides",
            stopping="function",
        ),
    }
)


@dataclass(frozen=True)
class NelderMeadRequest:
    expression: str = "(x-2)^2+2*(y+1)^2"
    simplex: tuple[tuple[float, float], ...] = ((6.0, 3.0), (7.0, 3.0), (6.0, 4.0))
    tolerance: float = 1e-6
    max_iterations: int = 200
    plot_min: float = -2.0
    plot_max: float = 8.0
    plot_y_min: float = -3.0
    plot_y_max: float = 5.0
    variant: str = "slides"
    stopping: str = "function"

    algorithm_name = "2D Nelder–Mead"

    def validate(self):
        if self.variant not in ("slides", "standard") or self.stopping not in (
            "function",
            "position",
        ):
            raise InputError(
                "Choose slides/standard branches and function/position stopping."
            )
        if len(self.simplex) != 3 or any(len(v) != 2 for v in self.simplex):
            raise InputError("Initial simplex: enter three 2D vertices A, B and C.")
        if any(
            not math.isfinite(v) or abs(v) > 1e50
            for point in self.simplex
            for v in point
        ):
            raise InputError(
                "Initial simplex coordinates must be finite and at most 1e50 in magnitude."
            )
        SteepestRequest(
            expression=self.expression,
            x0=self.simplex[0][0],
            y0=self.simplex[0][1],
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
            self.tolerance,
            self.max_iterations,
            self.plot_min,
            self.plot_max,
            self.plot_y_min,
            self.plot_y_max,
        )
        vertices = ["\t".join(map(repr, point)) for point in self.simplex]
        return (
            "NELDER_MEAD2_V1\n"
            + "\t".join(map(repr, options))
            + f"\t{self.variant}\t{self.stopping}"
            + "\n"
            + "\n".join(vertices)
            + "\n"
            + compile_expression(self.expression, dimensions=2)
            + "\n"
        )


@dataclass(frozen=True)
class Vertex:
    x: float
    y: float
    value: float


@dataclass(frozen=True)
class SimplexPoint:
    iteration: int
    spread: float
    diameter: float
    simplex: tuple[Vertex, ...]  # Sorted A (best), B, C (worst).

    @property
    def x(self):
        return self.simplex[0].x

    @property
    def y(self):
        return self.simplex[0].y

    @property
    def value(self):
        return self.simplex[0].value

    @property
    def best_radius(self):
        return max(math.hypot(v.x - self.x, v.y - self.y) for v in self.simplex)


@dataclass(frozen=True)
class SimplexStep:
    iteration: int
    action: str
    centroid: tuple[float, float]
    reflection: Vertex
    expansion: Vertex | None
    contraction: Vertex | None
    shrunk_b: Vertex | None
    shrunk_c: Vertex | None


@dataclass(frozen=True)
class FailedTrial:
    stage: str
    x: float
    y: float


@dataclass(frozen=True)
class NelderMeadResult:
    request: NelderMeadRequest
    status: str
    points: tuple[SimplexPoint, ...]
    steps: tuple[SimplexStep, ...]
    failure: FailedTrial | None
    grid: tuple[tuple[float | None, ...], ...]

    @property
    def final(self):
        return self.points[-1]


def _vertex(fields):
    return Vertex(*map(float, fields))


def _optional_vertex(fields):
    return None if fields == ["_", "_", "_"] else _vertex(fields)


def parse_output(output, request):
    status = failure = None
    points, steps, samples = [], [], []
    try:
        for line in output.splitlines():
            f = line.split("\t")
            if f[0] == "NMR" and len(f) == 2 and status is None:
                status = f[1]
            elif f[0] == "NMP" and len(f) == 13:
                points.append(
                    SimplexPoint(
                        int(f[1]),
                        float(f[2]),
                        float(f[3]),
                        tuple(_vertex(f[i : i + 3]) for i in (4, 7, 10)),
                    )
                )
            elif f[0] == "NMT" and len(f) == 20:
                steps.append(
                    SimplexStep(
                        int(f[1]),
                        f[2],
                        tuple(map(float, f[3:5])),
                        _vertex(f[5:8]),
                        _optional_vertex(f[8:11]),
                        _optional_vertex(f[11:14]),
                        _optional_vertex(f[14:17]),
                        _optional_vertex(f[17:20]),
                    )
                )
            elif f[0] == "NMF" and len(f) == 4 and failure is None:
                failure = FailedTrial(f[1], float(f[2]), float(f[3]))
            elif f[0] == "NMC" and len(f) == 3 and int(f[1]) == len(samples):
                samples.append(None if f[2] == "_" else float(f[2]))
            else:
                raise ValueError
        vertices = [v for p in points for v in p.simplex]
        vertices.extend(
            v
            for s in steps
            for v in (s.reflection, s.expansion, s.contraction, s.shrunk_b, s.shrunk_c)
            if v is not None
        )
        if (
            status
            not in {
                "function_tolerance",
                "simplex_tolerance",
                "iteration_limit",
                "numerical_limit",
                "stagnation",
            }
            or not points
            or len(points) > request.max_iterations + 1
            or len(steps) != len(points) - 1
            or len(samples) != GRID_SIZE**2
            or any(
                p.iteration != i
                or not math.isfinite(p.spread)
                or not math.isfinite(p.diameter)
                or p.spread < 0
                or p.diameter < 0
                or not p.simplex[0].value <= p.simplex[1].value <= p.simplex[2].value
                for i, p in enumerate(points)
            )
            or any(
                not all(
                    math.isfinite(x) and abs(x) <= 1e100 for x in (v.x, v.y, v.value)
                )
                for v in vertices
            )
            or any(
                s.iteration != i
                or s.action not in ACTIONS
                or not all(math.isfinite(x) for x in s.centroid)
                or (s.expansion is not None)
                != (s.action in ("expand", "reflect_after_expand"))
                or (s.contraction is not None)
                != (s.action in ("contract", "contract_outside", "shrink"))
                or (s.shrunk_b is not None) != (s.action == "shrink")
                or (s.shrunk_c is not None) != (s.action == "shrink")
                for i, s in enumerate(steps)
            )
            or any(v is not None and not math.isfinite(v) for v in samples)
            or (failure is not None) != (status == "numerical_limit")
            or (
                failure is not None
                and (
                    failure.stage
                    not in {
                        "reflection",
                        "expansion",
                        "contraction",
                        "shrink_B",
                        "shrink_C",
                    }
                    or not all(math.isfinite(v) for v in (failure.x, failure.y))
                )
            )
        ):
            raise ValueError
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Invalid Nelder–Mead response from Rust.") from exc
    grid = tuple(
        tuple(samples[i : i + GRID_SIZE]) for i in range(0, len(samples), GRID_SIZE)
    )
    return NelderMeadResult(request, status, tuple(points), tuple(steps), failure, grid)


def export_trace(path, result):
    r = result.request
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "algorithm",
                "status",
                "function",
                "function_tolerance",
                "branch_variant",
                "stopping_rule",
                "iteration_limit",
                "reflection_alpha",
                "expansion_gamma",
                "contraction_rho",
                "shrink_sigma",
                *[f"initial_{v}_{axis}" for v in "ABC" for axis in ("x", "y")],
                "plot_x_min",
                "plot_x_max",
                "plot_y_min",
                "plot_y_max",
                "k",
                "function_spread",
                "simplex_diameter",
                *[f"{v}_{col}" for v in "ABC" for col in ("x", "y", "f")],
                "action",
                "centroid_x",
                "centroid_y",
                *[
                    f"{name}_{col}"
                    for name in (
                        "reflection",
                        "expansion",
                        "contraction",
                        "shrunk_B",
                        "shrunk_C",
                    )
                    for col in ("x", "y", "f")
                ],
                *[f"next_{v}_{col}" for v in "ABC" for col in ("x", "y", "f")],
                "failed_stage",
                "failed_x",
                "failed_y",
            ]
        )
        for i, p in enumerate(result.points):
            step = result.steps[i] if i < len(result.steps) else None
            nxt = result.points[i + 1] if i + 1 < len(result.points) else None
            trials = (
                [
                    x
                    for v in (
                        step.reflection,
                        step.expansion,
                        step.contraction,
                        step.shrunk_b,
                        step.shrunk_c,
                    )
                    for x in ((v.x, v.y, v.value) if v else ("", "", ""))
                ]
                if step
                else [""] * 15
            )
            failure = result.failure if nxt is None else None
            writer.writerow(
                [
                    "NELDER_MEAD2D",
                    result.status,
                    r.expression,
                    r.tolerance,
                    r.variant,
                    r.stopping,
                    r.max_iterations,
                    1,
                    2,
                    0.5,
                    0.5,
                    *[x for v in r.simplex for x in v],
                    r.plot_min,
                    r.plot_max,
                    r.plot_y_min,
                    r.plot_y_max,
                    p.iteration,
                    p.spread,
                    p.diameter,
                    *[x for v in p.simplex for x in (v.x, v.y, v.value)],
                    step.action if step else "",
                    *(step.centroid if step else ("", "")),
                    *trials,
                    *(
                        [x for v in nxt.simplex for x in (v.x, v.y, v.value)]
                        if nxt
                        else [""] * 9
                    ),
                    failure.stage if failure else "",
                    failure.x if failure else "",
                    failure.y if failure else "",
                ]
            )
