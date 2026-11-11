import ast
import csv
from dataclasses import dataclass
import math
from pathlib import Path

from ..backend import InputError

STATUS = {
    "gradient_tolerance": "Gradient tolerance met",
    "iteration_limit": "Iteration limit reached",
    "numerical_limit": "Numerical limit reached",
    "stagnation": "Floating-point stagnation",
    "step_tolerance": "Step tolerance met",
    "zero_curvature": "Zero second derivative",
    "singular_hessian": "Singular Hessian",
    "function_tolerance": "Function spread tolerance met",
    "simplex_tolerance": "Simplex position tolerance met",
    "line_search_limit": "Line search evaluation limit",
    "line_search_failed": "No forward stationary point bracketed",
    "line_search_precision": "Line search precision limit",
    "line_search_increase": "Stationary line trial increased the objective",
    "powell_function_tolerance": "Cycle function-change tolerance met",
    "powell_position_tolerance": "Cycle displacement tolerance met",
    "powell_line_failed": "No line minimum bracketed",
}
GRADIENT = "Gradient Method · 1D"
NEWTON = "Newton's Method · 1D"
STEEPEST = "Steepest Gradient Descent · 2D"
NEWTON2D = "Newton's Method · 2D"
NELDER_MEAD = "Nelder–Mead · 2D"
NORMALIZED = "Normalized Gradient · 2D"
LINE_STEEPEST = "Steepest Descent · Line search · 2D"
POWELL = "Powell's Method · 2D"
METHODS = (
    GRADIENT,
    NEWTON,
    STEEPEST,
    LINE_STEEPEST,
    NEWTON2D,
    NELDER_MEAD,
    NORMALIZED,
    POWELL,
)
PRESETS = {
    "Quadratic · minimum at 2": ("(x - 2)^2 + 1", "6", "0.1", "-2", "7"),
    "Parabola · minimum at 0": ("x^2", "4", "0.1", "-5", "5"),
    "Double well · two minima": ("(x^2 - 1)^2", "2", "0.05", "-2.2", "2.2"),
    "Exponential + quadratic": ("exp(x) + x^2", "1", "0.1", "-3", "2"),
}

PRESETS.update(
    {
        "Slides p5 · exp(x)−2x, nine gradient steps": (
            "exp(x)-2*x",
            "-1",
            "0.35",
            "-1.2",
            "2",
        ),
        "04 · quartic gradient/Newton from 2": (
            "x^4-2*x^2",
            "2",
            "0.05",
            "-2.2",
            "2.2",
        ),
        "CV4 · Newton quartic from 1": ("x^4-3*x^3+2", "1", "0.05", "-1", "3"),
    }
)
SOURCE_PRESET_OPTIONS = {
    "Slides p5 · exp(x)−2x, nine gradient steps": {
        "tolerance": "0.001",
        "max_iterations": "9",
    }
}

FUNCTIONS = {"sin", "cos", "tan", "exp", "ln", "log", "sqrt"}


def compile_expression(text, *, dimensions=1):
    """Allowlisted AST to postfix instructions; never eval or execute user code."""
    variables = ("x", "y") if dimensions == 2 else ("x",)
    label = "f(x, y)" if dimensions == 2 else "f(x)"
    if not text.strip() or len(text) > 1000:
        raise InputError(f"{label}: enter an expression with 1–1,000 characters.")
    try:
        tree = ast.parse(text.strip().replace("^", "**"), mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise InputError(f"{label}: use explicit multiplication, such as 2*x.") from exc
    if sum(1 for _ in ast.walk(tree)) > 200:
        raise InputError("f(x): use fewer terms (at most 200 expression nodes).")

    def visit(node, depth=0):
        if depth > 30:
            raise InputError("f(x): parentheses are nested too deeply.")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if abs(node.value) > 1e100 or not math.isfinite(node.value):
                raise InputError("f(x): constants must be finite and at most 1e100.")
            return [f"c:{float(node.value)!r}"]
        if isinstance(node, ast.Name) and node.id in (*variables, "pi", "e"):
            return [
                node.id if node.id in variables else f"c:{getattr(math, node.id)!r}"
            ]
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            return visit(node.operand, depth + 1) + (
                ["neg"] if isinstance(node.op, ast.USub) else []
            )
        binary = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
            ast.Pow: "pow",
        }
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return (
                visit(node.left, depth + 1)
                + visit(node.right, depth + 1)
                + [binary[type(node.op)]]
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS
            and len(node.args) == 1
            and not node.keywords
        ):
            return visit(node.args[0], depth + 1) + [
                "ln" if node.func.id == "log" else node.func.id
            ]
        raise InputError(
            f"{label}: use {', '.join(variables)}, pi, e, + - * / ^ and sin, cos, tan, exp, ln, log, sqrt. Functions take one argument."
        )

    return " ".join(visit(tree.body))


@dataclass(frozen=True)
class GradientRequest:
    expression: str = "(x - 2)^2 + 1"
    x0: float = 6.0
    alpha: float = 0.1
    tolerance: float = 1e-6
    max_iterations: int = 200
    plot_min: float = -2.0
    plot_max: float = 7.0

    algorithm_name = "1D Gradient Method"

    def validate(self):
        compile_expression(self.expression)
        if any(
            not math.isfinite(v) or abs(v) > 1e50
            for v in (self.x0, self.plot_min, self.plot_max)
        ):
            raise InputError(
                "Starting x and plot bounds must be finite and at most 1e50 in magnitude."
            )
        if not math.isfinite(self.alpha) or not 0 < self.alpha <= 1e50:
            raise InputError(
                "Step length α must be positive, finite, and at most 1e50."
            )
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise InputError("Tolerance ε must be positive and finite.")
        if (
            type(self.max_iterations) is not int
            or not 1 <= self.max_iterations <= 100_000
        ):
            raise InputError(
                "Maximum iterations must be an integer between 1 and 100,000."
            )
        if self.plot_min >= self.plot_max:
            raise InputError("Plot minimum must be smaller than plot maximum.")

    def serialize(self):
        self.validate()
        options = (
            self.x0,
            self.alpha,
            self.tolerance,
            self.max_iterations,
            self.plot_min,
            self.plot_max,
        )
        return (
            "GRADIENT1_V1\n"
            + "\t".join(map(repr, options))
            + "\n"
            + compile_expression(self.expression)
            + "\n"
        )


@dataclass(frozen=True)
class NewtonRequest:
    expression: str = "(x - 2)^2 + 1"
    x0: float = 6.0
    tolerance: float = 1e-6
    max_iterations: int = 200
    plot_min: float = -2.0
    plot_max: float = 7.0
    first_derivative: str = ""
    second_derivative: str = ""

    algorithm_name = "1D Newton's Method"

    def validate(self):
        # Share objective/range/tolerance validation, not the gradient stopping rule.
        GradientRequest(
            self.expression,
            self.x0,
            tolerance=self.tolerance,
            plot_min=self.plot_min,
            plot_max=self.plot_max,
        ).validate()
        if (
            type(self.max_iterations) is not int
            or not 0 <= self.max_iterations <= 99_999
        ):
            raise InputError(
                "Newton Nmax must be an integer between 0 and 99,999. The slide loops through i=0…Nmax (at most 100,000 updates)."
            )
        for label, expression in (
            ("f′(x)", self.first_derivative),
            ("f″(x)", self.second_derivative),
        ):
            if expression.strip():
                try:
                    compile_expression(expression)
                except InputError as exc:
                    raise InputError(f"{label}: {exc}") from exc

    def serialize(self):
        self.validate()
        options = (
            self.x0,
            self.tolerance,
            self.max_iterations,
            self.plot_min,
            self.plot_max,
        )
        programs = [compile_expression(self.expression)]
        programs.extend(
            compile_expression(expr) if expr.strip() else "_"
            for expr in (self.first_derivative, self.second_derivative)
        )
        return (
            "NEWTON1_V1\n"
            + "\t".join(map(repr, options))
            + "\n"
            + "\n".join(programs)
            + "\n"
        )


@dataclass(frozen=True)
class GradientPoint:
    iteration: int
    x: float
    value: float
    gradient: float


@dataclass(frozen=True)
class GradientResult:
    request: GradientRequest
    status: str
    points: tuple[GradientPoint, ...]
    curve: tuple[tuple[float, float | None], ...]

    @property
    def final(self):
        return self.points[-1]


@dataclass(frozen=True)
class NewtonPoint:
    iteration: int
    x: float
    value: float
    gradient: float
    second_derivative: float


@dataclass(frozen=True)
class NewtonResult:
    request: NewtonRequest
    status: str
    points: tuple[NewtonPoint, ...]
    curve: tuple[tuple[float, float | None], ...]

    @property
    def final(self):
        return self.points[-1]


def parse_output(output, request):
    newton = isinstance(request, NewtonRequest)
    status = None
    points, curve = [], []
    try:
        for line in output.splitlines():
            fields = line.split("\t")
            if (
                fields[0] == ("NR" if newton else "GR")
                and len(fields) == 2
                and status is None
            ):
                status = fields[1]
            elif fields[0] == ("NP" if newton else "GP") and len(fields) == (
                6 if newton else 5
            ):
                point_type = NewtonPoint if newton else GradientPoint
                points.append(point_type(int(fields[1]), *map(float, fields[2:])))
            elif fields[0] == "GC" and len(fields) == 3:
                curve.append(
                    (float(fields[1]), None if fields[2] == "_" else float(fields[2]))
                )
            else:
                raise ValueError
        if (
            status
            not in (
                {
                    "step_tolerance",
                    "iteration_limit",
                    "numerical_limit",
                    "zero_curvature",
                }
                if newton
                else {
                    "gradient_tolerance",
                    "iteration_limit",
                    "numerical_limit",
                    "stagnation",
                }
            )
            or not points
            or len(curve) != 401
            or len(points) > request.max_iterations + (2 if newton else 1)
            or any(
                p.iteration != i
                or not all(
                    math.isfinite(v)
                    for v in (
                        p.x,
                        p.value,
                        p.gradient,
                        *((p.second_derivative,) if newton else ()),
                    )
                )
                for i, p in enumerate(points)
            )
            or any(
                not math.isfinite(x) or (y is not None and not math.isfinite(y))
                for x, y in curve
            )
        ):
            raise ValueError
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Invalid optimization response from Rust.") from exc
    result_type = NewtonResult if newton else GradientResult
    return result_type(request, status, tuple(points), tuple(curve))


def export_trace(path, result):
    from .powell import PowellRequest, export_trace as export_powell

    if isinstance(result.request, PowellRequest):
        return export_powell(path, result)
    from .normalized_gradient import (
        NormalizedRequest,
        export_trace as export_normalized,
    )

    if isinstance(result.request, NormalizedRequest):
        return export_normalized(path, result)
    from .nelder_mead import NelderMeadRequest, export_trace as export_nelder_mead

    if isinstance(result.request, NelderMeadRequest):
        return export_nelder_mead(path, result)
    from .newton2d import Newton2DRequest, export_trace as export_newton2d

    if isinstance(result.request, Newton2DRequest):
        return export_newton2d(path, result)
    from .steepest import SteepestRequest, export_trace as export_steepest

    if isinstance(result.request, SteepestRequest):
        return export_steepest(path, result)
    r = result.request
    newton = isinstance(r, NewtonRequest)
    if newton:
        return _export_newton(path, result)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "status",
                "function",
                "initial_x",
                "alpha",
                "gradient_tolerance",
                "iteration_limit",
                "k",
                "x_k",
                "f_x_k",
                "derivative_x_k",
                "delta_x",
                "x_next",
            ]
        )
        for i, p in enumerate(result.points):
            nxt = result.points[i + 1] if i + 1 < len(result.points) else None
            writer.writerow(
                [
                    result.status,
                    r.expression,
                    r.x0,
                    r.alpha,
                    r.tolerance,
                    r.max_iterations,
                    p.iteration,
                    p.x,
                    p.value,
                    p.gradient,
                    -r.alpha * p.gradient if nxt else "",
                    nxt.x if nxt else "",
                ]
            )


def _export_newton(path, result):
    r = result.request
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "algorithm",
                "status",
                "function",
                "first_derivative",
                "second_derivative",
                "initial_x",
                "step_tolerance",
                "Nmax_inclusive",
                "i",
                "x_i",
                "f_x_i",
                "derivative_x_i",
                "second_derivative_x_i",
                "delta_x",
                "x_next",
                "absolute_step",
                "step_tolerance_passed",
            ]
        )
        for i, p in enumerate(result.points):
            nxt = result.points[i + 1] if i + 1 < len(result.points) else None
            step = abs(nxt.x - p.x) if nxt else None
            writer.writerow(
                [
                    "NEWTON1D",
                    result.status,
                    r.expression,
                    r.first_derivative or "automatic",
                    r.second_derivative or "automatic",
                    r.x0,
                    r.tolerance,
                    r.max_iterations,
                    p.iteration,
                    p.x,
                    p.value,
                    p.gradient,
                    p.second_derivative,
                    -p.gradient / p.second_derivative if nxt else "",
                    nxt.x if nxt else "",
                    step if nxt else "",
                    step < r.tolerance if nxt else "",
                ]
            )
