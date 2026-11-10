"""Full-Hessian Newton formulas, substituted systems and diagnostics for the 2D view."""

from ..algorithms.optimization import STATUS
from .steepest_view import in_view

NOTES = """NEWTON'S METHOD · TWO DIMENSIONS · SUPPLIED SLIDE

The method uses both the gradient and the full Hessian at the SAME OLD pair:
    g = ∇f(xₖ,yₖ) = (fₓ, fᵧ)
    H = [[fₓₓ, fₓᵧ], [fᵧₓ, fᵧᵧ]]
    [xₖ₊₁,yₖ₊₁]ᵀ = [xₖ,yₖ]ᵀ − H⁻¹g

The function must be twice differentiable near the visited points. Rust obtains the gradient and all Hessian entries, including the mixed partial, using second-order forward automatic differentiation and the chain rule. For these smooth expressions fₓᵧ=fᵧₓ. No finite differences are used. Mixed derivatives are propagated directly, rather than recovered by subtracting large diagonal derivatives.

Full matrix update
Let D=fₓₓ fᵧᵧ−fₓᵧ fᵧₓ. When H is invertible:
    H⁻¹ = (1/D) [[fᵧᵧ, −fₓᵧ], [−fᵧₓ, fₓₓ]]
    sₓ = (fᵧᵧ fₓ − fₓᵧ fᵧ)/D
    sᵧ = (fₓₓ fᵧ − fᵧₓ fₓ)/D
    xₖ₊₁ = xₖ − sₓ,  yₖ₊₁ = yₖ − sᵧ.
Both gradient components enter the coupled update. It is not two independent scalar Newton divisions. Both coordinates are calculated before committing the new pair. There is no learning rate, damping, line search, diagonal approximation, Hessian regularization or replacement of negative curvature.

Numerical system
Rust computes s by solving Hs=g, which is algebraically the inverse-Hessian update in the image. It scales both sides by scale=max(|fₓₓ|,|fₓᵧ|,|fᵧᵧ|). For A=H/scale=[[a,b],[b,c]], D_scaled=ac−b²:
    sₓ = (c(gₓ/scale)−b(gᵧ/scale))/D_scaled
    sᵧ = (a(gᵧ/scale)−b(gₓ/scale))/D_scaled.
Scaling preserves the Newton system and avoids underflow in the unscaled determinant. The trace reports scale and det(H/scale). A zero scale or zero scaled determinant reports Singular Hessian at working precision, without committing a step. Small nonzero determinants are used literally. No pseudoinverse or fallback method is substituted.

Worked example
f=(x−2)²+(x−2)(y+1)+2(y+1)²; (x₀,y₀)=(6,3).
g=(12,20), H=[[2,1],[1,4]]. The solution of Hs=g is s=(4,4).
The new pair is (6,3)−(4,4)=(2,−1), whose gradient is zero.
The off-diagonal entries are essential: independent scalar divisions would produce a different pair.

Stopping convention
The supplied 2D slide contains no stopping rule. As in the existing 2D gradient workspace, this worksheet explicitly stops when ||∇f||₂ ≤ ε, including at the initial pair, or after the selected maximum number of updates (1–100,000). This is independent of the 1D Newton slide's strict post-update step test. A tiny step alone is not convergence. If both coordinates round to their old values while the gradient test fails, the status is Floating-point stagnation. An already stationary pair performs zero updates; no inverse is needed, even when its Hessian is singular.

Curvature and minima
An invertible indefinite or negative definite Hessian is used as written. Newton can reach a saddle or maximum, increase the objective, or cycle. Gradient tolerance alone does not prove a minimum. The Hessian's positive/negative definite, indefinite or singular classification is displayed as a diagnostic; it never changes the recurrence. A positive definite Hessian with a zero gradient indicates a strict local minimum, not a claim of global optimality.

Inputs and limits
Enter f(x,y), x₀, y₀, gradient tolerance and the update budget. The gradient and Hessian are automatic. Use x, y, pi, e, + − * / ^ or **, parentheses, sin, cos, tan, exp, ln/log and sqrt. Use explicit multiplication. Trigonometric arguments use radians; log is natural logarithm. Variable exponents require a positive base for differentiation. Values, gradient/Hessian entries and intermediate expression results must be real, finite and at most 1e100 in magnitude. Both attempted coordinate steps and new coordinates have the same guard. A failed attempted pair is rejected atomically; invalid starts, input errors and cancellation retain the prior completed result.

The function heatmap and path share the existing 2D plotting controls; ranges affect the view only. Gray cells are invalid function samples. Worked steps shows the old gradient, every Hessian entry, the inverse formula, numerical system and simultaneous update. The complete table and CSV include gradient, full symmetric Hessian, scaled determinant, scale, actual Newton coordinate steps and next pair. The final row has no outgoing update. Every method retains its own form, result and navigation in the same application window.
"""

TRACE_COLUMNS = [
    "k",
    "xₖ",
    "yₖ",
    "f(xₖ,yₖ)",
    "∂f/∂x",
    "∂f/∂y",
    "||∇f||₂",
    "Hₓₓ",
    "Hₓᵧ",
    "Hᵧₓ",
    "Hᵧᵧ",
    "H scale",
    "det(H/scale)",
    "Δx",
    "Δy",
    "xₖ₊₁",
    "yₖ₊₁",
]


def curvature(point):
    if point.hessian_scale == 0 or point.determinant_scaled == 0:
        return "singular at working precision"
    if point.determinant_scaled < 0:
        return "indefinite"
    return "positive definite" if point.hessian_xx > 0 else "negative definite"


def render_summary(panel):
    r, f = panel.result, panel.fmt
    p, request = r.final, r.request
    panel.snapshot.set(
        f"{request.algorithm_name}: f(x,y) = {request.expression}  ·  (x₀,y₀) = ({f(request.x0)}, {f(request.y0)})  ·  ε = {f(request.tolerance)}"
    )
    detail = {
        "gradient_tolerance": "The gradient norm test passed. Inspect curvature; stationarity alone does not prove a minimum.",
        "iteration_limit": "The update budget ended before the gradient norm test passed.",
        "singular_hessian": "The Hessian is singular at working precision. Its inverse is unavailable; no further step was committed.",
        "numerical_limit": "The next attempted pair failed a numerical/domain guard; neither coordinate was committed.",
        "stagnation": "Both new coordinates round to their old values while the gradient norm test still fails.",
    }[r.status]
    increases = sum(b.value > a.value for a, b in zip(r.points, r.points[1:]))
    outside = sum(not in_view(q, request) for q in r.points)
    panel.app._set_text(
        panel.summary_text,
        f"{STATUS[r.status]}\nEstimate (x,y) = ({f(p.x)}, {f(p.y)})     f(x,y) = {f(p.value)}\n∇f = ({f(p.gradient_x)}, {f(p.gradient_y)})     ||∇f||₂ = {f(p.gradient_norm)}\nH = [[{f(p.hessian_xx)}, {f(p.hessian_xy)}], [{f(p.hessian_xy)}, {f(p.hessian_yy)}]] · {curvature(p)}\nCompleted updates: {p.iteration:,}   ·   ||∇f||₂ ≤ ε: {p.gradient_norm <= request.tolerance}\n{detail}\nObjective increases: {increases:,}   ·   Estimates outside heatmap: {outside:,}",
    )


def worked_step(panel):
    r, f, i = panel.result, panel.fmt, panel.step_index
    p, request = r.points[i], r.request
    lines = [
        f"ITERATION k = {p.iteration}\nf(x,y) = {request.expression}",
        f"SAME OLD pair (xₖ,yₖ) = ({f(p.x)}, {f(p.y)})\nf(xₖ,yₖ) = {f(p.value)}\n∇f = ({f(p.gradient_x)}, {f(p.gradient_y)})",
        f"FULL HESSIAN · at the OLD pair\nfₓₓ = {f(p.hessian_xx)}\nfₓᵧ = fᵧₓ = {f(p.hessian_xy)}\nfᵧᵧ = {f(p.hessian_yy)}\nH = [[{f(p.hessian_xx)}, {f(p.hessian_xy)}],\n     [{f(p.hessian_xy)}, {f(p.hessian_yy)}]]\nCurvature: {curvature(p)}",
        f"Stopping test: ||∇f||₂ ≤ ε\n{f(p.gradient_norm)} ≤ {f(request.tolerance)} → {str(p.gradient_norm <= request.tolerance).upper()}",
    ]
    if i + 1 < len(r.points):
        nxt = r.points[i + 1]
        lines.extend(
            [
                "NEWTON MATRIX UPDATE\n[xₖ₊₁, yₖ₊₁]ᵀ = [xₖ, yₖ]ᵀ − H⁻¹∇f\nH⁻¹ = (1 / (HₓₓHᵧᵧ − HₓᵧHᵧₓ)) × [[Hᵧᵧ, −Hₓᵧ], [−Hᵧₓ, Hₓₓ]]",
                f"Solve the complete system Hs = ∇f\n({f(p.hessian_xx)})sₓ + ({f(p.hessian_xy)})sᵧ = {f(p.gradient_x)}\n({f(p.hessian_xy)})sₓ + ({f(p.hessian_yy)})sᵧ = {f(p.gradient_y)}\nscale = {f(p.hessian_scale)}; det(H/scale) = {f(p.determinant_scaled)}\ns = H⁻¹∇f = ({f(-p.delta_x)}, {f(-p.delta_y)})",
                f"Δ = −H⁻¹∇f = ({f(p.delta_x)}, {f(p.delta_y)})\nxₖ₊₁ = ({f(p.x)}) + ({f(p.delta_x)}) = {f(nxt.x)}\nyₖ₊₁ = ({f(p.y)}) + ({f(p.delta_y)}) = {f(nxt.y)}",
                f"Commit BOTH coordinates together: ({f(nxt.x)}, {f(nxt.y)})\nf(xₖ₊₁,yₖ₊₁) = {f(nxt.value)}\nRecompute the gradient and full Hessian at the new pair in iteration {nxt.iteration}.",
            ]
        )
    else:
        if r.status == "singular_hessian":
            lines.append(
                f"Cannot invert the singular Hessian.\nscale = {f(p.hessian_scale)}; det(H/scale) = {f(p.determinant_scaled)}\nNo diagonal replacement or pseudoinverse is used."
            )
        lines.append(f"Stop: {STATUS[r.status]}. No further update was committed.")
    return "\n\n".join(lines)


def trace_values(panel, index):
    r, f = panel.result, panel.fmt
    p = r.points[index]
    nxt = r.points[index + 1] if index + 1 < len(r.points) else None
    return [
        p.iteration,
        f(p.x),
        f(p.y),
        f(p.value),
        f(p.gradient_x),
        f(p.gradient_y),
        f(p.gradient_norm),
        f(p.hessian_xx),
        f(p.hessian_xy),
        f(p.hessian_xy),
        f(p.hessian_yy),
        f(p.hessian_scale),
        f(p.determinant_scaled),
        f(p.delta_x) if nxt else "—",
        f(p.delta_y) if nxt else "—",
        f(nxt.x) if nxt else "—",
        f(nxt.y) if nxt else "—",
    ]
