"""Substitutions and tables for the supplied one-pass Hebbian procedure."""

from .backend import STATUS

ALGORITHM = """Hebbian Learning · supplied lecture algorithm

Step 1 · Choose η > 0 and initialize to zero
    w₁ = w₂ = … = wₙ = 0
    b = 0

Step 2 · Visit every training pair (x, t) in the given order
Make exactly ONE pass through the displayed rows. No shuffling, convergence loop or repeated epochs are part of this procedure.

Step 3 · Compute the changes for the current pattern
    Δwᵢ = η xᵢ t
    Δb  = η t

Use the supplied target t directly. Every pattern contributes, regardless of its current classification. No prediction error is used in the update.

Step 4 · Update
    wᵢ ← wᵢ + Δwᵢ
    b  ← b  + Δb

Commit each pattern's new weights and bias before visiting the next row.

Step 5 · Verify all patterns after one complete pass
Use ONE fixed final model for every verification row:
    y_in = b + Σᵢ xᵢwᵢ
    y = +1 if y_in ≥ 0, otherwise −1

The page does not specify an activation function. Verification uses the same bipolar threshold as the other worksheets, including +1 at zero. This does not affect the Hebbian updates. Training targets remain −1 or +1; inputs can be bipolar, binary, or custom finite real values.

The screenshot shows two inputs. The implementation applies the same formula to 1–32 input coordinates, with 1–128 training patterns.

Outcome
pass_complete means every training pair was processed once and final verification was performed. It does not mean every prediction is correct. Accuracy is shown separately; a single Hebbian pass need not solve every dataset (for example, bipolar XOR).

The pass summary records the largest individual update and the net final change for inspection only. Neither value is a stopping condition. Classification SSE = Σ(t − y)²: every wrong bipolar prediction contributes 4. It is a verification diagnostic, not a minimized training loss.

Numerical guard
An attempted pattern update with a non-finite delta/new coefficient or magnitude above 1e100 is rejected as a whole. In that case the pass is incomplete, and the last valid weights and processed prefix remain available. Any predictions shown then describe that partial model, not a completed Step 5.

Each Run starts again from zero. Display precision affects formatting only. CSV exports preserve full Rust f64 precision and the immutable input snapshot of the completed calculation.
"""


def snapshot(app):
    r, f = app.result.request, app.fmt
    return f"Run snapshot: Hebbian Learning · η = {f(r.eta)} · {len(r.samples[0].inputs)} inputs · {len(r.samples)} patterns · initial weights/bias = 0 · exactly one pass"


def worked_step(app, s):
    r, f = app.result, app.fmt
    sample = r.request.samples[s.pattern - 1]
    lines = [
        "STEP 1 · Initial model for this run\nAll weights and bias were initialized to zero.",
        f"STEP 2 · Training pair {s.pattern}/{len(r.request.samples)} in the given order\nx = ({', '.join(map(f, sample.inputs))}), target t = {f(sample.target)}",
        f"STEP 3 · Compute changes directly from η, x and t\nΔb = ηt = ({f(r.request.eta)})×({f(sample.target)}) = {f(s.delta_bias)}",
    ]
    for i, (x, delta) in enumerate(zip(sample.inputs, s.delta_weights), 1):
        lines.append(
            f"Δw{i} = ηx{i}t = ({f(r.request.eta)})×({f(x)})×({f(sample.target)}) = {f(delta)}"
        )
    lines.append(
        f"STEP 4 · Apply the changes\nb ← b + Δb = ({f(s.before_bias)}) + ({f(s.delta_bias)}) = {f(s.after_bias)}"
    )
    for i, (before, delta, after) in enumerate(
        zip(s.before_weights, s.delta_weights, s.after_weights), 1
    ):
        lines.append(f"w{i} ← w{i} + Δw{i} = ({f(before)}) + ({f(delta)}) = {f(after)}")
    if s.pattern == len(r.request.samples):
        lines.append(
            f"STEP 5 · One complete pass finished; verify ALL patterns\nFinal weights = ({', '.join(map(f, r.weights))}), bias = {f(r.bias)}\nUse these SAME coefficients for every verification row."
        )
        for p in r.predictions:
            inputs = r.request.samples[p.pattern - 1]
            terms = " + ".join(
                f"({f(x)})×({f(w)})" for x, w in zip(inputs.inputs, r.weights)
            )
            lines.append(
                f"Pattern {p.pattern}: y_in = ({f(r.bias)}) + {terms} = {f(p.net)}\ny = {p.output:+d}, target = {f(inputs.target)} → {'correct' if p.output == inputs.target else 'incorrect'}"
            )
        lines.append(
            f"One pass complete · accuracy = {r.correct}/{len(r.request.samples)}\nClassification SSE = {f(r.sse)}. No second pass is performed."
        )
    else:
        lines.append(
            "STEP 5 · Verification waits until all training pairs have been processed once."
        )
    if app.step_index == len(r.steps) - 1 and r.failed_at:
        lines.append(
            f"The NEXT attempted update, pattern {r.failed_at[1]}, reached the numerical guard and was rejected. This pass is incomplete; final-model predictions are diagnostics for the last valid partial model."
        )
    app._set_text(app.worked_text, "\n\n".join(lines))


def render_trace(app, page_size):
    r, f = app.result, app.fmt
    n = len(r.weights)
    app._columns(
        app.trace_tree,
        [
            "Pass",
            "Pattern",
            *[f"x{i + 1}" for i in range(n)],
            "Target",
            *[f"w{i + 1} before" for i in range(n)],
            "b before",
            *[f"Δw{i + 1}" for i in range(n)],
            "Δb",
            *[f"w{i + 1} after" for i in range(n)],
            "b after",
            "Max |Δ|",
        ],
    )
    start = app.trace_page * page_size
    for index in range(start, min(start + page_size, len(r.steps))):
        s = r.steps[index]
        sample = r.request.samples[s.pattern - 1]
        app.trace_tree.insert(
            "",
            "end",
            iid=str(index),
            values=[
                1,
                s.pattern,
                *map(f, sample.inputs),
                f(sample.target),
                *map(f, s.before_weights),
                f(s.before_bias),
                *map(f, s.delta_weights),
                f(s.delta_bias),
                *map(f, s.after_weights),
                f(s.after_bias),
                f(s.max_change),
            ],
        )
    app._page_status("trace", len(r.steps))


def render_epochs(app, page_size):
    r, f = app.result, app.fmt
    app._columns(
        app.epoch_tree,
        [
            "Pass",
            "Max |Δ|",
            "Net pass Δ",
            "Class SSE",
            "Correct",
            *[f"w{i + 1}" for i in range(len(r.weights))],
            "Bias",
        ],
    )
    for e in r.epochs:
        app.epoch_tree.insert(
            "",
            "end",
            iid="1",
            values=[
                1,
                f(e.max_change),
                f(e.net_change),
                f(e.sse),
                f"{e.correct}/{len(r.request.samples)}",
                *map(f, e.weights),
                f(e.bias),
            ],
        )
    app._page_status("epoch", len(r.epochs))


def render_final(app):
    r, f = app.result, app.fmt
    equation = (
        " + ".join(f"({f(w)})x{i + 1}" for i, w in enumerate(r.weights))
        + f" + ({f(r.bias)})"
    )
    note = (
        "Every training pair was processed exactly once. Verification uses the final model on all patterns."
        if r.status == "pass_complete"
        else f"Pattern {r.failed_at[1]} was rejected by the numerical guard. The pass is incomplete; these predictions describe the last valid partial model."
    )
    app._set_text(
        app.final_text,
        f"Hebbian Learning · {STATUS[r.status]}\n{note}\n\ny_in = {equation}\ny = +1 if y_in ≥ 0, otherwise −1.\n\nAccuracy = {r.correct}/{len(r.request.samples)}    Classification SSE = {f(r.sse)}\nComplete passes: {r.completed_epochs}; committed pattern updates: {len(r.steps)}.",
    )
    app._columns(
        app.prediction_tree,
        [
            "Pattern",
            *[f"x{i + 1}" for i in range(len(r.weights))],
            "Target",
            "Final y_in",
            "Final y",
            "Correct",
        ],
    )
    for p in r.predictions:
        sample = r.request.samples[p.pattern - 1]
        app.prediction_tree.insert(
            "",
            "end",
            values=[
                p.pattern,
                *map(f, sample.inputs),
                f(sample.target),
                f(p.net),
                p.output,
                "Yes" if p.output == sample.target else "No",
            ],
        )
