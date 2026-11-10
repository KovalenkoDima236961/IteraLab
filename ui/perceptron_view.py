"""Worked calculations for binary, bipolar sign, and threshold-band perceptrons."""

from ..backend import STATUS
from ..algorithms.perceptron import MODE_LABELS

ALGORITHM = """Perceptron · binary, bipolar sign, and threshold-band variants

The supplied image combines binary and threshold-band rules. The worksheet also provides the bipolar sign perceptron. Choose the activation and target encoding used by your example.

Binary (0/1) · upper section
    net = b + Σ xᵢwᵢ
    y = 1 if net ≥ 0, otherwise 0
    e = t − y
    Δwᵢ = η e xᵢ
    Δb = η e

Targets are 0 or 1. At net = 0 the output is 1. Correct patterns have e = 0 and leave every coefficient unchanged.

Bipolar (−1/+1) · sign activation
    net = b + Σ xᵢwᵢ
    y = +1 if net ≥ 0, otherwise −1

Targets and predictions are −1 or +1; there is no neutral class. At net = 0 the output is +1. On a mistake, apply the lecture's target-based update:
    Δwᵢ = α t xᵢ
    Δb = α t
On a correct prediction, all deltas are zero. The classification error t − y is ±2 on a mistake, but this update uses αt, not α(t−y).

Bipolar · error update (CV6Perceptron)
    y = +1 if net ≥ 0, otherwise −1
    e = t − y
    Δwᵢ = α e xᵢ; Δb = α e
This is the CV6 handwritten rule: a mistaken bipolar prediction gives e=±2. It uses the same activation as Bipolar sign but twice the update at the same α. Select it to reproduce the handwritten AND, OR, XOR and majority-vote calculations.

Threshold band (−1/0/+1) · Steps 0–6 in the lower section
    net = b + Σ xᵢwᵢ
    y = +1 if net > θ
    y =  0 if −θ ≤ net ≤ θ
    y = −1 if net < −θ

Targets are −1 or +1. If y ≠ t:
    Δwᵢ = α t xᵢ
    Δb = α t
Otherwise every coefficient remains unchanged. This rule uses t in the update. In particular, when t = +1 and y = −1, e = 2 is shown for inspection but the bias update is α, not 2α. Both band boundaries belong to output 0. With θ = 0, net zero still gives output 0.

Step 0 · Initialize weights and bias; choose the learning rate
Selection defaults to zeros and a learning rate of 1. Manual initial values and seeded small random values are available. The supplied initialization range is 0 < rate ≤ 1. Theta is finite and nonnegative in threshold-band mode; binary and bipolar sign modes use threshold zero.

Steps 1–2 · Repeat complete epochs
Visit all training pairs in the displayed row order, without shuffling. Each pattern uses the model left by the previous pattern.

Step 3 · Set activations
    xᵢ = sᵢ

Step 4 · Compute net input and the selected activation function
Use the OLD weights and bias to calculate every update for the current pattern.

Step 5 · Apply the selected update rule
Record the old coefficients, output, classification error, every delta, new coefficients, and whether any coefficient actually changed. Correct predictions never update.

Step 6 · Test after a complete epoch
Stop only when no weight or bias changed anywhere in that full pass. Individual updates can cancel, so equality of the starting and ending models is insufficient. Perfect final classification after a changing epoch does not replace the required unchanged pass.

Outcomes
weights_stable: no coefficient changed during the last complete pass. Accuracy remains visible separately, including rare floating-point stagnation.
epoch_limit: the calculation budget ended before an unchanged pass. Nonseparable patterns such as XOR can reach this limit.
numerical_limit: a whole attempted pattern update was rejected because an intermediate or new coefficient was non-finite or exceeded 1e100 in magnitude. The last valid model and incomplete epoch are identified.

Diagnostics
Mistakes counts incorrect predictions as patterns were visited during training. Updates counts pattern visits that actually changed coefficients. Final accuracy and classification SSE = Σ(t−y)² are evaluated using one fixed model at each completed epoch endpoint. They do not control stopping. Binary SSE equals the number of wrong predictions. Bipolar sign errors contribute 4 each. Band-mode wrong outputs can contribute 1 (output 0) or 4 (opposite class).

The image has two inputs; the same sums support 1–32 input coordinates. Presets use binary inputs for the binary variant and bipolar inputs for the bipolar sign and band variants by default. The input encoding can be changed independently, and custom finite real inputs are accepted. Targets must match the selected variant. Custom data is never silently recoded.

Rust f64 performs all calculations; display precision only changes formatting. CSV exports preserve the full run snapshot, mode, theta, learning rate, row order and every recorded value.
"""


def snapshot(app):
    r, f = app.result.request, app.fmt
    threshold = (
        "net ≥ 0 → 1"
        if r.mode == "binary"
        else "net ≥ 0 → +1; otherwise −1"
        if r.mode in ("bipolar", "bipolar_error")
        else f"θ = {f(r.theta)}"
    )
    return f"Run snapshot: Perceptron · {MODE_LABELS[r.mode]} · rate = {f(r.alpha)} · {threshold} · {len(r.weights)} inputs · {len(r.samples)} patterns · initial w = ({', '.join(map(f, r.weights))}), b = {f(r.bias)}"


def activation_text(request, net, output, f):
    if request.mode == "binary":
        return f"y = {output} because net {'≥' if net >= 0 else '<'} 0."
    if request.mode in ("bipolar", "bipolar_error"):
        return f"y = {output:+d} because net {'≥' if net >= 0 else '<'} 0 (bipolar sign activation)."
    if net > request.theta:
        return f"y = +1 because net > θ = {f(request.theta)}."
    if net < -request.theta:
        return f"y = −1 because net < −θ = {f(-request.theta)}."
    return f"y = 0 because −θ ≤ net ≤ θ, including both boundaries (θ = {f(request.theta)})."


def worked_step(app, s):
    r, f = app.result, app.fmt
    request = r.request
    sample = request.samples[s.pattern - 1]
    terms = " + ".join(
        f"({f(x)})×({f(w)})" for x, w in zip(sample.inputs, s.before_weights)
    )
    lines = [
        f"STEP 3 · Set inputs\nx = s = ({', '.join(map(f, sample.inputs))}), target t = {f(sample.target)}",
        f"STEP 4 · Compute the OLD model's net input and activation\nnet = b + Σ xᵢwᵢ\n    = ({f(s.before_bias)}) + {terms}\n    = {f(s.net)}\n{activation_text(request, s.net, s.output, f)}",
        f"Classification error e = t − y = ({f(sample.target)}) − ({s.output}) = {f(s.error)}",
    ]
    if s.error == 0:
        lines.append(
            "STEP 5 · y = t: correct prediction\nAll deltas are zero; weights and bias remain unchanged."
        )
    else:
        factor = (
            s.error if request.mode in ("binary", "bipolar_error") else sample.target
        )
        label = "ηe" if request.mode in ("binary", "bipolar_error") else "αt"
        lines.append(
            f"STEP 5 · y ≠ t: apply the {MODE_LABELS[request.mode]} rule\nΔb = {label} = ({f(request.alpha)})×({f(factor)}) = {f(s.delta_bias)}"
        )
        for i, (x, delta) in enumerate(zip(sample.inputs, s.delta_weights), 1):
            lines.append(
                f"Δw{i} = {label}x{i} = ({f(request.alpha)})×({f(factor)})×({f(x)}) = {f(delta)}"
            )
        lines.append(
            f"b(new) = ({f(s.before_bias)}) + ({f(s.delta_bias)}) = {f(s.after_bias)}"
        )
        for i, (before, delta, after) in enumerate(
            zip(s.before_weights, s.delta_weights, s.after_weights), 1
        ):
            lines.append(f"w{i}(new) = ({f(before)}) + ({f(delta)}) = {f(after)}")
    lines.append(
        f"Any coefficient actually changed in this pattern? {'Yes' if s.changed else 'No'}. Largest requested |Δ| = {f(s.max_change)}."
    )
    if s.pattern == len(request.samples):
        e = r.epochs[s.epoch - 1]
        outcome = (
            "No weights or bias changed anywhere in this pass: stop."
            if e.updates == 0
            else "Epoch budget exhausted; stop with epoch_limit."
            if e.number == request.max_epochs
            else "Changes occurred: continue with the next complete pass."
        )
        lines.append(
            f"STEP 6 · Complete epoch {e.number}\nTraining mistakes = {e.mistakes}; pattern visits with changed coefficients = {e.updates}\n{outcome}\nNet difference between epoch endpoints = {f(e.net_change)}\nFixed-model accuracy = {e.correct}/{len(request.samples)}; classification SSE = {f(e.sse)}"
        )
    else:
        lines.append(
            "STEP 6 · Wait until every pattern in this epoch has been processed."
        )
    if app.step_index == len(r.steps) - 1 and r.failed_at:
        lines.append(
            f"The NEXT attempted update at epoch {r.failed_at[0]}, pattern {r.failed_at[1]} reached the numerical guard and was not committed."
        )
    app._set_text(app.worked_text, "\n\n".join(lines))


def render_trace(app, page_size):
    r, f = app.result, app.fmt
    n = len(r.weights)
    app._columns(
        app.trace_tree,
        [
            "Epoch",
            "Pattern",
            *[f"x{i + 1}" for i in range(n)],
            "Target",
            *[f"w{i + 1} before" for i in range(n)],
            "b before",
            "Net",
            "y",
            "e = t−y",
            "Changed",
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
                s.epoch,
                s.pattern,
                *map(f, sample.inputs),
                f(sample.target),
                *map(f, s.before_weights),
                f(s.before_bias),
                f(s.net),
                s.output,
                f(s.error),
                "Yes" if s.changed else "No",
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
            "Epoch",
            "Mistakes",
            "Changed visits",
            "Max |Δ|",
            "Net epoch Δ",
            "Class SSE",
            "Correct",
            *[f"w{i + 1}" for i in range(len(r.weights))],
            "Bias",
        ],
    )
    start = app.epoch_page * page_size
    for e in r.epochs[start : start + page_size]:
        app.epoch_tree.insert(
            "",
            "end",
            iid=str(e.number),
            values=[
                e.number,
                e.mistakes,
                e.updates,
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
    outcome = {
        "weights_stable": "No coefficient changed anywhere in the last complete pass. Accuracy is shown separately.",
        "epoch_limit": "The epoch budget ended before a complete unchanged pass occurred.",
        "numerical_limit": f"The update at epoch {r.failed_at[0]}, pattern {r.failed_at[1]} was rejected; predictions describe the last valid partial model."
        if r.failed_at
        else "Numerical guard reached.",
    }[r.status]
    activation = (
        "y = 1 if net ≥ 0, otherwise 0."
        if r.request.mode == "binary"
        else "y = +1 if net ≥ 0, otherwise −1."
        if r.request.mode in ("bipolar", "bipolar_error")
        else f"y = +1 if net > {f(r.request.theta)}; y = −1 if net < {f(-r.request.theta)}; otherwise y = 0."
    )
    app._set_text(
        app.final_text,
        f"Perceptron · {MODE_LABELS[r.request.mode]} · {STATUS[r.status]}\n{outcome}\n\nnet = {equation}\n{activation}\n\nAccuracy = {r.correct}/{len(r.request.samples)}    Classification SSE = {f(r.sse)}\nComplete epochs: {r.completed_epochs}; recorded pattern visits: {len(r.steps)}.",
    )
    app._columns(
        app.prediction_tree,
        [
            "Pattern",
            *[f"x{i + 1}" for i in range(len(r.weights))],
            "Target",
            "Final net",
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
