"""MADALINE worksheet presentation; training and predictions come from Rust."""

from .backend import STATUS
from .madaline import weight_names

ALGORITHM = """MADALINE · Rule I (MRI) · supplied lecture algorithm

Activation for all hidden units Z1 … ZH and output Y
    f(a) = +1 if a ≥ 0; otherwise −1.

Step 0 · Initialize
Choose H hidden ADALINE units (1–32). Set one input-weight vector and bias per unit. Small random initial values are usual. Set α as in ADALINE. The H output weights v1 … vH and output bias b_out stay FIXED throughout training.

The page does not specify numeric output constants. Editable defaults vⱼ = 0.5 and b_out = (H−1)/2 implement bipolar OR of all H hidden activations. For H = 2 this gives the earlier v1 = v2 = b3 = 0.5. Changing the hidden count restores these output defaults and keeps the existing hidden-unit inputs. Enter the constants from your particular exercise if different.

Steps 1–2 · Repeat complete epochs in the displayed row order
No shuffling is performed. Maximum epochs counts full passes (Step 2), not the number of modified hidden units.

Step 3 · Set inputs
    xᵢ = sᵢ

Step 4 · Compute hidden net inputs
    z_inⱼ = bⱼ + Σᵢ xᵢwᵢⱼ, for j = 1 … H

Step 5 · Activate hidden units
    zⱼ = f(z_inⱼ), for j = 1 … H

Step 6 · Compute output
    y_in = b_out + Σⱼ zⱼvⱼ
    y = f(y_in)

Step 7 · Determine the error case
If y = t: do not update any weight or bias.

Otherwise, if t = +1: select ONE hidden unit whose net input is closest to zero (smallest |z_in|) across ALL H units. Equal distances select the lowest-index tied unit.
    bⱼ(new)  = bⱼ(old)  + α(1 − z_inⱼ)
    wᵢⱼ(new) = wᵢⱼ(old) + α(1 − z_inⱼ)xᵢ

Otherwise, if t = −1: update ALL hidden units with POSITIVE net input (z_inⱼ > 0).
    bⱼ(new)  = bⱼ(old)  + α(−1 − z_inⱼ)
    wᵢⱼ(new) = wᵢⱼ(old) + α(−1 − z_inⱼ)xᵢ

Every selected unit uses its own pre-update z_in. Nonselected units and the entire output layer stay unchanged. All updates for a pattern are committed together before the next pattern.

Zero convention
The page's “positive net input” is implemented literally as > 0, although activation at exactly zero is +1. A zero-net unit is excluded from the negative-target update. Consequently, zero initialization can stop with classification errors. Seeded small random initialization is the default for MADALINE.

Step 8 · Stop after a complete epoch
The page permits stopped changes, an acceptable level of changes, or a maximum number of epochs. This worksheet reports:
• weights_stable: the largest individual hidden weight/bias change was exactly zero;
• tolerance_met: the largest absolute individual change across all hidden units and every pattern was strictly below ε;
• epoch_limit: the complete-epoch budget ended before either condition passed.

Bias changes are included. Net epoch endpoint change and classification accuracy are diagnostics; they do not replace this stopping test. A stable model may still misclassify patterns, and accuracy is always shown separately.

Diagnostics and safeguards
Classification SSE = Σ(t − y)², evaluated using a fixed final model on all patterns. Each wrong bipolar prediction contributes 4. It is not the ADALINE linear-response SSE and does not drive training.

A numerical guard rejects a whole attempted pattern update if any intermediate or new hidden coefficient is non-finite or exceeds 1e100 in magnitude. The last valid model and incomplete epoch are identified. No completed trace rows are dropped.

The screenshot illustrates two inputs and two hidden units. This worksheet generalizes the same Rule I to 1–32 input coordinates and 1–32 hidden units. Targets and all activations remain bipolar. Display precision affects formatting only; CSV keeps full Rust f64 precision and the original run snapshot. Multi-digit weight indices use an underscore, e.g. w1_12, to avoid ambiguous labels.
"""


def snapshot(app):
    r, f = app.result.request, app.fmt
    return (
        f"Run snapshot: MADALINE MRI · α = {f(r.alpha)} · ε = {f(r.tolerance)} · {len(r.hidden_weights[0])} inputs · "
        f"{len(r.hidden_weights)} hidden units · {len(r.samples)} patterns · fixed v = ({', '.join(map(f, r.output_weights))}), b_out = {f(r.output_bias)}"
    )


def worked_step(app, s):
    result, f = app.result, app.fmt
    r = result.request
    sample = r.samples[s.pattern - 1]
    n = len(sample.inputs)
    units = len(r.hidden_weights)
    names = weight_names(n, units)
    lines = [
        f"STEP 3 · Set input activations\nx = s = ({', '.join(map(f, sample.inputs))})    target t = {f(sample.target)}"
    ]
    hidden = []
    for j in range(units):
        products = " + ".join(
            f"({f(x)})×({f(w)})"
            for x, w in zip(sample.inputs, s.before_weights[j * n : (j + 1) * n])
        )
        hidden.append(
            f"z_in{j + 1} = b{j + 1} + Σ xᵢwᵢ{j + 1}\n      = ({f(s.before_biases[j])}) + {products}\n      = {f(s.nets[j])}"
        )
    lines.append(
        f"STEP 4 · Compute ALL {units} hidden net inputs from the OLD model\n"
        + "\n\n".join(hidden)
    )
    lines.append(
        "STEP 5 · Activate the hidden units\n"
        + "\n".join(
            f"z{j + 1} = f({f(s.nets[j])}) = {s.hidden_outputs[j]:+d} (net {'≥' if s.nets[j] >= 0 else '<'} 0)"
            for j in range(units)
        )
    )
    terms = " + ".join(
        f"({s.hidden_outputs[j]})×({f(r.output_weights[j])})" for j in range(units)
    )
    lines.append(
        f"STEP 6 · Compute the output with FIXED output weights and bias\ny_in = b_out + Σ zⱼvⱼ\n     = ({f(r.output_bias)}) + {terms}\n     = {f(s.net)}\ny = f(y_in) = {s.output:+d}"
    )
    if s.rule == "correct":
        lines.append(
            "STEP 7 · y = t: correct output\nNo weights or biases are updated."
        )
    elif s.rule == "positive_closest":
        chosen = next(j + 1 for j, selected in enumerate(s.selected) if selected)
        distances = "; ".join(
            f"|z_in{j + 1}| = {f(abs(net))}" for j, net in enumerate(s.nets)
        )
        lines.append(
            f"STEP 7 · y ≠ t, t = +1: choose the unit closest to zero\n{distances}\nSelect Z{chosen}. Equal distances select the lowest-index tied unit."
        )
    else:
        chosen = ", ".join(f"Z{j + 1}" for j in range(units) if s.selected[j]) or "none"
        lines.append(
            f"STEP 7 · y ≠ t, t = −1: select ALL units with z_in > 0\nSelected units: {chosen}. Zero-net units are excluded."
        )
    for j in range(units):
        if not s.selected[j]:
            lines.append(f"Z{j + 1}: unchanged; b{j + 1} = {f(s.after_biases[j])}.")
            continue
        lines.append(
            f"Z{j + 1}: e{j + 1} = ({f(sample.target)}) − ({f(s.nets[j])}) = {f(s.errors[j])}\nΔb{j + 1} = αe{j + 1} = ({f(r.alpha)})×({f(s.errors[j])}) = {f(s.delta_biases[j])}\nb{j + 1}(new) = ({f(s.before_biases[j])}) + ({f(s.delta_biases[j])}) = {f(s.after_biases[j])}"
        )
        for i, x in enumerate(sample.inputs):
            k = j * n + i
            name = names[k]
            lines.append(
                f"{name}(new) = {name}(old) + αe{j + 1}x{i + 1}\n         = ({f(s.before_weights[k])}) + ({f(s.delta_biases[j])})×({f(x)})\n         = ({f(s.before_weights[k])}) + ({f(s.delta_weights[k])}) = {f(s.after_weights[k])}"
            )
    lines.append(
        f"Output layer remains fixed. Largest |Δ| for this pattern = {f(s.max_change)}."
    )
    if s.pattern == len(r.samples):
        e = result.epochs[s.epoch - 1]
        outcome = (
            "Weight changes stopped: stop."
            if e.max_change == 0
            else "Acceptable change level reached: stop."
            if e.max_change < r.tolerance
            else "Epoch budget exhausted: stop with epoch_limit."
            if e.number == r.max_epochs
            else "Continue with the next epoch."
        )
        lines.append(
            f"STEP 8 · Complete epoch {e.number}: test stopping condition\nLargest individual hidden weight/bias change = {f(e.max_change)}\nmax |Δ| = 0? {'Yes' if e.max_change == 0 else 'No'}.  max |Δ| < ε? {f(e.max_change)} < {f(r.tolerance)} → {'TRUE' if e.max_change < r.tolerance else 'FALSE'}\n{outcome}\nNet epoch endpoint change = {f(e.net_change)}\nClassification SSE = {f(e.sse)}; correct = {e.correct}/{len(r.samples)}"
        )
    else:
        lines.append("STEP 8 · Wait until every pattern in this epoch is processed.")
    if app.step_index == len(result.steps) - 1 and result.failed_at:
        lines.append(
            f"Numerical guard reached on the NEXT attempted update: epoch {result.failed_at[0]}, pattern {result.failed_at[1]}. The whole attempted update was rejected."
        )
    app._set_text(app.worked_text, "\n\n".join(lines))


def render_trace(app, page_size):
    r, f = app.result, app.fmt
    n = len(r.request.hidden_weights[0])
    units = len(r.request.hidden_weights)
    names = weight_names(n, units)
    app._columns(
        app.trace_tree,
        [
            "Epoch",
            "Pattern",
            *[f"x{i + 1}" for i in range(n)],
            "Target",
            *[f"z_in{j + 1}" for j in range(units)],
            *[f"z{j + 1}" for j in range(units)],
            "y_in",
            "y",
            "Rule",
            *[f"Z{j + 1} selected" for j in range(units)],
            *[f"e{j + 1}" for j in range(units)],
            *[s + " before" for s in names],
            *[f"b{j + 1} before" for j in range(units)],
            *["Δ" + s for s in names],
            *[f"Δb{j + 1}" for j in range(units)],
            *[s + " after" for s in names],
            *[f"b{j + 1} after" for j in range(units)],
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
                *map(f, s.nets),
                *s.hidden_outputs,
                f(s.net),
                s.output,
                s.rule,
                *["Yes" if v else "No" for v in s.selected],
                *map(f, s.errors),
                *map(f, s.before_weights),
                *map(f, s.before_biases),
                *map(f, s.delta_weights),
                *map(f, s.delta_biases),
                *map(f, s.after_weights),
                *map(f, s.after_biases),
                f(s.max_change),
            ],
        )
    app._page_status("trace", len(r.steps))


def render_epochs(app, page_size):
    r, f = app.result, app.fmt
    units = len(r.request.hidden_weights)
    app._columns(
        app.epoch_tree,
        [
            "Epoch",
            "Max |Δ|",
            "Net epoch Δ",
            "Class SSE",
            "Correct",
            *weight_names(len(r.request.hidden_weights[0]), units),
            *[f"b{j + 1}" for j in range(units)],
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
                f(e.max_change),
                f(e.net_change),
                f(e.sse),
                f"{e.correct}/{len(r.request.samples)}",
                *map(f, e.weights),
                *map(f, e.biases),
            ],
        )
    app._page_status("epoch", len(r.epochs))


def render_final(app):
    r, f = app.result, app.fmt
    n = len(r.request.hidden_weights[0])
    units = len(r.request.hidden_weights)
    lines = [f"MADALINE MRI · {STATUS[r.status]}"]
    if r.status == "weights_stable":
        lines.append(
            "No individual hidden coefficient changed during the last complete epoch. Accuracy is shown separately."
        )
    elif r.status == "tolerance_met":
        lines.append(
            "The largest individual hidden weight/bias change was strictly below ε."
        )
    elif r.status == "epoch_limit":
        lines.append("The complete-epoch budget ended before the change test passed.")
    else:
        lines.append(
            f"Attempted update at epoch {r.failed_at[0]}, pattern {r.failed_at[1]} was rejected. This is the last valid model; its final attempted epoch is incomplete."
        )
    for j in range(units):
        equation = " + ".join(
            f"({f(w)})x{i + 1}" for i, w in enumerate(r.weights[j * n : (j + 1) * n])
        )
        lines.append(
            f"z_in{j + 1} = {equation} + ({f(r.biases[j])})\nz{j + 1} = f(z_in{j + 1})"
        )
    terms = " + ".join(
        f"({f(v)})z{j + 1}" for j, v in enumerate(r.request.output_weights)
    )
    lines.append(
        f"Fixed output: y_in = ({f(r.request.output_bias)}) + {terms}\ny = f(y_in); f(a) = +1 if a ≥ 0, otherwise −1."
    )
    lines.append(
        f"Classification SSE = {f(r.sse)}    Accuracy = {r.correct}/{len(r.request.samples)}\nComplete epochs: {r.completed_epochs}; recorded pattern visits: {len(r.steps)}."
    )
    app._set_text(app.final_text, "\n\n".join(lines))
    app._columns(
        app.prediction_tree,
        [
            "Pattern",
            *[f"x{i + 1}" for i in range(n)],
            "Target",
            *[f"z_in{j + 1}" for j in range(units)],
            *[f"z{j + 1}" for j in range(units)],
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
                *map(f, p.nets),
                *p.hidden_outputs,
                f(p.net),
                p.output,
                "Yes" if p.output == sample.target else "No",
            ],
        )
