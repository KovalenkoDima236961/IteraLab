"""Editable learning inputs transcribed from practical_examples (no answers)."""

LEARNING_EXAMPLES = {
    "06 p8 · Hebbian AND": dict(
        algorithm="Hebbian Learning",
        data="1,1,1\n-1,1,-1\n1,-1,-1\n-1,-1,-1",
        alpha="1",
    ),
    "06 pp11–13 · Perceptron AND, band": dict(
        algorithm="Perceptron",
        mode="band",
        data="1,1,1\n0,1,-1\n1,0,-1\n0,0,-1",
        alpha="1",
        theta="0.2",
        max_epochs="100",
    ),
    "06 pp14–16 · Perceptron OR, band": dict(
        algorithm="Perceptron",
        mode="band",
        data="1,1,1\n0,1,1\n1,0,1\n0,0,-1",
        alpha="1",
        theta="0.2",
        max_epochs="100",
    ),
    "CV6 · binary Perceptron AND": dict(
        algorithm="Perceptron",
        mode="binary",
        data="1,1,1\n1,0,0\n0,1,0\n0,0,0",
        alpha="1",
        max_epochs="100",
    ),
    "CV6 · bipolar error-rule AND, two epochs": dict(
        algorithm="Perceptron",
        mode="bipolar_error",
        data="1,1,1\n1,0,-1\n0,1,-1\n0,0,-1",
        alpha="1",
        max_epochs="2",
    ),
    "CV6 · bipolar error-rule OR, two epochs": dict(
        algorithm="Perceptron",
        mode="bipolar_error",
        data="1,1,1\n1,0,1\n0,1,1\n0,0,-1",
        alpha="1",
        max_epochs="2",
    ),
    "CV6 · bipolar error-rule XOR, two epochs": dict(
        algorithm="Perceptron",
        mode="bipolar_error",
        data="1,1,-1\n1,0,1\n0,1,1\n0,0,-1",
        alpha="1",
        max_epochs="2",
    ),
    "CV6 · majority vote, two epochs": dict(
        algorithm="Perceptron",
        mode="bipolar_error",
        data="1,1,1,1\n1,1,0,1\n1,0,1,1\n0,1,1,1\n1,0,0,-1\n0,1,0,-1\n0,0,1,-1\n0,0,0,-1",
        alpha="1",
        max_epochs="2",
    ),
    "07 pp3–4 · ADALINE AND, two epochs": dict(
        algorithm="ADALINE",
        data="1,1,1\n1,-1,-1\n-1,1,-1\n-1,-1,-1",
        alpha="0.2",
        max_epochs="2",
    ),
    "07 pp5–6 · ADALINE housing, two epochs": dict(
        algorithm="ADALINE", data="2,1,-1\n3,2,1\n4,3,1", alpha="0.1", max_epochs="2"
    ),
    "CV7 · ADALINE AND, two epochs": dict(
        algorithm="ADALINE",
        data="1,1,1\n1,-1,-1\n-1,1,-1\n-1,-1,-1",
        alpha="0.1",
        max_epochs="2",
    ),
    "CV7 · ADALINE OR, two epochs": dict(
        algorithm="ADALINE",
        data="1,1,1\n1,-1,1\n-1,1,1\n-1,-1,-1",
        alpha="0.1",
        max_epochs="2",
    ),
    "07 pp9–10 · MADALINE AND, zero start": dict(
        algorithm="MADALINE (MRI)",
        data="1,1,1\n1,-1,-1\n-1,1,-1\n-1,-1,-1",
        alpha="0.2",
        max_epochs="2",
    ),
    "CV7 · MADALINE OR, canonical ±1 targets": dict(
        algorithm="MADALINE (MRI)",
        data="-1,-1,-1\n-1,1,1\n1,-1,1\n1,1,1",
        alpha="0.1",
        max_epochs="1",
        output_weights="1,1",
        output_bias="0.5",
    ),
    "CV8 · ADALINE rate example": dict(
        algorithm="ADALINE", data="1,0,-1\n0,1,1\n1,1,1", alpha="0.45", max_epochs="1"
    ),
}
