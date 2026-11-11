"""File CLI compared with the existing Python adapters, using direct rustc builds."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from every_example_gui.backend import Request, Sample, parse_output, train
from every_example_gui.algorithms.hebbian import HebbianRequest
from every_example_gui.algorithms.lecture08 import Lecture08Request
from every_example_gui.algorithms.nelder_mead import NelderMeadRequest
from every_example_gui.algorithms.newton2d import Newton2DRequest
from every_example_gui.algorithms.normalized_gradient import (
    LineSteepestRequest,
    NormalizedRequest,
)
from every_example_gui.algorithms.optimization import GradientRequest
from every_example_gui.algorithms.perceptron import PerceptronRequest
from every_example_gui.algorithms.powell import PowellRequest
from every_example_gui.algorithms.steepest import SteepestRequest


ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = (
    "nelder-mead", "gradient-1d", "steepest-fixed", "steepest-line-search",
    "normalized-gradient", "powell", "newton-2d", "perceptron", "adaline", "hebb",
    "mcculloch-pitts",
)


def read_template(source):
    settings, sections, current = {}, {}, None
    for raw in source.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            current = line[1:-1]
            sections[current] = []
        elif current:
            sections[current].append(tuple(float(v.strip()) for v in line.split(",")))
        else:
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip()
    return settings, {key: tuple(rows) for key, rows in sections.items()}


def gui_request(algorithm, settings, sections, rate):
    if algorithm == "mcculloch-pitts":
        return Lecture08Request(
            "mcp", s=sections["patterns"], w=sections["weights"],
            t=(tuple(map(float, settings["thresholds"].split(","))),),
            expected=sections["expected"],
        )
    if algorithm in ("adaline", "perceptron", "hebb"):
        samples = tuple(Sample(row[:-1], row[-1]) for row in sections["samples"])
        if algorithm == "hebb":
            return HebbianRequest(samples, eta=rate)
        common = dict(
            samples=samples,
            weights=tuple(map(float, settings["initial_weights"].split(","))),
            bias=float(settings["initial_bias"]), alpha=rate,
            max_epochs=int(settings["max_epochs"]),
        )
        if algorithm == "adaline":
            return Request(**common, tolerance=float(settings["tolerance"]))
        return PerceptronRequest(
            **common, mode=settings["mode"], theta=float(settings.get("theta", 0)),
        )
    bounds = tuple(map(float, settings["plot_bounds"].split(",")))
    common = dict(
        expression=settings["objective"], tolerance=float(settings["tolerance"]),
        max_iterations=int(settings["max_iterations"]),
        plot_min=bounds[0], plot_max=bounds[1],
    )
    if algorithm == "gradient-1d":
        return GradientRequest(**common, x0=float(settings["x0"]), alpha=rate)
    common.update(plot_y_min=bounds[2], plot_y_max=bounds[3])
    if algorithm == "nelder-mead":
        return NelderMeadRequest(
            **common, simplex=sections["simplex"], variant=settings["variant"],
            stopping=settings["stopping"],
        )
    common.update(x0=float(settings["x0"]), y0=float(settings["y0"]))
    if algorithm == "steepest-fixed":
        return SteepestRequest(**common, alpha=rate)
    if algorithm == "newton-2d":
        return Newton2DRequest(**common)
    common.update(
        line_tolerance=float(settings["line_tolerance"]),
        max_line_evaluations=int(settings["max_line_evaluations"]),
    )
    probe = float(settings["initial_step"])
    if algorithm == "powell":
        return PowellRequest(**common, initial_step=probe)
    cls = NormalizedRequest if algorithm == "normalized-gradient" else LineSteepestRequest
    return cls(**common, initial_gamma=probe)


class CliAdapterIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("rustc"):
            raise unittest.SkipTest("rustc is unavailable")
        cls.scratch = tempfile.TemporaryDirectory(prefix="math cli integration ")
        cls.binary = Path(cls.scratch.name) / "direct Rust executable"
        compiled = subprocess.run(
            ["rustc", "--edition=2021", str(ROOT / "rust_backend/src/main.rs"),
             "-o", str(cls.binary)],
            capture_output=True, text=True, timeout=120,
        )
        if compiled.returncode:
            cls.scratch.cleanup()
            raise RuntimeError(compiled.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def test_every_cli_template_matches_the_existing_gui_adapter_in_full(self):
        paths = sorted((ROOT / "cli_templates").rglob("*.txt"))
        self.assertEqual(len(paths), 20)
        for path in paths:
            with self.subTest(template=path.name):
                algorithm = next(name for name in ALGORITHMS if path.stem.startswith(name))
                output = Path(self.scratch.name) / "complete calculation report.txt"
                run = subprocess.run(
                    [str(self.binary), algorithm, "--input", str(path), "--output", str(output)],
                    capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                report = output.read_text()
                rate = next(
                    (float(line.split(" = ", 1)[1]) for line in report.splitlines()
                     if line.startswith("resolved_learning_rate = ")), None,
                )
                settings, sections = read_template(path.read_text())
                request = gui_request(algorithm, settings, sections, rate)
                legacy = subprocess.run(
                    [str(self.binary)], input=request.serialize(), capture_output=True,
                    text=True, timeout=120,
                )
                self.assertEqual(legacy.returncode, 0, legacy.stderr)
                tags = {line.split("\t", 1)[0] for line in legacy.stdout.splitlines()}
                trace = report.split("Complete calculation trace\n", 1)[1].split(
                    "Resolved backend request (reproducibility)\n", 1
                )[0]
                actual = "".join(
                    line + "\n" for line in trace.splitlines()
                    if line.split("\t", 1)[0] in tags
                )
                self.assertEqual(actual, legacy.stdout)
                # Also exercise the unchanged optimized rustc/cache/GUI parser path.
                self.assertEqual(parse_output(actual, request), train(request))


if __name__ == "__main__":
    unittest.main()
