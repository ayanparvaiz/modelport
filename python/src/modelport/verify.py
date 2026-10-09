"""Check every variant of a bundle against its golden data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bundle import Bundle
from .errors import MissingDependencyError, ModelPortError
from .golden import golden_arrays
from .manifest import ClassificationPostprocess, Tolerance
from .runtimes import RuntimeNotSupported, run_variant


class VerifyError(ModelPortError):
    """The bundle cannot be verified at all."""


@dataclass(frozen=True)
class OutputCheck:
    name: str
    max_abs_diff: float
    cosine: float
    within_tolerance: bool
    top1_match: bool | None = None


@dataclass(frozen=True)
class VariantResult:
    variant_id: str
    runtime: str
    tolerance: Tolerance
    outputs: list[OutputCheck] = field(default_factory=list)
    skipped: str | None = None

    @property
    def passed(self) -> bool:
        return self.skipped is None and all(
            o.within_tolerance and o.top1_match is not False for o in self.outputs
        )


def compare(
    name: str, got: np.ndarray, expected: np.ndarray, tolerance: Tolerance, classify: bool
) -> OutputCheck:
    if got.shape != expected.shape:
        raise VerifyError(f"output '{name}' has shape {got.shape}, golden has {expected.shape}")
    a = got.astype(np.float64).ravel()
    b = expected.astype(np.float64).ravel()
    diff = np.abs(a - b)
    within = bool(np.all(diff <= tolerance.atol + tolerance.rtol * np.abs(b)))
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(np.dot(a, b) / norms) if norms > 0 else float(np.array_equal(a, b))
    top1 = None
    if classify:
        top1 = bool(np.array_equal(got.argmax(axis=-1), expected.argmax(axis=-1)))
    return OutputCheck(name, float(diff.max(initial=0.0)), cosine, within, top1)


def verify_bundle(path: str | Path) -> list[VariantResult]:
    bundle = Bundle(path)
    if not bundle.manifest_path.is_file():
        raise VerifyError(f"{bundle.manifest_path} not found")
    manifest = bundle.read_manifest()
    problems = bundle.problems(manifest)
    if problems:
        raise VerifyError("bundle files do not match the manifest:\n  " + "\n  ".join(problems))
    if manifest.golden is None:
        raise VerifyError(f"{manifest.id} has no golden data to verify against")

    inputs, expected = golden_arrays(bundle, manifest)
    classify = {
        o.name for o in manifest.outputs if isinstance(o.postprocess, ClassificationPostprocess)
    }
    results = []
    for variant in manifest.variants:
        tolerance = variant.tolerance or manifest.golden.tolerance
        try:
            got = run_variant(bundle, manifest, variant, inputs)
        except (MissingDependencyError, RuntimeNotSupported) as error:
            results.append(
                VariantResult(variant.id, variant.runtime, tolerance, skipped=str(error))
            )
            continue
        checks = [
            compare(name, got[name], want, tolerance, name in classify)
            for name, want in expected.items()
        ]
        results.append(VariantResult(variant.id, variant.runtime, tolerance, checks))
    return results
