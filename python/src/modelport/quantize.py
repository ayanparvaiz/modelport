"""Smaller ONNX variants: fp16 weights, or int8 weights for MatMul and Gemm layers."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .bundle import Bundle
from .errors import MissingDependencyError, ModelPortError
from .golden import golden_arrays
from .manifest import ClassificationPostprocess, Manifest, Runtime, Tolerance, Variant
from .runtimes import run_variant
from .verify import compare

Kind = Literal["fp16", "int8"]


class QuantizeError(ModelPortError):
    """A smaller variant could not be made or is not accurate enough."""


@dataclass(frozen=True)
class QuantizeResult:
    variant: Variant
    size_ratio: float
    max_abs_diff: float
    top1_match: bool | None


def tolerance_for(max_abs_diff: float, base: Tolerance) -> Tolerance:
    """Twice the measured error, rounded up to two significant digits, at least base.atol."""
    limit = max(base.atol, 2 * max_abs_diff)
    digits = 1 - math.floor(math.log10(limit))
    return Tolerance(atol=math.ceil(limit * 10**digits) / 10**digits, rtol=base.rtol)


def _write_variant(source: Path, target: Path, kind: Kind) -> None:
    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as error:
        raise MissingDependencyError("Quantizing ONNX models", "onnx") from error

    model = onnx.load(str(source))
    # Exporter shape annotations can disagree with ONNX shape inference; drop them.
    del model.graph.value_info[:]
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "fp16":
        # Inputs and outputs stay float32 so apps feed the same tensors to every variant.
        onnx.save(convert_float_to_float16(model, keep_io_types=True), str(target))
        return
    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / "model.onnx"
        onnx.save(model, str(clean))
        # Dynamic int8 on convolutions badly hurts CNN accuracy, so only quantize
        # MatMul and Gemm, where most transformer and classifier-head weights live.
        quantize_dynamic(
            str(clean),
            str(target),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            per_channel=True,
        )


def quantize_bundle(
    path: str | Path, kinds: list[Kind], *, allow_top1_change: bool = False
) -> list[QuantizeResult]:
    """Add one ONNX variant per kind, measure its error, and record a matching tolerance."""
    bundle = Bundle(path)
    manifest = bundle.read_manifest()
    if manifest.golden is None:
        raise QuantizeError("the bundle needs golden data to measure accuracy")
    base = next(
        (v for v in manifest.variants if v.runtime is Runtime.ONNX and v.precision == "fp32"),
        None,
    )
    if base is None or base.file.path is None:
        raise QuantizeError("the bundle has no onnx fp32 variant to start from")

    inputs, expected = golden_arrays(bundle, manifest)
    classify = {
        o.name for o in manifest.outputs if isinstance(o.postprocess, ClassificationPostprocess)
    }
    variants = list(manifest.variants)
    results = []
    for kind in dict.fromkeys(kinds):
        variant_id = f"onnx-{kind}"
        relative = f"{variant_id}/model.onnx"
        _write_variant(bundle.path(base.file.path), bundle.path(relative), kind)
        candidate = Variant(
            id=variant_id, runtime=Runtime.ONNX, precision=kind, file=bundle.add(relative)
        )
        got = run_variant(bundle, manifest, candidate, inputs)
        checks = [
            compare(name, got[name], want, manifest.golden.tolerance, name in classify)
            for name, want in expected.items()
        ]
        worst = max(check.max_abs_diff for check in checks)
        top1 = None if not classify else all(c.top1_match is not False for c in checks)
        if top1 is False and not allow_top1_change:
            raise QuantizeError(
                f"{variant_id} changes the top-1 class on the golden input "
                f"(max |diff| {worst:.3g}). Use --allow-top1-change to keep it anyway."
            )
        variant = candidate.model_copy(
            update={"tolerance": tolerance_for(worst, manifest.golden.tolerance)}
        )
        variants = [v for v in variants if v.id != variant_id] + [variant]
        results.append(QuantizeResult(variant, variant.file.size / base.file.size, worst, top1))

    updated = Manifest.model_validate(
        {**manifest.model_dump(by_alias=True), "variants": [v.model_dump() for v in variants]}
    )
    bundle.write_manifest(updated)
    return results
