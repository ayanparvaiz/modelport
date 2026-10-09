"""Run exported variants in Python, the same engines the Dart adapters wrap."""

from __future__ import annotations

import numpy as np

from .bundle import Bundle
from .errors import MissingDependencyError, ModelPortError
from .manifest import Manifest, Runtime, Variant


class RuntimeNotSupported(ModelPortError):
    """This runtime cannot run tensor inputs in Python."""


def run_variant(
    bundle: Bundle, manifest: Manifest, variant: Variant, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Run one variant on named inputs and return outputs keyed by manifest output name."""
    if variant.file.path is None:
        raise ModelPortError(f"{variant.id}: only files inside the bundle can be run")
    path = bundle.path(variant.file.path)
    output_names = [o.name for o in manifest.outputs]

    if variant.runtime is Runtime.ONNX:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise MissingDependencyError("Running ONNX variants", "onnx") from error
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        results = session.run(output_names, inputs)
        return {name: np.asarray(value) for name, value in zip(output_names, results, strict=True)}

    if variant.runtime is Runtime.EXECUTORCH:
        try:
            import torch
            from executorch.runtime import Runtime as ExecuTorchRuntime
        except ImportError as error:
            raise MissingDependencyError("Running ExecuTorch variants", "executorch") from error
        method = ExecuTorchRuntime.get().load_program(str(path)).load_method("forward")
        if method is None:
            raise ModelPortError(f"{variant.id}: the program has no 'forward' method")
        # ExecuTorch takes inputs by position, in manifest order.
        args = [torch.from_numpy(np.ascontiguousarray(inputs[s.name])) for s in manifest.inputs]
        results = method.execute(args)
        return {
            name: value.detach().numpy() for name, value in zip(output_names, results, strict=True)
        }

    raise RuntimeNotSupported(
        f"{variant.id}: runtime '{variant.runtime}' has no tensor golden test"
    )
