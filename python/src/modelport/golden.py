"""Golden data: saved inputs and expected outputs that prove a device matches Python."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .bundle import Bundle
from .errors import MissingDependencyError, ModelPortError
from .manifest import DType, FileRef, Golden, Manifest, TensorSpec
from .preprocess import preprocess_image, to_le_bytes
from .sources import SourceModel

NUMPY_DTYPES: dict[DType, str] = {
    DType.FLOAT32: "<f4",
    DType.FLOAT16: "<f2",
    DType.INT64: "<i8",
    DType.INT32: "<i4",
    DType.INT8: "i1",
    DType.UINT8: "u1",
    DType.BOOL: "?",
}


def dtype_of(array: np.ndarray) -> DType:
    for dtype, code in NUMPY_DTYPES.items():
        if np.dtype(code) == array.dtype.newbyteorder("<"):
            return dtype
    raise ModelPortError(f"unsupported tensor dtype {array.dtype}")


def sample_image(width: int = 320, height: int = 240) -> Image.Image:
    """A fixed, colourful test picture with gradients, a ring, and fine detail."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    red = x * 255 / (width - 1)
    green = y * 255 / (height - 1)
    blue = 128 + 127 * np.sin(x / 9.0) * np.cos(y / 7.0)
    distance = np.hypot(x - width * 0.6, y - height * 0.45)
    ring = (distance > 40) & (distance < 60)
    pixels = np.stack([red, green, blue], axis=-1)
    pixels[ring] = (250, 240, 20)
    return Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8), "RGB")


@dataclass(frozen=True)
class GoldenData:
    golden: Golden
    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]


def make_golden(source: SourceModel, bundle: Bundle, image: Image.Image) -> GoldenData:
    """Preprocess `image`, run the PyTorch model, and save both sides under golden/."""
    try:
        import torch
    except ImportError as error:
        raise MissingDependencyError("Creating golden data", "onnx") from error

    inputs: dict[str, np.ndarray] = {}
    for spec in source.inputs:
        if spec.preprocess is None:
            raise ModelPortError(f"input '{spec.name}' needs image preprocessing for golden data")
        inputs[spec.name] = preprocess_image(image, spec)

    with torch.no_grad():
        result = source.module(*(torch.from_numpy(inputs[s.name]) for s in source.inputs))
    results = result if isinstance(result, (tuple, list)) else (result,)
    if len(results) != len(source.output_names):
        raise ModelPortError(
            f"model returned {len(results)} outputs, expected {len(source.output_names)}"
        )
    outputs = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(source.output_names, results, strict=True)
    }

    input_refs = {
        name: bundle.write_bytes(f"golden/{name}.bin", to_le_bytes(a)) for name, a in inputs.items()
    }
    output_refs = {
        name: bundle.write_bytes(f"golden/{name}.bin", to_le_bytes(a))
        for name, a in outputs.items()
    }
    return GoldenData(Golden(inputs=input_refs, outputs=output_refs), inputs, outputs)


def read_tensor(bundle: Bundle, ref: FileRef, spec: TensorSpec) -> np.ndarray:
    """Load a golden file as an array with the tensor's dtype and shape."""
    if ref.path is None:
        raise ModelPortError("golden files must be inside the bundle")
    data = bundle.path(ref.path).read_bytes()
    # Copy so the array is writable; torch warns when wrapping read-only buffers.
    array = np.frombuffer(data, dtype=NUMPY_DTYPES[spec.dtype]).copy()
    if spec.shape.count(-1) <= 1:
        array = array.reshape(spec.shape)
    return array


def golden_arrays(bundle: Bundle, manifest: Manifest) -> tuple[dict, dict]:
    """Golden inputs and outputs of a manifest, keyed by tensor name."""
    if manifest.golden is None:
        raise ModelPortError(f"{manifest.id} has no golden data")
    specs = {s.name: s for s in [*manifest.inputs, *manifest.outputs]}
    inputs = {n: read_tensor(bundle, r, specs[n]) for n, r in manifest.golden.inputs.items()}
    outputs = {n: read_tensor(bundle, r, specs[n]) for n, r in manifest.golden.outputs.items()}
    return inputs, outputs
