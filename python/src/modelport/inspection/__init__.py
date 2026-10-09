"""Read inputs, outputs, and metadata from model files without running them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..errors import MissingDependencyError, ModelPortError
from ..hashing import sha256_file

ModelFormat = Literal["onnx", "executorch", "gguf"]


class InspectError(ModelPortError):
    """A model file could not be inspected."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: list[int | str]


@dataclass(frozen=True)
class ModelInfo:
    format: ModelFormat
    path: Path
    size: int
    sha256: str
    inputs: list[TensorInfo] = field(default_factory=list)
    outputs: list[TensorInfo] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


def detect_format(path: Path) -> ModelFormat:
    """Detect the format from magic bytes, falling back to the file extension."""
    with path.open("rb") as handle:
        head = handle.read(8)
    if head[:4] == b"GGUF":
        return "gguf"
    if head[4:8] == b"ET12":
        return "executorch"
    if path.suffix.lower() == ".onnx":
        return "onnx"
    raise InspectError(f"{path}: not a recognized model file (.onnx, .pte, or .gguf)")


def inspect_model(path: str | Path) -> ModelInfo:
    """Describe a model file. Raises InspectError if it cannot be read."""
    path = Path(path)
    if not path.is_file():
        raise InspectError(f"{path}: file not found")
    fmt = detect_format(path)
    if fmt == "onnx":
        from .onnx import read_onnx

        inputs, outputs, metadata = read_onnx(path)
    elif fmt == "executorch":
        from .executorch import read_executorch

        inputs, outputs, metadata = read_executorch(path)
    else:
        from .gguf import read_gguf

        inputs, outputs, metadata = read_gguf(path)
    return ModelInfo(
        format=fmt,
        path=path,
        size=path.stat().st_size,
        sha256=sha256_file(path),
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
    )


__all__ = [
    "InspectError",
    "MissingDependencyError",
    "ModelFormat",
    "ModelInfo",
    "TensorInfo",
    "detect_format",
    "inspect_model",
]
