"""Load a PyTorch model and describe it well enough to write a manifest.

Sources:
    torchvision:<model name>      e.g. torchvision:mobilenet_v3_small
    hf:<repo id or local folder>  e.g. hf:google/vit-base-patch16-224
    file:<script.py>[:function]   a function that returns a SourceModel
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..errors import ModelPortError
from ..manifest import InputSpec, Task

_ID_CLEAN = re.compile(r"[^a-z0-9._-]+")


class SourceError(ModelPortError):
    """A model source could not be loaded."""


@dataclass
class SourceModel:
    """A PyTorch module plus everything needed to describe it in a manifest.

    `example_inputs` must match `inputs` in order and shape. They are made contiguous
    before export, because ExecuTorch records the memory layout of example inputs.
    """

    module: Any
    example_inputs: tuple[Any, ...]
    id: str
    task: Task
    license: str
    inputs: list[InputSpec]
    output_names: list[str]
    labels: list[str] | None = None
    name: str | None = None
    description: str | None = None
    source: str | None = None
    top_k: int = 5
    extra: dict[str, Any] = field(default_factory=dict)


def model_id(text: str) -> str:
    """Turn a model name or repo id into a manifest id, such as 'google/ViT-B' -> 'vit-b'."""
    last = text.rstrip("/").split("/")[-1].lower()
    cleaned = _ID_CLEAN.sub("-", last).strip("-._")
    if not cleaned:
        raise SourceError(f"cannot make a model id from '{text}'")
    return cleaned


def load_source(spec: str, *, license: str | None = None) -> SourceModel:
    """Load a model from a source string such as 'torchvision:mobilenet_v3_small'."""
    kind, sep, rest = spec.partition(":")
    if not sep or not rest:
        raise SourceError(f"'{spec}' is not a source. Use torchvision:, hf:, or file:")
    if kind == "torchvision":
        from .torchvision import load_torchvision

        model = load_torchvision(rest)
    elif kind == "hf":
        from .hf import load_hf

        model = load_hf(rest, license=license)
    elif kind == "file":
        from .file import load_file

        model = load_file(rest)
    else:
        raise SourceError(f"unknown source type '{kind}'. Use torchvision:, hf:, or file:")
    if license is not None:
        model.license = license
    return model


__all__ = ["SourceError", "SourceModel", "load_source", "model_id"]
