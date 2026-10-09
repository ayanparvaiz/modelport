"""The top-level `modelport.json` manifest."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .base import StrictModel
from .files import FileRef
from .postprocess import ClassificationPostprocess, DetectionPostprocess
from .tensors import InputSpec, OutputSpec

SCHEMA_VERSION = "modelport/0.1"
MANIFEST_FILENAME = "modelport.json"

_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
_SEMVER_PATTERN = (
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


class Task(StrEnum):
    """What the model does. Decides which Dart task API can load it."""

    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    TEXT_GENERATION = "text-generation"


TENSOR_TASKS = frozenset({Task.IMAGE_CLASSIFICATION, Task.OBJECT_DETECTION})


class Runtime(StrEnum):
    """Inference engine that runs a variant."""

    ONNX = "onnx"
    EXECUTORCH = "executorch"
    LLAMACPP = "llamacpp"


class Tolerance(StrictModel):
    """How close two outputs must be: |a - b| <= atol + rtol * |b|."""

    atol: float = Field(default=1e-3, ge=0)
    rtol: float = Field(default=1e-3, ge=0)


class Variant(StrictModel):
    """One concrete form of the model, such as ONNX fp32 or GGUF Q4_K_M."""

    id: str = Field(pattern=_ID_PATTERN, examples=["onnx-fp32", "gguf-q4_k_m"])
    runtime: Runtime
    backend: str | None = Field(
        default=None, description="Engine backend, such as 'xnnpack' or 'coreml'."
    )
    precision: str = Field(pattern=r"^[a-z0-9_]+$", examples=["fp32", "fp16", "int8", "q4_k_m"])
    file: FileRef = Field(description="The main model file.")
    extra_files: list[FileRef] = Field(
        default_factory=list,
        description="Files the main file needs, such as ONNX external weights.",
    )
    min_ram_mb: int | None = Field(
        default=None, gt=0, description="Skip this variant on devices with less RAM."
    )
    tolerance: Tolerance | None = Field(
        default=None, description="Overrides the golden tolerance, for example for int8."
    )


class GenerationDefaults(StrictModel):
    """Default sampling settings. Apps can override each one per request."""

    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int | None = Field(default=None, gt=0)
    max_tokens: int = Field(default=512, gt=0)
    repeat_penalty: float | None = Field(default=None, gt=0)


class LlmConfig(StrictModel):
    """Settings for text-generation models."""

    context_length: int = Field(gt=0, description="Maximum tokens the model can attend to.")
    chat_template: Literal["from_gguf"] = Field(
        default="from_gguf",
        description="Where the chat template comes from. Templates are never stored as code here.",
    )
    defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)


class Golden(StrictModel):
    """Saved inputs and expected outputs, used to prove a device matches Python.

    Each file holds raw little-endian tensor data in the dtype and shape of the tensor
    with the same name.
    """

    inputs: dict[str, FileRef] = Field(min_length=1)
    outputs: dict[str, FileRef] = Field(min_length=1)
    tolerance: Tolerance = Field(default_factory=Tolerance)


class Manifest(StrictModel):
    """Everything an app needs to download, run, and interpret a model."""

    schema_version: Literal["modelport/0.1"] = Field(default=SCHEMA_VERSION, alias="schema")
    id: str = Field(pattern=_ID_PATTERN, examples=["mobilenet_v3_small"])
    version: str = Field(pattern=_SEMVER_PATTERN, examples=["1.0.0"])
    task: Task
    license: str = Field(min_length=1, description="SPDX license identifier of the weights.")
    name: str | None = Field(default=None, description="Human-friendly name.")
    description: str | None = None
    source: str | None = Field(default=None, examples=["torchvision:mobilenet_v3_small"])
    variants: list[Variant] = Field(min_length=1)
    inputs: list[InputSpec] = Field(default_factory=list)
    outputs: list[OutputSpec] = Field(default_factory=list)
    llm: LlmConfig | None = None
    golden: Golden | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Manifest:
        _require_unique("variant id", [v.id for v in self.variants])
        _require_unique("input name", [i.name for i in self.inputs])
        _require_unique("output name", [o.name for o in self.outputs])

        if self.task in TENSOR_TASKS:
            self._check_tensor_task()
        else:
            self._check_text_generation()

        if self.golden is not None:
            self._check_golden(self.golden)
        return self

    def _check_tensor_task(self) -> None:
        if not self.inputs or not self.outputs:
            raise ValueError(f"task '{self.task}' needs at least one input and one output")
        if self.llm is not None:
            raise ValueError(f"task '{self.task}' must not have an 'llm' section")
        if any(v.runtime is Runtime.LLAMACPP for v in self.variants):
            raise ValueError("runtime 'llamacpp' is only for task 'text-generation'")
        if not any(i.preprocess is not None for i in self.inputs):
            raise ValueError(f"task '{self.task}' needs an input with image preprocessing")
        wanted = (
            ClassificationPostprocess
            if self.task is Task.IMAGE_CLASSIFICATION
            else DetectionPostprocess
        )
        if not any(isinstance(o.postprocess, wanted) for o in self.outputs):
            raise ValueError(
                f"task '{self.task}' needs an output with '{wanted().type}' postprocess"
            )

    def _check_text_generation(self) -> None:
        if self.llm is None:
            raise ValueError("task 'text-generation' needs an 'llm' section")
        if any(v.runtime is not Runtime.LLAMACPP for v in self.variants):
            raise ValueError("task 'text-generation' supports only runtime 'llamacpp' in spec 0.1")
        if self.golden is not None:
            raise ValueError("golden tests are only for tensor tasks")

    def _check_golden(self, golden: Golden) -> None:
        input_names = {i.name for i in self.inputs}
        output_names = {o.name for o in self.outputs}
        for name in golden.inputs:
            if name not in input_names:
                raise ValueError(f"golden input '{name}' is not a model input")
        for name in golden.outputs:
            if name not in output_names:
                raise ValueError(f"golden output '{name}' is not a model output")

    def files(self) -> list[FileRef]:
        """Every file the manifest points to, in a stable order."""
        refs: list[FileRef] = []
        for variant in self.variants:
            refs.append(variant.file)
            refs.extend(variant.extra_files)
        for output in self.outputs:
            labels = getattr(output.postprocess, "labels", None)
            if labels is not None:
                refs.append(labels)
        if self.golden is not None:
            refs.extend(self.golden.inputs.values())
            refs.extend(self.golden.outputs.values())
        return refs

    def to_json(self) -> str:
        """Serialize the way the CLI writes `modelport.json`."""
        return self.model_dump_json(by_alias=True, exclude_none=True, indent=2) + "\n"

    @classmethod
    def from_file(cls, path: str | Path) -> Manifest:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _require_unique(label: str, values: list[str]) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label}: {', '.join(duplicates)}")
