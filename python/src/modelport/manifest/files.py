"""File references inside a model bundle."""

from __future__ import annotations

import re

from pydantic import Field, field_validator, model_validator

from .base import Sha256, StrictModel

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


class FileRef(StrictModel):
    """A file that belongs to a model, such as weights, labels, or golden test data.

    Exactly one of `path` or `url` must be set.
    """

    model_config = StrictModel.model_config | {
        "json_schema_extra": {"oneOf": [{"required": ["path"]}, {"required": ["url"]}]}
    }

    path: str | None = Field(
        default=None,
        description="POSIX path relative to the manifest file. Use for files inside the bundle.",
        examples=["onnx-fp32/model.onnx", "labels.txt"],
    )
    url: str | None = Field(
        default=None,
        pattern=r"^https://\S+$",
        description=(
            "Absolute HTTPS URL. Use for files hosted elsewhere, such as an original GGUF repo."
        ),
    )
    size: int = Field(gt=0, description="File size in bytes.")
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def _check_relative_path(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if value.startswith("/") or "\\" in value or _WINDOWS_DRIVE.match(value):
            raise ValueError("path must be a relative POSIX path, like 'onnx-fp32/model.onnx'")
        if any(part in ("", ".", "..") for part in value.split("/")):
            raise ValueError("path must not contain empty, '.' or '..' segments")
        return value

    @model_validator(mode="after")
    def _check_exactly_one_location(self) -> FileRef:
        if (self.path is None) == (self.url is None):
            raise ValueError("set exactly one of 'path' or 'url'")
        return self
