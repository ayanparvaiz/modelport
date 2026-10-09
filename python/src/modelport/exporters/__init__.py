"""Turn a SourceModel into model files for each target runtime."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from ..bundle import Bundle
from ..errors import ModelPortError
from ..manifest import Runtime
from ..sources import SourceModel


class ExportError(ModelPortError):
    """A model could not be exported."""


@dataclass(frozen=True)
class ExportedVariant:
    """Files written for one variant, as bundle-relative paths."""

    id: str
    runtime: Runtime
    precision: str
    main_file: str
    extra_files: list[str] = field(default_factory=list)
    backend: str | None = None


Exporter = Callable[[SourceModel, Bundle], ExportedVariant]


@contextlib.contextmanager
def quiet_output() -> Iterator[io.StringIO]:
    """Capture the progress lines that exporters print, to show them only on failure."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer


def get_exporter(target: str) -> Exporter:
    if target == "onnx":
        from .onnx import export_onnx

        return export_onnx
    if target == "executorch":
        from .executorch import export_executorch

        return export_executorch
    raise ExportError(f"unknown export target '{target}'. Available: onnx, executorch")


__all__ = ["ExportError", "ExportedVariant", "Exporter", "get_exporter", "quiet_output"]
