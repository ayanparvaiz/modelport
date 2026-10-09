"""Models described by a function in the user's own Python file."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from . import SourceError, SourceModel

DEFAULT_FUNCTION = "build"


def split_file_spec(spec: str) -> tuple[Path, str]:
    """'models/net.py:make' -> (models/net.py, 'make'). The function defaults to 'build'."""
    path_text, sep, function = spec.rpartition(":")
    if sep and path_text.endswith(".py") and function:
        return Path(path_text), function
    return Path(spec), DEFAULT_FUNCTION


def load_file(spec: str) -> SourceModel:
    """Import a Python file and call a function that returns a SourceModel.

    This runs the file's code, like `python file.py` would. Only use files you trust.
    """
    path, function = split_file_spec(spec)
    if not path.is_file():
        raise SourceError(f"{path} does not exist")
    module_spec = importlib.util.spec_from_file_location(f"_modelport_user_{path.stem}", path)
    if module_spec is None or module_spec.loader is None:
        raise SourceError(f"cannot import {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    factory = getattr(module, function, None)
    if not callable(factory):
        raise SourceError(f"{path} has no function named '{function}'")
    result = factory()
    if not isinstance(result, SourceModel):
        raise SourceError(
            f"{path}:{function} must return modelport.sources.SourceModel, "
            f"got {type(result).__name__}"
        )
    if result.source is None:
        result.source = f"file:{path.name}:{function}"
    return result
