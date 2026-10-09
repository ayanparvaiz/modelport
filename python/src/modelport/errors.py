"""Errors that the CLI turns into short messages instead of tracebacks."""

from __future__ import annotations


class ModelPortError(Exception):
    """Base class for expected, user-facing errors."""


class MissingDependencyError(ModelPortError):
    """An optional package needed for this feature is not installed."""

    def __init__(self, feature: str, extra: str) -> None:
        super().__init__(
            f"{feature} needs extra packages. Install them with: pip install 'modelport[{extra}]'"
        )
        self.extra = extra
