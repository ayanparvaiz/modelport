"""Check the local environment for what each CLI feature needs."""

from __future__ import annotations

import platform
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

from . import __version__

# extra name -> packages it installs, in the order shown to users
EXTRAS: dict[str, tuple[str, ...]] = {
    "onnx": ("onnx", "onnxruntime"),
    "executorch": ("executorch", "torch"),
    "gguf": ("gguf",),
}

LOW_DISK_BYTES = 5 * 1000**3


@dataclass(frozen=True)
class PackageStatus:
    name: str
    extra: str
    version: str | None

    @property
    def installed(self) -> bool:
        return self.version is not None


@dataclass(frozen=True)
class DoctorReport:
    modelport_version: str
    python_version: str
    platform: str
    free_disk_bytes: int
    packages: list[PackageStatus]
    problems: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_report(
    folder: Path | None = None,
    version_of: Callable[[str], str | None] = installed_version,
) -> DoctorReport:
    """Gather versions and disk space. Never imports heavy packages such as torch."""
    packages = [
        PackageStatus(name=name, extra=extra, version=version_of(name))
        for extra, names in EXTRAS.items()
        for name in names
    ]
    free = shutil.disk_usage(folder or Path.cwd()).free

    problems: list[str] = []
    hints: list[str] = []
    for extra, names in EXTRAS.items():
        statuses = [p for p in packages if p.name in names]
        missing = [p.name for p in statuses if not p.installed]
        if missing and len(missing) < len(statuses):
            problems.append(f"'{extra}' is only partly installed, missing: {', '.join(missing)}")
        if missing:
            hints.append(f"pip install 'modelport[{extra}]'")
    if free < LOW_DISK_BYTES:
        problems.append(f"Only {free / 1000**3:.1f} GB free. Exports and LLM downloads need more.")

    return DoctorReport(
        modelport_version=__version__,
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.machine()} ({sys.platform})",
        free_disk_bytes=free,
        packages=packages,
        problems=problems,
        hints=hints,
    )
