"""Finalize a bundle after manual edits: refresh hashes and find stray files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bundle import Bundle
from .errors import ModelPortError
from .manifest import MANIFEST_FILENAME, Manifest

# Files that may sit next to the manifest without being listed in it.
ALLOWED_EXTRA = {MANIFEST_FILENAME, "README.md", ".gitattributes"}


@dataclass(frozen=True)
class PackResult:
    manifest: Manifest
    changed: list[str]
    stray: list[str]
    total_bytes: int


def pack_bundle(path: str | Path) -> PackResult:
    bundle = Bundle(path)
    if not bundle.manifest_path.is_file():
        raise ModelPortError(f"{bundle.manifest_path} not found")
    before = bundle.read_manifest()
    missing = [p for p in bundle.problems(before) if p.endswith(": missing")]
    if missing:
        raise ModelPortError("files listed in the manifest are missing:\n  " + "\n  ".join(missing))

    after = bundle.refresh(before)
    old = {ref.path: ref for ref in before.files() if ref.path}
    changed = sorted(ref.path for ref in after.files() if ref.path and old.get(ref.path) != ref)
    if after != before:
        bundle.write_manifest(after)

    listed = {ref.path for ref in after.files() if ref.path}
    stray = sorted(
        p.relative_to(bundle.root).as_posix()
        for p in bundle.root.rglob("*")
        if p.is_file()
        and p.relative_to(bundle.root).as_posix() not in listed
        and p.relative_to(bundle.root).as_posix() not in ALLOWED_EXTRA
    )
    total = sum(ref.size for ref in after.files() if ref.path)
    return PackResult(after, changed, stray, total)
