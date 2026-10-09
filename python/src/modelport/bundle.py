"""A bundle folder: modelport.json plus the files it points to."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .manifest import MANIFEST_FILENAME, FileRef, Manifest

_SAFE_CHECK_SHA = "0" * 64


class Bundle:
    """Write and check the files of one model bundle."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, relative: str) -> Path:
        """Absolute path for a bundle-relative path. Rejects unsafe paths."""
        FileRef(path=relative, size=1, sha256=_SAFE_CHECK_SHA)  # validates the path rules
        return self.root / relative

    def add(self, relative: str) -> FileRef:
        """Reference a file that is already inside the bundle."""
        target = self.path(relative)
        if not target.is_file():
            raise FileNotFoundError(f"{target} does not exist")
        return FileRef(path=relative, size=target.stat().st_size, sha256=sha256_file(target))

    def write_bytes(self, relative: str, data: bytes) -> FileRef:
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return self.add(relative)

    def write_text(self, relative: str, text: str) -> FileRef:
        return self.write_bytes(relative, text.encode("utf-8"))

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILENAME

    def write_manifest(self, manifest: Manifest) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(manifest.to_json(), encoding="utf-8")
        return self.manifest_path

    def read_manifest(self) -> Manifest:
        return Manifest.from_file(self.manifest_path)

    def problems(self, manifest: Manifest) -> list[str]:
        """Bundle files that are missing or do not match their size or sha256."""
        found: list[str] = []
        for ref in manifest.files():
            if ref.path is None:
                continue
            target = self.root / ref.path
            if not target.is_file():
                found.append(f"{ref.path}: missing")
            elif target.stat().st_size != ref.size:
                found.append(
                    f"{ref.path}: size is {target.stat().st_size}, manifest says {ref.size}"
                )
            elif sha256_file(target) != ref.sha256:
                found.append(f"{ref.path}: sha256 does not match")
        return found

    def refresh(self, manifest: Manifest) -> Manifest:
        """Recompute size and sha256 of every bundle file the manifest points to."""
        data = manifest.model_dump(by_alias=True, exclude_none=True, mode="json")
        return Manifest.model_validate(self._refresh_refs(data))

    def _refresh_refs(self, node: Any) -> Any:
        if isinstance(node, dict):
            if "path" in node and "sha256" in node and "size" in node:
                ref = self.add(node["path"])
                return {**node, "size": ref.size, "sha256": ref.sha256}
            return {key: self._refresh_refs(value) for key, value in node.items()}
        if isinstance(node, list):
            return [self._refresh_refs(item) for item in node]
        return node
