"""File hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """Lowercase hex SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
