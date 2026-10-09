"""Check that every zoo manifest loads and every file it points at exists with its size.

Run: uv run python scripts/check_zoo.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from modelport.manifest import Manifest

INDEX = Path(__file__).resolve().parents[2] / "zoo" / "index.json"


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def main() -> int:
    failures = 0
    for entry in json.loads(INDEX.read_text(encoding="utf-8"))["models"]:
        with urllib.request.urlopen(entry["location"], timeout=60) as response:
            manifest = Manifest.model_validate_json(response.read())
        for ref in manifest.files():
            if ref.url is None:
                print(f"✗ {entry['id']}: {ref.path} has no url")
                failures += 1
                continue
            size = remote_size(ref.url)
            if size != ref.size:
                print(f"✗ {entry['id']}: {ref.url} is {size} bytes, manifest says {ref.size}")
                failures += 1
        print(f"✓ {entry['id']}: {len(manifest.files())} files")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
