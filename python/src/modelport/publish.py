"""Upload a bundle to the Hugging Face Hub, where apps can load it with hf://."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bundle import Bundle
from .errors import MissingDependencyError, ModelPortError
from .manifest import MANIFEST_FILENAME, Manifest

CARD_NAME = "README.md"


@dataclass(frozen=True)
class PublishResult:
    repo_id: str
    url: str
    hf_uri: str
    files: list[str]


def model_card(manifest: Manifest, repo_id: str) -> str:
    """A short Hub model card describing the bundle and how to load it."""
    rows = "\n".join(
        f"| `{v.id}` | {v.runtime} | {v.precision} | {v.file.size / 1e6:.1f} MB |"
        for v in manifest.variants
    )
    title = manifest.name or manifest.id
    description = manifest.description or ""
    return f"""---
license: {manifest.license.lower()}
tags:
  - modelport
  - flutter
  - on-device
  - {manifest.task}
---

# {title}

{description}

This is a [ModelPort](https://github.com/ayanparvaiz/modelport) bundle. The
`{MANIFEST_FILENAME}` file describes inputs, preprocessing, outputs, and checksums, so a
Flutter app can download and run it with one line.

| Variant | Runtime | Precision | Size |
|---|---|---|---|
{rows}

## Use in Flutter

```dart
final model = await ModelPort.load('hf://{repo_id}');
```

Source: `{manifest.source or "unknown"}`. License: {manifest.license}.
"""


def publish_bundle(path: str | Path, repo_id: str, *, private: bool = False) -> PublishResult:
    """Check the bundle, write a model card if missing, and upload the listed files."""
    if repo_id.count("/") != 1:
        raise ModelPortError(f"'{repo_id}' is not a repo id like 'org/name'")
    bundle = Bundle(path)
    manifest = bundle.read_manifest()
    problems = bundle.problems(manifest)
    if problems:
        raise ModelPortError(
            "bundle files do not match the manifest, run `modelport pack` first:\n  "
            + "\n  ".join(problems)
        )
    card = bundle.root / CARD_NAME
    if not card.exists():
        card.write_text(model_card(manifest, repo_id), encoding="utf-8")

    # Only files inside the bundle are uploaded; https urls stay where they are.
    files = [MANIFEST_FILENAME, CARD_NAME] + [ref.path for ref in manifest.files() if ref.path]
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise MissingDependencyError("Publishing to Hugging Face", "hf") from error

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(bundle.root),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=files,
        commit_message=f"modelport: {manifest.id} {manifest.version}",
    )
    return PublishResult(
        repo_id=repo_id,
        url=f"https://huggingface.co/{repo_id}",
        hf_uri=f"hf://{repo_id}",
        files=files,
    )
