"""Upload a bundle to the Hugging Face Hub, where apps can load it with hf://."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
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
    location: str
    """What an app passes to ModelPort.load()."""
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
        location=f"hf://{repo_id}",
        files=files,
    )


def github_asset_name(model_id: str, path: str) -> str:
    """Release assets are flat, so 'onnx-fp32/model.onnx' becomes 'id--onnx-fp32--model.onnx'."""
    return f"{model_id}--{path.replace('/', '--')}"


def github_manifest(manifest: Manifest, repo: str, tag: str) -> Manifest:
    """The manifest with every bundle path replaced by its release download URL."""
    base = f"https://github.com/{repo}/releases/download/{tag}"
    data = manifest.model_dump(by_alias=True, exclude_none=True, mode="json")

    def rewrite(node: object) -> object:
        if isinstance(node, dict):
            if "path" in node and "sha256" in node and "size" in node:
                name = github_asset_name(manifest.id, str(node["path"]))
                return {"url": f"{base}/{name}", "size": node["size"], "sha256": node["sha256"]}
            return {key: rewrite(value) for key, value in node.items()}
        if isinstance(node, list):
            return [rewrite(item) for item in node]
        return node

    return Manifest.model_validate(rewrite(data))


def publish_to_github(
    path: str | Path,
    repo: str,
    tag: str,
    *,
    run: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> PublishResult:
    """Upload a bundle as assets of a GitHub release, using the `gh` CLI.

    The uploaded manifest, `<id>.json`, points at the other assets by URL, so apps
    load the model from https://github.com/<repo>/releases/download/<tag>/<id>.json.
    """
    if repo.count("/") != 1:
        raise ModelPortError(f"'{repo}' is not a GitHub repo like 'owner/name'")
    bundle = Bundle(path)
    manifest = bundle.read_manifest()
    problems = bundle.problems(manifest)
    if problems:
        raise ModelPortError(
            "bundle files do not match the manifest, run `modelport pack` first:\n  "
            + "\n  ".join(problems)
        )
    execute = run or _run_gh
    if shutil.which("gh") is None and run is None:
        raise ModelPortError("publishing to GitHub needs the gh CLI: https://cli.github.com")

    if execute(["gh", "release", "view", tag, "-R", repo]).returncode != 0:
        created = execute(
            [
                "gh", "release", "create", tag, "-R", repo,
                "--title", f"Model zoo {tag}",
                "--notes", "ModelPort model bundles. Load them by their .json URL.",
                "--latest=false",
            ]
        )  # fmt: skip
        if created.returncode != 0:
            raise ModelPortError(f"could not create release {tag}: {created.stderr.strip()}")

    remote = github_manifest(manifest, repo, tag)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        uploads: list[str] = []
        for ref in manifest.files():
            if ref.path is None:
                continue
            target = staging / github_asset_name(manifest.id, ref.path)
            if not target.exists():
                shutil.copyfile(bundle.root / ref.path, target)
                uploads.append(str(target))
        manifest_file = staging / f"{manifest.id}.json"
        manifest_file.write_text(remote.to_json(), encoding="utf-8")
        uploads.append(str(manifest_file))
        result = execute(["gh", "release", "upload", tag, "-R", repo, "--clobber", *uploads])
        if result.returncode != 0:
            raise ModelPortError(f"upload to {repo} {tag} failed: {result.stderr.strip()}")

    url = f"https://github.com/{repo}/releases/download/{tag}/{manifest.id}.json"
    return PublishResult(repo_id=repo, url=url, location=url, files=[Path(u).name for u in uploads])


def _run_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)
