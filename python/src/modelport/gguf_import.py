"""Describe ready-made GGUF language models with a manifest.

Large GGUF files are not copied: the manifest points at them with pinned
Hugging Face URLs, so a bundle is just modelport.json.
"""

from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bundle import Bundle
from .errors import MissingDependencyError, ModelPortError
from .manifest import FileRef, LlmConfig, Manifest, Runtime, Task, Variant
from .sources import model_id as make_id

_QUANT = re.compile(r"[-_.]((?:i?q\d\w*)|bf16|f16|f32)\.gguf$", re.IGNORECASE)
_SPLIT = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


class GgufImportError(ModelPortError):
    """A GGUF model could not be described."""


@dataclass(frozen=True)
class GgufImport:
    manifest: Manifest
    warnings: list[str] = field(default_factory=list)


def quant_of(filename: str) -> str | None:
    """'qwen2.5-0.5b-instruct-q4_k_m.gguf' -> 'q4_k_m'."""
    match = _QUANT.search(filename)
    return match.group(1).lower() if match else None


def estimate_min_ram_mb(size_bytes: int, context: int) -> int:
    """A rough lower bound: weights, a KV cache for `context`, and runtime overhead."""
    mib = size_bytes / 2**20 * 1.1 + 256 + context / 1024 * 32
    return int(math.ceil(mib / 256) * 256)


def _bundle_id(name: str) -> str:
    base = make_id(name)
    return base[: -len("-gguf")] if base.endswith("-gguf") else base


def import_from_hub(
    repo: str,
    quants: list[str],
    *,
    revision: str | None = None,
    context: int = 4096,
    license: str | None = None,
    model_id: str | None = None,
) -> GgufImport:
    """Build a manifest for GGUF files in a Hugging Face repo, without downloading them."""
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise MissingDependencyError("Importing GGUF models from the Hub", "hf") from error

    api = HfApi()
    try:
        info = api.model_info(repo, revision=revision, files_metadata=True)
        summary = getattr(api.model_info(repo, revision=revision, expand=["gguf"]), "gguf", None)
    except Exception as error:
        raise GgufImportError(
            f"could not read {repo} from the Hugging Face Hub: {error}"
        ) from error

    files: dict[str, Any] = {}
    for sibling in info.siblings or []:
        name = sibling.rfilename
        if not name.endswith(".gguf") or _SPLIT.search(name):
            continue
        quant = quant_of(Path(name).name)
        if quant and quant not in files:
            files[quant] = sibling
    if not files:
        raise GgufImportError(f"{repo} has no single-file GGUF models")

    warnings: list[str] = []
    gguf = summary if isinstance(summary, dict) else {}
    trained = gguf.get("context_length")
    if trained and context > trained:
        warnings.append(f"context {context} is above the trained {trained}; using {trained}")
        context = int(trained)
    if not gguf.get("chat_template"):
        warnings.append("the GGUF has no chat template, so chat may not work")

    license = license or _card_license(info)
    if not license:
        raise GgufImportError(f"no license found for {repo}. Pass it with --license")

    variants = []
    for quant in [q.lower() for q in quants]:
        sibling = files.get(quant)
        if sibling is None:
            raise GgufImportError(
                f"{repo} has no {quant} file. Available: {', '.join(sorted(files))}"
            )
        sha256 = _lfs_sha256(sibling)
        if sha256 is None or not sibling.size:
            raise GgufImportError(f"{sibling.rfilename} has no size or sha256 on the Hub")
        url = f"https://huggingface.co/{repo}/resolve/{info.sha}/{sibling.rfilename}"
        variants.append(
            Variant(
                id=f"gguf-{quant}",
                runtime=Runtime.LLAMACPP,
                precision=quant,
                file=FileRef(url=url, size=sibling.size, sha256=sha256),
                min_ram_mb=estimate_min_ram_mb(sibling.size, context),
            )
        )

    manifest = Manifest(
        id=model_id or _bundle_id(repo),
        version="1.0.0",
        task=Task.TEXT_GENERATION,
        license=license,
        name=repo.split("/")[-1],
        description=f"{gguf.get('architecture', 'GGUF')} model from {repo}.",
        source=f"hf:{repo}@{(info.sha or '')[:12]}",
        variants=variants,
        llm=LlmConfig(context_length=context),
    )
    return GgufImport(manifest, warnings)


def import_from_file(
    path: str | Path,
    bundle: Bundle,
    *,
    context: int = 4096,
    license: str,
    model_id: str | None = None,
) -> GgufImport:
    """Copy a local GGUF file into a bundle and describe it."""
    from .inspection import inspect_model

    source = Path(path)
    info = inspect_model(source)
    if info.format != "gguf":
        raise GgufImportError(f"{source} is not a GGUF file")
    quant = quant_of(source.name) or info.metadata.get("quantization", "unknown").lower()
    warnings: list[str] = []
    trained = info.metadata.get("context_length", "")
    if trained.isdigit() and context > int(trained):
        warnings.append(f"context {context} is above the trained {trained}; using {trained}")
        context = int(trained)
    if info.metadata.get("chat_template") != "yes":
        warnings.append("the GGUF has no chat template, so chat may not work")

    variant_id = f"gguf-{quant}"
    relative = f"{variant_id}/model.gguf"
    target = bundle.path(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    manifest = Manifest(
        id=model_id or _bundle_id(info.metadata.get("name", source.stem)),
        version="1.0.0",
        task=Task.TEXT_GENERATION,
        license=license,
        name=info.metadata.get("name"),
        source=f"file:{source.name}",
        variants=[
            Variant(
                id=variant_id,
                runtime=Runtime.LLAMACPP,
                precision=quant,
                file=bundle.add(relative),
                min_ram_mb=estimate_min_ram_mb(info.size, context),
            )
        ],
        llm=LlmConfig(context_length=context),
    )
    return GgufImport(manifest, warnings)


def _card_license(info: Any) -> str | None:
    card = getattr(info, "card_data", None)
    value = getattr(card, "license", None) if card is not None else None
    if isinstance(value, list):
        value = value[0] if value else None
    return str(value) if value else None


def _lfs_sha256(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
