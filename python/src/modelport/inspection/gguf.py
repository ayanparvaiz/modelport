"""GGUF (llama.cpp) model inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import MissingDependencyError, TensorInfo


def read_gguf(path: Path) -> tuple[list[TensorInfo], list[TensorInfo], dict[str, str]]:
    try:
        import gguf
    except ImportError as error:
        raise MissingDependencyError("Inspecting .gguf files", "gguf") from error

    reader = gguf.GGUFReader(str(path))

    def value(key: str) -> Any:
        field = reader.fields.get(key)
        return field.contents() if field is not None else None

    arch = value("general.architecture") or "unknown"
    file_type = value("general.file_type")
    if file_type is not None:
        try:
            quant = gguf.LlamaFileType(file_type).name.removeprefix("MOSTLY_").lower()
        except ValueError:
            quant = str(file_type)
    else:
        quant = "unknown"

    metadata = {
        "architecture": str(arch),
        "name": str(value("general.name") or "unknown"),
        "quantization": quant,
        "context_length": str(value(f"{arch}.context_length") or "unknown"),
        "tensors": str(len(reader.tensors)),
        "chat_template": "yes" if "tokenizer.chat_template" in reader.fields else "no",
        "gguf_version": str(value("GGUF.version") or "unknown"),
    }
    return [], [], metadata
