from types import SimpleNamespace
from typing import ClassVar

import pytest

from modelport.bundle import Bundle
from modelport.gguf_import import (
    GgufImportError,
    estimate_min_ram_mb,
    import_from_file,
    import_from_hub,
    quant_of,
)
from modelport.manifest import Manifest, Runtime, Task

SHA = "9217f5db79a29953eb74d5343926648285ec7e67"


def sibling(name, size=491_400_032, sha="a" * 64):
    return SimpleNamespace(rfilename=name, size=size, lfs={"sha256": sha})


class FakeApi:
    gguf: ClassVar[dict] = {
        "architecture": "qwen2",
        "context_length": 8192,
        "chat_template": "{{x}}",
    }
    license: ClassVar[str | None] = "apache-2.0"

    def model_info(self, repo, revision=None, files_metadata=False, expand=None):
        if expand:
            return SimpleNamespace(gguf=FakeApi.gguf)
        return SimpleNamespace(
            sha=SHA,
            card_data=SimpleNamespace(license=FakeApi.license),
            siblings=[
                sibling("README.md", size=100),
                sibling("qwen2.5-0.5b-instruct-q4_k_m.gguf"),
                sibling("qwen2.5-0.5b-instruct-q8_0.gguf", size=675_710_816, sha="b" * 64),
                sibling("big-q4_0-00001-of-00002.gguf"),
            ],
        )


@pytest.fixture
def hub(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    FakeApi.gguf = {"architecture": "qwen2", "context_length": 8192, "chat_template": "{{x}}"}
    FakeApi.license = "apache-2.0"
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)


@pytest.mark.parametrize(
    ("name", "quant"),
    [
        ("qwen2.5-0.5b-instruct-q4_k_m.gguf", "q4_k_m"),
        ("Model-Q8_0.gguf", "q8_0"),
        ("llama-IQ4_XS.gguf", "iq4_xs"),
        ("phi-f16.gguf", "f16"),
        ("weights.gguf", None),
    ],
)
def test_quant_of(name, quant):
    assert quant_of(name) == quant


def test_ram_estimate_is_rounded_to_256_mb():
    assert estimate_min_ram_mb(491_400_032, 4096) == 1024
    assert estimate_min_ram_mb(675_710_816, 4096) % 256 == 0


def test_import_from_hub(hub):
    result = import_from_hub("Qwen/Qwen2.5-0.5B-Instruct-GGUF", ["q4_k_m", "Q8_0"])
    manifest = result.manifest
    assert manifest.id == "qwen2.5-0.5b-instruct"
    assert manifest.task is Task.TEXT_GENERATION
    assert manifest.license == "apache-2.0"
    assert [v.id for v in manifest.variants] == ["gguf-q4_k_m", "gguf-q8_0"]
    first = manifest.variants[0]
    assert first.runtime is Runtime.LLAMACPP
    assert first.file.url == (
        f"https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/{SHA}/"
        "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    )
    assert first.min_ram_mb == 1024
    assert manifest.llm is not None and manifest.llm.context_length == 4096
    assert result.warnings == []
    Manifest.model_validate_json(manifest.to_json())


def test_context_is_capped_at_the_trained_length(hub):
    result = import_from_hub("org/model-GGUF", ["q4_k_m"], context=32768)
    assert result.manifest.llm is not None
    assert result.manifest.llm.context_length == 8192
    assert "above the trained 8192" in result.warnings[0]


def test_missing_quant_lists_what_exists(hub):
    with pytest.raises(GgufImportError, match="Available: q4_k_m, q8_0"):
        import_from_hub("org/model-GGUF", ["q2_k"])


def test_missing_license_needs_flag(hub):
    FakeApi.license = None
    with pytest.raises(GgufImportError, match="--license"):
        import_from_hub("org/model-GGUF", ["q4_k_m"])
    assert import_from_hub("org/model-GGUF", ["q4_k_m"], license="MIT").manifest.license == "MIT"


def test_missing_chat_template_is_a_warning(hub):
    FakeApi.gguf = {"architecture": "llama", "context_length": 2048}
    result = import_from_hub("org/model-GGUF", ["q4_k_m"], context=1024)
    assert any("chat template" in w for w in result.warnings)


def test_import_local_file(tiny_gguf, tmp_path):
    bundle = Bundle(tmp_path / "bundle")
    result = import_from_file(tiny_gguf, bundle, license="MIT", context=4096)
    manifest = result.manifest
    assert manifest.id == "tiny-llama"
    assert manifest.variants[0].id == "gguf-q8_0"
    assert manifest.llm is not None and manifest.llm.context_length == 2048
    assert bundle.problems(manifest) == []
    assert any("chat template" in w for w in result.warnings)
