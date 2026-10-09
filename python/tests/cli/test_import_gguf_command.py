from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from modelport.cli import app
from modelport.manifest import Manifest

runner = CliRunner()


class FakeApi:
    def model_info(self, repo, revision=None, files_metadata=False, expand=None):
        if expand:
            return SimpleNamespace(gguf={"context_length": 8192, "chat_template": "t"})
        return SimpleNamespace(
            sha="c" * 40,
            card_data=SimpleNamespace(license="apache-2.0"),
            siblings=[
                SimpleNamespace(rfilename="m-q4_k_m.gguf", size=1000, lfs={"sha256": "d" * 64})
            ],
        )


def test_import_from_hub(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    result = runner.invoke(app, ["import-gguf", "org/m-GGUF", "-o", str(tmp_path)])
    assert result.exit_code == 0, result.output
    manifest = Manifest.from_file(tmp_path / "m" / "modelport.json")
    assert manifest.variants[0].id == "gguf-q4_k_m"
    again = runner.invoke(app, ["import-gguf", "org/m-GGUF", "-o", str(tmp_path)])
    assert again.exit_code == 1 and "--force" in again.output


def test_import_local_file(tiny_gguf, tmp_path):
    out = tmp_path / "dist"
    result = runner.invoke(app, ["import-gguf", str(tiny_gguf), "-o", str(out), "--license", "MIT"])
    assert result.exit_code == 0, result.output
    manifest = Manifest.from_file(out / "tiny-llama" / "modelport.json")
    assert (out / "tiny-llama" / "gguf-q8_0" / "model.gguf").is_file()
    assert manifest.license == "MIT"


def test_local_file_needs_license(tiny_gguf, tmp_path):
    result = runner.invoke(app, ["import-gguf", str(tiny_gguf), "-o", str(tmp_path)])
    assert result.exit_code == 1
