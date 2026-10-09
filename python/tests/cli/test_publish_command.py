import pytest
from typer.testing import CliRunner

from modelport.cli import app
from modelport.pipeline import export_bundle

runner = CliRunner()


class FakeApi:
    def create_repo(self, repo_id, **kwargs):
        pass

    def upload_folder(self, **kwargs):
        pass


def test_publish_prints_hf_uri(tiny_source, tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    pytest.importorskip("onnxscript")
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    bundle, _ = export_bundle(tiny_source, tmp_path, ["onnx"])
    result = runner.invoke(app, ["publish", str(bundle.root), "--hf", "someone/tiny"])
    assert result.exit_code == 0, result.output
    assert "hf://someone/tiny" in result.output


def test_publish_needs_repo_option(tmp_path):
    result = runner.invoke(app, ["publish", str(tmp_path)])
    assert result.exit_code == 2
