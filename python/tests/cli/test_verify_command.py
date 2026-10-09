import numpy as np
import pytest
from typer.testing import CliRunner

from modelport.cli import app
from modelport.pipeline import export_bundle

runner = CliRunner()


@pytest.fixture
def bundle(tiny_source, tmp_path):
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    bundle, _ = export_bundle(tiny_source, tmp_path, ["onnx"])
    return bundle


def test_verify_passes(bundle):
    result = runner.invoke(app, ["verify", str(bundle.root)])
    assert result.exit_code == 0, result.output
    assert "pass" in result.output


def test_verify_fails_on_wrong_golden(bundle):
    manifest = bundle.read_manifest()
    path = bundle.root / "golden" / "logits.bin"
    path.write_bytes((np.frombuffer(path.read_bytes(), "<f4") + 1).astype("<f4").tobytes())
    bundle.write_manifest(bundle.refresh(manifest))
    result = runner.invoke(app, ["verify", str(bundle.root)])
    assert result.exit_code == 1
    assert "fail" in result.output


def test_verify_missing_bundle(tmp_path):
    result = runner.invoke(app, ["verify", str(tmp_path)])
    assert result.exit_code == 1
