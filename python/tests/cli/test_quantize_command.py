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


def test_quantize_adds_variants(bundle):
    result = runner.invoke(app, ["quantize", str(bundle.root), "--fp16", "--int8"])
    assert result.exit_code == 0, result.output
    assert "onnx-fp16" in result.output and "onnx-int8" in result.output
    assert runner.invoke(app, ["verify", str(bundle.root)]).exit_code == 0


def test_quantize_needs_a_kind(bundle):
    result = runner.invoke(app, ["quantize", str(bundle.root)])
    assert result.exit_code == 2
