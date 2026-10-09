import pytest
from typer.testing import CliRunner

from modelport.cli import app
from modelport.pipeline import export_bundle

runner = CliRunner()


def test_pack_reports_changes(tiny_source, tmp_path):
    pytest.importorskip("onnxscript")
    bundle, _ = export_bundle(tiny_source, tmp_path, ["onnx"])
    (bundle.root / "labels.txt").write_text("r\ng\nb\n", encoding="utf-8")
    (bundle.root / "scratch.txt").write_text("x", encoding="utf-8")
    result = runner.invoke(app, ["pack", str(bundle.root)])
    assert result.exit_code == 0, result.output
    assert "updated hash: labels.txt" in result.output
    assert "scratch.txt" in result.output


def test_pack_missing_bundle(tmp_path):
    assert runner.invoke(app, ["pack", str(tmp_path)]).exit_code == 1
