import json

from typer.testing import CliRunner

from modelport.cli import app

runner = CliRunner()


def test_inspect_prints_tables(tiny_onnx):
    result = runner.invoke(app, ["inspect", str(tiny_onnx)])
    assert result.exit_code == 0, result.stdout
    assert "Inputs" in result.stdout
    assert "float32" in result.stdout
    assert "opset" in result.stdout


def test_inspect_json(tiny_gguf):
    result = runner.invoke(app, ["inspect", str(tiny_gguf), "--json"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["format"] == "gguf"
    assert data["metadata"]["quantization"] == "q8_0"
    assert len(data["sha256"]) == 64


def test_inspect_unknown_file_fails(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("nope", encoding="utf-8")
    result = runner.invoke(app, ["inspect", str(path)])
    assert result.exit_code == 1
