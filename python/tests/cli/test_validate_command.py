from pathlib import Path

from typer.testing import CliRunner

from modelport.cli import app

runner = CliRunner()
EXAMPLES = Path(__file__).resolve().parents[3] / "spec" / "examples"


def test_valid_examples_pass():
    files = [str(p) for p in sorted(EXAMPLES.glob("*.json"))]
    result = runner.invoke(app, ["validate", *files])
    assert result.exit_code == 0, result.stdout
    assert result.stdout.count("✓") == len(files)


def test_invalid_example_fails_with_field_path():
    result = runner.invoke(app, ["validate", str(EXAMPLES / "invalid" / "missing-sha256.json")])
    assert result.exit_code == 1
    assert "variants.0.file.sha256" in result.stdout


def test_folder_argument_reads_modelport_json(tmp_path):
    (tmp_path / "modelport.json").write_text(
        (EXAMPLES / "text-generation.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    result = runner.invoke(app, ["validate", str(tmp_path)])
    assert result.exit_code == 0, result.stdout


def test_missing_file_fails(tmp_path):
    result = runner.invoke(app, ["validate", str(tmp_path / "nope.json")])
    assert result.exit_code == 1


def test_bad_json_fails(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1
