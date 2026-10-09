import json

from typer.testing import CliRunner

from modelport.cli import app
from modelport.manifest.schema import render_schema

runner = CliRunner()


def test_schema_prints_json():
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["$id"].endswith("manifest.schema.json")


def test_schema_writes_into_folder(tmp_path):
    result = runner.invoke(app, ["schema", "-o", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "manifest.schema.json").read_text(encoding="utf-8") == render_schema()


def test_schema_writes_to_file(tmp_path):
    target = tmp_path / "custom.json"
    result = runner.invoke(app, ["schema", "--output", str(target)])
    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == render_schema()
