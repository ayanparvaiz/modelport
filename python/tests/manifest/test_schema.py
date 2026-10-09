import json
from pathlib import Path

from jsonschema import Draft202012Validator

from modelport.manifest.schema import manifest_json_schema, render_schema

SPEC_DIR = Path(__file__).resolve().parents[3] / "spec"


def test_committed_schema_is_up_to_date():
    committed = (SPEC_DIR / "manifest.schema.json").read_text(encoding="utf-8")
    hint = "spec/manifest.schema.json is stale. Run: uv run modelport schema -o ../spec"
    assert committed == render_schema(), hint


def test_schema_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(manifest_json_schema())


def test_schema_key_is_required():
    assert manifest_json_schema()["required"][0] == "schema"


def test_optional_fields_are_not_nullable():
    text = json.dumps(manifest_json_schema())
    assert '"type": "null"' not in text
