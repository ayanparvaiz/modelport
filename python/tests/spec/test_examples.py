"""Every example in spec/examples must load, and every invalid example must fail.

The Dart parser runs against the same files, which keeps both languages in step.
"""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from modelport.manifest import Manifest
from modelport.manifest.schema import manifest_json_schema

EXAMPLES = Path(__file__).resolve().parents[3] / "spec" / "examples"
VALID = sorted(EXAMPLES.glob("*.json"))

# file name -> (text the Python error must contain, whether JSON Schema also catches it)
INVALID = {
    "missing-sha256.json": ("sha256", True),
    "path-traversal.json": ("'..' segments", True),
    "http-url.json": ("https", True),
    "duplicate-variant-ids.json": ("duplicate variant id", False),
    "unknown-field.json": ("checksum", True),
    "missing-schema.json": ("missing 'schema' key", True),
    "text-generation-without-llm.json": ("needs an 'llm' section", False),
}

validator = Draft202012Validator(manifest_json_schema())


def test_examples_exist():
    assert len(VALID) >= 3
    assert sorted(p.name for p in (EXAMPLES / "invalid").glob("*.json")) == sorted(INVALID)


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.name)
def test_valid_example_passes_python(path):
    Manifest.from_file(path)


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.name)
def test_valid_example_passes_json_schema(path):
    validator.validate(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("name", sorted(INVALID))
def test_invalid_example_fails_python(name):
    message, _ = INVALID[name]
    with pytest.raises((ValidationError, ValueError), match=message):
        Manifest.from_file(EXAMPLES / "invalid" / name)


@pytest.mark.parametrize("name", sorted(n for n, (_, by_schema) in INVALID.items() if by_schema))
def test_invalid_example_fails_json_schema(name):
    data = json.loads((EXAMPLES / "invalid" / name).read_text(encoding="utf-8"))
    assert not validator.is_valid(data)
