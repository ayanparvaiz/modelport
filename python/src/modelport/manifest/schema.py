"""Generate the JSON Schema for `modelport.json` from the Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema

from .models import Manifest

SCHEMA_URL = (
    "https://raw.githubusercontent.com/ayanparvaiz/modelport/main/spec/manifest.schema.json"
)


class _ManifestSchemaGenerator(GenerateJsonSchema):
    """Optional fields are written by leaving them out, never as null."""

    def nullable_schema(self, schema: core_schema.NullableSchema) -> JsonSchemaValue:
        return self.generate_inner(schema["schema"])

    def default_schema(self, schema: core_schema.WithDefaultSchema) -> JsonSchemaValue:
        json_schema = super().default_schema(schema)
        if "default" in json_schema and json_schema["default"] is None:
            del json_schema["default"]
        return json_schema


def manifest_json_schema() -> dict[str, Any]:
    """The JSON Schema (draft 2020-12) that every `modelport.json` must satisfy."""
    schema = Manifest.model_json_schema(
        by_alias=True, mode="validation", schema_generator=_ManifestSchemaGenerator
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_URL,
        **schema,
    }


def render_schema() -> str:
    return json.dumps(manifest_json_schema(), indent=2, ensure_ascii=False) + "\n"


def write_schema(path: str | Path) -> None:
    Path(path).write_text(render_schema(), encoding="utf-8")
