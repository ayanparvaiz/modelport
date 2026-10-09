"""Shared building blocks for manifest models."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Sha256 = Annotated[
    str,
    Field(pattern=r"^[0-9a-f]{64}$", description="Lowercase hex SHA-256 digest of the file."),
]


class StrictModel(BaseModel):
    """Base for all manifest models.

    Unknown fields are rejected so that typos in hand-written manifests are caught early.
    Models are immutable once validated.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
