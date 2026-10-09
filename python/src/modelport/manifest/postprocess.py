"""Rules for turning raw output tensors into results people can use."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .base import StrictModel
from .files import FileRef


class ClassificationPostprocess(StrictModel):
    """Scores over a fixed set of classes, for example ImageNet logits of shape [1, 1000]."""

    type: Literal["classification"] = "classification"
    activation: Literal["softmax", "sigmoid", "none"] = Field(
        default="softmax", description="Applied to raw scores before ranking."
    )
    labels: FileRef | None = Field(
        default=None, description="UTF-8 text file with one class name per line."
    )
    top_k: int = Field(default=5, gt=0, description="How many results to return by default.")


class DetectionPostprocess(StrictModel):
    """Boxes with class scores, shaped [1, N, 4 + objectness + classes].

    Draft for spec 0.1. The exact layout will be confirmed when the first detection model
    is exported in Phase 7.
    """

    type: Literal["detection"] = "detection"
    box_format: Literal["xyxy", "cxcywh"] = Field(
        default="cxcywh", description="How the first four values of each row describe a box."
    )
    has_objectness: bool = Field(
        default=True, description="Whether a single objectness score follows the box values."
    )
    score_threshold: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(
        default=0.45, ge=0, le=1, description="Overlap above which non-max suppression drops boxes."
    )
    max_detections: int = Field(default=100, gt=0)
    labels: FileRef | None = None


Postprocess = Annotated[
    ClassificationPostprocess | DetectionPostprocess, Field(discriminator="type")
]
