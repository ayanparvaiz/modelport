"""Rules for turning raw output tensors into results people can use."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

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
    """Boxes with class scores. Two layouts are supported.

    * `rows` (YOLO style): this output is [1, N, 4 + objectness + classes].
    * `detr` (DETR, YOLOS, RT-DETR style): this output holds class scores
      [1, N, classes], and `boxes_output` names the [1, N, 4] boxes output.
    """

    type: Literal["detection"] = "detection"
    format: Literal["rows", "detr"] = "rows"
    boxes_output: str | None = Field(
        default=None, description="For format 'detr': the output with the [1, N, 4] boxes."
    )
    activation: Literal["softmax", "sigmoid", "none"] = Field(
        default="none", description="Applied to class scores before thresholding."
    )
    background_class: bool = Field(
        default=False,
        description="Whether the last class means 'no object' and is ignored, as in DETR.",
    )
    box_format: Literal["xyxy", "cxcywh"] = Field(
        default="cxcywh", description="How the four box values describe a box."
    )
    normalized: bool = Field(
        default=False,
        description="true: box values are 0-1 fractions of the input size; false: input pixels.",
    )
    has_objectness: bool = Field(
        default=False, description="For format 'rows': an objectness score follows the box."
    )
    score_threshold: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(
        default=0.45,
        ge=0,
        le=1,
        description="Non-max suppression overlap threshold. 1 turns suppression off.",
    )
    max_detections: int = Field(default=100, gt=0)
    labels: FileRef | None = None

    @model_validator(mode="after")
    def _check_format(self) -> DetectionPostprocess:
        if self.format == "detr" and not self.boxes_output:
            raise ValueError("format 'detr' needs 'boxes_output'")
        if self.format == "rows" and self.boxes_output:
            raise ValueError("'boxes_output' is only for format 'detr'")
        if self.format == "detr" and self.has_objectness:
            raise ValueError("'has_objectness' is only for format 'rows'")
        return self


Postprocess = Annotated[
    ClassificationPostprocess | DetectionPostprocess, Field(discriminator="type")
]
