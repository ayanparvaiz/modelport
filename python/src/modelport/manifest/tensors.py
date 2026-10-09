"""Tensor descriptions and the rules for turning raw data into model inputs."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .base import StrictModel
from .postprocess import Postprocess

TensorLayout = Literal["NCHW", "NHWC"]


class DType(StrEnum):
    """Element type of a tensor."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT64 = "int64"
    INT32 = "int32"
    INT8 = "int8"
    UINT8 = "uint8"
    BOOL = "bool"


class TensorSpec(StrictModel):
    """Name, element type, and shape of one model input or output."""

    name: str = Field(min_length=1, description="Tensor name as the engine knows it.")
    dtype: DType
    shape: list[int] = Field(
        description="Dimensions. Use -1 for a dimension that is only known at run time.",
        examples=[[1, 3, 224, 224]],
    )

    @field_validator("shape")
    @classmethod
    def _check_dims(cls, value: list[int]) -> list[int]:
        if any(dim == 0 or dim < -1 for dim in value):
            raise ValueError("every dimension must be a positive integer or -1")
        return value


class ResizeSpec(StrictModel):
    """How to resize an image. Set exactly one of `shorter_side` or `size`."""

    model_config = StrictModel.model_config | {
        "json_schema_extra": {"oneOf": [{"required": ["shorter_side"]}, {"required": ["size"]}]}
    }

    shorter_side: int | None = Field(
        default=None,
        gt=0,
        description="Scale so the shorter side has this length, keeping the aspect ratio.",
    )
    size: tuple[int, int] | None = Field(
        default=None, description="Exact output size as [height, width]. Ignores aspect ratio."
    )
    method: Literal["bilinear", "nearest", "bicubic"] = "bilinear"
    antialias: bool = Field(
        default=False,
        description="Whether to low-pass filter when shrinking. Python and Dart must agree.",
    )

    @field_validator("size")
    @classmethod
    def _check_size(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is not None and min(value) <= 0:
            raise ValueError("size must be positive")
        return value

    @model_validator(mode="after")
    def _check_exactly_one(self) -> ResizeSpec:
        if (self.shorter_side is None) == (self.size is None):
            raise ValueError("set exactly one of 'shorter_side' or 'size'")
        return self


class ImagePreprocess(StrictModel):
    """Turn a decoded image into a tensor.

    Steps run in this order: resize, center crop, channel order, multiply by `scale`,
    subtract `mean`, divide by `std`, then arrange in the input's layout.
    """

    type: Literal["image"] = "image"
    resize: ResizeSpec
    center_crop: tuple[int, int] | None = Field(
        default=None, description="Crop [height, width] from the center after resizing."
    )
    color: Literal["RGB", "BGR"] = "RGB"
    scale: float = Field(default=1 / 255, gt=0, description="Multiplier applied to 0-255 pixels.")
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @field_validator("std")
    @classmethod
    def _check_std(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if min(value) <= 0:
            raise ValueError("std values must be positive")
        return value

    @property
    def output_size(self) -> tuple[int, int] | None:
        """Final [height, width] if it is fixed, otherwise None."""
        if self.center_crop is not None:
            return self.center_crop
        return self.resize.size


class InputSpec(TensorSpec):
    """A model input, with optional preprocessing rules."""

    layout: TensorLayout | None = Field(
        default=None, description="Dimension order for image tensors."
    )
    preprocess: ImagePreprocess | None = None

    @model_validator(mode="after")
    def _check_image_input(self) -> InputSpec:
        if self.preprocess is None:
            return self
        if self.layout is None:
            raise ValueError(f"input '{self.name}' has image preprocessing but no layout")
        if len(self.shape) != 4:
            raise ValueError(f"image input '{self.name}' must have 4 dimensions")
        if self.dtype not in (DType.FLOAT32, DType.FLOAT16, DType.UINT8):
            raise ValueError(f"image input '{self.name}' must be float32, float16, or uint8")
        if self.layout == "NCHW":
            channels, height, width = self.shape[1], self.shape[2], self.shape[3]
        else:
            height, width, channels = self.shape[1], self.shape[2], self.shape[3]
        if channels not in (-1, 3):
            raise ValueError(f"image input '{self.name}' must have 3 channels")
        size = self.preprocess.output_size
        if size is not None:
            for expected, actual, label in ((size[0], height, "height"), (size[1], width, "width")):
                if actual != -1 and actual != expected:
                    raise ValueError(
                        f"input '{self.name}' {label} is {actual}, "
                        f"but preprocessing produces {expected}"
                    )
        return self


class OutputSpec(TensorSpec):
    """A model output, with optional rules for turning it into results."""

    postprocess: Postprocess | None = None
