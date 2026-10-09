"""torchvision classification models with their ImageNet preprocessing."""

from __future__ import annotations

from typing import Literal

from ..errors import MissingDependencyError
from ..manifest import DType, ImagePreprocess, InputSpec, ResizeSpec, Task
from . import SourceError, SourceModel

_METHODS: dict[str, Literal["bilinear", "bicubic", "nearest"]] = {
    "bilinear": "bilinear",
    "bicubic": "bicubic",
    "nearest": "nearest",
}


def load_torchvision(name: str, *, pretrained: bool = True) -> SourceModel:
    """Load a torchvision classifier.

    `pretrained=False` keeps the weights' metadata but skips the download. Tests use it.
    """
    try:
        import torch
        import torchvision.models as models
    except ImportError as error:
        raise MissingDependencyError("torchvision: sources", "torchvision") from error

    try:
        weights = models.get_model_weights(name)["DEFAULT"]
    except ValueError as error:
        raise SourceError(f"torchvision has no model named '{name}'") from error

    transform = weights.transforms()
    if type(transform).__name__ != "ImageClassification":
        raise SourceError(
            f"torchvision:{name} is a {type(transform).__name__} model. "
            "Only image classification is supported for now."
        )

    method = _METHODS.get(transform.interpolation.value)
    if method is None:
        raise SourceError(f"unsupported interpolation {transform.interpolation}")
    crop = int(transform.crop_size[0])
    preprocess = ImagePreprocess(
        # torchvision resizes PIL images with PIL, which always antialiases.
        resize=ResizeSpec(
            shorter_side=int(transform.resize_size[0]),
            method=method,
            antialias=method != "nearest",
        ),
        center_crop=(crop, crop),
        mean=tuple(transform.mean),
        std=tuple(transform.std),
    )
    module = models.get_model(name, weights=weights if pretrained else None).eval()
    return SourceModel(
        module=module,
        example_inputs=(torch.zeros(1, 3, crop, crop),),
        id=name.lower(),
        task=Task.IMAGE_CLASSIFICATION,
        # torchvision code and its pretrained weights are released under BSD-3-Clause.
        license="BSD-3-Clause",
        inputs=[
            InputSpec(
                name="pixel_values",
                dtype=DType.FLOAT32,
                shape=[1, 3, crop, crop],
                layout="NCHW",
                preprocess=preprocess,
            )
        ],
        output_names=["logits"],
        labels=list(weights.meta["categories"]),
        name=name,
        description=f"torchvision {name} with {weights.name} weights.",
        source=f"torchvision:{name}",
    )
