"""Hugging Face transformers image classifiers, from the Hub or a local folder."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from ..errors import MissingDependencyError
from ..manifest import DetectionPostprocess, DType, ImagePreprocess, InputSpec, ResizeSpec, Task
from . import SourceError, SourceModel, model_id

# PIL resampling codes used by transformers image processors.
_RESAMPLE: dict[int, Literal["nearest", "bilinear", "bicubic"]] = {
    0: "nearest",
    2: "bilinear",
    3: "bicubic",
}


def load_hf(repo: str, *, license: str | None = None, image_size: int | None = None) -> SourceModel:
    """Load an image classifier or object detector with its image processor's preprocessing."""
    try:
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
    except ImportError as error:
        raise MissingDependencyError("hf: sources", "hf") from error

    try:
        architectures = AutoConfig.from_pretrained(repo).architectures or []
    except (OSError, ValueError) as error:
        raise SourceError(f"could not load '{repo}': {error}") from error
    if any(name.endswith("ForObjectDetection") for name in architectures):
        return _load_detector(repo, license=license, image_size=image_size)

    try:
        processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModelForImageClassification.from_pretrained(repo).eval()
    except (OSError, ValueError) as error:
        raise SourceError(f"could not load '{repo}' as an image classifier: {error}") from error

    preprocess, (height, width) = preprocess_from_processor(processor)
    config = model.config
    labels = [str(config.id2label[i]) for i in range(config.num_labels)]

    if license is None and not Path(repo).is_dir():
        license = _hub_license(repo)
    if license is None:
        raise SourceError(f"no license found for '{repo}'. Pass it with --license, e.g. MIT")

    return SourceModel(
        module=_logits_only(model),
        example_inputs=(torch.zeros(1, 3, height, width),),
        id=model_id(repo),
        task=Task.IMAGE_CLASSIFICATION,
        license=license,
        inputs=[
            InputSpec(
                name="pixel_values",
                dtype=DType.FLOAT32,
                shape=[1, 3, height, width],
                layout="NCHW",
                preprocess=preprocess,
            )
        ],
        output_names=["logits"],
        labels=labels,
        name=repo.rstrip("/").split("/")[-1],
        description=f"Hugging Face {type(model).__name__} from {repo}.",
        source=f"hf:{repo}",
    )


def _size_value(size: Any, key: str) -> int | None:
    value = size.get(key) if isinstance(size, dict) else getattr(size, key, None)
    return int(value) if value else None


def preprocess_from_processor(processor: Any) -> tuple[ImagePreprocess, tuple[int, int]]:
    """Translate a transformers image processor into a manifest preprocessing rule.

    Returns the rule and the final (height, width) of the input tensor.
    """
    resample = getattr(processor, "resample", 2)
    code = int(getattr(resample, "value", resample))
    method = _RESAMPLE.get(code)
    if method is None:
        raise SourceError(f"unsupported resample mode {resample!r}")
    antialias = method != "nearest"  # transformers resizes with PIL or antialiased torchvision

    size = processor.size
    height, width = _size_value(size, "height"), _size_value(size, "width")
    shortest = _size_value(size, "shortest_edge")
    crop_pct = getattr(processor, "crop_pct", None)
    crop: tuple[int, int] | None = None

    if height and width:
        resize = ResizeSpec(size=(height, width), method=method, antialias=antialias)
        final = (height, width)
    elif shortest and crop_pct:
        # ConvNext style: below 384, resize to shortest / crop_pct and crop back.
        if shortest < 384:
            resize = ResizeSpec(
                shorter_side=int(shortest / crop_pct), method=method, antialias=antialias
            )
            crop = (shortest, shortest)
        else:
            resize = ResizeSpec(size=(shortest, shortest), method=method, antialias=antialias)
        final = (shortest, shortest)
    elif shortest:
        resize = ResizeSpec(shorter_side=shortest, method=method, antialias=antialias)
        crop_size = getattr(processor, "crop_size", None)
        if not (getattr(processor, "do_center_crop", False) and crop_size):
            raise SourceError("this processor gives a variable input size; a center crop is needed")
        crop = (_size_value(crop_size, "height") or 0, _size_value(crop_size, "width") or 0)
        final = crop
    else:
        raise SourceError(f"unsupported image processor size {size!r}")

    if crop is None and getattr(processor, "do_center_crop", False):
        crop_size = getattr(processor, "crop_size", None)
        if crop_size:
            crop = (_size_value(crop_size, "height") or 0, _size_value(crop_size, "width") or 0)
            final = crop

    do_rescale = getattr(processor, "do_rescale", True)
    do_normalize = getattr(processor, "do_normalize", True)
    preprocess = ImagePreprocess(
        resize=resize,
        center_crop=crop,
        scale=float(processor.rescale_factor) if do_rescale else 1.0,
        mean=_triple(processor.image_mean, "image_mean") if do_normalize else (0.0, 0.0, 0.0),
        std=_triple(processor.image_std, "image_std") if do_normalize else (1.0, 1.0, 1.0),
    )
    return preprocess, final


def _triple(values: Any, name: str) -> tuple[float, float, float]:
    numbers = [float(v) for v in values]
    if len(numbers) != 3:
        raise SourceError(f"{name} must have 3 values, got {len(numbers)}")
    return numbers[0], numbers[1], numbers[2]


def _hub_license(repo: str) -> str | None:
    try:
        from huggingface_hub import model_info

        card = model_info(repo).card_data
    except Exception:
        return None
    license = getattr(card, "license", None) if card is not None else None
    if isinstance(license, list):
        license = license[0] if license else None
    return str(license) if license else None


def _logits_only(model: Any) -> Any:
    """Wrap a transformers model so forward(pixel_values) returns only the logits tensor."""
    import torch

    class LogitsOnly(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.inner(pixel_values=pixel_values).logits

    return LogitsOnly(model).eval()


def _load_detector(repo: str, *, license: str | None, image_size: int | None) -> SourceModel:
    """DETR-family detectors (DETR, YOLOS, RT-DETR, D-FINE) exported at a fixed square size."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    try:
        processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModelForObjectDetection.from_pretrained(repo).eval()
    except (OSError, ValueError) as error:
        raise SourceError(f"could not load '{repo}' as an object detector: {error}") from error

    try:
        preprocess, (height, width) = preprocess_from_processor(processor)
    except SourceError:
        # Detectors usually accept any size; export at a fixed square instead.
        size = image_size or _size_value(processor.size, "shortest_edge") or 512
        fixed = SimpleNamespace(
            resample=getattr(processor, "resample", 2),
            do_rescale=getattr(processor, "do_rescale", True) is not False,
            rescale_factor=getattr(processor, "rescale_factor", 1 / 255),
            do_normalize=getattr(processor, "do_normalize", True) is not False,
            image_mean=processor.image_mean,
            image_std=processor.image_std,
            size={"height": size, "width": size},
            do_center_crop=False,
        )
        preprocess, (height, width) = preprocess_from_processor(fixed)
    if image_size and (height, width) != (image_size, image_size):
        raise SourceError(f"{repo} has a fixed input size of {height}x{width}")

    wrapped = _detector_outputs(model)
    example = torch.zeros(1, 3, height, width)
    with torch.no_grad():
        logits, _ = wrapped(example)
    config = model.config
    classes = int(logits.shape[-1])
    background = classes == config.num_labels + 1
    labels = [str(config.id2label[i]) for i in range(config.num_labels)]

    if license is None and not Path(repo).is_dir():
        license = _hub_license(repo)
    if license is None:
        raise SourceError(f"no license found for '{repo}'. Pass it with --license, e.g. MIT")

    return SourceModel(
        module=wrapped,
        example_inputs=(example,),
        id=model_id(repo),
        task=Task.OBJECT_DETECTION,
        license=license,
        inputs=[
            InputSpec(
                name="pixel_values",
                dtype=DType.FLOAT32,
                shape=[1, 3, height, width],
                layout="NCHW",
                preprocess=preprocess,
            )
        ],
        output_names=["logits", "pred_boxes"],
        labels=labels,
        name=repo.rstrip("/").split("/")[-1],
        description=f"Hugging Face {type(model).__name__} from {repo}, at {width}x{height}.",
        source=f"hf:{repo}",
        detection=DetectionPostprocess(
            format="detr",
            boxes_output="pred_boxes",
            activation="softmax" if background else "sigmoid",
            background_class=background,
            box_format="cxcywh",
            normalized=True,
            score_threshold=0.5,
            # DETR-style models predict each object once, so no suppression is needed.
            iou_threshold=1.0,
            max_detections=int(logits.shape[1]),
        ),
    )


def _detector_outputs(model: Any) -> Any:
    """Wrap a detector so forward(pixel_values) returns (logits, pred_boxes)."""
    import torch

    class DetectorOutputs(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.inner(pixel_values=pixel_values)
            return out.logits, out.pred_boxes

    return DetectorOutputs(model).eval()
