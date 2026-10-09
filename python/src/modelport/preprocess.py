"""Reference implementation of manifest image preprocessing.

The Dart packages implement the same steps, and cross-language tests compare the two.
See docs/spec.md, "Image preprocessing", for the exact rules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .manifest.tensors import DType, ImagePreprocess, InputSpec, ResizeSpec

_PIL_FILTERS = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "nearest": Image.Resampling.NEAREST,
}


def load_image(path: str | Path) -> Image.Image:
    """Open an image, apply its EXIF orientation, and convert it to 8-bit RGB."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def resized_size(width: int, height: int, spec: ResizeSpec) -> tuple[int, int]:
    """Output (width, height) for a resize rule."""
    if spec.size is not None:
        out_height, out_width = spec.size
        return out_width, out_height
    assert spec.shorter_side is not None
    short = spec.shorter_side
    if width <= height:
        return short, int(short * height / width)
    return int(short * width / height), short


def resize(image: Image.Image, spec: ResizeSpec) -> np.ndarray:
    """Resize to a float32 array of shape (height, width, 3) with values in 0-255."""
    width, height = resized_size(image.width, image.height, spec)
    if spec.antialias or spec.method == "nearest":
        if (width, height) != image.size:
            image = image.resize((width, height), resample=_PIL_FILTERS[spec.method])
        return np.asarray(image, dtype=np.float32)
    return bilinear_half_pixel(np.asarray(image, dtype=np.float32), height, width)


def _axis(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = in_size / out_size
    src = np.maximum((np.arange(out_size, dtype=np.float64) + 0.5) * scale - 0.5, 0.0)
    low = np.minimum(np.floor(src).astype(np.int64), in_size - 1)
    high = np.minimum(low + 1, in_size - 1)
    return low, high, src - low


def bilinear_half_pixel(array: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    """Bilinear resize with half-pixel centers and no antialiasing, in floating point.

    Matches PyTorch `interpolate(mode="bilinear", align_corners=False, antialias=False)`
    and OpenCV `INTER_LINEAR`.
    """
    data = array.astype(np.float64)
    y_low, y_high, y_frac = _axis(data.shape[0], out_height)
    x_low, x_high, x_frac = _axis(data.shape[1], out_width)
    y_frac = y_frac[:, None, None]
    rows = data[y_low] * (1 - y_frac) + data[y_high] * y_frac
    x_frac = x_frac[None, :, None]
    out = rows[:, x_low] * (1 - x_frac) + rows[:, x_high] * x_frac
    return out.astype(np.float32)


def center_crop(array: np.ndarray, crop_height: int, crop_width: int) -> np.ndarray:
    """Crop the center, padding with zeros first if the image is too small."""
    height, width = array.shape[:2]
    if crop_height > height or crop_width > width:
        dh, dw = max(crop_height - height, 0), max(crop_width - width, 0)
        array = np.pad(array, ((dh // 2, (dh + 1) // 2), (dw // 2, (dw + 1) // 2), (0, 0)))
        height, width = array.shape[:2]
    # Python's round() is round-half-to-even, as in torchvision.
    top = round((height - crop_height) / 2)
    left = round((width - crop_width) / 2)
    return array[top : top + crop_height, left : left + crop_width]


def preprocess_array(image: Image.Image, pre: ImagePreprocess) -> np.ndarray:
    """Run resize, crop, channel order, scale, and normalize. Returns (H, W, 3) float64."""
    array = resize(image.convert("RGB"), pre.resize)
    if pre.center_crop is not None:
        array = center_crop(array, *pre.center_crop)
    if pre.color == "BGR":
        array = array[..., ::-1]
    values = array.astype(np.float64) * pre.scale
    return (values - np.asarray(pre.mean)) / np.asarray(pre.std)


def preprocess_image(image: Image.Image, spec: InputSpec) -> np.ndarray:
    """Build the input tensor for one image, including the batch dimension."""
    if spec.preprocess is None:
        raise ValueError(f"input '{spec.name}' has no image preprocessing")
    values = preprocess_array(image, spec.preprocess)
    if spec.dtype is DType.UINT8:
        tensor = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    elif spec.dtype is DType.FLOAT16:
        tensor = values.astype(np.float16)
    else:
        tensor = values.astype(np.float32)
    tensor = tensor.transpose(2, 0, 1)[None] if spec.layout == "NCHW" else tensor[None]
    for axis, (expected, actual) in enumerate(zip(spec.shape, tensor.shape, strict=True)):
        if expected != -1 and expected != actual:
            raise ValueError(
                f"input '{spec.name}' dimension {axis} is {actual}, manifest says {expected}"
            )
    return np.ascontiguousarray(tensor)


def to_le_bytes(array: np.ndarray) -> bytes:
    """Raw little-endian bytes, the format of golden files."""
    return np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes()
