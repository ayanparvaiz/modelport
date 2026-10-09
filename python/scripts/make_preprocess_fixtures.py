"""Write spec/fixtures/preprocess: images, preprocessing cases, and expected tensors.

The Dart package runs the same cases and must produce the same tensors, which keeps
the two implementations of the spec in step. A Python test fails if these files
drift from the reference implementation.

Run: uv run python scripts/make_preprocess_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from modelport.golden import sample_image
from modelport.manifest import InputSpec
from modelport.preprocess import preprocess_image, to_le_bytes

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "spec" / "fixtures" / "preprocess"
IMAGENET = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def images() -> dict[str, Image.Image]:
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    tall = np.zeros((70, 30, 3), dtype=np.uint8)
    tall[..., 0] = np.arange(70, dtype=np.uint8)[:, None] * 3
    tall[..., 1] = np.arange(30, dtype=np.uint8)[None, :] * 8
    tall[::7, :, 2] = 255
    return {
        "gradient": sample_image(64, 48),
        "noise": Image.fromarray(noise, "RGB"),
        "tall": Image.fromarray(tall, "RGB"),
    }


def case(name, image, preprocess, shape, layout="NCHW", dtype="float32"):
    return {
        "name": name,
        "image": f"{image}.png",
        "input": {
            "name": "x",
            "dtype": dtype,
            "shape": shape,
            "layout": layout,
            "preprocess": {"type": "image", **preprocess},
        },
        "expected": f"expected/{name}.bin",
    }


CASES = [
    case(
        "imagenet_like",
        "gradient",
        {"resize": {"shorter_side": 40, "antialias": True}, "center_crop": [32, 32], **IMAGENET},
        [1, 3, 32, 32],
    ),
    case(
        "bicubic_odd_crop",
        "noise",
        {
            "resize": {"shorter_side": 31, "method": "bicubic", "antialias": True},
            "center_crop": [24, 24],
        },
        [1, 3, 24, 24],
    ),
    case(
        "upscale_bilinear_aa",
        "noise",
        {"resize": {"size": [80, 60], "antialias": True}},
        [1, 3, 80, 60],
    ),
    case("bilinear_plain_down", "gradient", {"resize": {"size": [20, 30]}}, [1, 3, 20, 30]),
    case("bilinear_plain_up", "tall", {"resize": {"size": [90, 45]}}, [1, 3, 90, 45]),
    case(
        "nearest_down", "noise", {"resize": {"size": [17, 23], "method": "nearest"}}, [1, 3, 17, 23]
    ),
    case(
        "nearest_up", "tall", {"resize": {"size": [100, 50], "method": "nearest"}}, [1, 3, 100, 50]
    ),
    case(
        "bgr_nhwc_unscaled",
        "gradient",
        {"resize": {"size": [16, 16], "antialias": True}, "color": "BGR", "scale": 1.0},
        [1, 16, 16, 3],
        layout="NHWC",
    ),
    case(
        "uint8_input",
        "noise",
        {"resize": {"size": [16, 16], "antialias": True}, "scale": 1.0},
        [1, 3, 16, 16],
        dtype="uint8",
    ),
    case(
        "float16_input",
        "gradient",
        {"resize": {"shorter_side": 40, "antialias": True}, "center_crop": [32, 32], **IMAGENET},
        [1, 3, 32, 32],
        dtype="float16",
    ),
    case(
        "crop_pads_small_image",
        "tall",
        {"resize": {"size": [20, 10], "antialias": True}, "center_crop": [24, 24]},
        [1, 3, 24, 24],
    ),
    case("same_size_is_unchanged", "gradient", {"resize": {"size": [48, 64]}}, [1, 3, 48, 64]),
]


def write(out: Path) -> None:
    (out / "expected").mkdir(parents=True, exist_ok=True)
    pictures = images()
    for name, picture in pictures.items():
        picture.save(out / f"{name}.png", optimize=False)
    for item in CASES:
        spec = InputSpec.model_validate(item["input"])
        picture = Image.open(out / item["image"]).convert("RGB")
        tensor = preprocess_image(picture, spec)
        (out / item["expected"]).write_bytes(to_le_bytes(tensor))
    (out / "cases.json").write_text(json.dumps({"cases": CASES}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    write(target)
    print(f"wrote {len(CASES)} cases to {target}")
