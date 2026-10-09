import numpy as np
import pytest
from PIL import Image

from modelport.manifest import DType, ImagePreprocess, InputSpec, ResizeSpec
from modelport.preprocess import (
    bilinear_half_pixel,
    center_crop,
    preprocess_image,
    resized_size,
    to_le_bytes,
)


def random_image(width: int, height: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8), "RGB")


def image_input(pre: dict, shape: list[int], layout: str = "NCHW", dtype: str = "float32"):
    return InputSpec.model_validate(
        {"name": "x", "dtype": dtype, "shape": shape, "layout": layout, "preprocess": pre}
    )


@pytest.mark.parametrize(
    ("size", "expected"),
    [((640, 480), (341, 256)), ((480, 640), (256, 341)), ((256, 256), (256, 256))],
)
def test_shorter_side_floors_the_long_side(size, expected):
    width, height = size
    assert resized_size(width, height, ResizeSpec(shorter_side=256)) == expected


def test_exact_size_is_height_width():
    assert resized_size(640, 480, ResizeSpec(size=(100, 200))) == (200, 100)


def test_center_crop_rounds_half_to_even():
    rows = np.arange(293, dtype=np.float32)[:, None, None] * np.ones((1, 300, 3), np.float32)
    cropped = center_crop(rows, 224, 224)
    # (293 - 224) / 2 = 34.5, which rounds to 34
    assert cropped[0, 0, 0] == 34
    assert cropped.shape == (224, 224, 3)


def test_center_crop_pads_small_images():
    small = np.ones((10, 11, 3), np.float32)
    cropped = center_crop(small, 14, 14)
    assert cropped.shape == (14, 14, 3)
    assert cropped[:, :, 0].sum() == 110
    assert cropped[0].sum() == 0 and cropped[1, 1, 0] == 0 and cropped[2, 1, 0] == 1


def test_normalize_constant_color():
    image = Image.new("RGB", (8, 8), (255, 0, 51))
    pre = {"resize": {"size": [4, 4]}, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    tensor = preprocess_image(image, image_input(pre, [1, 3, 4, 4]))
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0, :, 0, 0], [1.0, -1.0, -0.6], atol=1e-6)


def test_bgr_and_nhwc():
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    pre = {"resize": {"size": [4, 4]}, "color": "BGR", "scale": 1.0}
    tensor = preprocess_image(image, image_input(pre, [1, 4, 4, 3], layout="NHWC"))
    assert tensor.shape == (1, 4, 4, 3)
    assert tensor[0, 0, 0].tolist() == [30.0, 20.0, 10.0]


def test_uint8_input_is_rounded_and_clipped():
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    pre = {"resize": {"size": [4, 4]}, "scale": 1.0}
    tensor = preprocess_image(image, image_input(pre, [1, 3, 4, 4], dtype="uint8"))
    assert tensor.dtype == np.uint8
    assert tensor[0, :, 0, 0].tolist() == [10, 20, 30]


def test_shape_mismatch_is_reported():
    image = Image.new("RGB", (8, 8))
    spec = image_input({"resize": {"size": [4, 4]}}, [1, 3, -1, -1])
    tensor = preprocess_image(image, spec)
    assert tensor.shape == (1, 3, 4, 4)
    bad = ImagePreprocess.model_validate({"resize": {"shorter_side": 6}})
    spec = InputSpec(name="x", dtype=DType.FLOAT32, shape=[1, 3, 4, 4], layout="NCHW")
    spec = spec.model_copy(update={"preprocess": bad})
    with pytest.raises(ValueError, match="dimension 2 is 6"):
        preprocess_image(image, spec)


def test_golden_bytes_are_little_endian():
    data = to_le_bytes(np.array([1.0], dtype=">f4"))
    assert data == np.array([1.0], dtype="<f4").tobytes()


def test_matches_torch_interpolate_without_antialias():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    source = rng.uniform(0, 255, (37, 53, 3)).astype(np.float32)
    for out_h, out_w in [(20, 31), (80, 90), (37, 53)]:
        ours = bilinear_half_pixel(source, out_h, out_w)
        theirs = torch.nn.functional.interpolate(
            torch.from_numpy(source).permute(2, 0, 1)[None],
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )[0].permute(1, 2, 0)
        np.testing.assert_allclose(ours, theirs.numpy(), atol=1e-3)


def test_matches_torchvision_imagenet_transform():
    pytest.importorskip("torchvision")
    from torchvision.models import MobileNet_V3_Small_Weights

    transform = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    pre = {
        "resize": {"shorter_side": transform.resize_size[0], "antialias": True},
        "center_crop": [transform.crop_size[0], transform.crop_size[0]],
        "mean": transform.mean,
        "std": transform.std,
    }
    for size in [(300, 400), (513, 257), (224, 224)]:
        image = random_image(*size)
        ours = preprocess_image(image, image_input(pre, [1, 3, 224, 224]))
        theirs = transform(image).unsqueeze(0).numpy()
        np.testing.assert_allclose(ours, theirs, atol=1e-5)
