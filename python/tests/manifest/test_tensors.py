import pytest
from pydantic import ValidationError

from modelport.manifest.tensors import DType, ImagePreprocess, InputSpec, ResizeSpec, TensorSpec

IMAGENET = {
    "type": "image",
    "resize": {"shorter_side": 256},
    "center_crop": [224, 224],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def image_input(**overrides):
    data = {
        "name": "pixel_values",
        "dtype": "float32",
        "shape": [1, 3, 224, 224],
        "layout": "NCHW",
        "preprocess": IMAGENET,
    }
    data.update(overrides)
    return InputSpec.model_validate(data)


def test_imagenet_input_is_valid():
    spec = image_input()
    assert spec.dtype is DType.FLOAT32
    assert spec.preprocess is not None
    assert spec.preprocess.output_size == (224, 224)
    assert spec.preprocess.scale == pytest.approx(1 / 255)


def test_nhwc_layout_is_valid():
    spec = image_input(shape=[1, 224, 224, 3], layout="NHWC")
    assert spec.layout == "NHWC"


def test_dynamic_dimensions_are_allowed():
    assert image_input(shape=[-1, 3, -1, -1]).shape == [-1, 3, -1, -1]


@pytest.mark.parametrize("shape", [[1, 0, 2], [1, -2]])
def test_bad_dimensions_are_rejected(shape):
    with pytest.raises(ValidationError):
        TensorSpec(name="x", dtype=DType.FLOAT32, shape=shape)


def test_scalar_shape_is_allowed():
    assert TensorSpec(name="x", dtype=DType.FLOAT32, shape=[]).shape == []


def test_image_input_needs_layout():
    with pytest.raises(ValidationError, match="no layout"):
        image_input(layout=None)


def test_image_input_needs_four_dims():
    with pytest.raises(ValidationError, match="4 dimensions"):
        image_input(shape=[3, 224, 224])


def test_image_input_needs_three_channels():
    with pytest.raises(ValidationError, match="3 channels"):
        image_input(shape=[1, 1, 224, 224])


def test_crop_must_match_input_size():
    with pytest.raises(ValidationError, match="preprocessing produces 224"):
        image_input(shape=[1, 3, 256, 224])


def test_image_input_rejects_int64():
    with pytest.raises(ValidationError, match="float32, float16, or uint8"):
        image_input(dtype="int64")


def test_resize_needs_exactly_one_rule():
    with pytest.raises(ValidationError, match="exactly one"):
        ResizeSpec(shorter_side=256, size=(224, 224))
    with pytest.raises(ValidationError, match="exactly one"):
        ResizeSpec()


def test_exact_resize_sets_output_size():
    pre = ImagePreprocess(resize=ResizeSpec(size=(320, 320)))
    assert pre.output_size == (320, 320)


def test_std_must_be_positive():
    with pytest.raises(ValidationError):
        ImagePreprocess.model_validate({"resize": {"size": [1, 1]}, "std": [1, 0, 1]})


def test_bicubic_needs_antialias():
    with pytest.raises(ValidationError, match="bicubic"):
        ResizeSpec(size=(224, 224), method="bicubic", antialias=False)
    assert ResizeSpec(size=(224, 224), method="bicubic", antialias=True).antialias


def test_nearest_cannot_antialias():
    with pytest.raises(ValidationError, match="nearest"):
        ResizeSpec(size=(224, 224), method="nearest", antialias=True)
