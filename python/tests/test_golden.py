import numpy as np
import pytest

from modelport.bundle import Bundle
from modelport.golden import dtype_of, make_golden, read_tensor, sample_image
from modelport.manifest import DType, TensorSpec


def test_sample_image_is_stable():
    first, second = sample_image(), sample_image()
    assert first.size == (320, 240)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert len(np.unique(np.asarray(first).reshape(-1, 3), axis=0)) > 1000


@pytest.mark.parametrize(
    ("array", "dtype"),
    [
        (np.zeros(1, np.float32), DType.FLOAT32),
        (np.zeros(1, ">f4"), DType.FLOAT32),
        (np.zeros(1, np.int64), DType.INT64),
        (np.zeros(1, np.uint8), DType.UINT8),
        (np.zeros(1, bool), DType.BOOL),
    ],
)
def test_dtype_of(array, dtype):
    assert dtype_of(array) is dtype


def test_make_golden_writes_and_reads_back(tiny_source, tmp_path):
    bundle = Bundle(tmp_path)
    data = make_golden(tiny_source, bundle, sample_image())
    assert set(data.golden.inputs) == {"pixel_values"}
    assert set(data.golden.outputs) == {"logits"}
    assert data.golden.inputs["pixel_values"].size == 1 * 3 * 16 * 16 * 4
    assert data.outputs["logits"].shape == (1, 3)

    spec = TensorSpec(name="logits", dtype=DType.FLOAT32, shape=[1, 3])
    back = read_tensor(bundle, data.golden.outputs["logits"], spec)
    np.testing.assert_array_equal(back, data.outputs["logits"])
