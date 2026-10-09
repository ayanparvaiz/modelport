import pytest
from pydantic import ValidationError

from modelport.manifest.postprocess import ClassificationPostprocess, DetectionPostprocess
from modelport.manifest.tensors import OutputSpec

LABELS = {"path": "labels.txt", "size": 100, "sha256": "b" * 64}


def test_classification_output():
    out = OutputSpec.model_validate(
        {
            "name": "logits",
            "dtype": "float32",
            "shape": [1, 1000],
            "postprocess": {"type": "classification", "labels": LABELS, "top_k": 3},
        }
    )
    assert isinstance(out.postprocess, ClassificationPostprocess)
    assert out.postprocess.activation == "softmax"
    assert out.postprocess.top_k == 3


def test_detection_output_defaults():
    out = OutputSpec.model_validate(
        {
            "name": "boxes",
            "dtype": "float32",
            "shape": [1, -1, 85],
            "postprocess": {"type": "detection"},
        }
    )
    assert isinstance(out.postprocess, DetectionPostprocess)
    assert out.postprocess.box_format == "cxcywh"
    assert out.postprocess.iou_threshold == pytest.approx(0.45)


def test_unknown_postprocess_type_is_rejected():
    with pytest.raises(ValidationError):
        OutputSpec.model_validate(
            {"name": "x", "dtype": "float32", "shape": [1], "postprocess": {"type": "magic"}}
        )


@pytest.mark.parametrize("field", ["score_threshold", "iou_threshold"])
def test_thresholds_must_be_between_zero_and_one(field):
    with pytest.raises(ValidationError):
        DetectionPostprocess.model_validate({field: 1.5})


def test_top_k_must_be_positive():
    with pytest.raises(ValidationError):
        ClassificationPostprocess(top_k=0)
