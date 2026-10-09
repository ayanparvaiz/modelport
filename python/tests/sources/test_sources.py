import pytest

from modelport.manifest import Task
from modelport.sources import SourceError, load_source, model_id


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mobilenet_v3_small", "mobilenet_v3_small"),
        ("google/vit-base-patch16-224", "vit-base-patch16-224"),
        ("Org/My Model!", "my-model"),
        ("/tmp/models/tiny_vit/", "tiny_vit"),
    ],
)
def test_model_id(text, expected):
    assert model_id(text) == expected


@pytest.mark.parametrize("spec", ["mobilenet", "torchvision:", "pip:thing"])
def test_bad_source_strings(spec):
    with pytest.raises(SourceError):
        load_source(spec)


def test_torchvision_metadata_without_download():
    pytest.importorskip("torchvision")
    from modelport.sources.torchvision import load_torchvision

    model = load_torchvision("efficientnet_b0", pretrained=False)
    assert model.task is Task.IMAGE_CLASSIFICATION
    assert model.id == "efficientnet_b0"
    assert model.labels is not None and len(model.labels) == 1000
    pre = model.inputs[0].preprocess
    assert pre is not None
    assert pre.resize.shorter_side == 256
    assert pre.resize.method == "bicubic" and pre.resize.antialias
    assert pre.center_crop == (224, 224)
    assert tuple(model.example_inputs[0].shape) == (1, 3, 224, 224)


def test_torchvision_unknown_model():
    pytest.importorskip("torchvision")
    with pytest.raises(SourceError, match="no model named"):
        load_source("torchvision:not_a_model")


def test_torchvision_rejects_detection_for_now():
    pytest.importorskip("torchvision")
    with pytest.raises(SourceError, match="Only image classification"):
        load_source("torchvision:fasterrcnn_resnet50_fpn")


def test_license_override():
    pytest.importorskip("torchvision")
    from modelport.sources.torchvision import load_torchvision

    model = load_torchvision("mobilenet_v3_small", pretrained=False)
    assert model.license == "BSD-3-Clause"
