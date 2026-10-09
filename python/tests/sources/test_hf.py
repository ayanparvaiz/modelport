import pytest

from modelport.manifest import Task
from modelport.sources import SourceError, load_source


@pytest.fixture(scope="module")
def tiny_vit(tmp_path_factory):
    """A tiny, randomly initialised ViT classifier saved like a Hub repo."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    folder = tmp_path_factory.mktemp("tiny_vit")
    config = transformers.ViTConfig(
        image_size=32,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=37,
        num_labels=3,
        id2label={0: "cat", 1: "dog", 2: "bird"},
        label2id={"cat": 0, "dog": 1, "bird": 2},
    )
    transformers.ViTForImageClassification(config).save_pretrained(folder)
    transformers.ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(folder)
    return folder


def test_local_vit(tiny_vit):
    model = load_source(f"hf:{tiny_vit}", license="MIT")
    assert model.task is Task.IMAGE_CLASSIFICATION
    assert model.id == tiny_vit.name
    assert model.labels == ["cat", "dog", "bird"]
    assert model.license == "MIT"
    spec = model.inputs[0]
    assert spec.shape == [1, 3, 32, 32]
    assert spec.preprocess is not None
    assert spec.preprocess.resize.size == (32, 32)
    assert spec.preprocess.mean == (0.5, 0.5, 0.5)
    logits = model.module(*model.example_inputs)
    assert tuple(logits.shape) == (1, 3)


def test_local_folder_needs_license(tiny_vit):
    with pytest.raises(SourceError, match="--license"):
        load_source(f"hf:{tiny_vit}")


def test_convnext_crop_pct_rule():
    transformers = pytest.importorskip("transformers")
    from modelport.sources.hf import preprocess_from_processor

    processor = transformers.ConvNextImageProcessor(size={"shortest_edge": 224}, crop_pct=0.875)
    pre, final = preprocess_from_processor(processor)
    assert pre.resize.shorter_side == 256
    assert pre.center_crop == (224, 224)
    assert pre.resize.method == "bicubic" and pre.resize.antialias
    assert final == (224, 224)


def test_not_a_model_folder(tmp_path):
    pytest.importorskip("transformers")
    with pytest.raises(SourceError, match="could not load"):
        load_source(f"hf:{tmp_path}", license="MIT")


@pytest.fixture(scope="module")
def tiny_yolos(tmp_path_factory):
    """A tiny, randomly initialised YOLOS detector saved like a Hub repo."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    folder = tmp_path_factory.mktemp("tiny_yolos")
    config = transformers.YolosConfig(
        image_size=[32, 32],
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=37,
        num_detection_tokens=5,
        num_labels=3,
        id2label={0: "cat", 1: "dog", 2: "bird"},
        label2id={"cat": 0, "dog": 1, "bird": 2},
    )
    transformers.YolosForObjectDetection(config).save_pretrained(folder)
    transformers.YolosImageProcessor(
        size={"shortest_edge": 32, "longest_edge": 64}
    ).save_pretrained(folder)
    return folder


def test_local_detector(tiny_yolos):
    model = load_source(f"hf:{tiny_yolos}", license="MIT")
    assert model.task is Task.OBJECT_DETECTION
    assert model.output_names == ["logits", "pred_boxes"]
    assert model.inputs[0].shape == [1, 3, 32, 32]
    detection = model.detection
    assert detection is not None
    assert detection.format == "detr" and detection.boxes_output == "pred_boxes"
    assert detection.background_class and detection.activation == "softmax"
    assert detection.max_detections == 5
    logits, boxes = model.module(*model.example_inputs)
    assert tuple(logits.shape) == (1, 5, 4) and tuple(boxes.shape) == (1, 5, 4)


def test_detector_exports_and_verifies(tiny_yolos, tmp_path):
    pytest.importorskip("onnxscript")
    from modelport.manifest import DetectionPostprocess
    from modelport.pipeline import export_bundle
    from modelport.verify import verify_bundle

    source = load_source(f"hf:{tiny_yolos}", license="MIT", image_size=32)
    bundle, manifest = export_bundle(source, tmp_path, ["onnx"])
    post = manifest.outputs[0].postprocess
    assert isinstance(post, DetectionPostprocess) and post.labels is not None
    assert [o.name for o in manifest.outputs] == ["logits", "pred_boxes"]
    assert all(result.passed for result in verify_bundle(bundle.root))
