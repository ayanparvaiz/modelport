import json

import pytest
from pydantic import ValidationError

from modelport.manifest import Manifest, Runtime, Task


def test_classifier_manifest_is_valid(classifier):
    manifest = Manifest.model_validate(classifier)
    assert manifest.task is Task.IMAGE_CLASSIFICATION
    assert manifest.schema_version == "modelport/0.1"
    assert manifest.variants[0].runtime is Runtime.ONNX


def test_llm_manifest_is_valid(llm):
    manifest = Manifest.model_validate(llm)
    assert manifest.llm is not None
    assert manifest.llm.chat_template == "from_gguf"
    assert manifest.llm.defaults.max_tokens == 512


def test_json_round_trip_uses_schema_key(classifier):
    manifest = Manifest.model_validate(classifier)
    data = json.loads(manifest.to_json())
    assert data["schema"] == "modelport/0.1"
    assert "schema_version" not in data
    assert Manifest.model_validate(data) == manifest


def test_files_lists_every_reference(classifier):
    paths = [ref.path for ref in Manifest.model_validate(classifier).files()]
    assert paths == [
        "onnx-fp32/model.onnx",
        "labels.txt",
        "golden/pixel_values.bin",
        "golden/logits.bin",
    ]


def test_unknown_schema_version_is_rejected(classifier):
    classifier["schema"] = "modelport/9.0"
    with pytest.raises(ValidationError):
        Manifest.model_validate(classifier)


def test_duplicate_variant_ids_are_rejected(classifier):
    classifier["variants"].append(classifier["variants"][0])
    with pytest.raises(ValidationError, match="duplicate variant id: onnx-fp32"):
        Manifest.model_validate(classifier)


def test_variants_are_required(classifier):
    classifier["variants"] = []
    with pytest.raises(ValidationError):
        Manifest.model_validate(classifier)


@pytest.mark.parametrize("bad", ["MobileNet", "-x", "a b", ""])
def test_bad_ids_are_rejected(classifier, bad):
    classifier["id"] = bad
    with pytest.raises(ValidationError):
        Manifest.model_validate(classifier)


@pytest.mark.parametrize("bad", ["1.0", "v1.0.0", "01.0.0"])
def test_bad_versions_are_rejected(classifier, bad):
    classifier["version"] = bad
    with pytest.raises(ValidationError):
        Manifest.model_validate(classifier)


def test_classifier_needs_classification_output(classifier):
    del classifier["outputs"][0]["postprocess"]
    with pytest.raises(ValidationError, match="'classification' postprocess"):
        Manifest.model_validate(classifier)


def test_classifier_needs_image_input(classifier):
    del classifier["inputs"][0]["preprocess"]
    with pytest.raises(ValidationError, match="image preprocessing"):
        Manifest.model_validate(classifier)


def test_tensor_task_rejects_llamacpp(classifier):
    classifier["variants"][0]["runtime"] = "llamacpp"
    with pytest.raises(ValidationError, match="only for task 'text-generation'"):
        Manifest.model_validate(classifier)


def test_text_generation_needs_llm_section(llm):
    del llm["llm"]
    with pytest.raises(ValidationError, match="needs an 'llm' section"):
        Manifest.model_validate(llm)


def test_text_generation_rejects_onnx(llm):
    llm["variants"][0]["runtime"] = "onnx"
    with pytest.raises(ValidationError, match="only runtime 'llamacpp'"):
        Manifest.model_validate(llm)


def test_golden_names_must_match_tensors(classifier):
    classifier["golden"]["inputs"] = {"image": classifier["golden"]["inputs"]["pixel_values"]}
    with pytest.raises(ValidationError, match="golden input 'image'"):
        Manifest.model_validate(classifier)


def test_from_file(tmp_path, classifier):
    path = tmp_path / "modelport.json"
    path.write_text(json.dumps(classifier), encoding="utf-8")
    assert Manifest.from_file(path).id == "mobilenet_v3_small"
