import dataclasses

import pytest

from modelport.exporters import ExportError
from modelport.manifest import ClassificationPostprocess, Manifest
from modelport.pipeline import export_bundle


@pytest.fixture(autouse=True)
def _needs_onnx_export():
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnx")


def test_export_writes_a_valid_bundle(tiny_source, tmp_path):
    bundle, manifest = export_bundle(tiny_source, tmp_path, ["onnx"])
    assert bundle.root == tmp_path / "tiny_cnn"
    assert Manifest.from_file(bundle.manifest_path) == manifest
    assert bundle.problems(manifest) == []
    assert [v.id for v in manifest.variants] == ["onnx-fp32"]
    out = manifest.outputs[0]
    assert out.shape == [1, 3]
    assert isinstance(out.postprocess, ClassificationPostprocess)
    assert out.postprocess.top_k == 2
    assert (bundle.root / "labels.txt").read_text(encoding="utf-8") == "red\ngreen\nblue\n"
    assert manifest.golden is not None


def test_existing_folder_needs_overwrite(tiny_source, tmp_path):
    export_bundle(tiny_source, tmp_path, ["onnx"])
    with pytest.raises(ExportError, match="--force"):
        export_bundle(tiny_source, tmp_path, ["onnx"])
    export_bundle(tiny_source, tmp_path, ["onnx"], overwrite=True)


def test_label_count_must_match_classes(tiny_source, tmp_path):
    source = dataclasses.replace(tiny_source, labels=["only", "two"])
    with pytest.raises(ExportError, match="3 classes but 2 labels"):
        export_bundle(source, tmp_path, ["onnx"])


def test_unknown_target_fails_before_writing(tiny_source, tmp_path):
    with pytest.raises(ExportError, match="unknown export target"):
        export_bundle(tiny_source, tmp_path, ["onnx", "tflite"])
    assert not (tmp_path / "tiny_cnn" / "golden").exists()
