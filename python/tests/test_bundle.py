import pytest
from pydantic import ValidationError

from modelport.bundle import Bundle
from modelport.manifest import Manifest


@pytest.fixture
def bundle_with_manifest(tmp_path):
    bundle = Bundle(tmp_path / "toy")
    model = bundle.write_bytes("onnx-fp32/model.onnx", b"fake model")
    labels = bundle.write_text("labels.txt", "cat\ndog\n")
    manifest = Manifest.model_validate(
        {
            "schema": "modelport/0.1",
            "id": "toy",
            "version": "1.0.0",
            "task": "image-classification",
            "license": "MIT",
            "variants": [
                {
                    "id": "onnx-fp32",
                    "runtime": "onnx",
                    "precision": "fp32",
                    "file": model.model_dump(exclude_none=True),
                }
            ],
            "inputs": [
                {
                    "name": "x",
                    "dtype": "float32",
                    "shape": [1, 3, 4, 4],
                    "layout": "NCHW",
                    "preprocess": {"resize": {"size": [4, 4]}},
                }
            ],
            "outputs": [
                {
                    "name": "y",
                    "dtype": "float32",
                    "shape": [1, 2],
                    "postprocess": {
                        "type": "classification",
                        "labels": labels.model_dump(exclude_none=True),
                    },
                }
            ],
        }
    )
    bundle.write_manifest(manifest)
    return bundle, manifest


def test_write_bytes_returns_file_ref(tmp_path):
    ref = Bundle(tmp_path).write_bytes("a/b.bin", b"abc")
    assert ref.path == "a/b.bin"
    assert ref.size == 3
    assert ref.sha256 == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_unsafe_path_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        Bundle(tmp_path).write_bytes("../escape.bin", b"x")


def test_manifest_round_trip(bundle_with_manifest):
    bundle, manifest = bundle_with_manifest
    assert bundle.read_manifest() == manifest
    assert bundle.problems(manifest) == []


def test_problems_find_changed_and_missing_files(bundle_with_manifest):
    bundle, manifest = bundle_with_manifest
    (bundle.root / "labels.txt").write_text("cat\ndig\n", encoding="utf-8")  # same size
    (bundle.root / "onnx-fp32" / "model.onnx").unlink()
    found = bundle.problems(manifest)
    assert "onnx-fp32/model.onnx: missing" in found
    assert "labels.txt: sha256 does not match" in found


def test_refresh_recomputes_hashes(bundle_with_manifest):
    bundle, manifest = bundle_with_manifest
    (bundle.root / "labels.txt").write_text("cat\ndog\nbird\n", encoding="utf-8")
    refreshed = bundle.refresh(manifest)
    assert bundle.problems(refreshed) == []
    assert refreshed.variants == manifest.variants
