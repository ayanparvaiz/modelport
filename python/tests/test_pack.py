import pytest

from modelport.errors import ModelPortError
from modelport.pack import pack_bundle
from modelport.pipeline import export_bundle


@pytest.fixture
def bundle(tiny_source, tmp_path):
    pytest.importorskip("onnxscript")
    bundle, _ = export_bundle(tiny_source, tmp_path, ["onnx"])
    return bundle


def test_clean_bundle_has_nothing_to_change(bundle):
    result = pack_bundle(bundle.root)
    assert result.changed == []
    assert result.stray == []
    assert result.total_bytes > 0


def test_edited_file_gets_new_hash(bundle):
    (bundle.root / "labels.txt").write_text("r\ng\nb\n", encoding="utf-8")
    result = pack_bundle(bundle.root)
    assert result.changed == ["labels.txt"]
    assert bundle.problems(bundle.read_manifest()) == []


def test_stray_files_are_listed(bundle):
    (bundle.root / "notes.txt").write_text("hi", encoding="utf-8")
    (bundle.root / "README.md").write_text("card", encoding="utf-8")
    assert pack_bundle(bundle.root).stray == ["notes.txt"]


def test_missing_file_is_an_error(bundle):
    (bundle.root / "labels.txt").unlink()
    with pytest.raises(ModelPortError, match="missing"):
        pack_bundle(bundle.root)
