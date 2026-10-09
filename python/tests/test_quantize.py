import pytest

from modelport.manifest import Tolerance
from modelport.pipeline import export_bundle
from modelport.quantize import QuantizeError, quantize_bundle, tolerance_for
from modelport.verify import verify_bundle


@pytest.fixture
def bundle(tiny_source, tmp_path):
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    bundle, _ = export_bundle(tiny_source, tmp_path, ["onnx"])
    return bundle


@pytest.mark.parametrize(
    ("diff", "atol"),
    [(0.0, 0.001), (0.0001, 0.001), (0.0123, 0.025), (0.104, 0.21), (3.2, 6.4)],
)
def test_tolerance_for(diff, atol):
    assert tolerance_for(diff, Tolerance()).atol == pytest.approx(atol)


def test_fp16_and_int8_variants_verify(bundle):
    results = quantize_bundle(bundle.root, ["fp16", "int8"])
    assert [r.variant.id for r in results] == ["onnx-fp16", "onnx-int8"]
    manifest = bundle.read_manifest()
    assert [v.id for v in manifest.variants] == ["onnx-fp32", "onnx-fp16", "onnx-int8"]
    assert all(v.tolerance is not None for v in manifest.variants[1:])
    assert bundle.problems(manifest) == []
    assert all(r.passed for r in verify_bundle(bundle.root))


def test_running_twice_replaces_the_variant(bundle):
    quantize_bundle(bundle.root, ["fp16"])
    quantize_bundle(bundle.root, ["fp16"])
    assert [v.id for v in bundle.read_manifest().variants] == ["onnx-fp32", "onnx-fp16"]


def test_needs_an_onnx_fp32_variant(bundle):
    manifest = bundle.read_manifest()
    variant = manifest.variants[0].model_copy(update={"precision": "fp16"})
    bundle.write_manifest(manifest.model_copy(update={"variants": [variant]}))
    with pytest.raises(QuantizeError, match="no onnx fp32 variant"):
        quantize_bundle(bundle.root, ["int8"])
