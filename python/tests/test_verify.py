import numpy as np
import pytest

from modelport.manifest import Tolerance
from modelport.pipeline import export_bundle
from modelport.verify import VerifyError, compare, verify_bundle


@pytest.fixture
def exported(tiny_source, tmp_path):
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    bundle, _ = export_bundle(tiny_source, tmp_path, ["onnx"])
    return bundle


def test_fresh_export_passes(exported):
    (result,) = verify_bundle(exported.root)
    assert result.passed
    assert result.outputs[0].max_abs_diff < 1e-4
    assert result.outputs[0].top1_match is True


def test_wrong_golden_values_fail(exported):
    manifest = exported.read_manifest()
    path = exported.root / "golden" / "logits.bin"
    values = np.frombuffer(path.read_bytes(), "<f4").copy()
    values += 0.5
    path.write_bytes(values.astype("<f4").tobytes())
    exported.write_manifest(exported.refresh(manifest))  # keep hashes consistent

    (result,) = verify_bundle(exported.root)
    assert not result.passed
    assert result.outputs[0].max_abs_diff == pytest.approx(0.5, abs=1e-3)


def test_changed_file_is_reported_before_running(exported):
    (exported.root / "labels.txt").write_text("x\ny\nz\n", encoding="utf-8")
    with pytest.raises(VerifyError, match=r"labels\.txt"):
        verify_bundle(exported.root)


def test_missing_manifest(tmp_path):
    with pytest.raises(VerifyError, match="not found"):
        verify_bundle(tmp_path)


def test_compare_uses_atol_and_rtol():
    tol = Tolerance(atol=0.01, rtol=0.1)
    expected = np.array([[1.0, 10.0]])
    assert compare("y", np.array([[1.005, 10.9]]), expected, tol, True).within_tolerance
    # 0.2 > 0.01 + 0.1 * 1.0
    assert not compare("y", np.array([[1.2, 10.0]]), expected, tol, True).within_tolerance


def test_compare_top1():
    tol = Tolerance(atol=10, rtol=0)
    check = compare("y", np.array([[0.0, 1.0]]), np.array([[1.0, 0.0]]), tol, True)
    assert check.within_tolerance and check.top1_match is False
