import numpy as np
import pytest

from modelport.bundle import Bundle
from modelport.exporters import ExportError, get_exporter
from modelport.manifest import Runtime


def test_onnx_export_matches_pytorch(tiny_source, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    import torch

    bundle = Bundle(tmp_path)
    exported = get_exporter("onnx")(tiny_source, bundle)
    assert exported.runtime is Runtime.ONNX
    assert exported.main_file == "onnx-fp32/model.onnx"
    assert exported.extra_files == []

    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        expected = tiny_source.module(x).numpy()
    session = ort.InferenceSession(str(bundle.path(exported.main_file)))
    assert [i.name for i in session.get_inputs()] == ["pixel_values"]
    got = session.run(["logits"], {"pixel_values": x.numpy()})[0]
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_unknown_target():
    with pytest.raises(ExportError, match="unknown export target"):
        get_exporter("tflite")
