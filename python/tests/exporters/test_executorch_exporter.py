import numpy as np
import pytest

from modelport.bundle import Bundle
from modelport.exporters import get_exporter
from modelport.manifest import Runtime
from modelport.pipeline import export_bundle
from modelport.verify import verify_bundle


@pytest.fixture(autouse=True)
def _needs_executorch():
    pytest.importorskip("torch")
    pytest.importorskip("executorch.runtime")


def test_executorch_export_matches_pytorch(tiny_source, tmp_path):
    import torch
    from executorch.runtime import Runtime as ExecuTorchRuntime

    bundle = Bundle(tmp_path)
    exported = get_exporter("executorch")(tiny_source, bundle)
    assert exported.runtime is Runtime.EXECUTORCH
    assert exported.backend == "xnnpack"

    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        expected = tiny_source.module(x).numpy()
    program = ExecuTorchRuntime.get().load_program(str(bundle.path(exported.main_file)))
    method = program.load_method("forward")
    assert method is not None
    got = method.execute([x])[0].numpy()
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_channels_last_example_still_accepts_nchw(tiny_source, tmp_path):
    """The Phase 0 bug: a channels_last example input broke ordinary NCHW inputs."""
    import torch

    channels_last = torch.zeros(1, 3, 16, 16).to(memory_format=torch.channels_last)
    tiny_source.example_inputs = (channels_last,)
    bundle, _ = export_bundle(tiny_source, tmp_path, ["executorch"])
    (result,) = verify_bundle(bundle.root)
    assert result.passed


def test_both_targets_in_one_bundle(tiny_source, tmp_path):
    pytest.importorskip("onnxscript")
    bundle, manifest = export_bundle(tiny_source, tmp_path, ["onnx", "executorch"])
    assert [v.id for v in manifest.variants] == ["onnx-fp32", "executorch-xnnpack-fp32"]
    assert all(result.passed for result in verify_bundle(bundle.root))
