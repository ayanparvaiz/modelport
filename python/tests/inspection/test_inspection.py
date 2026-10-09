import shutil
import sys

import pytest

from modelport.hashing import sha256_file
from modelport.inspection import (
    InspectError,
    MissingDependencyError,
    TensorInfo,
    detect_format,
    inspect_model,
)


def test_onnx_inputs_outputs_and_metadata(tiny_onnx):
    info = inspect_model(tiny_onnx)
    assert info.format == "onnx"
    assert info.inputs == [TensorInfo("a", "float32", ["batch", 3])]
    assert info.outputs == [TensorInfo("out", "float32", ["batch", 3])]
    assert info.metadata["opset"] == "17"
    assert info.metadata["producer"] == "modelport-tests"
    assert info.metadata["external_data"] == "no"
    assert info.size == tiny_onnx.stat().st_size
    assert info.sha256 == sha256_file(tiny_onnx)


def test_gguf_metadata(tiny_gguf):
    info = inspect_model(tiny_gguf)
    assert info.format == "gguf"
    assert info.inputs == [] and info.outputs == []
    assert info.metadata["architecture"] == "llama"
    assert info.metadata["name"] == "tiny-llama"
    assert info.metadata["context_length"] == "2048"
    assert info.metadata["quantization"] == "q8_0"
    assert info.metadata["tensors"] == "1"
    assert info.metadata["chat_template"] == "no"


def test_format_comes_from_magic_bytes_not_extension(tiny_gguf, tmp_path):
    renamed = tmp_path / "model.bin"
    shutil.copy(tiny_gguf, renamed)
    assert detect_format(renamed) == "gguf"


def test_executorch_program(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("executorch.runtime")
    from executorch.exir import (  # pyright: ignore[reportMissingImports]
        to_edge_transform_and_lower,
    )

    model = torch.nn.Linear(4, 2).eval()
    program = to_edge_transform_and_lower(torch.export.export(model, (torch.randn(1, 4),)))
    path = tmp_path / "linear.pte"
    path.write_bytes(program.to_executorch().buffer)

    info = inspect_model(path)
    assert info.format == "executorch"
    assert info.inputs == [TensorInfo("input_0", "float32", [1, 4])]
    assert info.outputs == [TensorInfo("output_0", "float32", [1, 2])]
    assert info.metadata["methods"] == "forward"


def test_unknown_file_is_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(InspectError, match="not a recognized model file"):
        inspect_model(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(InspectError, match="file not found"):
        inspect_model(tmp_path / "missing.onnx")


def test_missing_optional_package_explains_the_extra(tiny_onnx, monkeypatch):
    # A None entry makes `import onnx` raise ImportError, as if it were not installed.
    monkeypatch.setitem(sys.modules, "onnx", None)
    with pytest.raises(MissingDependencyError, match=r"modelport\[onnx\]"):
        inspect_model(tiny_onnx)
