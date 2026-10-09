"""Fixtures that build tiny model files, so tests never download real models."""

from pathlib import Path

import pytest


@pytest.fixture
def tiny_onnx(tmp_path: Path) -> Path:
    """An ONNX graph that adds a constant to input 'a' of shape [batch, 3]."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    weight = helper.make_tensor("w", TensorProto.FLOAT, [3], [1.0, 2.0, 3.0])
    graph = helper.make_graph(
        [helper.make_node("Add", ["a", "w"], ["out"])],
        "tiny",
        # Older exporters also list initializers as inputs; inspection must hide them.
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, ["batch", 3]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [3]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", 3])],
        initializer=[weight],
    )
    model = helper.make_model(
        graph, producer_name="modelport-tests", opset_imports=[helper.make_opsetid("", 17)]
    )
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def tiny_gguf(tmp_path: Path) -> Path:
    """A GGUF file with llama metadata and one small tensor."""
    gguf = pytest.importorskip("gguf")
    import numpy as np

    path = tmp_path / "tiny.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_name("tiny-llama")
    writer.add_context_length(2048)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
    writer.add_tensor("weight", np.zeros(32, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


@pytest.fixture
def tiny_source():
    """A small CNN classifier described as a SourceModel."""
    torch = pytest.importorskip("torch")
    from modelport.manifest import DType, ImagePreprocess, InputSpec, ResizeSpec, Task
    from modelport.sources import SourceModel

    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 3),
    ).eval()
    return SourceModel(
        module=net,
        example_inputs=(torch.zeros(1, 3, 16, 16),),
        id="tiny_cnn",
        task=Task.IMAGE_CLASSIFICATION,
        license="MIT",
        inputs=[
            InputSpec(
                name="pixel_values",
                dtype=DType.FLOAT32,
                shape=[1, 3, 16, 16],
                layout="NCHW",
                preprocess=ImagePreprocess(
                    resize=ResizeSpec(shorter_side=20, antialias=True),
                    center_crop=(16, 16),
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            )
        ],
        output_names=["logits"],
        labels=["red", "green", "blue"],
        top_k=2,
    )
