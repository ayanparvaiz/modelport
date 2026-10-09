"""ONNX model inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import MissingDependencyError, TensorInfo

# onnx.TensorProto element types that map to manifest dtypes.
_DTYPES = {1: "float32", 10: "float16", 7: "int64", 6: "int32", 3: "int8", 2: "uint8", 9: "bool"}


def read_onnx(path: Path) -> tuple[list[TensorInfo], list[TensorInfo], dict[str, str]]:
    try:
        import onnx
    except ImportError as error:
        raise MissingDependencyError("Inspecting .onnx files", "onnx") from error

    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    initializers = {init.name for init in graph.initializer}

    def describe(value: Any) -> TensorInfo:
        tensor_type = value.type.tensor_type
        elem = tensor_type.elem_type
        dtype = _DTYPES.get(elem, onnx.TensorProto.DataType.Name(elem).lower())
        shape: list[int | str] = [
            dim.dim_value if dim.HasField("dim_value") else (dim.dim_param or "?")
            for dim in tensor_type.shape.dim
        ]
        return TensorInfo(name=value.name, dtype=dtype, shape=shape)

    inputs = [describe(v) for v in graph.input if v.name not in initializers]
    outputs = [describe(v) for v in graph.output]
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), None)
    external = any(init.data_location == onnx.TensorProto.EXTERNAL for init in graph.initializer)
    metadata = {
        "ir_version": str(model.ir_version),
        "opset": str(opset) if opset is not None else "unknown",
        "producer": f"{model.producer_name} {model.producer_version}".strip() or "unknown",
        "external_data": "yes" if external else "no",
    }
    return inputs, outputs, metadata
