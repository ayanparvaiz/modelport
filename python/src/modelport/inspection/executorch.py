"""ExecuTorch (.pte) program inspection."""

from __future__ import annotations

from pathlib import Path

from . import MissingDependencyError, TensorInfo

# c10::ScalarType codes that map to manifest dtypes.
_DTYPES = {0: "uint8", 1: "int8", 3: "int32", 4: "int64", 5: "float16", 6: "float32", 11: "bool"}


def read_executorch(path: Path) -> tuple[list[TensorInfo], list[TensorInfo], dict[str, str]]:
    try:
        from executorch.runtime import Runtime  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise MissingDependencyError(".pte", "executorch") from error

    program = Runtime.get().load_program(str(path))
    methods = sorted(program.method_names)
    method = "forward" if "forward" in methods else methods[0]
    meta = program.metadata(method)

    def describe(prefix: str, info: object, index: int) -> TensorInfo:
        dtype = int(info.dtype())  # type: ignore[attr-defined]
        sizes = list(info.sizes())  # type: ignore[attr-defined]
        return TensorInfo(f"{prefix}_{index}", _DTYPES.get(dtype, f"scalar_type_{dtype}"), sizes)

    inputs = [describe("input", meta.input_tensor_meta(i), i) for i in range(meta.num_inputs())]
    outputs = [describe("output", meta.output_tensor_meta(i), i) for i in range(meta.num_outputs())]
    metadata = {"methods": ", ".join(methods), "inspected_method": method}
    return inputs, outputs, metadata
