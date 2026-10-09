"""Export to ONNX with the torch.export-based (dynamo) exporter."""

from __future__ import annotations

from ..bundle import Bundle
from ..errors import MissingDependencyError
from ..manifest import Runtime
from ..sources import SourceModel
from . import ExportedVariant, ExportError, quiet_output

VARIANT_ID = "onnx-fp32"
MODEL_PATH = f"{VARIANT_ID}/model.onnx"
# ONNX protobuf files cannot exceed 2 GB, so larger weights go to a side file.
EXTERNAL_DATA_BYTES = 1_800_000_000


def export_onnx(source: SourceModel, bundle: Bundle) -> ExportedVariant:
    try:
        import onnx
        import torch
    except ImportError as error:
        raise MissingDependencyError("Exporting to ONNX", "onnx") from error

    target = bundle.path(MODEL_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    weight_bytes = sum(p.numel() * p.element_size() for p in source.module.parameters())
    external = weight_bytes > EXTERNAL_DATA_BYTES
    # Contiguous inputs keep the exported graph in plain NCHW memory layout.
    example = tuple(t.contiguous() for t in source.example_inputs)

    with quiet_output() as log, torch.no_grad():
        try:
            torch.onnx.export(
                source.module,
                example,
                str(target),
                dynamo=True,
                input_names=[spec.name for spec in source.inputs],
                output_names=source.output_names,
                external_data=external,
            )
        except Exception as error:
            raise ExportError(f"ONNX export failed: {error}\n{log.getvalue()}") from error

    onnx.checker.check_model(str(target))
    graph = onnx.load(str(target), load_external_data=False).graph
    initializers = {init.name for init in graph.initializer}
    got_inputs = [v.name for v in graph.input if v.name not in initializers]
    got_outputs = [v.name for v in graph.output]
    if got_inputs != [s.name for s in source.inputs] or got_outputs != source.output_names:
        raise ExportError(
            f"exported names {got_inputs} -> {got_outputs} do not match "
            f"{[s.name for s in source.inputs]} -> {source.output_names}"
        )

    extra = [f"{MODEL_PATH}.data"] if bundle.path(f"{MODEL_PATH}.data").is_file() else []
    return ExportedVariant(
        id=VARIANT_ID,
        runtime=Runtime.ONNX,
        precision="fp32",
        main_file=MODEL_PATH,
        extra_files=extra,
    )
