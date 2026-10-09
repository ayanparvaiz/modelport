"""Export to an ExecuTorch program (.pte) delegated to XNNPACK."""

from __future__ import annotations

from ..bundle import Bundle
from ..errors import MissingDependencyError
from ..manifest import Runtime
from ..sources import SourceModel
from . import ExportedVariant, ExportError, quiet_output

VARIANT_ID = "executorch-xnnpack-fp32"
MODEL_PATH = f"{VARIANT_ID}/model.pte"


def export_executorch(source: SourceModel, bundle: Bundle) -> ExportedVariant:
    try:
        import torch
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        from executorch.exir import to_edge_transform_and_lower
    except ImportError as error:
        raise MissingDependencyError("Exporting to ExecuTorch", "executorch") from error

    # ExecuTorch records the memory layout of example inputs. A channels_last example
    # makes the program reject ordinary NCHW tensors at run time (found in Phase 0).
    example = tuple(t.contiguous() for t in source.example_inputs)
    with quiet_output() as log, torch.no_grad():
        try:
            exported = torch.export.export(source.module, example)
            program = to_edge_transform_and_lower(
                exported, partitioner=[XnnpackPartitioner()]
            ).to_executorch()
        except Exception as error:
            raise ExportError(f"ExecuTorch export failed: {error}\n{log.getvalue()}") from error

    target = bundle.path(MODEL_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(program.buffer)
    return ExportedVariant(
        id=VARIANT_ID,
        runtime=Runtime.EXECUTORCH,
        precision="fp32",
        backend="xnnpack",
        main_file=MODEL_PATH,
    )
