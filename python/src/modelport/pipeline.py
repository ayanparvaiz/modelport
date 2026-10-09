"""The export pipeline: SourceModel in, verified-ready bundle folder out."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from .bundle import Bundle
from .exporters import ExportError, get_exporter
from .golden import dtype_of, make_golden, sample_image
from .manifest import (
    ClassificationPostprocess,
    DetectionPostprocess,
    Manifest,
    OutputSpec,
    Task,
    Variant,
)
from .sources import SourceModel


def export_bundle(
    source: SourceModel,
    out_dir: str | Path,
    targets: Sequence[str],
    *,
    image: Image.Image | None = None,
    version: str = "1.0.0",
    overwrite: bool = False,
) -> tuple[Bundle, Manifest]:
    """Write `<out_dir>/<id>/` with model files, labels, golden data, and modelport.json."""
    if not targets:
        raise ExportError("choose at least one export target")
    root = Path(out_dir) / source.id
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise ExportError(f"{root} already exists. Use --force to replace it.")
        shutil.rmtree(root)
    bundle = Bundle(root)
    # Fail fast on unknown targets before doing any work.
    exporters = [get_exporter(target) for target in dict.fromkeys(targets)]

    labels = None
    if source.labels:
        labels = bundle.write_text("labels.txt", "\n".join(source.labels) + "\n")
    golden = make_golden(source, bundle, image or sample_image())

    variants = []
    for exporter in exporters:
        exported = exporter(source, bundle)
        variants.append(
            Variant(
                id=exported.id,
                runtime=exported.runtime,
                backend=exported.backend,
                precision=exported.precision,
                file=bundle.add(exported.main_file),
                extra_files=[bundle.add(path) for path in exported.extra_files],
            )
        )

    outputs = []
    for index, name in enumerate(source.output_names):
        array = golden.outputs[name]
        postprocess: ClassificationPostprocess | DetectionPostprocess | None = None
        if index == 0 and source.task is Task.IMAGE_CLASSIFICATION:
            classes = int(array.shape[-1])
            _check_labels(source, classes)
            postprocess = ClassificationPostprocess(labels=labels, top_k=min(source.top_k, classes))
        elif index == 0 and source.task is Task.OBJECT_DETECTION:
            if source.detection is None:
                raise ExportError("object detection sources must describe their outputs")
            detection = source.detection
            classes = int(array.shape[-1])
            if detection.format == "rows":
                classes -= 4 + (1 if detection.has_objectness else 0)
            elif detection.background_class:
                classes -= 1
            _check_labels(source, classes)
            postprocess = detection.model_copy(update={"labels": labels})
        outputs.append(
            OutputSpec(
                name=name, dtype=dtype_of(array), shape=list(array.shape), postprocess=postprocess
            )
        )

    manifest = Manifest(
        id=source.id,
        version=version,
        task=source.task,
        license=source.license,
        name=source.name,
        description=source.description,
        source=source.source,
        variants=variants,
        inputs=source.inputs,
        outputs=outputs,
        golden=golden.golden,
    )
    bundle.write_manifest(manifest)
    return bundle, manifest


def _check_labels(source: SourceModel, classes: int) -> None:
    if source.labels and len(source.labels) != classes:
        raise ExportError(f"model has {classes} classes but {len(source.labels)} labels were given")
