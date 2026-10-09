"""Spike A + B (Python side): export MobileNetV3 Small to ONNX and ExecuTorch.

Checks that both exported models give the same output as PyTorch on a real photo,
then writes golden files for the Flutter side of the spike.

Run: uv run python export_mobilenet.py
"""

from __future__ import annotations

import hashlib
import io
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

OUT = Path(__file__).parent / "out" / "mobilenet"
DOG_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(name: str, ref: np.ndarray, got: np.ndarray, labels: list[str]) -> None:
    diff = float(np.max(np.abs(ref - got)))
    cos = float(np.dot(ref.ravel(), got.ravel()) / (np.linalg.norm(ref) * np.linalg.norm(got)))
    same_top1 = int(ref.argmax()) == int(got.argmax())
    print(f"  {name:<11} max|diff|={diff:.2e}  cosine={cos:.8f}  top1 same={same_top1}")
    print(f"  {'':<11} top1 = {labels[int(got.argmax())]}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights).eval()
    labels: list[str] = weights.meta["categories"]

    image = Image.open(io.BytesIO(urllib.request.urlopen(DOG_URL).read())).convert("RGB")
    # torchvision transforms return a channels_last tensor. torch.export records that
    # memory layout, and the ExecuTorch program then rejects ordinary NCHW inputs.
    pixel_values = weights.transforms()(image).unsqueeze(0).contiguous()
    print(f"input {tuple(pixel_values.shape)} {pixel_values.dtype}")

    with torch.no_grad():
        ref = model(pixel_values).numpy()
    print(f"PyTorch top1 = {labels[int(ref.argmax())]}")

    # ONNX
    onnx_path = OUT / "model.onnx"
    t = time.perf_counter()
    torch.onnx.export(
        model,
        (pixel_values,),
        str(onnx_path),
        dynamo=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        external_data=False,
    )
    print(f"ONNX export {time.perf_counter() - t:.1f}s, {onnx_path.stat().st_size:,} bytes")

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print("  inputs ", [(i.name, i.shape, i.type) for i in session.get_inputs()])
    print("  outputs", [(o.name, o.shape, o.type) for o in session.get_outputs()])
    onnx_out = session.run(["logits"], {"pixel_values": pixel_values.numpy()})[0]

    # ExecuTorch with XNNPACK
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from executorch.runtime import Runtime

    pte_path = OUT / "model.pte"
    t = time.perf_counter()
    exported = torch.export.export(model, (pixel_values,))
    program = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    pte_path.write_bytes(program.to_executorch().buffer)
    print(f"ExecuTorch export {time.perf_counter() - t:.1f}s, {pte_path.stat().st_size:,} bytes")

    method = Runtime.get().load_program(str(pte_path)).load_method("forward")
    et_out = method.execute([pixel_values])[0].numpy()

    print("compare against PyTorch:")
    report("ONNX", ref, onnx_out, labels)
    report("ExecuTorch", ref, et_out, labels)

    # Golden files for the Flutter side
    (OUT / "pixel_values.bin").write_bytes(pixel_values.numpy().astype("<f4").tobytes())
    (OUT / "logits.bin").write_bytes(ref.astype("<f4").tobytes())
    (OUT / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    for name in ["model.onnx", "model.pte", "pixel_values.bin", "logits.bin", "labels.txt"]:
        path = OUT / name
        print(f"{name:<17} {path.stat().st_size:>10,}  {sha256(path)}")


if __name__ == "__main__":
    main()
