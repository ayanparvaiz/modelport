# Phase 0 spikes

Spike mane chhoto, fela dewar moto experiment. Lokkho: design er age dekha je engine gulo asholei chole, ar kon jhamela ache.

Machine: MacBook (Apple Silicon), Flutter 3.44.9, Dart 3.12.2, Python 3.12 (uv).

---

## Spike A + B, Python side: MobileNetV3 Small → ONNX ar ExecuTorch

Script: `spikes/python/export_mobilenet.py`
Versions: torch 2.14.1, torchvision 0.29.1, onnx 1.23.2, onnxruntime 1.31.0, executorch 1.5.1

**Result (dog.jpg, PyTorch er sathe tulona):**

| Engine | File size | Export time | max abs diff | top-1 |
|---|---|---|---|---|
| PyTorch | | | | Samoyed |
| ONNX (dynamo) | 10.5 MB | ~2 s | 1.8e-05 | Samoyed ✅ |
| ExecuTorch XNNPACK | 10.2 MB | ~7 s | 4.7e-05 | Samoyed ✅ |

**Shikkha:**

1. **ExecuTorch e input er memory layout export er sathe atke jay.** torchvision transform `channels_last` tensor dey. Oi tensor diye export korle `.pte` sadharon NCHW input e fail kore (`XNNExecutor: Propagating input shapes failed`). Solution: export er age example input e `.contiguous()`. **CLI te eta sobsomoy korte hobe.**
2. ONNX dynamo export e `input_names` ar `output_names` kaj kore. Manifest er naam gulo sorasori ONNX e boshano jay.
3. ExecuTorch Python runtime (`executorch.runtime.Runtime`) diye `verify` command banano jabe. Method metadata theke input shape ar dtype o pawa jay.
4. Golden file size manifest er sathe mile: `pixel_values.bin` 602,112 byte, `logits.bin` 4,000 byte.
