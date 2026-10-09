# modelport-cli

Prepare models for Flutter apps. Part of [ModelPort](https://github.com/ayanparvaiz/modelport).

Install `modelport-cli`; the command and the Python package are both called `modelport`.

> **Status: 0.1.0, alpha.** Works end to end and is tested on a real Android phone. The spec may still change before 1.0.

The CLI converts PyTorch, Hugging Face, and torchvision models into mobile formats, checks that the converted model gives the same output as the original, and writes a `modelport.json` manifest that the ModelPort Dart packages read.

## Install

```bash
pip install "modelport-cli[onnx,torchvision]"
```

| Extra | Adds | Needed for |
|---|---|---|
| `onnx` | onnx, onnxruntime, onnxscript, torch | ONNX export, quantize, verify |
| `executorch` | executorch, torch | ExecuTorch export and verify |
| `torchvision` | torchvision, torch | `torchvision:` sources |
| `hf` | transformers, huggingface_hub, torch | `hf:` sources and `publish` |
| `gguf` | gguf | Inspecting GGUF files |

Run `modelport doctor` to see what is installed.

## Quick start

```bash
modelport export torchvision:mobilenet_v3_small --target onnx,executorch
modelport quantize dist/mobilenet_v3_small --fp16 --int8
modelport verify dist/mobilenet_v3_small
modelport publish dist/mobilenet_v3_small --hf your-name/mobilenet_v3_small
```

`verify` runs every variant on the saved golden input and compares it with PyTorch's output:

```
 variant                 ┃ output ┃ max diff ┃ cosine   ┃ top-1 ┃ result
━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━
 onnx-fp32               │ logits │ 3.34e-05 │ 1.000000 │ same  │ pass
 executorch-xnnpack-fp32 │ logits │ 3.34e-05 │ 1.000000 │ same  │ pass
 onnx-fp16               │ logits │ 1.05e-01 │ 0.999952 │ same  │ pass
 onnx-int8               │ logits │ 1.27e-01 │ 0.999933 │ same  │ pass
```

## Commands

| Command | What it does |
|---|---|
| `export SOURCE` | Convert a model and write a bundle with `modelport.json`, labels, and golden data. |
| `quantize BUNDLE` | Add `--fp16` and `--int8` ONNX variants and record how far each drifts. |
| `verify BUNDLE` | Run every variant on the golden input and compare with the expected output. |
| `pack BUNDLE` | Refresh sizes and hashes after manual edits and list unlisted files. |
| `publish BUNDLE --hf org/name` | Upload to the Hugging Face Hub. Log in first with `hf auth login`. |
| `inspect FILE` | Show inputs, outputs, and metadata of an `.onnx`, `.pte`, or `.gguf` file. |
| `validate MANIFEST...` | Check manifests against the spec. |
| `schema` | Print the manifest JSON Schema. |
| `doctor` | Check Python, optional packages, and disk space. |

## Sources

| Source | Example |
|---|---|
| torchvision classifier | `torchvision:efficientnet_b0` |
| Hugging Face image classifier | `hf:facebook/deit-tiny-patch16-224` |
| Your own model | `file:my_model.py:build` |

For `file:`, write a function that returns a `SourceModel`:

```python
import torch
from modelport.manifest import DType, ImagePreprocess, InputSpec, ResizeSpec, Task
from modelport.sources import SourceModel


def build() -> SourceModel:
    net = MyNet()
    net.load_state_dict(torch.load("weights.pt", weights_only=True))
    return SourceModel(
        module=net.eval(),
        example_inputs=(torch.zeros(1, 3, 224, 224),),
        id="my_net",
        task=Task.IMAGE_CLASSIFICATION,
        license="MIT",
        inputs=[
            InputSpec(
                name="pixel_values",
                dtype=DType.FLOAT32,
                shape=[1, 3, 224, 224],
                layout="NCHW",
                preprocess=ImagePreprocess(resize=ResizeSpec(size=(224, 224))),
            )
        ],
        output_names=["logits"],
        labels=["cat", "dog"],
    )
```

`file:` runs the code in that file, so only use files you trust.

## Notes on quantization

- `--fp16` halves the size and keeps inputs and outputs in float32.
- `--int8` quantizes only MatMul and Gemm weights. Dynamic int8 on convolutions changed MobileNetV3's top-1 class in testing, so convolution-heavy models shrink less. Transformers shrink to about a third.
- Each new variant gets a tolerance of twice its measured error, so devices are checked against a realistic bar.
