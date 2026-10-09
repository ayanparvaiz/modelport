# Getting started: Python and ML developers

This page turns a PyTorch model into a bundle that any Flutter app can load.

## Install

```bash
pip install "modelport-cli[onnx,executorch,torchvision]"
modelport doctor
```

`doctor` lists which optional packages are installed and what to add for each feature.

## Export

```bash
modelport export torchvision:mobilenet_v3_small --target onnx,executorch
```

This writes `dist/mobilenet_v3_small/`:

| File | What it is |
|---|---|
| `modelport.json` | The manifest: inputs, preprocessing, outputs, labels, files, checksums |
| `onnx-fp32/model.onnx` | ONNX variant |
| `executorch-xnnpack-fp32/model.pte` | ExecuTorch variant |
| `labels.txt` | Class names |
| `golden/*.bin` | A saved input and PyTorch's output for it |

Sources can also be Hugging Face models (`hf:facebook/deit-tiny-patch16-224`) or your own code (`file:my_model.py:build`). See the [CLI reference](../cli.md#sources).

## Make smaller variants

```bash
modelport quantize dist/mobilenet_v3_small --fp16 --int8
```

Each new variant is measured against PyTorch and gets a matching tolerance. See [Smaller models](../guides/quantization.md).

## Verify

```bash
modelport verify dist/mobilenet_v3_small
```

```
 variant                 ┃ output ┃ max diff ┃ cosine   ┃ top-1 ┃ result
━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━
 onnx-fp32               │ logits │ 3.34e-05 │ 1.000000 │ same  │ pass
 executorch-xnnpack-fp32 │ logits │ 3.34e-05 │ 1.000000 │ same  │ pass
 onnx-fp16               │ logits │ 1.05e-01 │ 0.999952 │ same  │ pass
 onnx-int8               │ logits │ 1.27e-01 │ 0.999933 │ same  │ pass
```

## Publish

=== "GitHub release"

    ```bash
    modelport publish dist/mobilenet_v3_small --github you/models --tag v1
    ```

    Needs the [gh CLI](https://cli.github.com), logged in. Apps load
    `https://github.com/you/models/releases/download/v1/mobilenet_v3_small.json`.

=== "Hugging Face"

    ```bash
    hf auth login
    modelport publish dist/mobilenet_v3_small --hf you/mobilenet_v3_small
    ```

    Apps load `hf://you/mobilenet_v3_small`.

## Use it from Dart with types

```bash
modelport gen-dart dist/mobilenet_v3_small -o lib/models --location hf://you/mobilenet_v3_small
```

This writes a class whose inputs and outputs are named fields, so a typo in a tensor name is a compile error.
