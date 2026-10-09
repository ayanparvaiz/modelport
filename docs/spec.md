# Manifest spec 0.1

A `modelport.json` manifest describes everything an app needs to download, run, and interpret a model. The Python CLI writes it and the Dart packages read it.

- JSON Schema: [`spec/manifest.schema.json`](../spec/manifest.schema.json)
- Examples: [`spec/examples/`](../spec/examples/)
- Check a manifest: `modelport validate path/to/modelport.json`

> **Status: draft.** Spec 0.1 may still change before ModelPort v0.1.0 ships. Detection output is a draft until the first detection model is exported.

## Bundle layout

A bundle is a folder with `modelport.json` at its root and the files it points to:

```
mobilenet_v3_small/
├── modelport.json
├── labels.txt
├── onnx-fp32/model.onnx
├── executorch-xnnpack-fp32/model.pte
└── golden/
    ├── pixel_values.bin
    └── logits.bin
```

The same bundle can live on Hugging Face, on any HTTPS server, or inside a Flutter app's assets. Relative paths keep working in every location.

## Top-level fields

| Field | Required | Description |
|---|---|---|
| `schema` | yes | Spec version. Always `"modelport/0.1"` for this spec. |
| `id` | yes | Lowercase id: letters, digits, `.`, `_`, `-`. Example: `mobilenet_v3_small`. |
| `version` | yes | Semantic version of this model bundle, such as `1.0.0`. |
| `task` | yes | `image-classification`, `object-detection`, or `text-generation`. |
| `license` | yes | SPDX identifier of the model weights, such as `Apache-2.0`. |
| `name` | no | Human-friendly name. |
| `description` | no | One or two sentences about the model. |
| `source` | no | Where the model came from, such as `torchvision:mobilenet_v3_small`. |
| `variants` | yes | One or more concrete forms of the model. |
| `inputs` | tensor tasks | Model inputs. |
| `outputs` | tensor tasks | Model outputs. |
| `llm` | text-generation | Settings for language models. |
| `golden` | no | Saved inputs and expected outputs for on-device checks. |

## Files

Every file is a `FileRef` with **exactly one** of `path` or `url`, plus `size` and `sha256`.

| Field | Description |
|---|---|
| `path` | POSIX path relative to `modelport.json`. No leading `/`, no `\`, no drive letters, and no empty, `.`, or `..` segments. |
| `url` | Absolute `https://` URL. Use it for files that stay in another repo, such as an official GGUF release. |
| `size` | File size in bytes. |
| `sha256` | Lowercase hex SHA-256 digest. |

Readers must reject a downloaded file whose size or digest does not match.

## Variants

| Field | Required | Description |
|---|---|---|
| `id` | yes | Unique within the manifest, such as `onnx-fp32`. |
| `runtime` | yes | `onnx`, `executorch`, or `llamacpp`. |
| `backend` | no | Engine backend, such as `xnnpack`. |
| `precision` | yes | Such as `fp32`, `fp16`, `int8`, `q4_k_m`. |
| `file` | yes | The main model file. |
| `extra_files` | no | Files the main file needs, such as ONNX external weights. |
| `min_ram_mb` | no | Readers should skip this variant on devices with less RAM. |
| `tolerance` | no | Overrides the golden tolerance for this variant. |

Readers pick the first variant whose runtime has a registered adapter, that the platform supports, and whose `min_ram_mb` fits the device.

## Inputs and outputs

| Field | Description |
|---|---|
| `name` | Tensor name. ONNX uses names directly. ExecuTorch takes inputs in the order they are listed. |
| `dtype` | `float32`, `float16`, `int64`, `int32`, `int8`, `uint8`, or `bool`. |
| `shape` | List of dimensions. `-1` means the size is only known at run time. |
| `layout` | Inputs only. `NCHW` or `NHWC`. Required when `preprocess` is set. |
| `preprocess` | Inputs only. How to build the tensor from raw data. |
| `postprocess` | Outputs only. How to turn the tensor into results. |

### Image preprocessing

Steps always run in this order:

1. **Resize.** Either `shorter_side` (keep the aspect ratio) or `size` as `[height, width]`. `method` is `bilinear`, `nearest`, or `bicubic`. `antialias` says whether to low-pass filter when shrinking.
2. **Center crop** to `center_crop` as `[height, width]`, if set.
3. **Channel order** from `color`: `RGB` or `BGR`.
4. **Scale** each 0–255 pixel by `scale`. The default is `1/255`.
5. **Normalize** each channel: `(value - mean) / std`.
6. **Arrange** in the input's `layout`.

The resize method and antialias setting are explicit because image libraries in Python and Dart do not resize the same way by default. Both sides must implement the same algorithm.

### Classification postprocessing

| Field | Default | Description |
|---|---|---|
| `activation` | `softmax` | `softmax`, `sigmoid`, or `none`. |
| `labels` | none | UTF-8 text file, one class name per line, in class-index order. |
| `top_k` | `5` | How many results to return by default. |

### Detection postprocessing (draft)

Output rows are `[box(4), objectness(1, optional), class scores(C)]`.

| Field | Default | Description |
|---|---|---|
| `box_format` | `cxcywh` | `cxcywh` or `xyxy`. |
| `has_objectness` | `true` | Whether an objectness score follows the box. |
| `score_threshold` | `0.25` | Drop boxes with a lower score. |
| `iou_threshold` | `0.45` | Non-max suppression overlap threshold. |
| `max_detections` | `100` | Keep at most this many boxes. |
| `labels` | none | Class names, one per line. |

## LLM settings

| Field | Default | Description |
|---|---|---|
| `context_length` | required | Context window to allocate on device. Smaller windows use less RAM. |
| `chat_template` | `from_gguf` | Templates come from the GGUF file. Manifests never store template code. |
| `defaults.temperature` | `0.7` | 0 to 2. |
| `defaults.top_p` | `0.9` | Greater than 0, at most 1. |
| `defaults.top_k` | none | Optional. |
| `defaults.max_tokens` | `512` | Maximum new tokens per reply. |
| `defaults.repeat_penalty` | none | Optional. |

In spec 0.1, `text-generation` supports only the `llamacpp` runtime.

## Golden tests

`golden.inputs` and `golden.outputs` map tensor names to files of raw little-endian tensor data in the tensor's dtype and shape. A float32 tensor of shape `[1, 3, 224, 224]` is `1 × 3 × 224 × 224 × 4 = 602112` bytes.

A device passes when every output element satisfies:

```
|device - expected| <= atol + rtol * |expected|
```

The default `tolerance` is `atol = 0.001` and `rtol = 0.001`.

## Task rules

| Task | Needs |
|---|---|
| `image-classification` | An input with image `preprocess` and an output with `classification` postprocess. |
| `object-detection` | An input with image `preprocess` and an output with `detection` postprocess. |
| `text-generation` | An `llm` section and only `llamacpp` variants. No `golden`. |

## Versioning

- The `schema` value names the spec version.
- Adding an optional field raises the minor version, for example `0.1` to `0.2`.
- Removing a field or changing its meaning raises the major version.
- Readers must refuse a manifest with a newer major version and should ignore unknown fields from a newer minor version.

## Security rules

- Manifests are data only. They never contain code, scripts, or commands.
- Remote files are downloaded over HTTPS only.
- Every file is checked against `size` and `sha256` before use.
- Paths cannot leave the bundle folder.
