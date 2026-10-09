# Concepts

## Bundle

A folder with `modelport.json` and the files it lists. The same bundle works from a GitHub release, the Hugging Face Hub, any HTTPS server, or a Flutter app's assets.

## Manifest

`modelport.json` describes everything an app needs: the task, inputs and their preprocessing, outputs and their postprocessing, labels, every file with its size and sha256, and golden data. The Dart side reads it, so supporting a new model never needs new Dart code. See the [spec](spec.md).

## Variant

One concrete form of the model, such as `onnx-fp32`, `onnx-int8`, `executorch-xnnpack-fp32`, or `gguf-q4_k_m`. Variants are listed in order of preference. An app gets the first one that a registered adapter can run and that fits in the device's RAM. Pass `variantId` to choose one yourself.

## Adapter

A small package that connects one inference engine to ModelPort: `modelport_onnx`, `modelport_executorch`, or `modelport_llamacpp`. Engines are big, so apps add only the ones they use.

## Task APIs

| Task | Dart API |
|---|---|
| `image-classification` | `ImageClassifier` |
| `object-detection` | `ObjectDetector` |
| `text-generation` | `TextGenerator` |
| any tensor model | `ModelPort.load()` and `TensorModel.run()` |

## Golden data

A saved input and Python's output for it. `modelport verify` checks every variant against it on your computer, and `TensorModel.checkGolden()` does the same on a device. Quantized variants carry a looser tolerance that `modelport quantize` measured.

## Cache

Downloaded files live in the app's support folder under `modelport/`. A file is used only after its size and sha256 match. Verified files are marked, so big models are hashed once. The last manifest of each location is kept, so models load offline. `ModelPort.store.cacheSize()` and `delete()` manage space.
