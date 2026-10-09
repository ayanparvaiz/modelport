# ModelPort

**Run any AI model in your Flutter app.** Prepare a model once with Python. Run it on the device with one line of Dart.

```dart
final classifier = await ImageClassifier.load(
  'https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/mobilenet_v3_small.json',
);
final results = await classifier.classify(jpegBytes);
print(results.first); // Samoyed (0.76)
```

## Why ModelPort

Engine packages such as `flutter_onnxruntime`, `executorch_flutter`, and `llm_llamacpp` run models well. Everything around them is left to you: converting the model, guessing input shapes, writing image preprocessing in Dart, downloading and caching big files, and hoping the phone gives the same answer as Python. ModelPort does that part.

- **One manifest describes the model.** `modelport.json` lists inputs, outputs, preprocessing, labels, files, sizes, and checksums. The Dart side needs no model-specific code.
- **Preprocessing matches Python byte for byte.** Resize, crop, and normalize follow an exact spec. Twelve cross-language fixtures produce identical tensors in Python and Dart.
- **Golden checks prove the phone matches Python.** Each bundle carries a saved input and Python's output. `model.checkGolden()` runs it on the device. On a 2019 mid-range Android phone the difference was 3.3e-5.
- **Downloads resume and are verified.** Files are checked against their sha256 before use, interrupted downloads continue where they stopped, and models load offline after the first download.
- **One API, several engines.** ONNX Runtime, ExecuTorch, and llama.cpp plug in as adapters. Add only the engines you use, because each one adds to app size.
- **Tested on a real phone.** Every model in the zoo passed its golden check on an OPPO CPH1937 (Android 11, Snapdragon 665).

## How it works

```
  Python (your computer)                    Flutter (the user's phone)
 ┌──────────────────────────┐              ┌──────────────────────────────┐
 │ modelport CLI            │   bundle     │ modelport (Dart)             │
 │ export → verify → pack   │ ───────────► │ download · cache · preprocess│
 │ writes modelport.json    │ GitHub, HF,  │ run · postprocess            │
 └──────────────────────────┘ assets, URL  └──────────────┬───────────────┘
                                                          ▼
                                     ONNX Runtime · ExecuTorch · llama.cpp
```

## Packages

| Package | What it does |
|---|---|
| [`modelport`](https://pub.dev/packages/modelport) | Pure Dart core: manifests, downloads, preprocessing, task APIs |
| [`modelport_flutter`](https://pub.dev/packages/modelport_flutter) | Flutter setup: cache folder, assets, native image decoding, device RAM |
| [`modelport_onnx`](https://pub.dev/packages/modelport_onnx) | ONNX Runtime adapter |
| [`modelport_executorch`](https://pub.dev/packages/modelport_executorch) | PyTorch ExecuTorch adapter |
| [`modelport_llamacpp`](https://pub.dev/packages/modelport_llamacpp) | llama.cpp adapter for GGUF language models |
| `modelport` on PyPI | The CLI: export, quantize, verify, pack, publish, gen-dart |

## Where to start

- **You build Flutter apps:** [Getting started for Flutter developers](getting-started/flutter.md) uses a ready model from the [zoo](zoo.md). No Python needed.
- **You have your own model:** [Getting started for Python and ML developers](getting-started/python.md).
