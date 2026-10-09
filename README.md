# ModelPort

**Run any AI model in your Flutter app.**
Prepare a model once with Python. Run it on-device in Flutter with one line of Dart.

> **Status: 0.1.0.** Tested on a real Android phone and macOS. iOS is not verified yet.
>
> 📖 **Docs:** https://ayanparvaiz.github.io/modelport/ · 📱 **Demo APK:** [Releases](https://github.com/ayanparvaiz/modelport/releases) · 🧠 **Ready models:** [zoo](zoo/README.md)

---

## Why

Today, putting your own PyTorch or Hugging Face model into a Flutter app means doing all of this by hand:

- converting the model to a mobile format (ONNX, ExecuTorch `.pte`, GGUF)
- guessing input names, shapes, and dtypes
- re-implementing image preprocessing in Dart, where a single wrong number gives silently wrong results
- wiring labels, tokenizers, and chat templates
- writing download, resume, checksum, and cache logic for large model files
- learning a different API for every inference engine
- hoping the phone gives the same output as Python

Engine packages such as `flutter_onnxruntime`, `executorch_flutter`, and `llm_llamacpp` run models well. ModelPort does everything around them.

## How it works

```
  Python (your computer)                    Flutter (user's phone)
 ┌──────────────────────────┐              ┌──────────────────────────────┐
 │ modelport CLI            │   bundle     │ modelport (Dart)             │
 │ export → verify → pack   │ ───────────► │ download · cache · preprocess│
 │ writes modelport.json    │  HF Hub /    │ run · postprocess            │
 └──────────────────────────┘  asset / URL └──────────────┬───────────────┘
                                                          ▼
                                     ONNX Runtime · ExecuTorch · llama.cpp
```

1. **`modelport` CLI (Python)** converts a model, checks that the converted model matches the original, and writes a `modelport.json` manifest describing inputs, outputs, preprocessing, labels, files, and checksums.
2. **`modelport` (Dart)** reads the manifest, downloads and verifies the files, prepares inputs, runs the model through an adapter, and returns typed results.
3. **Adapters** (`modelport_onnx`, `modelport_executorch`, `modelport_llamacpp`) connect existing engines to one API. Add only the engines you need.

## Usage

### Flutter

```bash
flutter pub add modelport_flutter modelport_onnx
```

```dart
await ModelPortFlutter.init(adapters: [OnnxAdapter()]);

final classifier = await ImageClassifier.load(
  'https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/mobilenet_v3_small.json',
);
print((await classifier.classify(jpegBytes)).first); // Samoyed (0.76)

// Prove this phone gives the same answer as Python.
print(await classifier.model.checkGolden()); // PASS
```

Object detection (`ObjectDetector`) and chat with local language models (`TextGenerator`, with `modelport_llamacpp`) work the same way.

### Python

```bash
pip install "modelport[onnx,executorch,torchvision]"
modelport export torchvision:mobilenet_v3_small --target onnx,executorch
modelport quantize dist/mobilenet_v3_small --fp16 --int8
modelport verify dist/mobilenet_v3_small
modelport publish dist/mobilenet_v3_small --github you/models --tag v1
```

See the [CLI README](python/README.md) for every command.

## Tested on a real phone

OPPO CPH1937 (Android 11, Snapdragon 665, 2019), profile mode:

| | Result |
|---|---|
| Golden check, every zoo model and variant | Pass (MobileNetV3 fp32 within 3.7e-5 of PyTorch) |
| MobileNetV3 run, ExecuTorch / ONNX Runtime | 17 ms / 77 ms |
| Classify a 1546x1213 JPEG end to end | 382 ms |
| SmolLM2 135M: download, verify, first words | about a minute, then 0.9 s |

More in [Performance](https://ayanparvaiz.github.io/modelport/performance/).

## Repository layout

| Path | What lives there |
|---|---|
| `spec/` | The `modelport.json` manifest schema and examples |
| `python/` | The `modelport` CLI (PyPI) |
| `dart/packages/` | Dart and Flutter packages (pub.dev) |
| `dart/apps/demo/` | Demo Flutter app |
| `zoo/` | Curated, tested model manifests |
| `docs/` | Documentation site |

## Roadmap

The full plan is in [PLAN.md](PLAN.md) (written in Bangla). In short:

- [x] Manifest spec v0.1 ([docs](docs/spec.md))
- [x] Engine spikes: ONNX Runtime, ExecuTorch, and llama.cpp all run on Android and macOS and match Python ([notes](notes/spikes.md), in Bangla)
- [x] Python CLI: export to ONNX and ExecuTorch, quantize, verify, pack, and publish
- [x] Dart core with download, cache, and preprocessing
- [x] ONNX, ExecuTorch, and llama.cpp adapters
- [x] Image classification, object detection, and text generation task APIs
- [x] Demo app, model zoo, and docs
- [ ] v0.1.0 on pub.dev and PyPI
- [ ] iOS verification

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

## License

[Apache-2.0](LICENSE)
