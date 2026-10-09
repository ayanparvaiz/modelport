# ModelPort

**Run any AI model in your Flutter app.**
Prepare a model once with Python. Run it on-device in Flutter with one line of Dart.

> **Status: early development.** Nothing is published yet. Follow along or star the repo to get notified when v0.1.0 ships.

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

The Python CLI works today (install from source until it is on PyPI):

```bash
cd python && uv sync --all-extras
uv run modelport export torchvision:mobilenet_v3_small --target onnx,executorch
uv run modelport quantize dist/mobilenet_v3_small --fp16 --int8
uv run modelport verify dist/mobilenet_v3_small
```

See the [CLI README](python/README.md) for every command.

The Flutter side is planned:

```dart
final classifier = await ImageClassifier.load('hf://modelport-dev/mobilenet_v3_small');
final results = await classifier.classify(imageBytes);
print(results.first); // golden retriever (0.93)

final llm = await TextGenerator.load('hf://modelport-dev/qwen2.5-0.5b-instruct');
await for (final piece in llm.chat([ChatMessage.user('What is Flutter?')])) {
  stdout.write(piece);
}
```

## Repository layout

| Path | What lives there |
|---|---|
| `spec/` | The `modelport.json` manifest schema and examples |
| `python/` | The `modelport` CLI (PyPI) |
| `dart/packages/` | Dart and Flutter packages (pub.dev) |
| `apps/demo/` | Demo Flutter app |
| `zoo/` | Curated, tested model manifests |
| `docs/` | Documentation site |

## Roadmap

The full plan is in [PLAN.md](PLAN.md) (written in Bangla). In short:

- [x] Manifest spec v0.1 ([docs](docs/spec.md))
- [x] Engine spikes: ONNX Runtime, ExecuTorch, and llama.cpp all run on Android and macOS and match Python ([notes](notes/spikes.md), in Bangla)
- [x] Python CLI: export to ONNX and ExecuTorch, quantize, verify, pack, and publish
- [ ] Dart core with download, cache, and preprocessing
- [ ] ONNX, ExecuTorch, and llama.cpp adapters
- [ ] Image classification, object detection, and text generation task APIs
- [ ] Demo app, model zoo, and docs
- [ ] v0.1.0 release

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

## License

[Apache-2.0](LICENSE)
