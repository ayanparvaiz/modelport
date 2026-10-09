# modelport

Run AI models in Flutter and Dart apps from a `modelport.json` bundle: image classification, object detection, and chat with local language models, through ONNX Runtime, PyTorch ExecuTorch, or llama.cpp, with one API.

```dart
final classifier = await ImageClassifier.load(
  'https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/mobilenet_v3_small.json',
);
print((await classifier.classify(jpegBytes)).first); // Samoyed (0.76)
```

Engine packages run models well. Everything around them is usually left to you: converting the model, guessing input shapes, writing image preprocessing in Dart, downloading and caching big files, and hoping the phone gives the same answer as Python. This package does that part:

- **No model-specific Dart code.** A `modelport.json` manifest describes inputs, preprocessing, outputs, labels, every file, and its checksum. The Python CLI writes it.
- **Preprocessing matches Python byte for byte.** Resizing follows Pillow's algorithm, including its fixed-point rounding. Twelve cross-language fixtures produce identical tensors in Python and Dart.
- **The phone is checked against Python.** Bundles carry a saved input and PyTorch's output. `model.checkGolden()` runs it on the device. On a 2019 mid-range Android phone, MobileNetV3 differed from PyTorch by 3.3e-5.
- **Downloads resume and are verified.** Interrupted downloads continue with HTTP Range requests, every file is checked against its sha256 before use, and models load offline after the first download.
- **Variants fit the device.** A bundle can hold fp32, fp16, int8, and ExecuTorch variants, or several GGUF quantizations. The first one a registered engine can run, and that fits in the device's RAM, is used.
- **Errors say how to fix them.** A missing engine names the package to add; a llama.cpp backend failure on Android names the Gradle setting.

## Packages

| Package | Use it for |
|---|---|
| `modelport` | This package: manifests, downloads, preprocessing, task APIs. Pure Dart. |
| [`modelport_flutter`](https://pub.dev/packages/modelport_flutter) | Flutter setup: cache folder, assets, native image decoding, device RAM |
| [`modelport_onnx`](https://pub.dev/packages/modelport_onnx) | ONNX Runtime engine |
| [`modelport_executorch`](https://pub.dev/packages/modelport_executorch) | PyTorch ExecuTorch engine, the fastest on the test phone |
| [`modelport_llamacpp`](https://pub.dev/packages/modelport_llamacpp) | llama.cpp engine for GGUF language models |

In a Flutter app, add `modelport_flutter` and the engines you need. It re-exports this package.

## Usage

```dart
await ModelPortFlutter.init(adapters: [OnnxAdapter(), LlamaCppAdapter()]);

// Image classification
final classifier = await ImageClassifier.load(location);
final top = await classifier.classify(jpegBytes, topK: 3);

// Object detection: boxes are in pixels of your image
final detector = await ObjectDetector.load(location);
for (final d in await detector.detect(jpegBytes)) {
  print('${d.label} ${d.score} ${d.box}');
}

// Chat
final llm = await TextGenerator.load(location);
await for (final piece in llm.chat([ChatMessage.user('What is Flutter?')])) {
  stdout.write(piece);
}

// Any tensor model
final model = await ModelPort.load(location);
final outputs = await model.run({'input': Tensor.float32([1, 4], data)});

// Prove this device matches Python
print(await model.checkGolden());
```

`location` can be a GitHub release or any HTTPS URL, `hf://org/name`, `asset://assets/models/name`, or a local folder.

## Models

Ready models are in the [model zoo](https://ayanparvaiz.github.io/modelport/zoo/): MobileNetV3, DeiT Tiny, YOLOS Tiny, SmolLM2 135M, and Qwen2.5 0.5B. Make your own bundle from a PyTorch, torchvision, Hugging Face, or GGUF model with the `modelport` Python CLI:

```bash
pip install "modelport-cli[onnx,executorch,torchvision]"
modelport export torchvision:mobilenet_v3_small --target onnx,executorch
modelport verify dist/mobilenet_v3_small
```

## Platform support

| Platform | Status |
|---|---|
| Android | Tested on an OPPO CPH1937 (Android 11, Snapdragon 665) with all three engines |
| macOS | Tested with all three engines |
| iOS | Expected to work through the engines' iOS support, not verified yet |
| Windows, Linux | Core package works; engines depend on their own support |
| Web | Not supported yet (`dart:io`) |

## Documentation

[ayanparvaiz.github.io/modelport](https://ayanparvaiz.github.io/modelport/): guides, the manifest spec, measured performance, and troubleshooting.

## License

Apache-2.0
