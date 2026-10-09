**Title:** I made ModelPort: run PyTorch/Hugging Face/GGUF models in Flutter with one line, and check the phone matches Python

**Body:**

Hi r/FlutterDev! I kept rewriting the same glue for every on-device model: convert it, guess input shapes, port the image preprocessing to Dart, download and cache big files. When the preprocessing was slightly off, nothing crashed, the results were just a bit wrong.

ModelPort is a Python CLI plus Dart packages:

```bash
modelport export torchvision:mobilenet_v3_small --target onnx,executorch
```

```dart
final classifier = await ImageClassifier.load(bundleUrl);
print((await classifier.classify(jpegBytes)).first); // Samoyed (0.76)
```

What it does differently:

- A `modelport.json` manifest describes inputs, preprocessing, outputs, and checksums, so there is no model-specific Dart code
- Dart preprocessing matches Python byte for byte (Pillow's resize is reproduced exactly)
- `model.checkGolden()` runs Python's saved input on the device and compares outputs
- Downloads resume, files are sha256-checked, models work offline after the first download
- Engines are separate packages: ONNX Runtime, ExecuTorch, llama.cpp

Tested on a 2019 Snapdragon 665 phone: ExecuTorch ran MobileNetV3 in 17 ms vs 77 ms on ONNX Runtime, and SmolLM2 chat works fully offline after a 105 MB download.

Demo APK, code, and docs: https://github.com/ayanparvaiz/modelport

It's 0.1.0. I'd love feedback, especially from iOS users since I haven't verified iOS yet.
