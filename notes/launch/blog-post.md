# Your PyTorch model in a Flutter app, checked against Python on a real phone

Running a model in a Flutter app is mostly not about the model. Engine packages like `flutter_onnxruntime`, `executorch_flutter`, and `llm_llamacpp` run models well. The work is everything around them: converting the model, guessing its input shape, rewriting image preprocessing in Dart, downloading and caching a 500 MB file, and then hoping the phone gives the same answer as Python. When it does not, nothing crashes. You just get slightly wrong results.

I built [ModelPort](https://github.com/ayanparvaiz/modelport) to take that work away.

## Three commands in Python

```bash
pip install "modelport[onnx,executorch,torchvision]"
modelport export torchvision:mobilenet_v3_small --target onnx,executorch
modelport verify dist/mobilenet_v3_small
```

`export` writes a bundle: the model files, the labels, and a `modelport.json` manifest that describes the inputs, the exact preprocessing, the outputs, and the sha256 of every file. It also saves one preprocessed input and PyTorch's output for it. `verify` runs every variant against that saved output.

## One line in Flutter

```dart
await ModelPortFlutter.init(adapters: [ExecuTorchAdapter()]);
final classifier = await ImageClassifier.load(bundleUrl);
print((await classifier.classify(jpegBytes)).first); // Samoyed (0.76)
```

The Dart side reads the manifest, downloads and verifies the files, decodes and preprocesses the image, runs the model, and turns logits into labels. Supporting a new model needs no new Dart code.

## Proving the phone matches Python

```dart
print(await classifier.model.checkGolden());
// Golden check for executorch-xnnpack-fp32: PASS
//   logits: max diff 3.72e-5 (allowed 0.001 + 0.001·|x|), top-1 same
```

This runs PyTorch's saved input on the device and compares the output. I ran it on an OPPO CPH1937, a 2019 phone with a Snapdragon 665. Every variant passed, including fp16 and int8.

Preprocessing has its own guarantee. The Dart code reproduces Pillow's resize, including its 22-bit fixed-point rounding, and twelve cross-language fixtures produce byte-identical tensors in Python and Dart.

## What I learned on a real phone

- **ExecuTorch was four times faster than ONNX Runtime** on this phone: 17 ms against 77 ms for MobileNetV3. It also adds 7.6 MB to the APK, against 29 MB.
- **The slowest step was decoding the JPEG.** The pure Dart decoder took 717 ms. Switching to the Flutter engine's native decoder brought the whole classification from 954 ms to 584 ms.
- **llama.cpp's Vulkan backend alone is 44 MB.** Engines are separate packages so apps only pay for what they use.
- **A channels_last example input silently breaks ExecuTorch exports.** The exported program rejects ordinary tensors. The CLI now always makes example inputs contiguous.

## Language models too

```bash
modelport import-gguf Qwen/Qwen2.5-0.5B-Instruct-GGUF -q q4_k_m,q8_0
```

This writes only a manifest, pinned to the repo's commit. The app downloads the GGUF on first use, checks its sha256, and picks the smaller file on phones with less RAM. On the test phone, SmolLM2 135M downloaded, loaded, and answered in about a minute.

## Try it

- Demo app APK and code: https://github.com/ayanparvaiz/modelport
- Docs: https://ayanparvaiz.github.io/modelport/
- Ready models: MobileNetV3, DeiT Tiny, YOLOS Tiny, SmolLM2, Qwen2.5

It is 0.1.0 and Apache-2.0. iOS should work through the engines' iOS support but I have not verified it yet, and I would love reports from more devices.
