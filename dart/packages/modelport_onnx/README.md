# modelport_onnx

ONNX Runtime adapter for [ModelPort](https://github.com/ayanparvaiz/modelport). Runs `onnx` variants of `modelport.json` bundles through [flutter_onnxruntime](https://pub.dev/packages/flutter_onnxruntime).

> **Status: early development.** Not published yet.

```dart
await ModelPortFlutter.init(adapters: [OnnxAdapter()]);
final classifier = await ImageClassifier.load('hf://org/mobilenet_v3_small');
print((await classifier.classify(jpegBytes)).first);
```

## Platform setup

- **macOS:** set the deployment target to 14.0 or newer.
- **Android:** add `-keep class ai.onnxruntime.** { *; }` to `android/app/proguard-rules.pro`.
