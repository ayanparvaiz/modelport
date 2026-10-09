# modelport_onnx

ONNX Runtime engine for [ModelPort](https://pub.dev/packages/modelport). Runs the `onnx` variants of `modelport.json` bundles through [flutter_onnxruntime](https://pub.dev/packages/flutter_onnxruntime).

```dart
await ModelPortFlutter.init(adapters: [OnnxAdapter()]);
final classifier = await ImageClassifier.load(
  'https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/mobilenet_v3_small.json',
);
```

- **Tensor details come from the manifest.** ONNX Runtime cannot report input and output names, types, or shapes on iOS and macOS. The manifest has them, so the same code works everywhere.
- **fp32, fp16, and int8 variants.** All three MobileNetV3 variants passed their golden checks on an ARMv8.0 phone, including fp16.
- **Native memory is freed after every run.** Input and output tensors are disposed as soon as the results are copied out.

## Setup

**Android:** add this line to `android/app/proguard-rules.pro` for release builds:

```
-keep class ai.onnxruntime.** { *; }
```

**macOS:** set the deployment target to 14.0, which `flutter_onnxruntime` requires.

## Choosing a variant

```dart
await ImageClassifier.load(location, variantId: 'onnx-int8');
```

Without `variantId`, the first variant listed in the manifest that this engine can run is used.

## Speed

MobileNetV3 Small on an OPPO CPH1937 (Snapdragon 665): 77 ms per run. ExecuTorch ran the same model in 17 ms, so consider [`modelport_executorch`](https://pub.dev/packages/modelport_executorch) too. ONNX Runtime adds about 29 MB to an arm64 APK.

## Example

The [example app](https://github.com/ayanparvaiz/modelport/tree/main/dart/packages/modelport_onnx/example) classifies a photo with each variant and runs the golden check.

## License

Apache-2.0
