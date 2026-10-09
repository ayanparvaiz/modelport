# modelport_flutter

Flutter setup for [ModelPort](https://pub.dev/packages/modelport). One call configures the model cache, `asset://` bundles, native image decoding, and device RAM.

```dart
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ModelPortFlutter.init(adapters: [OnnxAdapter()]);
  runApp(const MyApp());
}
```

What `init` does:

- **Cache folder.** Models are kept in the app support folder under `modelport/`, verified once, and reused offline.
- **Native image decoding.** JPEG and PNG are decoded by the Flutter engine instead of pure Dart. On the test phone a 1546x1213 JPEG went from 717 ms to 302 ms. EXIF orientation is applied, like Pillow does in Python. Pass `nativeImageDecoding: false` to opt out.
- **Assets.** `asset://assets/models/<name>` bundles load from your app's assets and are copied into the cache.
- **Device RAM.** Read with `device_info_plus`, so variants that need more memory than the device has are skipped.

This package re-exports `package:modelport/modelport.dart`, so one import is enough.

## Engines

Add the engines you need and pass them to `init`:

| Package | Engine |
|---|---|
| [`modelport_onnx`](https://pub.dev/packages/modelport_onnx) | ONNX Runtime |
| [`modelport_executorch`](https://pub.dev/packages/modelport_executorch) | PyTorch ExecuTorch |
| [`modelport_llamacpp`](https://pub.dev/packages/modelport_llamacpp) | llama.cpp for GGUF language models |

See the [documentation](https://ayanparvaiz.github.io/modelport/) and the [demo app](https://github.com/ayanparvaiz/modelport/tree/main/dart/apps/demo).

## License

Apache-2.0
