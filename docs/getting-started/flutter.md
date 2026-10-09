# Getting started: Flutter developers

This page runs an image classifier from the [model zoo](../zoo.md) in a new Flutter app. You do not need Python.

## 1. Add the packages

```bash
flutter pub add modelport_flutter modelport_onnx
```

`modelport_flutter` brings the core package and re-exports it, so one import is enough. Add more engines later if you need them: `modelport_executorch` or `modelport_llamacpp`.

## 2. Platform setup

=== "Android"

    Add this line to `android/app/proguard-rules.pro`, so release builds keep ONNX Runtime's Java classes:

    ```
    -keep class ai.onnxruntime.** { *; }
    ```

    Models download from the internet, so release builds need the permission in `android/app/src/main/AndroidManifest.xml`:

    ```xml
    <uses-permission android:name="android.permission.INTERNET" />
    ```

=== "macOS"

    Set the deployment target to 14.0 in Xcode, and allow outgoing connections in both `.entitlements` files:

    ```xml
    <key>com.apple.security.network.client</key>
    <true/>
    ```

## 3. Initialize once

```dart
import 'package:flutter/material.dart';
import 'package:modelport_flutter/modelport_flutter.dart';
import 'package:modelport_onnx/modelport_onnx.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ModelPortFlutter.init(adapters: [OnnxAdapter()]);
  runApp(const MyApp());
}
```

`init` picks the app's cache folder, registers the engines you pass, and reads the device's RAM so variants that need more memory are skipped.

## 4. Classify a photo

```dart
const mobilenet =
    'https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/mobilenet_v3_small.json';

final classifier = await ImageClassifier.load(
  mobilenet,
  onProgress: (p) => debugPrint('downloading ${(p.fraction * 100).round()}%'),
);
final results = await classifier.classify(jpegBytes, topK: 3);
for (final r in results) {
  debugPrint('${r.label}: ${(r.score * 100).toStringAsFixed(1)}%');
}
```

The first call downloads about 10 MB, checks every file's sha256, and caches it. Later calls load from the cache, also offline.

## 5. Check the device against Python

```dart
final report = await classifier.model.checkGolden();
debugPrint('$report'); // Golden check for onnx-fp32: PASS
```

This runs the input Python saved in the bundle and compares the output with Python's. Run it once on each new kind of device you support.

## Next steps

- [Object detection](../guides/object-detection.md) and [chat with a language model](../guides/llm.md)
- [Ship a model inside the app](../guides/hosting.md#flutter-assets) instead of downloading it
- [Try the demo app](https://github.com/ayanparvaiz/modelport/tree/main/dart/apps/demo)
