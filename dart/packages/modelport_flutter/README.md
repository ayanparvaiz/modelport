# modelport_flutter

Flutter setup for [ModelPort](https://github.com/ayanparvaiz/modelport): the app's model cache folder, `asset://` bundles, and device RAM for picking model variants.

> **Status: early development.** Not published yet.

```dart
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ModelPortFlutter.init(adapters: [OnnxAdapter()]);
  runApp(const MyApp());
}
```

This package re-exports `package:modelport/modelport.dart`, so one import is enough.
