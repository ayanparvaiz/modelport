# modelport_executorch

PyTorch [ExecuTorch](https://pytorch.org/executorch) adapter for [ModelPort](https://github.com/ayanparvaiz/modelport). Runs `executorch` variants (`.pte` programs) of `modelport.json` bundles through [executorch_flutter](https://pub.dev/packages/executorch_flutter).

> **Status: early development.** Not published yet.

```dart
await ModelPortFlutter.init(adapters: [ExecuTorchAdapter()]);
final classifier = await ImageClassifier.load('hf://org/mobilenet_v3_small');
```

## Requirements

- Flutter 3.38 or newer (native assets).
- `cmake` on the build machine, for example `brew install cmake`.
- Export `.pte` files with the ExecuTorch version that `executorch_flutter` ships (1.5 at the time of writing). `modelport export --target executorch` does this.
