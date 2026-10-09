# modelport_executorch

PyTorch [ExecuTorch](https://pytorch.org/executorch) engine for [ModelPort](https://pub.dev/packages/modelport). Runs the `executorch` variants (`.pte` programs) of `modelport.json` bundles through [executorch_flutter](https://pub.dev/packages/executorch_flutter).

```dart
await ModelPortFlutter.init(adapters: [ExecuTorchAdapter()]);
final detector = await ObjectDetector.load(
  'https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/yolos-tiny.json',
);
```

- **The fastest engine on the test phone.** MobileNetV3 Small ran in 17 ms on an OPPO CPH1937, against 77 ms with ONNX Runtime, and passed its golden check.
- **Small.** About 7.6 MB in an arm64 APK.
- **Inputs go in manifest order.** ExecuTorch takes inputs by position, so the adapter orders them from the manifest and checks every output's type.
- **Exports that just work.** `modelport export --target executorch` lowers to XNNPACK and makes example inputs contiguous; a `channels_last` example input otherwise produces a program that rejects ordinary inputs.

## Requirements

- Flutter 3.38 or newer.
- `cmake` on the build machine, for example `brew install cmake`. `executorch_flutter` downloads prebuilt ExecuTorch but builds a small wrapper.
- `.pte` files exported with the ExecuTorch version `executorch_flutter` ships (1.5 at the time of writing).

## Troubleshooting

**macOS build fails with deployment target 11.0.** `executorch_dart` passes 11.0 to CMake, which Xcode 27 rejects, and it can stick in the build cache. Set your app's macOS target to 14.0, delete `.dart_tool/hooks_runner/shared/executorch_dart/build/`, and build again.

## Example

The [example app](https://github.com/ayanparvaiz/modelport/tree/main/dart/packages/modelport_executorch/example) classifies a photo with MobileNetV3, detects objects with YOLOS Tiny, and runs both golden checks.

## License

Apache-2.0
