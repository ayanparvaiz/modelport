# Golden checks

Every bundle made by `modelport export` carries golden data: one preprocessed input and PyTorch's output for it, stored as raw little-endian tensors under `golden/`.

## On your computer

```bash
modelport verify dist/<bundle>
```

Runs each variant with ONNX Runtime or ExecuTorch in Python and compares the outputs with the recorded tolerance.

## On a device

```dart
final model = await ModelPort.load(location, variantId: 'onnx-int8');
final report = await model.checkGolden();
print(report);
```

```
Golden check for onnx-int8: PASS
  logits: max diff 1.27e-1 (allowed 0.26 + 0.001·|x|), top-1 same
```

This catches problems no unit test can: a wrong export setting, an engine bug on one CPU, or a variant that does not work on older phones. Run it in an integration test on each kind of device you support.

## Preprocessing parity

Golden checks compare engines, so they start from an already preprocessed tensor. Preprocessing has its own guarantee: the Dart implementation produces byte-identical tensors to the Python reference on twelve fixtures in `spec/fixtures/preprocess`, covering filtered and plain resizing, nearest, odd crops, padding, BGR, NHWC, uint8, and float16.
