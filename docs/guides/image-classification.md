# Image classification

```dart
final classifier = await ImageClassifier.load(location);
final results = await classifier.classify(jpegBytes, topK: 5);
```

- `classify` takes encoded bytes (JPEG, PNG, WebP). With `modelport_flutter`, decoding uses the Flutter engine's native codecs and applies EXIF orientation.
- `classifyImage` takes an already decoded `RgbImage`.
- Decoding and preprocessing run in a background isolate, so the UI stays smooth.
- Each `Classification` has `index`, `label`, and `score`. Scores are probabilities when the manifest's activation is `softmax` or `sigmoid`.

## Choosing an engine

```dart
await ImageClassifier.load(location, variantId: 'executorch-xnnpack-fp32');
```

On the test phone, ExecuTorch ran MobileNetV3 in 17 ms and ONNX Runtime in 77 ms. See [Performance](../performance.md).

## Bring your own model

```bash
modelport export hf:facebook/deit-tiny-patch16-224 --target onnx,executorch
```

Any Hugging Face image classifier whose image processor uses a fixed size, or a shortest edge plus a center crop, is supported, including the ConvNext `crop_pct` rule.
