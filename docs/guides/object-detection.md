# Object detection

```dart
final detector = await ObjectDetector.load(
  'https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/yolos-tiny.json',
);
for (final d in await detector.detect(jpegBytes, minScore: 0.5)) {
  print('${d.label} ${d.score} ${d.box}');
}
```

Boxes are in pixels of the image you passed in. ModelPort undoes the resize and center crop for you. To draw them, put the image and a `CustomPaint` in a `SizedBox` of the image's size inside a `FittedBox`; the demo app's detect page does this.

## Exporting a detector

```bash
modelport export hf:hustvl/yolos-tiny --image-size 320 --target onnx,executorch
```

DETR-family models on Hugging Face (DETR, YOLOS, RT-DETR, D-FINE) export directly. They accept any input size, so pick one with `--image-size`. Smaller is much faster on phones: YOLOS-tiny took 3.6 s at 512x512 on the test phone.

## Output layouts

The manifest describes either YOLO-style rows or DETR-style score and box outputs. See [Detection postprocessing](../spec.md#detection-postprocessing).
