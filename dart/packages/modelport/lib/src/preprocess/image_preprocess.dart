import 'dart:typed_data';

import '../errors.dart';
import '../manifest/tensors.dart';
import '../numeric.dart';
import '../tensor.dart';
import 'resample.dart';
import 'rgb_image.dart';

/// Output (width, height) for a resize rule. Matches torchvision's rounding.
(int, int) resizedSize(int width, int height, ResizeSpec spec) {
  final size = spec.size;
  if (size != null) return (size.width, size.height);
  final short = spec.shorterSide!;
  if (width <= height) return (short, (short * height / width).floor());
  return ((short * width / height).floor(), short);
}

/// Resizes to height x width x 3 values in 0-255.
Float32List resizeImage(RgbImage image, ResizeSpec spec) {
  final (width, height) = resizedSize(image.width, image.height, spec);
  if (spec.antialias || spec.method == ResizeMethod.nearest) {
    final resized = spec.method == ResizeMethod.nearest
        ? resampleNearest(image, width, height)
        : resamplePillow(
            image,
            width,
            height,
            bicubic: spec.method == ResizeMethod.bicubic,
          );
    return Float32List(resized.pixels.length)
      ..setAll(0, resized.pixels.map((v) => v.toDouble()));
  }
  return resampleBilinearPlain(image, width, height);
}

/// Center crop with zero padding when the image is smaller than the crop.
({Float32List values, int width, int height}) centerCrop(
  Float32List values,
  int width,
  int height,
  ImageSize crop,
) {
  var data = values;
  var w = width;
  var h = height;
  if (crop.height > h || crop.width > w) {
    final dh = crop.height > h ? crop.height - h : 0;
    final dw = crop.width > w ? crop.width - w : 0;
    final top = dh ~/ 2;
    final left = dw ~/ 2;
    final padded = Float32List((h + dh) * (w + dw) * 3);
    for (var y = 0; y < h; y++) {
      padded.setRange(
        ((y + top) * (w + dw) + left) * 3,
        ((y + top) * (w + dw) + left + w) * 3,
        data,
        y * w * 3,
      );
    }
    data = padded;
    h += dh;
    w += dw;
  }
  final top = roundHalfEven((h - crop.height) / 2);
  final left = roundHalfEven((w - crop.width) / 2);
  final out = Float32List(crop.height * crop.width * 3);
  for (var y = 0; y < crop.height; y++) {
    out.setRange(
      y * crop.width * 3,
      (y + 1) * crop.width * 3,
      data,
      ((y + top) * w + left) * 3,
    );
  }
  return (values: out, width: crop.width, height: crop.height);
}

/// Runs the manifest's preprocessing on one image and returns the input tensor,
/// including the batch dimension. Matches `modelport.preprocess` in Python.
Tensor preprocessImage(RgbImage image, InputSpec spec) {
  final pre = spec.preprocess;
  final layout = spec.layout;
  if (pre == null || layout == null) {
    throw ModelPortException('input "${spec.name}" has no image preprocessing');
  }

  var values = resizeImage(image, pre.resize);
  var (width, height) = resizedSize(image.width, image.height, pre.resize);
  final crop = pre.centerCrop;
  if (crop != null) {
    final cropped = centerCrop(values, width, height, crop);
    (values, width, height) = (cropped.values, cropped.width, cropped.height);
  }

  final shape = layout == TensorLayout.nchw
      ? [1, 3, height, width]
      : [1, height, width, 3];
  for (var axis = 0; axis < 4; axis++) {
    if (spec.shape[axis] != -1 && spec.shape[axis] != shape[axis]) {
      throw ModelPortException(
        'input "${spec.name}" dimension $axis is ${shape[axis]}, manifest says ${spec.shape[axis]}',
      );
    }
  }

  final count = width * height * 3;
  final normalized = Float64List(count);
  final plane = width * height;
  for (var i = 0; i < plane; i++) {
    for (var c = 0; c < 3; c++) {
      final source = pre.color == ColorOrder.bgr ? 2 - c : c;
      final value = values[i * 3 + source] * pre.scale;
      final index = layout == TensorLayout.nchw ? c * plane + i : i * 3 + c;
      normalized[index] = (value - pre.mean[c]) / pre.std[c];
    }
  }

  return switch (spec.dtype) {
    DType.float32 => Tensor.float32(shape, Float32List.fromList(normalized)),
    DType.float16 => Tensor.float16(
      shape,
      Uint16List.fromList([for (final v in normalized) doubleToHalfBits(v)]),
    ),
    DType.uint8 => Tensor.uint8(
      shape,
      Uint8List.fromList([
        for (final v in normalized) roundHalfEven(v).clamp(0, 255),
      ]),
    ),
    _ => throw ModelPortException(
      'image input "${spec.name}" cannot be ${spec.dtype.jsonName}',
    ),
  };
}
