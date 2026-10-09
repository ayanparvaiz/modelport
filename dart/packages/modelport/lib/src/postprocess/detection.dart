import 'dart:math' as math;

import '../errors.dart';
import '../manifest/postprocess.dart';
import '../manifest/tensors.dart';
import '../numeric.dart';
import '../preprocess/image_preprocess.dart';
import '../tensor.dart';
import 'classification.dart';

/// A rectangle in pixels of the original image.
class BoundingBox {
  const BoundingBox(this.left, this.top, this.right, this.bottom);

  final double left;
  final double top;
  final double right;
  final double bottom;

  double get width => right - left;
  double get height => bottom - top;
  double get area => math.max(0, width) * math.max(0, height);

  /// Intersection over union with [other].
  double iou(BoundingBox other) {
    final w = math.min(right, other.right) - math.max(left, other.left);
    final h = math.min(bottom, other.bottom) - math.max(top, other.top);
    if (w <= 0 || h <= 0) return 0;
    final overlap = w * h;
    return overlap / (area + other.area - overlap);
  }

  @override
  String toString() =>
      'BoundingBox(${left.toStringAsFixed(1)}, ${top.toStringAsFixed(1)}, '
      '${right.toStringAsFixed(1)}, ${bottom.toStringAsFixed(1)})';
}

/// One detected object.
class Detection {
  const Detection({
    required this.index,
    required this.label,
    required this.score,
    required this.box,
  });

  final int index;
  final String label;
  final double score;
  final BoundingBox box;

  @override
  String toString() => '$label (${score.toStringAsFixed(2)}) at $box';
}

/// Maps model-input pixels back to the original image by undoing the
/// manifest's center crop and resize.
class InputMapping {
  const InputMapping({
    required this.imageWidth,
    required this.imageHeight,
    required this.inputWidth,
    required this.inputHeight,
    required this.scaleX,
    required this.scaleY,
    required this.offsetX,
    required this.offsetY,
  });

  factory InputMapping.forImage(int width, int height, ImagePreprocess pre) {
    final (resizedWidth, resizedHeight) = resizedSize(
      width,
      height,
      pre.resize,
    );
    var offsetX = 0;
    var offsetY = 0;
    var inputWidth = resizedWidth;
    var inputHeight = resizedHeight;
    final crop = pre.centerCrop;
    if (crop != null) {
      // Mirrors centerCrop(): pad first if needed, then round-half-even offsets.
      final dh = crop.height > resizedHeight ? crop.height - resizedHeight : 0;
      final dw = crop.width > resizedWidth ? crop.width - resizedWidth : 0;
      final top = roundHalfEven((resizedHeight + dh - crop.height) / 2);
      final left = roundHalfEven((resizedWidth + dw - crop.width) / 2);
      offsetY = top - dh ~/ 2;
      offsetX = left - dw ~/ 2;
      inputWidth = crop.width;
      inputHeight = crop.height;
    }
    return InputMapping(
      imageWidth: width,
      imageHeight: height,
      inputWidth: inputWidth,
      inputHeight: inputHeight,
      scaleX: width / resizedWidth,
      scaleY: height / resizedHeight,
      offsetX: offsetX.toDouble(),
      offsetY: offsetY.toDouble(),
    );
  }

  final int imageWidth;
  final int imageHeight;
  final int inputWidth;
  final int inputHeight;
  final double scaleX;
  final double scaleY;
  final double offsetX;
  final double offsetY;

  /// A box in model-input pixels, as a box in image pixels clamped to the image.
  BoundingBox toImage(double left, double top, double right, double bottom) {
    double x(double v) =>
        ((v + offsetX) * scaleX).clamp(0, imageWidth.toDouble());
    double y(double v) =>
        ((v + offsetY) * scaleY).clamp(0, imageHeight.toDouble());
    return BoundingBox(x(left), y(top), x(right), y(bottom));
  }
}

/// Decodes detector outputs into boxes on the original image.
List<Detection> detectObjects(
  Map<String, Tensor> outputs,
  OutputSpec scoresOutput,
  DetectionPostprocess rule,
  InputMapping mapping, {
  List<String>? labels,
  double? scoreThreshold,
  int? maxDetections,
}) {
  final threshold = scoreThreshold ?? rule.scoreThreshold;
  final scores = outputs[scoresOutput.name];
  if (scores == null) {
    throw ModelPortException('missing output "${scoresOutput.name}"');
  }
  final shape = scores.shape;
  if (shape.length != 3 || shape.first != 1) {
    throw ModelPortException(
      'detection output must be [1, N, values], got $shape',
    );
  }
  final count = shape[1];
  final width = shape[2];
  final values = scores.toDoubles();

  List<double>? boxValues;
  if (rule.format == DetectionFormat.detr) {
    final boxes = outputs[rule.boxesOutput];
    if (boxes == null) {
      throw ModelPortException('missing output "${rule.boxesOutput}"');
    }
    if (boxes.shape.length != 3 ||
        boxes.shape[1] != count ||
        boxes.shape[2] != 4) {
      throw ModelPortException(
        'boxes output must be [1, $count, 4], got ${boxes.shape}',
      );
    }
    boxValues = boxes.toDoubles();
  }

  final candidates = <Detection>[];
  for (var i = 0; i < count; i++) {
    final row = values.sublist(i * width, (i + 1) * width);
    final List<double> box;
    List<double> classScores;
    var objectness = 1.0;
    if (rule.format == DetectionFormat.rows) {
      box = row.sublist(0, 4);
      final start = 4 + (rule.hasObjectness ? 1 : 0);
      if (rule.hasObjectness) objectness = _activate1(row[4], rule.activation);
      classScores = _activate(row.sublist(start), rule.activation);
    } else {
      box = boxValues!.sublist(i * 4, i * 4 + 4);
      classScores = _activate(row, rule.activation);
      if (rule.backgroundClass) {
        classScores = classScores.sublist(0, classScores.length - 1);
      }
    }
    if (labels != null && labels.length != classScores.length) {
      throw ModelPortException(
        'model has ${classScores.length} classes but ${labels.length} labels',
      );
    }
    var best = 0;
    for (var c = 1; c < classScores.length; c++) {
      if (classScores[c] > classScores[best]) best = c;
    }
    final score = objectness * classScores[best];
    if (score < threshold) continue;
    candidates.add(
      Detection(
        index: best,
        label: labels?[best] ?? '$best',
        score: score,
        box: _toImageBox(box, rule, mapping),
      ),
    );
  }

  candidates.sort((a, b) => b.score.compareTo(a.score));
  final kept = rule.iouThreshold >= 1
      ? candidates
      : _suppress(candidates, rule.iouThreshold);
  return kept.take(maxDetections ?? rule.maxDetections).toList();
}

double _activate1(double value, Activation activation) =>
    activation == Activation.sigmoid ? sigmoid(value) : value;

List<double> _activate(List<double> values, Activation activation) =>
    switch (activation) {
      Activation.softmax => softmax(values),
      Activation.sigmoid => [for (final v in values) sigmoid(v)],
      Activation.none => values,
    };

BoundingBox _toImageBox(
  List<double> b,
  DetectionPostprocess rule,
  InputMapping mapping,
) {
  var (x1, y1, x2, y2) = rule.boxFormat == BoxFormat.cxcywh
      ? (b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2)
      : (b[0], b[1], b[2], b[3]);
  if (rule.normalized) {
    x1 *= mapping.inputWidth;
    x2 *= mapping.inputWidth;
    y1 *= mapping.inputHeight;
    y2 *= mapping.inputHeight;
  }
  return mapping.toImage(x1, y1, x2, y2);
}

/// Greedy non-max suppression within each class. [sorted] is best first.
List<Detection> _suppress(List<Detection> sorted, double iouThreshold) {
  final kept = <Detection>[];
  for (final candidate in sorted) {
    final overlaps = kept.any(
      (k) =>
          k.index == candidate.index && k.box.iou(candidate.box) > iouThreshold,
    );
    if (!overlaps) kept.add(candidate);
  }
  return kept;
}
