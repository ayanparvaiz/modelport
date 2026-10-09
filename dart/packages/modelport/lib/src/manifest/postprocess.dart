import '../errors.dart';
import 'files.dart';
import 'json_reader.dart';

/// A rule for turning a raw output tensor into results.
sealed class Postprocess {
  const Postprocess();

  static Postprocess fromJson(JsonReader json) {
    final type = json.string('type');
    return switch (type) {
      'classification' => ClassificationPostprocess.fromJson(json),
      'detection' => DetectionPostprocess.fromJson(json),
      // A newer minor spec version may add types. Keep them so the manifest
      // still loads, and let task APIs reject what they cannot handle.
      _ => UnknownPostprocess(type),
    };
  }
}

enum Activation { softmax, sigmoid, none }

/// Scores over a fixed set of classes.
class ClassificationPostprocess extends Postprocess {
  const ClassificationPostprocess({
    this.activation = Activation.softmax,
    this.labels,
    this.topK = 5,
  });

  factory ClassificationPostprocess.fromJson(JsonReader json) {
    final topK = json.optionalInteger('top_k') ?? 5;
    if (topK <= 0) {
      throw ManifestException('${json.at('top_k')} must be positive');
    }
    return ClassificationPostprocess(
      activation: json.oneOf(
        'activation',
        Activation.values,
        (a) => a.name,
        fallback: Activation.softmax,
      ),
      labels: json.has('labels')
          ? FileRef.fromJson(json.object('labels'))
          : null,
      topK: topK,
    );
  }

  final Activation activation;

  /// Text file with one class name per line.
  final FileRef? labels;
  final int topK;
}

enum BoxFormat { xyxy, cxcywh }

/// How detection outputs are laid out.
enum DetectionFormat {
  /// YOLO style: one output of `[1, N, 4 + objectness + classes]`.
  rows,

  /// DETR style: class scores `[1, N, classes]` plus a separate boxes output.
  detr,
}

/// Boxes with class scores. See docs/spec.md, "Detection postprocessing".
class DetectionPostprocess extends Postprocess {
  const DetectionPostprocess({
    this.format = DetectionFormat.rows,
    this.boxesOutput,
    this.activation = Activation.none,
    this.backgroundClass = false,
    this.boxFormat = BoxFormat.cxcywh,
    this.normalized = false,
    this.hasObjectness = false,
    this.scoreThreshold = 0.25,
    this.iouThreshold = 0.45,
    this.maxDetections = 100,
    this.labels,
  });

  factory DetectionPostprocess.fromJson(JsonReader json) {
    double fraction(String key, double fallback) {
      final value = json.number(key, fallback);
      if (value < 0 || value > 1) {
        throw ManifestException('${json.at(key)} must be between 0 and 1');
      }
      return value;
    }

    final maxDetections = json.optionalInteger('max_detections') ?? 100;
    if (maxDetections <= 0) {
      throw ManifestException('${json.at('max_detections')} must be positive');
    }
    final format = json.oneOf(
      'format',
      DetectionFormat.values,
      (f) => f.name,
      fallback: DetectionFormat.rows,
    );
    final boxesOutput = json.optionalString('boxes_output');
    final hasObjectness = json.boolean('has_objectness', false);
    if (format == DetectionFormat.detr && boxesOutput == null) {
      throw ManifestException(
        '${json.path}: format "detr" needs "boxes_output"',
      );
    }
    if (format == DetectionFormat.rows && boxesOutput != null) {
      throw ManifestException(
        '${json.path}: "boxes_output" is only for format "detr"',
      );
    }
    if (format == DetectionFormat.detr && hasObjectness) {
      throw ManifestException(
        '${json.path}: "has_objectness" is only for format "rows"',
      );
    }
    return DetectionPostprocess(
      format: format,
      boxesOutput: boxesOutput,
      activation: json.oneOf(
        'activation',
        Activation.values,
        (a) => a.name,
        fallback: Activation.none,
      ),
      backgroundClass: json.boolean('background_class', false),
      boxFormat: json.oneOf(
        'box_format',
        BoxFormat.values,
        (b) => b.name,
        fallback: BoxFormat.cxcywh,
      ),
      normalized: json.boolean('normalized', false),
      hasObjectness: hasObjectness,
      scoreThreshold: fraction('score_threshold', 0.25),
      iouThreshold: fraction('iou_threshold', 0.45),
      maxDetections: maxDetections,
      labels: json.has('labels')
          ? FileRef.fromJson(json.object('labels'))
          : null,
    );
  }

  final DetectionFormat format;

  /// For [DetectionFormat.detr]: the output holding `[1, N, 4]` boxes.
  final String? boxesOutput;
  final Activation activation;

  /// Whether the last class means "no object" and is ignored.
  final bool backgroundClass;
  final BoxFormat boxFormat;

  /// true: box values are 0-1 fractions of the model input size.
  final bool normalized;
  final bool hasObjectness;
  final double scoreThreshold;

  /// Per-class non-max suppression threshold. 1 turns suppression off.
  final double iouThreshold;
  final int maxDetections;
  final FileRef? labels;
}

/// A postprocess type from a newer spec version.
class UnknownPostprocess extends Postprocess {
  const UnknownPostprocess(this.type);
  final String type;
}
