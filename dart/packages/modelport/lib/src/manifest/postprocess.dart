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

/// Boxes with class scores, shaped [1, N, 4 + objectness + classes]. Draft.
class DetectionPostprocess extends Postprocess {
  const DetectionPostprocess({
    this.boxFormat = BoxFormat.cxcywh,
    this.hasObjectness = true,
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
    return DetectionPostprocess(
      boxFormat: json.oneOf(
        'box_format',
        BoxFormat.values,
        (b) => b.name,
        fallback: BoxFormat.cxcywh,
      ),
      hasObjectness: json.boolean('has_objectness', true),
      scoreThreshold: fraction('score_threshold', 0.25),
      iouThreshold: fraction('iou_threshold', 0.45),
      maxDetections: maxDetections,
      labels: json.has('labels')
          ? FileRef.fromJson(json.object('labels'))
          : null,
    );
  }

  final BoxFormat boxFormat;
  final bool hasObjectness;
  final double scoreThreshold;
  final double iouThreshold;
  final int maxDetections;
  final FileRef? labels;
}

/// A postprocess type from a newer spec version.
class UnknownPostprocess extends Postprocess {
  const UnknownPostprocess(this.type);
  final String type;
}
