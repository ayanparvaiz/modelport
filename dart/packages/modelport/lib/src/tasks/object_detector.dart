import 'dart:isolate';
import 'dart:typed_data';

import '../errors.dart';
import '../manifest/manifest.dart';
import '../manifest/postprocess.dart';
import '../manifest/tensors.dart';
import '../model_port.dart';
import '../postprocess/detection.dart';
import '../preprocess/image_preprocess.dart';
import '../preprocess/rgb_image.dart';
import '../store/model_store.dart';

/// Finds objects in images with any object-detection bundle.
///
/// ```dart
/// final detector = await ObjectDetector.load('hf://org/yolos-tiny');
/// for (final d in await detector.detect(jpegBytes)) {
///   print('${d.label} ${d.score} ${d.box}');
/// }
/// ```
class ObjectDetector {
  ObjectDetector._(
    this.model,
    this._input,
    this._output,
    this._rule,
    this._labels,
  );

  static Future<ObjectDetector> load(
    String location, {
    String? variantId,
    void Function(DownloadProgress)? onProgress,
    CancelToken? cancel,
  }) async {
    final model = await ModelPort.load(
      location,
      variantId: variantId,
      onProgress: onProgress,
      cancel: cancel,
    );
    try {
      if (model.manifest.task != Task.objectDetection) {
        throw ModelPortException(
          '${model.manifest.id} is a ${model.manifest.task.jsonName} model, not object detection',
        );
      }
      final input = model.manifest.inputs.firstWhere(
        (i) => i.preprocess != null,
      );
      final output = model.manifest.outputs.firstWhere(
        (o) => o.postprocess is DetectionPostprocess,
      );
      final labels = await model.labelsFor(output);
      return ObjectDetector._(
        model,
        input,
        output,
        output.postprocess! as DetectionPostprocess,
        labels,
      );
    } catch (_) {
      await model.close();
      rethrow;
    }
  }

  /// The underlying model, for golden checks or raw runs.
  final TensorModel model;
  final InputSpec _input;
  final OutputSpec _output;
  final DetectionPostprocess _rule;
  final List<String>? _labels;

  List<String>? get labels => _labels;

  /// Detects objects in an encoded image such as JPEG or PNG bytes.
  Future<List<Detection>> detect(
    Uint8List encodedImage, {
    double? minScore,
    int? maxDetections,
  }) async {
    final decoder = ModelPort.imageDecoder;
    final image = decoder != null
        ? await decoder(encodedImage)
        : await Isolate.run(() => decodeRgbImage(encodedImage));
    return detectImage(image, minScore: minScore, maxDetections: maxDetections);
  }

  /// Detects objects in a decoded image. Boxes are in this image's pixels.
  Future<List<Detection>> detectImage(
    RgbImage image, {
    double? minScore,
    int? maxDetections,
  }) async {
    final input = _input;
    final tensor = await Isolate.run(() => preprocessImage(image, input));
    final outputs = await model.run({input.name: tensor});
    final mapping = InputMapping.forImage(
      image.width,
      image.height,
      input.preprocess!,
    );
    return detectObjects(
      outputs,
      _output,
      _rule,
      mapping,
      labels: _labels,
      scoreThreshold: minScore,
      maxDetections: maxDetections,
    );
  }

  Future<void> close() => model.close();
}
