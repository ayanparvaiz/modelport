import 'dart:isolate';
import 'dart:typed_data';

import '../errors.dart';
import '../manifest/manifest.dart';
import '../manifest/postprocess.dart';
import '../manifest/tensors.dart';
import '../model_port.dart';
import '../postprocess/classification.dart';
import '../preprocess/image_preprocess.dart';
import '../preprocess/rgb_image.dart';
import '../store/model_store.dart';

/// Classifies images with any image-classification bundle.
///
/// ```dart
/// final classifier = await ImageClassifier.load('hf://org/mobilenet_v3_small');
/// final results = await classifier.classify(jpegBytes);
/// print(results.first); // golden retriever (0.93)
/// ```
class ImageClassifier {
  ImageClassifier._(
    this.model,
    this._input,
    this._output,
    this._rule,
    this._labels,
  );

  static Future<ImageClassifier> load(
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
      if (model.manifest.task != Task.imageClassification) {
        throw ModelPortException(
          '${model.manifest.id} is a ${model.manifest.task.jsonName} model, not image classification',
        );
      }
      final input = model.manifest.inputs.firstWhere(
        (i) => i.preprocess != null,
      );
      final output = model.manifest.outputs.firstWhere(
        (o) => o.postprocess is ClassificationPostprocess,
      );
      final labels = await model.labelsFor(output);
      return ImageClassifier._(
        model,
        input,
        output,
        output.postprocess! as ClassificationPostprocess,
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
  final ClassificationPostprocess _rule;
  final List<String>? _labels;

  /// Class names, if the bundle has a labels file.
  List<String>? get labels => _labels;

  /// Classifies an encoded image such as JPEG or PNG bytes.
  ///
  /// Decoding uses [ModelPort.imageDecoder] when it is set, and otherwise the
  /// pure Dart decoder in a background isolate.
  Future<List<Classification>> classify(
    Uint8List encodedImage, {
    int? topK,
  }) async {
    final decoder = ModelPort.imageDecoder;
    final image = decoder != null
        ? await decoder(encodedImage)
        : await Isolate.run(() => decodeRgbImage(encodedImage));
    return classifyImage(image, topK: topK);
  }

  /// Classifies an already decoded image. Preprocessing runs in a background
  /// isolate so the UI stays smooth.
  Future<List<Classification>> classifyImage(
    RgbImage image, {
    int? topK,
  }) async {
    final input = _input;
    final tensor = await Isolate.run(() => preprocessImage(image, input));
    final outputs = await model.run({input.name: tensor});
    return topClasses(
      outputs[_output.name]!,
      _rule,
      labels: _labels,
      topK: topK,
    );
  }

  Future<void> close() => model.close();
}
