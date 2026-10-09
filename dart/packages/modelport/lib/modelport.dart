/// Run AI models from a `modelport.json` manifest.
///
/// This is the pure Dart core of ModelPort. Engines plug in through adapter
/// packages such as `modelport_onnx`.
library;

export 'src/errors.dart';
export 'src/manifest/files.dart' show FileRef;
export 'src/manifest/manifest.dart';
export 'src/manifest/postprocess.dart';
export 'src/manifest/tensors.dart';
export 'src/model_port.dart';
export 'src/numeric.dart';
export 'src/postprocess/classification.dart';
export 'src/postprocess/detection.dart';
export 'src/preprocess/image_preprocess.dart' show preprocessImage, resizedSize;
export 'src/preprocess/rgb_image.dart';
export 'src/runtime/adapter.dart';
export 'src/runtime/select.dart';
export 'src/store/location.dart';
export 'src/store/model_store.dart';
export 'src/tasks/image_classifier.dart';
export 'src/tasks/text_generator.dart';
export 'src/tensor.dart';
