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
export 'src/tensor.dart';
