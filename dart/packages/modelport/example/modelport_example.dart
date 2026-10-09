// A command-line tour of the core package: read a model bundle, inspect its
// variants, and build the input tensor for an image exactly as Python would.
//
// Running a model also needs an engine adapter, such as modelport_llamacpp in
// Dart, or modelport_onnx and modelport_executorch in Flutter apps.
//
//   dart run example/modelport_example.dart path/to/photo.jpg
import 'dart:io';

import 'package:modelport/modelport.dart';

const mobilenet =
    'https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/mobilenet_v3_small.json';

Future<void> main(List<String> args) async {
  final store = ModelStore(
    root: Directory('${Directory.systemTemp.path}/modelport_example'),
  );
  final manifest = await store.manifest(ModelLocation.parse(mobilenet));

  print(
    '${manifest.name ?? manifest.id} ${manifest.version} (${manifest.task.jsonName})',
  );
  for (final variant in manifest.variants) {
    print(
      '  ${variant.id}: ${variant.runtime} ${variant.precision}, '
      '${(variant.file.size / 1e6).toStringAsFixed(1)} MB',
    );
  }

  if (args.isEmpty) return;
  final input = manifest.inputs.single;
  final image = decodeRgbImage(await File(args.first).readAsBytes());
  final tensor = preprocessImage(image, input);
  print('Input "${input.name}": ${tensor.dtype.jsonName} ${tensor.shape}');
  store.close();
}
