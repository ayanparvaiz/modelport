import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:modelport_flutter/modelport_flutter.dart';

/// Serves a tiny classification bundle as Flutter assets.
class FakeAssets extends CachingAssetBundle {
  FakeAssets(this.files);
  final Map<String, Uint8List> files;
  final List<String> loaded = [];

  @override
  Future<ByteData> load(String key) async {
    loaded.add(key);
    final bytes = files[key];
    if (bytes == null) throw StateError('no asset $key');
    return ByteData.sublistView(bytes);
  }
}

class EchoAdapter implements TensorAdapter {
  @override
  String get runtime => Runtimes.onnx;

  @override
  bool canRun(Variant variant) => true;

  @override
  Future<TensorSession> open(LoadedVariant model) async => _Session(model);
}

class _Session implements TensorSession {
  _Session(this.model);
  final LoadedVariant model;

  @override
  Future<Map<String, Tensor>> run(Map<String, Tensor> inputs) async => {
    'logits': Tensor.float32([1, 2], Float32List.fromList([0.1, 0.9])),
  };

  @override
  Future<void> close() async {}
}

Map<String, Object?> ref(Uint8List bytes, String path) => {
  'path': path,
  'size': bytes.length,
  'sha256': sha256.convert(bytes).toString(),
};

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late Directory cache;
  setUp(
    () => cache = Directory.systemTemp.createTempSync('modelport_flutter_'),
  );
  tearDown(() {
    ModelPort.reset();
    cache.deleteSync(recursive: true);
  });

  test('init configures the store and adapters', () async {
    await ModelPortFlutter.init(adapters: [EchoAdapter()], cacheDir: cache);
    expect(ModelPort.store.root.path, cache.path);
    expect(ModelPort.adapters.single, isA<EchoAdapter>());
  });

  test('device RAM is unknown on a test host', () async {
    expect(await ModelPortFlutter.deviceRamMb(), isNull);
  });

  test('asset:// bundles load through the asset bundle', () async {
    final model = Uint8List.fromList(List.generate(100, (i) => i));
    final labels = Uint8List.fromList(utf8.encode('cat\ndog\n'));
    final manifest = {
      'schema': 'modelport/0.1',
      'id': 'pets',
      'version': '1.0.0',
      'task': 'image-classification',
      'license': 'MIT',
      'variants': [
        {
          'id': 'onnx-fp32',
          'runtime': 'onnx',
          'precision': 'fp32',
          'file': ref(model, 'model.onnx'),
        },
      ],
      'inputs': [
        {
          'name': 'pixel_values',
          'dtype': 'float32',
          'shape': [1, 3, 2, 2],
          'layout': 'NCHW',
          'preprocess': {
            'resize': {
              'size': [2, 2],
            },
          },
        },
      ],
      'outputs': [
        {
          'name': 'logits',
          'dtype': 'float32',
          'shape': [1, 2],
          'postprocess': {
            'type': 'classification',
            'labels': ref(labels, 'labels.txt'),
          },
        },
      ],
    };
    final assets = FakeAssets({
      'assets/models/pets/modelport.json': Uint8List.fromList(
        utf8.encode(jsonEncode(manifest)),
      ),
      'assets/models/pets/model.onnx': model,
      'assets/models/pets/labels.txt': labels,
    });
    await ModelPortFlutter.init(
      adapters: [EchoAdapter()],
      cacheDir: cache,
      assets: assets,
    );

    final classifier = await ImageClassifier.load('asset://assets/models/pets');
    final image = RgbImage(2, 2, Uint8List(12));
    final results = await classifier.classifyImage(image);
    expect(results.first.label, 'dog');
    expect(assets.loaded, contains('assets/models/pets/model.onnx'));
  });
}
