import 'dart:convert';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

/// Builds a small, valid image-classification bundle in memory.
class FakeBundle {
  FakeBundle({int modelBytes = 50000, this.id = 'toy', this.version = '1.0.0'})
    : files = {
        'onnx-fp32/model.onnx': Uint8List.fromList(
          List.generate(modelBytes, (i) => (i * 31 + 7) % 256),
        ),
        'labels.txt': Uint8List.fromList(utf8.encode('red\ngreen\nblue\n')),
      };

  final String id;
  final String version;
  final Map<String, Uint8List> files;

  Map<String, Object?> ref(String path) => {
    'path': path,
    'size': files[path]!.length,
    'sha256': sha256.convert(files[path]!).toString(),
  };

  /// Adds golden input and expected output files.
  void addGolden(Float32List input, Float32List output) {
    files['golden/pixel_values.bin'] = input.buffer.asUint8List();
    files['golden/logits.bin'] = output.buffer.asUint8List();
  }

  bool get hasGolden => files.containsKey('golden/logits.bin');

  Map<String, Object?> manifestJson({Map<String, Object?>? modelRef}) => {
    'schema': 'modelport/0.1',
    'id': id,
    'version': version,
    'task': 'image-classification',
    'license': 'MIT',
    'variants': [
      {
        'id': 'onnx-fp32',
        'runtime': 'onnx',
        'precision': 'fp32',
        'file': modelRef ?? ref('onnx-fp32/model.onnx'),
      },
    ],
    'inputs': [
      {
        'name': 'pixel_values',
        'dtype': 'float32',
        'shape': [1, 3, 4, 4],
        'layout': 'NCHW',
        'preprocess': {
          'resize': {
            'size': [4, 4],
          },
        },
      },
    ],
    'outputs': [
      {
        'name': 'logits',
        'dtype': 'float32',
        'shape': [1, 3],
        'postprocess': {'type': 'classification', 'labels': ref('labels.txt')},
      },
    ],
    if (hasGolden)
      'golden': {
        'inputs': {'pixel_values': ref('golden/pixel_values.bin')},
        'outputs': {'logits': ref('golden/logits.bin')},
      },
  };

  String manifestText({Map<String, Object?>? modelRef}) =>
      jsonEncode(manifestJson(modelRef: modelRef));
}

/// An in-memory HTTP server that understands Range requests and can misbehave.
class FakeServer {
  final Map<String, Uint8List> files = {};
  final List<http.BaseRequest> requests = [];
  bool offline = false;
  bool ignoreRange = false;

  /// Drop the connection after this many body bytes, once.
  int? failAfterBytes;

  void serve(String baseUrl, FakeBundle bundle, {String? manifest}) {
    files['$baseUrl/modelport.json'] = Uint8List.fromList(
      utf8.encode(manifest ?? bundle.manifestText()),
    );
    bundle.files.forEach((path, bytes) => files['$baseUrl/$path'] = bytes);
  }

  List<String> get requestedPaths => [for (final r in requests) r.url.path];

  late final http.Client client = MockClient.streaming((request, _) async {
    requests.add(request);
    if (offline) {
      throw http.ClientException('network is unreachable', request.url);
    }
    final body = files[request.url.toString()];
    if (body == null) return http.StreamedResponse(const Stream.empty(), 404);
    var start = 0;
    var status = 200;
    final range = request.headers['range'];
    if (range != null && !ignoreRange) {
      start = int.parse(RegExp(r'bytes=(\d+)-').firstMatch(range)![1]!);
      status = 206;
    }
    final data = body.sublist(start);
    final failAt = failAfterBytes;
    failAfterBytes = null;
    return http.StreamedResponse(_chunks(data, failAt, request.url), status);
  });

  Stream<List<int>> _chunks(Uint8List data, int? failAt, Uri url) async* {
    const size = 4096;
    for (var i = 0; i < data.length; i += size) {
      final end = i + size < data.length ? i + size : data.length;
      if (failAt != null && end > failAt) {
        if (failAt > i) yield data.sublist(i, failAt);
        throw http.ClientException('connection reset', url);
      }
      yield data.sublist(i, end);
    }
  }
}
