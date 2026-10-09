import 'dart:convert';
import 'dart:io';

import 'package:modelport/modelport.dart';
import 'package:modelport/src/manifest/json_reader.dart';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';

import 'support/spec.dart';

Map<String, Object?> example(String name) =>
    jsonDecode(readSpecFile('examples/$name')) as Map<String, Object?>;

void main() {
  group('spec examples shared with Python', () {
    final examples = Directory(p.join(specDir().path, 'examples'))
        .listSync()
        .whereType<File>()
        .where((f) => f.path.endsWith('.json'))
        .toList();

    test(
      'there are at least three',
      () => expect(examples, hasLength(greaterThanOrEqualTo(3))),
    );

    for (final file in examples) {
      test('${p.basename(file.path)} parses', () {
        expect(() => Manifest.parse(file.readAsStringSync()), returnsNormally);
      });
    }

    // Python rejects unknown fields to catch typos. Dart ignores them so that
    // manifests from a newer minor spec version still load.
    const invalid = {
      'missing-sha256.json': 'sha256',
      'path-traversal.json': "'..' segments",
      'http-url.json': 'https://',
      'duplicate-variant-ids.json': 'duplicate variant id',
      'missing-schema.json': 'schema is required',
      'text-generation-without-llm.json': '"llm" section',
    };
    for (final MapEntry(key: name, value: message) in invalid.entries) {
      test('invalid/$name is rejected', () {
        expect(
          () => Manifest.parse(readSpecFile('examples/invalid/$name')),
          throwsA(
            isA<ManifestException>().having(
              (e) => e.message,
              'message',
              contains(message),
            ),
          ),
        );
      });
    }

    test(
      'invalid/unknown-field.json loads, because Dart ignores unknown fields',
      () {
        expect(
          () => Manifest.parse(
            readSpecFile('examples/invalid/unknown-field.json'),
          ),
          returnsNormally,
        );
      },
    );
  });

  group('image classification example', () {
    final manifest = Manifest.fromJson(example('image-classification.json'));

    test('top-level fields', () {
      expect(manifest.id, 'mobilenet_v3_small');
      expect(manifest.task, Task.imageClassification);
      expect(manifest.schema.toString(), 'modelport/0.1');
      expect(manifest.variants.map((v) => v.runtime), [
        Runtimes.onnx,
        Runtimes.executorch,
      ]);
    });

    test('input and preprocessing', () {
      final input = manifest.inputs.single;
      expect(input.shape, [1, 3, 224, 224]);
      expect(input.layout, TensorLayout.nchw);
      expect(input.elementCount, 150528);
      final pre = input.preprocess!;
      expect(pre.resize.shorterSide, 256);
      expect(pre.resize.antialias, isTrue);
      expect(pre.centerCrop, (height: 224, width: 224));
      expect(pre.mean, [0.485, 0.456, 0.406]);
      expect(pre.scale, closeTo(1 / 255, 1e-12));
    });

    test('output and files', () {
      final post =
          manifest.outputs.single.postprocess as ClassificationPostprocess;
      expect(post.topK, 5);
      expect(post.labels!.path, 'labels.txt');
      expect(manifest.files.map((f) => f.path), [
        'onnx-fp32/model.onnx',
        'executorch-xnnpack-fp32/model.pte',
        'labels.txt',
        'golden/pixel_values.bin',
        'golden/logits.bin',
      ]);
      expect(manifest.golden!.tolerance.atol, 0.001);
    });
  });

  group('text generation example', () {
    final manifest = Manifest.fromJson(example('text-generation.json'));

    test('llm settings and url files', () {
      expect(manifest.llm!.contextLength, 4096);
      expect(manifest.llm!.defaults.maxTokens, 512);
      expect(
        manifest.variants.first.file.url,
        startsWith('https://huggingface.co/'),
      );
      expect(manifest.variants.first.minRamMb, 1024);
    });
  });

  group('schema versions', () {
    Map<String, Object?> withSchema(String schema) => {
      ...example('text-generation.json'),
      'schema': schema,
    };

    test('a newer major version is refused with a hint', () {
      expect(
        () => Manifest.fromJson(withSchema('modelport/1.0')),
        throwsA(
          isA<ManifestException>().having(
            (e) => e.hint,
            'hint',
            contains('Update'),
          ),
        ),
      );
    });

    test('a newer minor version with new fields still loads', () {
      final data = {...withSchema('modelport/0.9'), 'brand_new_field': true};
      final manifest = Manifest.fromJson(data);
      expect(manifest.schema.isFullySupported, isFalse);
    });

    test('garbage is refused', () {
      expect(
        () => Manifest.fromJson(withSchema('v1')),
        throwsA(isA<ManifestException>()),
      );
    });

    test('not JSON is refused', () {
      expect(() => Manifest.parse('{nope'), throwsA(isA<ManifestException>()));
    });
  });

  group('file paths', () {
    // The same cases as the Python tests and the JSON Schema pattern.
    const cases = {
      'model.onnx': true,
      'onnx-fp32/model.onnx': true,
      '.hidden/model.onnx': true,
      'a/.b': true,
      '/etc/passwd': false,
      '../secret': false,
      'a/../b': false,
      './model.onnx': false,
      'a//b': false,
      'a/': false,
      r'a\b': false,
      'C:/model.onnx': false,
      '..': false,
    };
    for (final MapEntry(key: path, value: ok) in cases.entries) {
      test('"$path" is ${ok ? 'allowed' : 'rejected'}', () {
        void parse() => FileRef.fromJson(
          JsonReader.of({'path': path, 'size': 1, 'sha256': 'a' * 64}, 'file'),
        );
        ok
            ? expect(parse, returnsNormally)
            : expect(parse, throwsA(isA<ManifestException>()));
      });
    }
  });
}
