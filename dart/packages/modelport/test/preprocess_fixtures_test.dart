// Runs the Python-generated preprocessing fixtures in spec/fixtures/preprocess.
// Dart must produce the same tensors as the Python reference implementation.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:modelport/modelport.dart';
import 'package:modelport/src/manifest/json_reader.dart';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';

import 'support/spec.dart';

void main() {
  final root = p.join(specDir().path, 'fixtures', 'preprocess');
  final cases =
      (jsonDecode(File(p.join(root, 'cases.json')).readAsStringSync())
              as Map<String, Object?>)['cases']!
          as List<Object?>;

  for (final item in cases.cast<Map<String, Object?>>()) {
    test(item['name'], () {
      final spec = InputSpec.fromJson(JsonReader.of(item['input'], 'input'));
      final image = decodeRgbImage(
        File(p.join(root, item['image']! as String)).readAsBytesSync(),
      );
      final expectedBytes = File(
        p.join(root, item['expected']! as String),
      ).readAsBytesSync();
      final expected = Tensor.fromBytes(
        spec.dtype,
        spec.shape,
        Uint8List.fromList(expectedBytes),
      );

      final actual = preprocessImage(image, spec);

      expect(actual.shape, spec.shape);
      expect(actual.dtype, spec.dtype);
      final a = actual.toDoubles();
      final b = expected.toDoubles();
      var worst = 0.0;
      var worstAt = -1;
      for (var i = 0; i < a.length; i++) {
        final d = (a[i] - b[i]).abs();
        if (d > worst) {
          worst = d;
          worstAt = i;
        }
      }
      expect(
        worst,
        0,
        reason:
            'max diff $worst at index $worstAt: dart ${a[worstAt < 0 ? 0 : worstAt]} vs python ${b[worstAt < 0 ? 0 : worstAt]}',
      );
      expect(
        actual.bytes,
        expectedBytes,
        reason: 'tensors must be byte-identical',
      );
    });
  }
}
