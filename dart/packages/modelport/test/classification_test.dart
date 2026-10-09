import 'dart:typed_data';

import 'package:modelport/modelport.dart';
import 'package:test/test.dart';

Tensor logits(List<double> values) =>
    Tensor.float32([1, values.length], Float32List.fromList(values));

void main() {
  test('parseLabels handles CRLF and trailing newlines', () {
    expect(parseLabels('cat\r\ndog\nbird\n\n'), ['cat', 'dog', 'bird']);
    expect(parseLabels('a\n\nc'), ['a', '', 'c']);
  });

  test('softmax top-k with labels', () {
    final results = topClasses(
      logits([1, 3, 2]),
      const ClassificationPostprocess(topK: 2),
      labels: ['cat', 'dog', 'bird'],
    );
    expect(results.map((r) => r.label), ['dog', 'bird']);
    expect(results.first.score, closeTo(0.6652, 1e-4));
    expect(results.first.toString(), 'dog (0.67)');
  });

  test('softmax is stable for large values', () {
    final results = topClasses(
      logits([1000, 1001]),
      const ClassificationPostprocess(),
    );
    expect(results.first.score, closeTo(0.7311, 1e-4));
  });

  test('sigmoid and none activations', () {
    final sig = topClasses(
      logits([0, 2]),
      const ClassificationPostprocess(activation: Activation.sigmoid),
    );
    expect(sig.first.score, closeTo(0.8808, 1e-4));
    final raw = topClasses(
      logits([0.5, -1]),
      const ClassificationPostprocess(activation: Activation.none),
    );
    expect(raw.first.score, 0.5);
  });

  test('ties keep class order and indexes are default labels', () {
    final results = topClasses(
      logits([1, 1, 0]),
      const ClassificationPostprocess(topK: 3),
    );
    expect(results.map((r) => r.label), ['0', '1', '2']);
  });

  test('topK argument overrides the manifest', () {
    expect(
      topClasses(logits([1, 2, 3]), const ClassificationPostprocess(), topK: 1),
      hasLength(1),
    );
  });

  test('label count must match', () {
    expect(
      () => topClasses(
        logits([1, 2]),
        const ClassificationPostprocess(),
        labels: ['only'],
      ),
      throwsA(isA<ModelPortException>()),
    );
  });
}
