import 'dart:math' as math;

import '../errors.dart';
import '../manifest/postprocess.dart';
import '../tensor.dart';

/// One class and its score.
class Classification {
  const Classification({
    required this.index,
    required this.label,
    required this.score,
  });

  final int index;
  final String label;

  /// Probability after the manifest's activation, or the raw score for `none`.
  final double score;

  @override
  String toString() => '$label (${score.toStringAsFixed(2)})';
}

/// Reads a labels file: one class name per line, in class-index order.
List<String> parseLabels(String text) {
  final lines = text
      .split('\n')
      .map((l) => l.endsWith('\r') ? l.substring(0, l.length - 1) : l)
      .toList();
  while (lines.isNotEmpty && lines.last.isEmpty) {
    lines.removeLast();
  }
  return lines;
}

List<double> softmax(List<double> values) {
  final top = values.reduce(math.max);
  final exps = [for (final v in values) math.exp(v - top)];
  final sum = exps.fold<double>(0, (a, b) => a + b);
  return [for (final e in exps) e / sum];
}

double sigmoid(double x) =>
    x >= 0 ? 1 / (1 + math.exp(-x)) : math.exp(x) / (1 + math.exp(x));

/// Turns a classifier output of shape `[1, classes]` (or `[classes]`) into the best classes.
List<Classification> topClasses(
  Tensor scores,
  ClassificationPostprocess rule, {
  List<String>? labels,
  int? topK,
}) {
  if (scores.shape.length > 2 ||
      (scores.shape.length == 2 && scores.shape.first != 1)) {
    throw ModelPortException(
      'expected scores of shape [1, classes], got ${scores.shape}',
    );
  }
  final raw = scores.toDoubles();
  if (labels != null && labels.length != raw.length) {
    throw ModelPortException(
      'model has ${raw.length} classes but ${labels.length} labels',
    );
  }
  final values = switch (rule.activation) {
    Activation.softmax => softmax(raw),
    Activation.sigmoid => [for (final v in raw) sigmoid(v)],
    Activation.none => raw,
  };
  final order = List<int>.generate(values.length, (i) => i)
    ..sort((a, b) {
      final byScore = values[b].compareTo(values[a]);
      return byScore != 0 ? byScore : a.compareTo(b);
    });
  final count = math.min(topK ?? rule.topK, values.length);
  return [
    for (final i in order.take(count))
      Classification(index: i, label: labels?[i] ?? '$i', score: values[i]),
  ];
}
