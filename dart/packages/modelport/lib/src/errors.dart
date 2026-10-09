/// An error that ModelPort expects and can explain.
///
/// [hint] tells the app developer how to fix it.
class ModelPortException implements Exception {
  ModelPortException(this.message, {this.hint});

  final String message;
  final String? hint;

  @override
  String toString() {
    final name = runtimeType.toString();
    return hint == null ? '$name: $message' : '$name: $message\nFix: $hint';
  }
}

/// A `modelport.json` file is missing data or breaks a spec rule.
class ManifestException extends ModelPortException {
  ManifestException(super.message, {super.hint});
}
