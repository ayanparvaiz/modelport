import '../errors.dart';
import 'json_reader.dart';

final _sha256 = RegExp(r'^[0-9a-f]{64}$');
final _windowsDrive = RegExp(r'^[A-Za-z]:');
final _whitespace = RegExp(r'\s');

/// Throws if [path] is not a relative POSIX path that stays inside the bundle.
void checkSafePath(String path, String where) {
  if (path.startsWith('/') ||
      path.contains(r'\') ||
      _windowsDrive.hasMatch(path)) {
    throw ManifestException('$where must be a relative POSIX path');
  }
  if (path
      .split('/')
      .any((part) => part.isEmpty || part == '.' || part == '..')) {
    throw ManifestException(
      "$where must not contain empty, '.' or '..' segments",
    );
  }
}

/// A file that belongs to a model: weights, labels, or golden test data.
///
/// Exactly one of [path] (inside the bundle) or [url] (an https link) is set.
class FileRef {
  const FileRef({
    this.path,
    this.url,
    required this.size,
    required this.sha256,
  });

  factory FileRef.fromJson(JsonReader json) {
    final path = json.optionalString('path');
    final url = json.optionalString('url');
    if ((path == null) == (url == null)) {
      throw ManifestException(
        '${json.path}: set exactly one of "path" or "url"',
      );
    }
    if (path != null) checkSafePath(path, json.at('path'));
    if (url != null &&
        (!url.startsWith('https://') || _whitespace.hasMatch(url))) {
      throw ManifestException('${json.at('url')} must be an https:// URL');
    }
    final size = json.integer('size');
    if (size <= 0) {
      throw ManifestException('${json.at('size')} must be positive');
    }
    final sha256 = json.string('sha256');
    if (!_sha256.hasMatch(sha256)) {
      throw ManifestException(
        '${json.at('sha256')} must be 64 lowercase hex characters',
      );
    }
    return FileRef(path: path, url: url, size: size, sha256: sha256);
  }

  /// POSIX path relative to `modelport.json`, for files inside the bundle.
  final String? path;

  /// Absolute https URL, for files hosted elsewhere.
  final String? url;

  /// Size in bytes.
  final int size;

  /// Lowercase hex SHA-256 digest.
  final String sha256;

  @override
  String toString() => 'FileRef(${path ?? url}, $size bytes)';
}
