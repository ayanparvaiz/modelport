import 'dart:convert';

import 'package:crypto/crypto.dart';
import 'package:path/path.dart' as p;

import '../errors.dart';
import '../manifest/files.dart';

const _manifestName = 'modelport.json';

/// Where a model bundle lives, and how to find each of its files.
///
/// Supported forms:
/// * `hf://org/repo`, `hf://org/repo@revision`, `hf://org/repo/sub/folder`
/// * `https://host/path/modelport.json` or `https://host/path/` (the folder)
/// * `file:///abs/path` or a plain absolute path to a folder or manifest
/// * `asset://assets/models/name` for Flutter assets (needs modelport_flutter)
class ModelLocation {
  ModelLocation._(this.original, this.manifestUri);

  factory ModelLocation.parse(String text) {
    final input = text.trim();
    if (input.startsWith('hf://')) {
      return ModelLocation._(input, _huggingFace(input));
    }
    if (input.startsWith('/')) {
      return ModelLocation._(input, _withManifest(Uri.file(input)));
    }
    final uri = Uri.tryParse(input);
    if (uri == null || !uri.hasScheme) {
      throw ModelPortException(
        '"$text" is not a model location',
        hint:
            'Use hf://org/repo, an https:// URL, an absolute path, or asset://path.',
      );
    }
    switch (uri.scheme) {
      case 'https':
      case 'file':
        return ModelLocation._(input, _withManifest(uri));
      case 'http':
        if (uri.host == 'localhost' || uri.host == '127.0.0.1') {
          return ModelLocation._(input, _withManifest(uri));
        }
        throw ModelPortException(
          'refusing to download a model over plain http',
          hint: 'Use https://. Plain http is only allowed for localhost.',
        );
      case 'asset':
        final path = [
          uri.host,
          ...uri.pathSegments,
        ].where((s) => s.isNotEmpty).join('/');
        return ModelLocation._(
          input,
          _withManifest(Uri(scheme: 'asset', path: path)),
        );
      default:
        throw ModelPortException(
          'unsupported location scheme "${uri.scheme}" in "$text"',
        );
    }
  }

  /// What the app passed in.
  final String original;

  /// Absolute URI of `modelport.json`.
  final Uri manifestUri;

  bool get isRemote =>
      manifestUri.scheme == 'https' || manifestUri.scheme == 'http';
  bool get isFile => manifestUri.scheme == 'file';
  bool get isAsset => manifestUri.scheme == 'asset';

  /// Where to fetch [file] from: its own URL, or its path next to the manifest.
  Uri uriFor(FileRef file) {
    final url = file.url;
    if (url != null) return Uri.parse(url);
    final relative = Uri(path: file.path);
    if (isAsset) {
      final folder = p.posix.dirname(manifestUri.path);
      return Uri(scheme: 'asset', path: p.posix.join(folder, relative.path));
    }
    return manifestUri.resolveUri(relative);
  }

  /// A short, stable key for caching things about this location.
  String get cacheKey => sha256
      .convert(utf8.encode(manifestUri.toString()))
      .toString()
      .substring(0, 16);

  @override
  String toString() => original;
}

Uri _withManifest(Uri uri) {
  if (uri.path.endsWith('.json')) return uri;
  final path = uri.path.endsWith('/') ? uri.path : '${uri.path}/';
  return uri.replace(path: '$path$_manifestName');
}

Uri _huggingFace(String text) {
  final parts = text
      .substring('hf://'.length)
      .split('/')
      .where((s) => s.isNotEmpty)
      .toList();
  if (parts.length < 2) {
    throw ModelPortException(
      '"$text" needs an org and a repo, like hf://org/name',
    );
  }
  final org = parts[0];
  var repo = parts[1];
  var revision = 'main';
  final at = repo.indexOf('@');
  if (at >= 0) {
    revision = repo.substring(at + 1);
    repo = repo.substring(0, at);
    if (revision.isEmpty || repo.isEmpty) {
      throw ModelPortException('"$text" has an empty repo or revision');
    }
  }
  final subfolder = parts.skip(2).join('/');
  final path = [
    org,
    repo,
    'resolve',
    revision,
    if (subfolder.isNotEmpty) subfolder,
  ].join('/');
  return _withManifest(Uri.https('huggingface.co', '/$path/'));
}
