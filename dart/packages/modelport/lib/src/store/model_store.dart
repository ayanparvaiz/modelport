import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;

import '../errors.dart';
import '../manifest/files.dart';
import '../manifest/manifest.dart';
import 'location.dart';

/// Reads a Flutter asset by key. Supplied by `modelport_flutter`.
typedef AssetReader = Future<Uint8List> Function(String key);

/// Download progress across every file of one fetch.
class DownloadProgress {
  const DownloadProgress({
    required this.receivedBytes,
    required this.totalBytes,
    required this.file,
  });

  final int receivedBytes;
  final int totalBytes;

  /// The file being downloaded right now.
  final String file;

  /// 0.0 to 1.0.
  double get fraction => totalBytes == 0 ? 1 : receivedBytes / totalBytes;

  @override
  String toString() =>
      'DownloadProgress(${(fraction * 100).toStringAsFixed(1)}%, $file)';
}

/// Lets an app stop a download. The partial file is kept so it can resume.
class CancelToken {
  bool _cancelled = false;
  bool get isCancelled => _cancelled;
  void cancel() => _cancelled = true;
}

/// A download was stopped with a [CancelToken].
class DownloadCancelled extends ModelPortException {
  DownloadCancelled() : super('download cancelled');
}

/// Downloads, verifies, and caches model files.
///
/// Layout under [root]:
/// * `models/<id>/<version>/<path>` for bundle files
/// * `models/<id>/<version>/_external/<sha>_<name>` for files given by url
/// * `locations/<key>.json`, the last manifest seen at each location, for offline use
///
/// A `.sha256` marker next to a file records that it was verified, so large
/// models are hashed once, not on every app start.
class ModelStore {
  ModelStore({required this.root, http.Client? client, this.assetReader})
    : _client = client ?? http.Client();

  final Directory root;
  final AssetReader? assetReader;
  final http.Client _client;

  /// Loads and validates the manifest. Remote manifests are cached so the
  /// model still loads offline after the first successful download.
  Future<Manifest> manifest(ModelLocation location) async {
    final cached = File(
      p.join(root.path, 'locations', '${location.cacheKey}.json'),
    );
    final Uint8List bytes;
    try {
      bytes = await _readWhole(location, location.manifestUri);
    } on _NetworkError catch (error) {
      if (cached.existsSync()) {
        return Manifest.parse(await cached.readAsString());
      }
      throw ModelPortException(
        'could not download ${location.manifestUri}: ${error.message}',
        hint:
            'Check the internet connection. Models load offline after one download.',
      );
    }
    final manifest = Manifest.parse(utf8.decode(bytes));
    if (location.isRemote) {
      await cached.parent.create(recursive: true);
      await cached.writeAsBytes(bytes);
    }
    return manifest;
  }

  /// Where [file] of [manifest] is kept in the cache.
  File cachedFile(Manifest manifest, FileRef file) {
    final folder = p.join(root.path, 'models', manifest.id, manifest.version);
    final path = file.path;
    if (path != null) return File(p.joinAll([folder, ...path.split('/')]));
    final name = Uri.parse(
      file.url!,
    ).pathSegments.lastWhere((s) => s.isNotEmpty, orElse: () => 'file');
    return File(
      p.join(folder, '_external', '${file.sha256.substring(0, 16)}_$name'),
    );
  }

  /// Makes sure every file in [files] is on disk and verified, downloading
  /// what is missing. Returns the local file for each reference.
  Future<Map<FileRef, File>> fetch(
    ModelLocation location,
    Manifest manifest,
    List<FileRef> files, {
    void Function(DownloadProgress)? onProgress,
    CancelToken? cancel,
  }) async {
    final result = <FileRef, File>{};
    final missing = <FileRef>[];
    for (final file in files) {
      if (location.isFile && file.path != null) {
        result[file] = await _localFile(location, file);
        continue;
      }
      final target = cachedFile(manifest, file);
      if (await _isVerified(target, file)) {
        result[file] = target;
      } else {
        missing.add(file);
      }
    }

    final total = missing.fold<int>(0, (sum, f) => sum + f.size);
    var done = 0;
    for (final file in missing) {
      final target = cachedFile(manifest, file);
      final name = file.path ?? file.url!;
      await _download(location, file, target, cancel, (received) {
        onProgress?.call(
          DownloadProgress(
            receivedBytes: done + received,
            totalBytes: total,
            file: name,
          ),
        );
      });
      done += file.size;
      result[file] = target;
    }
    if (missing.isNotEmpty) {
      onProgress?.call(
        DownloadProgress(receivedBytes: total, totalBytes: total, file: ''),
      );
    }
    return result;
  }

  /// Bytes used by cached models.
  Future<int> cacheSize() async {
    final folder = Directory(p.join(root.path, 'models'));
    if (!folder.existsSync()) return 0;
    var total = 0;
    await for (final entity in folder.list(recursive: true)) {
      if (entity is File) total += await entity.length();
    }
    return total;
  }

  /// Removes a cached model, or one version of it.
  Future<void> delete(String id, {String? version}) async {
    final folder = Directory(p.joinAll([root.path, 'models', id, ?version]));
    if (folder.existsSync()) await folder.delete(recursive: true);
  }

  /// Closes the HTTP client.
  void close() => _client.close();

  Future<File> _localFile(ModelLocation location, FileRef file) async {
    final local = File.fromUri(location.uriFor(file));
    if (!local.existsSync()) {
      throw ModelPortException('${local.path} is missing from the bundle');
    }
    final size = await local.length();
    if (size != file.size) {
      throw ModelPortException(
        '${local.path} is $size bytes, the manifest says ${file.size}',
        hint: 'Run `modelport pack` on the bundle after editing files.',
      );
    }
    return local;
  }

  Future<bool> _isVerified(File target, FileRef file) async {
    final marker = File('${target.path}.sha256');
    if (!target.existsSync() || !marker.existsSync()) return false;
    return await target.length() == file.size &&
        (await marker.readAsString()).trim() == file.sha256;
  }

  Future<void> _download(
    ModelLocation location,
    FileRef file,
    File target,
    CancelToken? cancel,
    void Function(int received) progress,
  ) async {
    await target.parent.create(recursive: true);
    final part = File('${target.path}.part');
    final uri = location.uriFor(file);

    if (uri.scheme == 'asset' || uri.scheme == 'file') {
      await part.writeAsBytes(await _readWhole(location, uri));
      progress(file.size);
    } else {
      var start = part.existsSync() ? await part.length() : 0;
      if (start > file.size) {
        await part.delete();
        start = 0;
      }
      if (start < file.size) {
        await _downloadRange(uri, part, start, file, cancel, progress);
      }
    }
    await _verifyAndKeep(part, target, file);
  }

  Future<void> _downloadRange(
    Uri uri,
    File part,
    int start,
    FileRef file,
    CancelToken? cancel,
    void Function(int received) progress,
  ) async {
    final request = http.Request('GET', uri);
    if (start > 0) request.headers['range'] = 'bytes=$start-';
    final http.StreamedResponse response;
    try {
      response = await _client.send(request);
    } on Object catch (error) {
      throw _wrapNetwork(error, uri);
    }
    var offset = start;
    if (response.statusCode == 200) {
      offset = 0; // the server ignored the range; start over
    } else if (response.statusCode != 206) {
      await response.stream.drain<void>();
      throw ModelPortException('HTTP ${response.statusCode} for $uri');
    }

    final sink = part.openWrite(
      mode: offset > 0 ? FileMode.append : FileMode.write,
    );
    var received = offset;
    progress(received);
    try {
      await for (final chunk in response.stream) {
        if (cancel?.isCancelled ?? false) throw DownloadCancelled();
        sink.add(chunk);
        received += chunk.length;
        if (received > file.size) {
          throw ModelPortException('$uri sent more than ${file.size} bytes');
        }
        progress(received);
      }
    } on ModelPortException {
      rethrow;
    } on Object catch (error) {
      throw _wrapNetwork(error, uri);
    } finally {
      await sink.close();
    }
  }

  Future<void> _verifyAndKeep(File part, File target, FileRef file) async {
    final size = await part.length();
    if (size != file.size) {
      if (size > file.size) await part.delete();
      throw ModelPortException(
        'download stopped at $size of ${file.size} bytes',
      );
    }
    final digest = (await sha256.bind(part.openRead()).first).toString();
    if (digest != file.sha256) {
      await part.delete();
      throw ModelPortException(
        'sha256 of ${file.path ?? file.url} does not match the manifest',
        hint:
            'The file on the server changed or is corrupt. Republish the bundle.',
      );
    }
    if (target.existsSync()) await target.delete();
    await part.rename(target.path);
    await File('${target.path}.sha256').writeAsString(file.sha256);
  }

  Future<Uint8List> _readWhole(ModelLocation location, Uri uri) async {
    switch (uri.scheme) {
      case 'file':
        final file = File.fromUri(uri);
        if (!file.existsSync()) {
          throw ModelPortException('${file.path} does not exist');
        }
        return file.readAsBytes();
      case 'asset':
        final reader = assetReader;
        if (reader == null) {
          throw ModelPortException(
            'asset:// locations need Flutter',
            hint: 'Add modelport_flutter and call ModelPortFlutter.init().',
          );
        }
        return reader(uri.path);
      default:
        final http.Response response;
        try {
          response = await _client.get(uri);
        } on Object catch (error) {
          throw _wrapNetwork(error, uri);
        }
        if (response.statusCode != 200) {
          throw ModelPortException('HTTP ${response.statusCode} for $uri');
        }
        return response.bodyBytes;
    }
  }
}

class _NetworkError extends ModelPortException {
  _NetworkError(super.message);
}

ModelPortException _wrapNetwork(Object error, Uri uri) {
  if (error is ModelPortException) return error;
  if (error is http.ClientException ||
      error is SocketException ||
      error is TimeoutException) {
    return _NetworkError('$error');
  }
  return ModelPortException('downloading $uri failed: $error');
}
