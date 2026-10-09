import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:modelport/modelport.dart';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';

import 'support/fake_bundle.dart';

void main() {
  const base = 'https://models.test/toy';
  late Directory cache;
  late FakeServer server;
  late FakeBundle bundle;
  late ModelStore store;
  final location = ModelLocation.parse(base);

  setUp(() {
    cache = Directory.systemTemp.createTempSync('modelport_store_');
    server = FakeServer();
    bundle = FakeBundle();
    server.serve(base, bundle);
    store = ModelStore(root: cache, client: server.client);
  });

  tearDown(() => cache.deleteSync(recursive: true));

  Future<Map<FileRef, File>> fetchAll(
    Manifest m, {
    void Function(DownloadProgress)? onProgress,
    CancelToken? cancel,
  }) =>
      store.fetch(location, m, m.files, onProgress: onProgress, cancel: cancel);

  test('downloads, verifies, and reuses cached files', () async {
    final manifest = await store.manifest(location);
    final progress = <DownloadProgress>[];
    final files = await fetchAll(manifest, onProgress: progress.add);

    expect(files, hasLength(2));
    final model = files[manifest.variants.single.file]!;
    expect(
      model.path,
      p.join(cache.path, 'models', 'toy', '1.0.0', 'onnx-fp32', 'model.onnx'),
    );
    expect(await model.readAsBytes(), bundle.files['onnx-fp32/model.onnx']);
    expect(progress.last.fraction, 1.0);
    expect(
      progress.map((e) => e.receivedBytes),
      orderedEquals([...progress.map((e) => e.receivedBytes)]..sort()),
    );

    server.requests.clear();
    await fetchAll(manifest);
    expect(
      server.requests,
      isEmpty,
      reason: 'verified files are not downloaded again',
    );
  });

  test('a wrong sha256 fails and leaves nothing behind', () async {
    final bad = {...bundle.ref('onnx-fp32/model.onnx'), 'sha256': 'b' * 64};
    server.serve(base, bundle, manifest: bundle.manifestText(modelRef: bad));
    final manifest = await store.manifest(location);
    await expectLater(
      fetchAll(manifest),
      throwsA(
        isA<ModelPortException>().having(
          (e) => e.message,
          'message',
          contains('sha256'),
        ),
      ),
    );
    final target = store.cachedFile(manifest, manifest.variants.single.file);
    expect(target.existsSync(), isFalse);
    expect(File('${target.path}.part').existsSync(), isFalse);
  });

  test('an interrupted download resumes with a Range request', () async {
    final manifest = await store.manifest(location);
    server.failAfterBytes = 20000;
    await expectLater(fetchAll(manifest), throwsA(isA<ModelPortException>()));
    final part = File(
      '${store.cachedFile(manifest, manifest.variants.single.file).path}.part',
    );
    expect(part.lengthSync(), 20000);

    server.requests.clear();
    await fetchAll(manifest);
    final resumed = server.requests.firstWhere(
      (r) => r.url.path.endsWith('model.onnx'),
    );
    expect(resumed.headers['range'], 'bytes=20000-');
    final model = store.cachedFile(manifest, manifest.variants.single.file);
    expect(await model.readAsBytes(), bundle.files['onnx-fp32/model.onnx']);
  });

  test('a server that ignores Range restarts cleanly', () async {
    final manifest = await store.manifest(location);
    server.failAfterBytes = 10000;
    await expectLater(fetchAll(manifest), throwsA(isA<ModelPortException>()));
    server.ignoreRange = true;
    await fetchAll(manifest);
    final model = store.cachedFile(manifest, manifest.variants.single.file);
    expect(await model.readAsBytes(), bundle.files['onnx-fp32/model.onnx']);
  });

  test('cancelling keeps the partial file for later', () async {
    final manifest = await store.manifest(location);
    final cancel = CancelToken();
    await expectLater(
      fetchAll(
        manifest,
        cancel: cancel,
        onProgress: (e) {
          if (e.receivedBytes > 8000) cancel.cancel();
        },
      ),
      throwsA(isA<DownloadCancelled>()),
    );
    final part = File(
      '${store.cachedFile(manifest, manifest.variants.single.file).path}.part',
    );
    expect(part.existsSync(), isTrue);
    await fetchAll(manifest);
  });

  test('the manifest loads offline after one successful download', () async {
    await store.manifest(location);
    server.offline = true;
    final manifest = await store.manifest(location);
    expect(manifest.id, 'toy');
  });

  test('offline with no cached manifest explains the problem', () async {
    server.offline = true;
    await expectLater(
      store.manifest(location),
      throwsA(
        isA<ModelPortException>().having(
          (e) => e.hint,
          'hint',
          contains('internet'),
        ),
      ),
    );
  });

  test('HTTP errors are reported', () async {
    await expectLater(
      store.manifest(ModelLocation.parse('https://models.test/missing')),
      throwsA(
        isA<ModelPortException>().having(
          (e) => e.message,
          'message',
          contains('404'),
        ),
      ),
    );
  });

  test('local bundles are used in place and size-checked', () async {
    final folder = Directory(p.join(cache.path, 'bundle'));
    for (final entry in bundle.files.entries) {
      File(p.joinAll([folder.path, ...entry.key.split('/')]))
        ..createSync(recursive: true)
        ..writeAsBytesSync(entry.value);
    }
    File(
      p.join(folder.path, 'modelport.json'),
    ).writeAsStringSync(bundle.manifestText());
    final local = ModelLocation.parse(folder.path);
    final manifest = await store.manifest(local);
    final files = await store.fetch(local, manifest, manifest.files);
    expect(files.values.first.path, startsWith(folder.path));

    File(p.join(folder.path, 'labels.txt')).writeAsStringSync('changed');
    await expectLater(
      store.fetch(local, manifest, manifest.files),
      throwsA(
        isA<ModelPortException>().having(
          (e) => e.hint,
          'hint',
          contains('modelport pack'),
        ),
      ),
    );
  });

  test('asset bundles are copied into the cache and verified', () async {
    final assets = {
      'assets/models/toy/modelport.json': utf8.encode(bundle.manifestText()),
      for (final e in bundle.files.entries)
        'assets/models/toy/${e.key}': e.value,
    };
    final assetStore = ModelStore(
      root: cache,
      client: server.client,
      assetReader: (key) async => assets[key] != null
          ? Uint8List.fromList(assets[key]!)
          : throw ModelPortException('no asset $key'),
    );
    final asset = ModelLocation.parse('asset://assets/models/toy');
    final manifest = await assetStore.manifest(asset);
    final files = await assetStore.fetch(asset, manifest, manifest.files);
    expect(files.values.first.path, startsWith(cache.path));
    expect(server.requests, isEmpty);
  });

  test('cache size and delete', () async {
    final manifest = await store.manifest(location);
    await fetchAll(manifest);
    expect(await store.cacheSize(), greaterThan(50000));
    await store.delete('toy');
    expect(await store.cacheSize(), 0);
  });
}
