import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:modelport_flutter/modelport_flutter.dart';
import 'package:path/path.dart' as p;

String repoFile(String relative) {
  var dir = Directory.current.absolute;
  while (!File(p.join(dir.path, 'spec', 'manifest.schema.json')).existsSync()) {
    dir = dir.parent;
  }
  return p.join(dir.path, relative);
}

void main() {
  testWidgets('PNG pixels match the pure Dart decoder exactly', (tester) async {
    await tester.runAsync(() async {
      final bytes = File(
        repoFile('spec/fixtures/preprocess/noise.png'),
      ).readAsBytesSync();
      final native = await decodeImageWithFlutter(bytes);
      final dart = decodeRgbImage(bytes);
      expect((native.width, native.height), (dart.width, dart.height));
      expect(native.pixels, dart.pixels);
    });
  });

  testWidgets('EXIF orientation is applied, like Pillow', (tester) async {
    await tester.runAsync(() async {
      // A 40x20 picture, red on the left and blue on the right, tagged
      // "rotate 90° clockwise". Upright it is 20x40 with red on top.
      final bytes = File('test/fixtures/exif_rotate90.jpg').readAsBytesSync();
      final image = await decodeImageWithFlutter(bytes);
      expect((image.width, image.height), (20, 40));
      int at(int x, int y, int c) =>
          image.pixels[(y * image.width + x) * 3 + c];
      expect(at(10, 5, 0), greaterThan(200), reason: 'top is red');
      expect(at(10, 35, 2), greaterThan(200), reason: 'bottom is blue');
    });
  });

  test('init registers the native decoder by default', () async {
    final cache = Directory.systemTemp.createTempSync('modelport_decode_');
    await ModelPortFlutter.init(cacheDir: cache);
    expect(ModelPort.imageDecoder, isNotNull);
    await ModelPortFlutter.init(cacheDir: cache, nativeImageDecoding: false);
    expect(ModelPort.imageDecoder, isNull);
    ModelPort.reset();
    cache.deleteSync(recursive: true);
  });
}
