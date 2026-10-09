/// Flutter setup for ModelPort.
///
/// ```dart
/// Future<void> main() async {
///   WidgetsFlutterBinding.ensureInitialized();
///   await ModelPortFlutter.init(adapters: [OnnxAdapter()]);
///   runApp(const MyApp());
/// }
/// ```
library;

import 'dart:io';
import 'dart:ui' as ui;

import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:modelport/modelport.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

export 'package:modelport/modelport.dart';

/// Configures ModelPort for a Flutter app.
abstract final class ModelPortFlutter {
  /// Sets up the model cache in the app's support folder, lets `asset://`
  /// locations read Flutter assets, reads the device's RAM, and registers
  /// [adapters].
  ///
  /// [cacheDir], [assets], and [client] are for tests and special setups.
  static Future<void> init({
    List<RuntimeAdapter> adapters = const [],
    Directory? cacheDir,
    AssetBundle? assets,
    http.Client? client,
    bool nativeImageDecoding = true,
  }) async {
    final root =
        cacheDir ??
        Directory(
          p.join((await getApplicationSupportDirectory()).path, 'modelport'),
        );
    final bundle = assets ?? rootBundle;
    ModelPort.configure(
      store: ModelStore(
        root: root,
        client: client,
        assetReader: (key) async {
          final data = await bundle.load(key);
          return Uint8List.sublistView(data);
        },
      ),
      adapters: adapters,
      deviceRamMb: await deviceRamMb(),
    );
    ModelPort.imageDecoder = nativeImageDecoding
        ? decodeImageWithFlutter
        : null;
  }

  /// Total device memory in MB, or null if it cannot be read.
  static Future<int?> deviceRamMb() async {
    try {
      final info = DeviceInfoPlugin();
      if (Platform.isAndroid) return (await info.androidInfo).physicalRamSize;
      if (Platform.isIOS) return (await info.iosInfo).physicalRamSize;
      if (Platform.isMacOS) {
        return (await info.macOsInfo).memorySize ~/ (1024 * 1024);
      }
    } on Object {
      // Unknown platforms and test hosts without the plugin: no RAM limit.
    }
    return null;
  }
}

/// Decodes image bytes with the Flutter engine's native codecs.
///
/// Much faster than the pure Dart decoder, and like Pillow it applies the
/// EXIF orientation. JPEG pixels can differ by a level or two from other
/// decoders; PNG pixels are identical.
Future<RgbImage> decodeImageWithFlutter(Uint8List bytes) async {
  final codec = await ui.instantiateImageCodec(bytes);
  try {
    final frame = await codec.getNextFrame();
    final image = frame.image;
    try {
      final data = await image.toByteData(
        format: ui.ImageByteFormat.rawStraightRgba,
      );
      if (data == null) throw ModelPortException('the image could not be read');
      final rgba = Uint8List.sublistView(data);
      final rgb = Uint8List(image.width * image.height * 3);
      for (var i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
        rgb[i] = rgba[j];
        rgb[i + 1] = rgba[j + 1];
        rgb[i + 2] = rgba[j + 2];
      }
      return RgbImage(image.width, image.height, rgb);
    } finally {
      image.dispose();
    }
  } finally {
    codec.dispose();
  }
}
