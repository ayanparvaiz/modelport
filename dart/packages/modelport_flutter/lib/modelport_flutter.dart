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
