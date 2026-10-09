// Set up ModelPort in a Flutter app and show where models are cached.
//
// Add an engine package, such as modelport_onnx, and pass its adapter to
// ModelPortFlutter.init to run models. See the demo app in the repository.
import 'package:flutter/material.dart';
import 'package:modelport_flutter/modelport_flutter.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ModelPortFlutter.init(); // adapters: [OnnxAdapter(), ...]
  runApp(const MaterialApp(home: CacheInfo()));
}

class CacheInfo extends StatelessWidget {
  const CacheInfo({super.key});

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('ModelPort')),
    body: FutureBuilder<int>(
      future: ModelPort.store.cacheSize(),
      builder: (context, snapshot) => ListTile(
        title: Text('Cache: ${ModelPort.store.root.path}'),
        subtitle: Text(
          'Downloaded: ${((snapshot.data ?? 0) / 1e6).toStringAsFixed(1)} MB · '
          'Device RAM: ${ModelPort.deviceRamMb ?? 'unknown'} MB',
        ),
      ),
    ),
  );
}
