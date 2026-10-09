import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:modelport_executorch/modelport_executorch.dart';
import 'package:modelport_flutter/modelport_flutter.dart';

/// The bundle copied in by tool/copy_bundle.sh.
const bundle = 'asset://assets/models/mobilenet_v3_small';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ModelPortFlutter.init(adapters: [ExecuTorchAdapter()]);
  runApp(
    MaterialApp(
      title: 'ModelPort ExecuTorch',
      theme: ThemeData(colorSchemeSeed: Colors.deepOrange),
      home: const ClassifyPage(),
    ),
  );
}

class ClassifyPage extends StatefulWidget {
  const ClassifyPage({super.key});

  @override
  State<ClassifyPage> createState() => _ClassifyPageState();
}

class _ClassifyPageState extends State<ClassifyPage> {
  String _status = 'Loading…';
  List<Classification> _results = const [];
  GoldenReport? _golden;

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    try {
      final classifier = await ImageClassifier.load(bundle);
      final photo = await rootBundle.load('assets/images/dog.jpg');
      final watch = Stopwatch()..start();
      final results = await classifier.classify(
        Uint8List.sublistView(photo),
        topK: 3,
      );
      final ms = watch.elapsedMilliseconds;
      final golden = await classifier.model.checkGolden();
      await classifier.close();
      setState(() {
        _results = results;
        _golden = golden;
        _status = 'Classified in $ms ms with ${classifier.model.variant.id}';
      });
    } on ModelPortException catch (error) {
      setState(() => _status = error.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    final golden = _golden;
    return Scaffold(
      appBar: AppBar(title: const Text('ModelPort · ExecuTorch')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Image.asset(
              'assets/images/dog.jpg',
              height: 220,
              fit: BoxFit.cover,
            ),
          ),
          const SizedBox(height: 16),
          Text(_status),
          for (final result in _results)
            ListTile(
              title: Text(result.label),
              trailing: Text('${(result.score * 100).toStringAsFixed(1)}%'),
            ),
          if (golden != null)
            Card(
              color: golden.passed ? Colors.green.shade50 : Colors.red.shade50,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Text('$golden'),
              ),
            ),
        ],
      ),
    );
  }
}
