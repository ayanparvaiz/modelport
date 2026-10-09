import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:modelport_flutter/modelport_flutter.dart';
import 'package:modelport_onnx/modelport_onnx.dart';

/// The bundle copied in by tool/copy_bundle.sh.
const bundle = 'asset://assets/models/mobilenet_v3_small';
const variants = ['onnx-fp32', 'onnx-fp16', 'onnx-int8'];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ModelPortFlutter.init(adapters: [OnnxAdapter()]);
  runApp(const ExampleApp());
}

class ExampleApp extends StatelessWidget {
  const ExampleApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'ModelPort ONNX',
    theme: ThemeData(colorSchemeSeed: Colors.teal),
    home: const ClassifyPage(),
  );
}

class ClassifyPage extends StatefulWidget {
  const ClassifyPage({super.key});

  @override
  State<ClassifyPage> createState() => _ClassifyPageState();
}

class _ClassifyPageState extends State<ClassifyPage> {
  String _variant = variants.first;
  String _status = 'Loading…';
  List<Classification> _results = const [];
  GoldenReport? _golden;

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    setState(() {
      _status = 'Loading $_variant…';
      _results = const [];
      _golden = null;
    });
    try {
      final classifier = await ImageClassifier.load(
        bundle,
        variantId: _variant,
      );
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
        _status = 'Classified in $ms ms';
      });
    } on ModelPortException catch (error) {
      setState(() => _status = error.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    final golden = _golden;
    return Scaffold(
      appBar: AppBar(title: const Text('ModelPort · ONNX')),
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
          DropdownButton<String>(
            value: _variant,
            items: [
              for (final v in variants)
                DropdownMenuItem(value: v, child: Text(v)),
            ],
            onChanged: (value) {
              if (value == null) return;
              _variant = value;
              _run();
            },
          ),
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
