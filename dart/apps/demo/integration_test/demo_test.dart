// Smoke test for the demo app on a real device. Models download from the
// ModelPort zoo (GitHub release assets) on first run.
//
//   flutter test integration_test -d <device> --no-uninstall
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:modelport_demo/main.dart';
import 'package:modelport_demo/widgets/common.dart';
import 'package:modelport_demo/zoo.dart';
import 'package:modelport_executorch/modelport_executorch.dart';
import 'package:modelport_flutter/modelport_flutter.dart';
import 'package:modelport_llamacpp/modelport_llamacpp.dart';
import 'package:modelport_onnx/modelport_onnx.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  setUpAll(
    () => ModelPortFlutter.init(
      adapters: [OnnxAdapter(), ExecuTorchAdapter(), LlamaCppAdapter()],
    ),
  );

  for (final engine in classifierEngines) {
    testWidgets('classify with ${engine.label}', (tester) async {
      final classifier = await ImageClassifier.load(
        classifiers.first.location,
        variantId: engine.variantId,
      );
      final results = await classifier.classify(await samplePhoto(), topK: 3);
      final golden = await classifier.model.checkGolden();
      debugPrint('DEMO| ${engine.label}: $results, golden ${golden.passed}');
      expect(results.first.label, 'Samoyed');
      expect(golden.passed, isTrue, reason: '$golden');
      await classifier.close();
    });
  }

  for (final engine in detectorEngines) {
    testWidgets('detect with ${engine.label}', (tester) async {
      final detector_ = await ObjectDetector.load(
        detector.location,
        variantId: engine.variantId,
      );
      final watch = Stopwatch()..start();
      final found = await detector_.detect(await samplePhoto(), minScore: 0.5);
      debugPrint(
        'DEMO| detect ${engine.label} ${watch.elapsedMilliseconds}ms: $found',
      );
      expect(found.any((d) => d.label == 'dog'), isTrue, reason: '$found');
      await detector_.close();
    });
  }

  testWidgets('chat with SmolLM2', (tester) async {
    final llm = await TextGenerator.load(chatModels.first.location);
    final config = llm.model.defaults.copyWith(temperature: 0, maxTokens: 40);
    final reply = await llm
        .generate('What is the capital of France?', config: config)
        .join();
    debugPrint('DEMO| chat: ${reply.trim()}');
    expect(reply.toLowerCase(), contains('paris'));
    await llm.close();
  });

  testWidgets('the app shows every tab', (tester) async {
    await tester.pumpWidget(const DemoApp());
    await tester.pumpAndSettle();
    for (final label in ['Detect', 'Chat', 'Models', 'Classify']) {
      await tester.tap(find.text(label).last);
      await tester.pumpAndSettle();
      expect(find.text('ModelPort · $label'), findsOneWidget);
    }
    await tester.tap(find.text('Models').last);
    await tester.pumpAndSettle();
    expect(
      find.textContaining('Engines: onnx, executorch, llamacpp'),
      findsOneWidget,
    );
  });
}
