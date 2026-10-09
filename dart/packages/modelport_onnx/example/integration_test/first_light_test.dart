// Phase 4 "First Light": a ModelPort bundle made by the Python CLI runs on the
// device through the ONNX adapter and gives the same answer as Python.
//
// Setup: tool/copy_bundle.sh, then
//   flutter test integration_test -d <device> --no-uninstall
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:modelport_flutter/modelport_flutter.dart';
import 'package:modelport_onnx/modelport_onnx.dart';

const bundle = 'asset://assets/models/mobilenet_v3_small';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  setUpAll(() => ModelPortFlutter.init(adapters: [OnnxAdapter()]));

  for (final variant in ['onnx-fp32', 'onnx-int8', 'onnx-fp16']) {
    testWidgets('golden check: $variant', (tester) async {
      final watch = Stopwatch()..start();
      final model = await ModelPort.load(bundle, variantId: variant);
      final loadMs = watch.elapsedMilliseconds;
      final report = await model.checkGolden();
      debugPrint('FIRSTLIGHT| $variant load=${loadMs}ms\n$report');
      expect(report.passed, isTrue, reason: '$report');
      await model.close();
    });
  }

  testWidgets('classifies a photo of a Samoyed', (tester) async {
    final classifier = await ImageClassifier.load(bundle);
    final photo = await rootBundle.load('assets/images/dog.jpg');
    final bytes = Uint8List.sublistView(photo);
    await classifier.classify(bytes); // warm up
    final watch = Stopwatch()..start();
    final results = await classifier.classify(bytes, topK: 3);
    debugPrint(
      'FIRSTLIGHT| classify ${watch.elapsedMilliseconds}ms '
      '${classifier.model.variant.id}: $results',
    );
    expect(results.first.label, 'Samoyed');
    await classifier.close();
  });

  testWidgets('time each step of one classification', (tester) async {
    final model = await ModelPort.load(bundle);
    final input = model.manifest.inputs.single;
    final photo = Uint8List.sublistView(
      await rootBundle.load('assets/images/dog.jpg'),
    );
    final watch = Stopwatch()..start();
    final image = decodeRgbImage(photo);
    final decodeMs = watch.elapsedMilliseconds;
    watch.reset();
    final tensor = preprocessImage(image, input);
    final preprocessMs = watch.elapsedMilliseconds;
    await model.run({input.name: tensor}); // warm up
    watch.reset();
    await model.run({input.name: tensor});
    final runMs = watch.elapsedMilliseconds;
    debugPrint(
      'FIRSTLIGHT| steps for ${image.width}x${image.height}: decode=${decodeMs}ms '
      'preprocess=${preprocessMs}ms run=${runMs}ms',
    );
    await model.close();
  });
}
