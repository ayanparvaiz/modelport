// The MobileNetV3 bundle runs on the device through the ExecuTorch adapter
// and gives the same answer as Python.
//
// Setup: tool/copy_bundle.sh, then
//   flutter test integration_test -d <device> --no-uninstall
//   flutter drive --driver=test_driver/integration_test.dart \
//     --target=integration_test/first_light_test.dart -d <device> --profile
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:modelport_executorch/modelport_executorch.dart';
import 'package:modelport_flutter/modelport_flutter.dart';

const bundle = 'asset://assets/models/mobilenet_v3_small';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  setUpAll(() => ModelPortFlutter.init(adapters: [ExecuTorchAdapter()]));

  testWidgets('golden check: executorch-xnnpack-fp32', (tester) async {
    final watch = Stopwatch()..start();
    final model = await ModelPort.load(bundle);
    final loadMs = watch.elapsedMilliseconds;
    expect(model.variant.id, 'executorch-xnnpack-fp32');
    final report = await model.checkGolden();
    debugPrint('EXECUTORCH| load=${loadMs}ms\n$report');
    expect(report.passed, isTrue, reason: '$report');
    await model.close();
  });

  testWidgets('classifies a photo of a Samoyed', (tester) async {
    final classifier = await ImageClassifier.load(bundle);
    final photo = Uint8List.sublistView(
      await rootBundle.load('assets/images/dog.jpg'),
    );
    await classifier.classify(photo); // warm up
    final watch = Stopwatch()..start();
    final results = await classifier.classify(photo, topK: 3);
    debugPrint('EXECUTORCH| classify ${watch.elapsedMilliseconds}ms: $results');
    expect(results.first.label, 'Samoyed');

    final input = classifier.model.manifest.inputs.single;
    final tensor = preprocessImage(await ModelPort.imageDecoder!(photo), input);
    watch.reset();
    await classifier.model.run({input.name: tensor});
    debugPrint('EXECUTORCH| model run ${watch.elapsedMilliseconds}ms');
    await classifier.close();
  });

  testWidgets('detects the dog with YOLOS-tiny', (tester) async {
    const yolos = 'asset://assets/models/yolos-tiny';
    final detector = await ObjectDetector.load(yolos);
    final report = await detector.model.checkGolden();
    debugPrint('EXECUTORCH| yolos $report');
    expect(report.passed, isTrue, reason: '$report');

    final photo = Uint8List.sublistView(
      await rootBundle.load('assets/images/dog.jpg'),
    );
    await detector.detect(photo); // warm up
    final watch = Stopwatch()..start();
    final found = await detector.detect(photo);
    debugPrint('EXECUTORCH| detect ${watch.elapsedMilliseconds}ms: $found');
    expect(
      found.any((d) => d.label == 'dog' && d.score > 0.5),
      isTrue,
      reason: '$found',
    );
    await detector.close();
  });
}
