// Phase 0 spikes, Flutter side.
//
// A: ONNX Runtime runs MobileNetV3 and matches Python.
// B: ExecuTorch runs the same model and matches Python.
// C: llama.cpp streams a chat reply from a GGUF model.
//
// Run on macOS:
//   ./tool/copy_assets.sh
//   flutter test integration_test/spike_test.dart -d macos \
//     --dart-define=GGUF_PATH=/abs/path/qwen2.5-0.5b-instruct-q4_k_m.gguf

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:llm_llamacpp/llm_llamacpp.dart';
// Workaround, see notes/spikes.md: ModelLoader in llm_llamacpp 0.7.0 does not load
// the CPU backend .so files on Android. BackendInitializer does, but is not exported.
// ignore: implementation_imports
import 'package:llm_llamacpp/src/backend_initializer.dart';

const samoyedIndex = 258;
const inputShape = [1, 3, 224, 224];
const ggufPath = String.fromEnvironment('GGUF_PATH');

void log(String message) => debugPrint('SPIKE| $message');

Future<Uint8List> assetBytes(String key) async {
  final data = await rootBundle.load(key);
  // Copy so the buffer starts at offset 0 and is safe to view as Float32List.
  return Uint8List.fromList(data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes));
}

Float32List floats(Uint8List bytes) => Float32List.view(Uint8List.fromList(bytes).buffer);

int argmax(List<double> values) {
  var best = 0;
  for (var i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

double maxAbsDiff(List<double> a, List<double> b) {
  expect(a.length, b.length);
  var worst = 0.0;
  for (var i = 0; i < a.length; i++) {
    final d = (a[i] - b[i]).abs();
    if (d > worst) worst = d;
  }
  return worst;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late Uint8List inputBytes;
  late Float32List expected;

  setUpAll(() async {
    inputBytes = await assetBytes('assets/pixel_values.bin');
    expected = floats(await assetBytes('assets/logits.bin'));
  });

  testWidgets('A: ONNX Runtime matches Python', (tester) async {
    final watch = Stopwatch()..start();
    final session = await OnnxRuntime().createSessionFromAsset('assets/model.onnx');
    final loadMs = watch.elapsedMilliseconds;

    Future<Float32List> runOnce() async {
      final input = await OrtValue.fromList(floats(inputBytes), inputShape);
      final outputs = await session.run({'pixel_values': input});
      final flat = await outputs['logits']!.asFlattenedList();
      await input.dispose();
      for (final value in outputs.values) {
        await value.dispose();
      }
      return Float32List.fromList([for (final v in flat) (v as num).toDouble()]);
    }

    watch.reset();
    final logits = await runOnce();
    final firstMs = watch.elapsedMilliseconds;
    watch.reset();
    await runOnce();
    final warmMs = watch.elapsedMilliseconds;

    final diff = maxAbsDiff(logits, expected);
    log('A onnx load=${loadMs}ms first=${firstMs}ms warm=${warmMs}ms '
        'top1=${argmax(logits)} maxDiff=${diff.toStringAsExponential(2)}');
    expect(argmax(logits), samoyedIndex);
    expect(diff, lessThan(1e-3));
    await session.close();
  });

  testWidgets('B: ExecuTorch matches Python', (tester) async {
    final watch = Stopwatch()..start();
    final model = await ExecuTorchModel.loadFromBytes(await assetBytes('assets/model.pte'));
    final loadMs = watch.elapsedMilliseconds;

    Future<Float32List> runOnce() async {
      final outputs = await model.forward([
        TensorData(shape: inputShape, dataType: TensorType.float32, data: inputBytes),
      ]);
      return floats(outputs.first.data);
    }

    watch.reset();
    final logits = await runOnce();
    final firstMs = watch.elapsedMilliseconds;
    watch.reset();
    await runOnce();
    final warmMs = watch.elapsedMilliseconds;

    final diff = maxAbsDiff(logits, expected);
    log('B executorch load=${loadMs}ms first=${firstMs}ms warm=${warmMs}ms '
        'top1=${argmax(logits)} maxDiff=${diff.toStringAsExponential(2)}');
    expect(argmax(logits), samoyedIndex);
    expect(diff, lessThan(1e-3));
    await model.dispose();
  });

  testWidgets('C: llama.cpp streams a chat reply', (tester) async {
    BackendInitializer.initializeBackend();
    final repo = LlamaCppRepository();
    final watch = Stopwatch()..start();
    final model = await repo.loadModel(ggufPath);
    final loadMs = watch.elapsedMilliseconds;
    final chat = LlamaCppChatRepository.withModel(
      model,
      repo.bindings,
      contextSize: 2048,
      nGpuLayers: 0,
    );

    final reply = StringBuffer();
    var pieces = 0;
    int? firstPieceMs;
    watch.reset();
    await for (final chunk in chat.streamChatWithGenerationOptions(
      ggufPath,
      messages: [LLMMessage(role: LLMRole.user, content: 'In one sentence, what is Flutter?')],
      generationOptions: const GenerationOptions(temperature: 0, maxTokens: 64),
    )) {
      final text = chunk.message?.content ?? '';
      if (text.isEmpty) continue;
      firstPieceMs ??= watch.elapsedMilliseconds;
      pieces++;
      reply.write(text);
    }
    final totalMs = watch.elapsedMilliseconds;
    final perSecond = pieces / ((totalMs - (firstPieceMs ?? 0)) / 1000);
    log('C llamacpp load=${loadMs}ms firstPiece=${firstPieceMs}ms pieces=$pieces '
        'total=${totalMs}ms speed=${perSecond.toStringAsFixed(1)}/s');
    log('C reply: ${reply.toString().trim()}');
    expect(reply.toString().trim(), isNotEmpty);
    chat.dispose();
    repo.dispose();
  }, skip: ggufPath.isEmpty);
}
