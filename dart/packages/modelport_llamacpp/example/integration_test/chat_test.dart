// A GGUF model, described by a manifest that ships as an asset, downloads
// from the Hugging Face Hub on the device and answers through llama.cpp.
//
//   flutter test integration_test -d <device> --no-uninstall
import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:modelport_flutter/modelport_flutter.dart';
import 'package:modelport_llamacpp/modelport_llamacpp.dart';

const bundle = 'asset://assets/manifests/smollm2-135m-instruct';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  setUpAll(() => ModelPortFlutter.init(adapters: [LlamaCppAdapter()]));

  testWidgets('downloads, loads, and chats', (tester) async {
    var lastPercent = -10;
    final watch = Stopwatch()..start();
    final llm = await TextGenerator.load(
      bundle,
      onProgress: (p) {
        final percent = (p.fraction * 100).floor();
        if (percent >= lastPercent + 10) {
          lastPercent = percent;
          debugPrint('LLM| download $percent%');
        }
      },
    );
    debugPrint(
      'LLM| ready in ${watch.elapsedMilliseconds}ms (${llm.model.variant.id})',
    );

    final config = llm.model.defaults.copyWith(temperature: 0, maxTokens: 48);
    watch.reset();
    var pieces = 0;
    int? firstMs;
    final reply = StringBuffer();
    await for (final piece in llm.generate(
      'What is the capital of France? Answer in one sentence.',
      config: config,
    )) {
      firstMs ??= watch.elapsedMilliseconds;
      pieces++;
      reply.write(piece);
    }
    final totalMs = watch.elapsedMilliseconds;
    debugPrint('LLM| first piece ${firstMs}ms, $pieces pieces in ${totalMs}ms');
    debugPrint('LLM| reply: ${reply.toString().trim()}');
    expect(reply.toString().trim(), isNotEmpty);
    expect(reply.toString().toLowerCase(), contains('paris'));
    await llm.close();
  });
}
