import 'package:modelport/modelport.dart';
import 'package:test/test.dart';

import 'support/fake_adapters.dart';
import 'support/spec.dart';

void main() {
  final classifier = Manifest.parse(
    readSpecFile('examples/image-classification.json'),
  );
  final llm = Manifest.parse(readSpecFile('examples/text-generation.json'));

  test('picks the first variant a registered adapter can run', () {
    final (variant, _) = selectVariant<TensorAdapter>(classifier, [
      ChannelMeanAdapter(runtime: Runtimes.executorch),
      ChannelMeanAdapter(),
    ]);
    expect(variant.id, 'onnx-fp32');
  });

  test('skips variants the adapter cannot run', () {
    final (variant, _) = selectVariant<TensorAdapter>(classifier, [
      ChannelMeanAdapter(supports: (v) => false),
      ChannelMeanAdapter(runtime: Runtimes.executorch),
    ]);
    expect(variant.id, 'executorch-xnnpack-fp32');
  });

  test('explains which package is missing', () {
    expect(
      () => selectVariant<TensorAdapter>(classifier, []),
      throwsA(
        isA<ModelPortException>().having(
          (e) => e.hint,
          'hint',
          allOf(contains('modelport_onnx'), contains('modelport_executorch')),
        ),
      ),
    );
  });

  test('respects device RAM', () {
    final (small, _) = selectVariant<TextGenerationAdapter>(llm, [
      EchoTextAdapter(),
    ], deviceRamMb: 1500);
    expect(small.id, 'gguf-q4_k_m');
    expect(
      () => selectVariant<TextGenerationAdapter>(llm, [
        EchoTextAdapter(),
      ], deviceRamMb: 512),
      throwsA(
        isA<ModelPortException>().having(
          (e) => e.message,
          'message',
          contains('1024 MB'),
        ),
      ),
    );
  });

  test('a specific variant can be requested', () {
    final (variant, _) = selectVariant<TextGenerationAdapter>(llm, [
      EchoTextAdapter(),
    ], variantId: 'gguf-q8_0');
    expect(variant.id, 'gguf-q8_0');
    expect(
      () => selectVariant<TextGenerationAdapter>(llm, [
        EchoTextAdapter(),
      ], variantId: 'nope'),
      throwsA(isA<ModelPortException>()),
    );
  });

  test('tensor adapters are not used for language models', () {
    expect(
      () => selectVariant<TextGenerationAdapter>(llm, [
        ChannelMeanAdapter(runtime: Runtimes.llamacpp),
      ]),
      throwsA(isA<ModelPortException>()),
    );
  });

  test('generation config from manifest defaults', () {
    final config = GenerationConfig.fromDefaults(llm.llm!.defaults);
    expect(config.maxTokens, 512);
    expect(config.copyWith(maxTokens: 8).maxTokens, 8);
  });
}
