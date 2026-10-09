import 'package:llm_llamacpp/llm_llamacpp.dart';
import 'package:modelport/modelport.dart';
import 'package:modelport_llamacpp/modelport_llamacpp.dart';
import 'package:test/test.dart';

void main() {
  test('chat roles map to llm_llamacpp roles', () {
    expect(toLlmMessage(const ChatMessage.system('s')).role, LLMRole.system);
    expect(toLlmMessage(const ChatMessage.user('u')).role, LLMRole.user);
    final assistant = toLlmMessage(const ChatMessage.assistant('a'));
    expect(assistant.role, LLMRole.assistant);
    expect(assistant.content, 'a');
  });

  test('generation config maps to options', () {
    final options = toGenerationOptions(
      const GenerationConfig(
        temperature: 0.2,
        topP: 0.8,
        maxTokens: 64,
        repeatPenalty: 1.1,
      ),
    );
    expect(options.temperature, 0.2);
    expect(options.topP, 0.8);
    expect(options.topK, 40);
    expect(options.maxTokens, 64);
    expect(options.repeatPenalty, 1.1);
  });

  test('missing backends explain the Android fix', () {
    final error = describeLoadError(
      Exception('llama.cpp errors: no backends are loaded'),
      'gguf-q4_k_m',
    );
    expect(error.hint, contains('useLegacyPackaging'));
    expect(describeLoadError(Exception('bad file'), 'v').hint, isNull);
  });

  test('adapter metadata', () {
    expect(LlamaCppAdapter().runtime, Runtimes.llamacpp);
  });
}
