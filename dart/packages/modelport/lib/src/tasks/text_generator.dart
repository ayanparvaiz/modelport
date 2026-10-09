import '../model_port.dart';
import '../runtime/adapter.dart';
import '../store/model_store.dart';

/// Chats with any text-generation bundle.
///
/// ```dart
/// final llm = await TextGenerator.load('hf://org/qwen2.5-0.5b-instruct');
/// await for (final piece in llm.chat([ChatMessage.user('What is Flutter?')])) {
///   stdout.write(piece);
/// }
/// ```
class TextGenerator {
  TextGenerator._(this.model);

  static Future<TextGenerator> load(
    String location, {
    String? variantId,
    void Function(DownloadProgress)? onProgress,
    CancelToken? cancel,
  }) async => TextGenerator._(
    await ModelPort.loadText(
      location,
      variantId: variantId,
      onProgress: onProgress,
      cancel: cancel,
    ),
  );

  final TextModel model;

  /// Streams the reply to a conversation.
  Stream<String> chat(List<ChatMessage> messages, {GenerationConfig? config}) =>
      model.chat(messages, config: config);

  /// Streams the reply to a single prompt.
  Stream<String> generate(String prompt, {GenerationConfig? config}) =>
      chat([ChatMessage.user(prompt)], config: config);

  Future<void> cancel() => model.cancel();
  Future<void> close() => model.close();
}
