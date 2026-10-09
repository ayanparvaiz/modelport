/// llama.cpp adapter for ModelPort.
///
/// ```dart
/// await ModelPortFlutter.init(adapters: [LlamaCppAdapter()]);
/// final llm = await TextGenerator.load('hf://org/qwen2.5-0.5b-instruct');
/// await for (final piece in llm.chat([ChatMessage.user('Hi!')])) {
///   stdout.write(piece);
/// }
/// ```
///
/// On Android, native libraries must be extracted from the APK so llama.cpp
/// can find its CPU backends. Add this to `android/app/build.gradle.kts`:
///
/// ```kotlin
/// android {
///     packaging { jniLibs { useLegacyPackaging = true } }
/// }
/// ```
library;

import 'dart:async';

import 'package:llm_llamacpp/llm_llamacpp.dart';
// llm_llamacpp's own model loader finds no CPU backend on Android; its
// BackendInitializer does, but is not exported. See notes/spikes.md.
// ignore: implementation_imports
import 'package:llm_llamacpp/src/backend_initializer.dart';
import 'package:modelport/modelport.dart';

/// Runs `llamacpp` variants (GGUF files) through `llm_llamacpp`.
class LlamaCppAdapter implements TextGenerationAdapter {
  LlamaCppAdapter({
    this.gpuLayers = 0,
    this.threads,
    this.stopTokens = const [],
  });

  /// Layers to offload to the GPU. 0 runs on the CPU, the most portable choice.
  final int gpuLayers;

  /// CPU threads. Null lets llama.cpp choose.
  final int? threads;

  /// Extra stop strings for models whose GGUF does not mark the end of a turn.
  final List<String> stopTokens;

  static bool _backendsReady = false;

  @override
  String get runtime => Runtimes.llamacpp;

  @override
  bool canRun(Variant variant) => variant.runtime == Runtimes.llamacpp;

  @override
  Future<TextGenerationSession> open(LoadedVariant model) async {
    if (!_backendsReady) {
      BackendInitializer.initializeBackend();
      _backendsReady = true;
    }
    final repository = LlamaCppRepository();
    final LlamaCppModel loaded;
    try {
      loaded = await repository.loadModel(model.modelFile.path);
    } on Object catch (error) {
      repository.dispose();
      throw describeLoadError(error, model.variant.id);
    }
    final chat = LlamaCppChatRepository.withModel(
      loaded,
      repository.bindings,
      contextSize: model.manifest.llm?.contextLength ?? 4096,
      nGpuLayers: gpuLayers,
      threads: threads,
      stopTokens: stopTokens,
    );
    return LlamaCppSession._(repository, chat, model.variant.id);
  }
}

/// Turns llama.cpp load failures into errors that say how to fix them.
ModelPortException describeLoadError(Object error, String variantId) {
  final text = '$error';
  if (text.contains('no backends are loaded')) {
    return ModelPortException(
      'llama.cpp found no CPU backend for $variantId',
      hint:
          'On Android, add packaging { jniLibs { useLegacyPackaging = true } } '
          'to android/app/build.gradle.kts so the backend libraries are extracted.',
    );
  }
  return ModelPortException('llama.cpp could not load $variantId: $text');
}

/// A GGUF model loaded with llama.cpp.
class LlamaCppSession implements TextGenerationSession {
  LlamaCppSession._(this._repository, this._chat, this._modelId);

  final LlamaCppRepository _repository;
  final LlamaCppChatRepository _chat;
  final String _modelId;
  StreamSubscription<LLMChunk>? _active;
  StreamController<String>? _controller;

  @override
  Stream<String> chat(List<ChatMessage> messages, GenerationConfig config) {
    if (_active != null) {
      throw ModelPortException(
        'a reply is already being generated',
        hint: 'Wait for it to finish or call cancel() first.',
      );
    }
    final controller = StreamController<String>(onCancel: cancel);
    _controller = controller;
    final source = _chat.streamChatWithGenerationOptions(
      _modelId,
      messages: [for (final m in messages) toLlmMessage(m)],
      generationOptions: toGenerationOptions(config),
    );
    _active = source.listen(
      (chunk) {
        final text = chunk.message?.content ?? '';
        if (text.isNotEmpty) controller.add(text);
      },
      onError: controller.addError,
      onDone: () {
        _active = null;
        controller.close();
      },
    );
    return controller.stream;
  }

  @override
  Future<void> cancel() async {
    final active = _active;
    _active = null;
    await active?.cancel();
    final controller = _controller;
    if (controller != null && !controller.isClosed) await controller.close();
  }

  @override
  Future<void> close() async {
    await cancel();
    _chat.dispose();
    _repository.dispose();
  }
}

/// Converts a ModelPort chat message to llm_llamacpp's type.
LLMMessage toLlmMessage(ChatMessage message) => LLMMessage(
  role: switch (message.role) {
    ChatRole.system => LLMRole.system,
    ChatRole.user => LLMRole.user,
    ChatRole.assistant => LLMRole.assistant,
  },
  content: message.content,
);

/// Converts ModelPort sampling settings to llm_llamacpp's options.
GenerationOptions toGenerationOptions(GenerationConfig config) =>
    GenerationOptions(
      temperature: config.temperature,
      topP: config.topP,
      topK: config.topK ?? 40,
      maxTokens: config.maxTokens,
      repeatPenalty: config.repeatPenalty,
    );
