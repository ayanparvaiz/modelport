import 'dart:io';

import '../manifest/manifest.dart';
import '../tensor.dart';

/// A variant whose files are downloaded, verified, and on disk.
class LoadedVariant {
  const LoadedVariant({
    required this.manifest,
    required this.variant,
    required this.modelFile,
    this.extraFiles = const [],
  });

  final Manifest manifest;
  final Variant variant;

  /// The main model file, such as `model.onnx`, `model.pte`, or `model.gguf`.
  final File modelFile;

  /// Files the main file needs, in manifest order.
  final List<File> extraFiles;
}

/// Connects one inference engine to ModelPort.
///
/// Adapter packages such as `modelport_onnx` implement [TensorAdapter] or
/// [TextGenerationAdapter], and apps register them once at startup.
abstract interface class RuntimeAdapter {
  /// The manifest runtime this adapter runs, such as `onnx`. See [Runtimes].
  String get runtime;

  /// Whether this adapter can run [variant] on this device, for example
  /// whether it supports the variant's backend.
  bool canRun(Variant variant);
}

/// Runs tensor-in, tensor-out models.
abstract interface class TensorAdapter implements RuntimeAdapter {
  Future<TensorSession> open(LoadedVariant model);
}

/// A loaded tensor model.
abstract interface class TensorSession {
  /// Runs the model. Inputs and outputs are keyed by manifest tensor name.
  Future<Map<String, Tensor>> run(Map<String, Tensor> inputs);

  Future<void> close();
}

/// Runs language models.
abstract interface class TextGenerationAdapter implements RuntimeAdapter {
  Future<TextGenerationSession> open(LoadedVariant model);
}

/// A loaded language model.
abstract interface class TextGenerationSession {
  /// Streams the reply to [messages], piece by piece.
  Stream<String> chat(List<ChatMessage> messages, GenerationConfig config);

  /// Stops the reply that is being generated.
  Future<void> cancel();

  Future<void> close();
}

enum ChatRole { system, user, assistant }

/// One message in a conversation.
class ChatMessage {
  const ChatMessage(this.role, this.content);
  const ChatMessage.system(this.content) : role = ChatRole.system;
  const ChatMessage.user(this.content) : role = ChatRole.user;
  const ChatMessage.assistant(this.content) : role = ChatRole.assistant;

  final ChatRole role;
  final String content;

  @override
  String toString() => '${role.name}: $content';
}

/// Sampling settings for one reply.
class GenerationConfig {
  const GenerationConfig({
    this.temperature = 0.7,
    this.topP = 0.9,
    this.topK,
    this.maxTokens = 512,
    this.repeatPenalty,
  });

  /// The manifest's defaults.
  factory GenerationConfig.fromDefaults(GenerationDefaults defaults) =>
      GenerationConfig(
        temperature: defaults.temperature,
        topP: defaults.topP,
        topK: defaults.topK,
        maxTokens: defaults.maxTokens,
        repeatPenalty: defaults.repeatPenalty,
      );

  final double temperature;
  final double topP;
  final int? topK;
  final int maxTokens;
  final double? repeatPenalty;

  GenerationConfig copyWith({
    double? temperature,
    double? topP,
    int? topK,
    int? maxTokens,
    double? repeatPenalty,
  }) => GenerationConfig(
    temperature: temperature ?? this.temperature,
    topP: topP ?? this.topP,
    topK: topK ?? this.topK,
    maxTokens: maxTokens ?? this.maxTokens,
    repeatPenalty: repeatPenalty ?? this.repeatPenalty,
  );
}
