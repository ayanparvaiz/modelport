import 'dart:typed_data';

import 'package:modelport/modelport.dart';

/// A pretend engine: logits are the mean of each input channel, so a red
/// image scores highest on class 0.
class ChannelMeanAdapter implements TensorAdapter {
  ChannelMeanAdapter({this.runtime = Runtimes.onnx, this.supports});

  @override
  final String runtime;
  final bool Function(Variant)? supports;
  int opened = 0;

  @override
  bool canRun(Variant variant) => supports?.call(variant) ?? true;

  @override
  Future<TensorSession> open(LoadedVariant model) async {
    opened++;
    return ChannelMeanSession(model.manifest);
  }
}

class ChannelMeanSession implements TensorSession {
  ChannelMeanSession(this.manifest);
  final Manifest manifest;
  bool closed = false;

  @override
  Future<Map<String, Tensor>> run(Map<String, Tensor> inputs) async {
    final input = inputs.values.single;
    final values = input.float32;
    final plane = values.length ~/ 3;
    final means = Float32List(3);
    for (var c = 0; c < 3; c++) {
      var sum = 0.0;
      for (var i = 0; i < plane; i++) {
        sum += values[c * plane + i];
      }
      means[c] = sum / plane;
    }
    return {
      manifest.outputs.single.name: Tensor.float32([1, 3], means),
    };
  }

  @override
  Future<void> close() async => closed = true;
}

/// A pretend language model that echoes the last user message word by word.
class EchoTextAdapter implements TextGenerationAdapter {
  @override
  String get runtime => Runtimes.llamacpp;

  @override
  bool canRun(Variant variant) => true;

  GenerationConfig? lastConfig;

  @override
  Future<TextGenerationSession> open(LoadedVariant model) async =>
      _EchoSession(this);
}

class _EchoSession implements TextGenerationSession {
  _EchoSession(this.adapter);
  final EchoTextAdapter adapter;

  @override
  Stream<String> chat(
    List<ChatMessage> messages,
    GenerationConfig config,
  ) async* {
    adapter.lastConfig = config;
    final words = messages
        .lastWhere((m) => m.role == ChatRole.user)
        .content
        .split(' ');
    for (final word in words.take(config.maxTokens)) {
      yield '$word ';
    }
  }

  @override
  Future<void> cancel() async {}

  @override
  Future<void> close() async {}
}
