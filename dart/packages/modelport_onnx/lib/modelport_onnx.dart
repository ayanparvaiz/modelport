/// ONNX Runtime adapter for ModelPort.
///
/// ```dart
/// await ModelPortFlutter.init(adapters: [OnnxAdapter()]);
/// final classifier = await ImageClassifier.load('hf://org/mobilenet_v3_small');
/// ```
library;

import 'dart:typed_data';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:modelport/modelport.dart';

/// Runs `onnx` variants with ONNX Runtime through `flutter_onnxruntime`.
///
/// ONNX Runtime cannot report input and output details on iOS and macOS, so
/// this adapter takes tensor names, types, and shapes from the manifest.
class OnnxAdapter implements TensorAdapter {
  OnnxAdapter({this.threads, this.providers});

  /// Threads per inference. Null lets ONNX Runtime choose.
  final int? threads;

  /// Execution providers in order of preference, such as
  /// `[OrtProvider.XNNPACK, OrtProvider.CPU]`. Null means CPU.
  final List<OrtProvider>? providers;

  final OnnxRuntime _runtime = OnnxRuntime();

  @override
  String get runtime => Runtimes.onnx;

  @override
  bool canRun(Variant variant) => variant.runtime == Runtimes.onnx;

  @override
  Future<TensorSession> open(LoadedVariant model) async {
    final options = OrtSessionOptions(
      intraOpNumThreads: threads,
      providers: providers,
    );
    try {
      final session = await _runtime.createSession(
        model.modelFile.path,
        options: options,
      );
      return OnnxSession._(session, model.manifest);
    } on Object catch (error) {
      throw ModelPortException(
        'ONNX Runtime could not open ${model.variant.id} of ${model.manifest.id}: $error',
        hint:
            'Check that the variant runs on this device, for example with checkGolden().',
      );
    }
  }
}

/// A model opened with ONNX Runtime.
class OnnxSession implements TensorSession {
  OnnxSession._(this._session, this._manifest);

  final OrtSession _session;
  final Manifest _manifest;

  @override
  Future<Map<String, Tensor>> run(Map<String, Tensor> inputs) async {
    final values = <String, OrtValue>{};
    Map<String, OrtValue>? outputs;
    try {
      for (final MapEntry(key: name, value: tensor) in inputs.entries) {
        values[name] = await OrtValue.fromList(
          _ortData(name, tensor),
          tensor.shape,
        );
      }
      outputs = await _session.run(values);
      final result = <String, Tensor>{};
      for (final spec in _manifest.outputs) {
        final value = outputs[spec.name];
        if (value == null) continue;
        final data = await value.asFlattenedList();
        result[spec.name] = toTensor(spec.dtype, value.shape, data);
      }
      return result;
    } finally {
      for (final value in [...values.values, ...?outputs?.values]) {
        await value.dispose();
      }
    }
  }

  @override
  Future<void> close() => _session.close();
}

Object _ortData(String name, Tensor tensor) => switch (tensor.dtype) {
  DType.float32 => tensor.float32,
  DType.int64 => tensor.data as Int64List,
  DType.int32 => tensor.data as Int32List,
  DType.uint8 => tensor.data as Uint8List,
  DType.boolean => [for (final b in tensor.data as Uint8List) b != 0],
  _ => throw ModelPortException(
    'the onnx adapter cannot send ${tensor.dtype.jsonName} input "$name"',
    hint: 'Export the model with float32, int32, int64, uint8, or bool inputs.',
  ),
};

/// Converts ONNX Runtime's flattened output into a [Tensor] of the manifest dtype.
Tensor toTensor(DType dtype, List<int> shape, List<dynamic> values) {
  double asDouble(dynamic v) => v is bool ? (v ? 1 : 0) : (v as num).toDouble();
  int asInt(dynamic v) => v is bool ? (v ? 1 : 0) : (v as num).toInt();
  return switch (dtype) {
    DType.float32 => Tensor.float32(
      shape,
      Float32List.fromList([for (final v in values) asDouble(v)]),
    ),
    DType.float16 => Tensor.float16(
      shape,
      Uint16List.fromList([
        for (final v in values) doubleToHalfBits(asDouble(v)),
      ]),
    ),
    DType.int64 => Tensor.int64(
      shape,
      Int64List.fromList([for (final v in values) asInt(v)]),
    ),
    DType.int32 => Tensor.int32(
      shape,
      Int32List.fromList([for (final v in values) asInt(v)]),
    ),
    DType.int8 => Tensor.int8(
      shape,
      Int8List.fromList([for (final v in values) asInt(v)]),
    ),
    DType.uint8 => Tensor.uint8(
      shape,
      Uint8List.fromList([for (final v in values) asInt(v)]),
    ),
    DType.boolean => Tensor.boolean(
      shape,
      Uint8List.fromList([for (final v in values) asInt(v)]),
    ),
  };
}
