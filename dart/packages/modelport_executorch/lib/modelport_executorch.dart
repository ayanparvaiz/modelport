/// PyTorch ExecuTorch adapter for ModelPort.
///
/// ```dart
/// await ModelPortFlutter.init(adapters: [ExecuTorchAdapter()]);
/// final classifier = await ImageClassifier.load('hf://org/mobilenet_v3_small');
/// ```
library;

import 'dart:io';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:modelport/modelport.dart';

/// Runs `executorch` variants (`.pte` programs) through `executorch_flutter`.
///
/// ExecuTorch takes inputs by position, so they are passed in the order the
/// manifest lists them, and outputs are matched to manifest outputs by order.
class ExecuTorchAdapter implements TensorAdapter {
  ExecuTorchAdapter({Set<String>? backends})
    : backends = backends ?? defaultBackends();

  /// Variant backends this adapter accepts. A variant without a backend is
  /// always accepted.
  final Set<String> backends;

  /// XNNPACK everywhere, plus CoreML and MPS on Apple platforms.
  static Set<String> defaultBackends() => {
    'xnnpack',
    'portable',
    if (Platform.isIOS || Platform.isMacOS) ...{'coreml', 'mps'},
  };

  @override
  String get runtime => Runtimes.executorch;

  @override
  bool canRun(Variant variant) =>
      variant.runtime == Runtimes.executorch &&
      (variant.backend == null || backends.contains(variant.backend));

  @override
  Future<TensorSession> open(LoadedVariant model) async {
    try {
      final program = await ExecuTorchModel.load(model.modelFile.path);
      return ExecuTorchSession._(program, model.manifest);
    } on Object catch (error) {
      throw ModelPortException(
        'ExecuTorch could not load ${model.variant.id} of ${model.manifest.id}: $error',
        hint:
            'The .pte must be made with an ExecuTorch version this package supports.',
      );
    }
  }
}

/// A `.pte` program loaded with ExecuTorch.
class ExecuTorchSession implements TensorSession {
  ExecuTorchSession._(this._program, this._manifest);

  final ExecuTorchModel _program;
  final Manifest _manifest;

  @override
  Future<Map<String, Tensor>> run(Map<String, Tensor> inputs) async {
    final ordered = [
      for (final spec in _manifest.inputs)
        TensorData(
          shape: inputs[spec.name]!.shape,
          dataType: tensorTypeOf(spec.dtype),
          data: inputs[spec.name]!.bytes,
        ),
    ];
    final outputs = await _program.forward(ordered);
    if (outputs.length < _manifest.outputs.length) {
      throw ModelPortException(
        'the program returned ${outputs.length} outputs, the manifest lists ${_manifest.outputs.length}',
      );
    }
    return {
      for (final (i, spec) in _manifest.outputs.indexed)
        spec.name: toTensor(spec.dtype, outputs[i]),
    };
  }

  @override
  Future<void> close() => _program.dispose();
}

/// The ExecuTorch element type for a manifest dtype.
TensorType tensorTypeOf(DType dtype) => switch (dtype) {
  DType.float32 => TensorType.float32,
  DType.float16 => TensorType.float16,
  DType.int64 => TensorType.int64,
  DType.int32 => TensorType.int32,
  DType.int8 => TensorType.int8,
  DType.uint8 => TensorType.uint8,
  DType.boolean => TensorType.bool_,
};

/// Wraps an ExecuTorch output as a [Tensor], checking it has the manifest dtype.
Tensor toTensor(DType expected, TensorData output) {
  if (output.dataType != tensorTypeOf(expected)) {
    throw ModelPortException(
      'output is ${output.dataType.displayName}, the manifest says ${expected.jsonName}',
    );
  }
  final shape = [for (final d in output.shape) d ?? -1];
  if (shape.contains(-1)) {
    throw ModelPortException(
      'ExecuTorch returned an output with an unknown dimension: $shape',
    );
  }
  return Tensor.fromBytes(expected, shape, output.data);
}
