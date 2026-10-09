import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:modelport/modelport.dart';
import 'package:modelport_executorch/modelport_executorch.dart';

void main() {
  test('every manifest dtype has an ExecuTorch type of the same size', () {
    for (final dtype in DType.values) {
      expect(
        tensorTypeOf(dtype).sizeInBytes,
        dtype.bytesPerElement,
        reason: dtype.name,
      );
    }
  });

  test('outputs become tensors', () {
    final data = Float32List.fromList([1, 2, 3]).buffer.asUint8List();
    final tensor = toTensor(
      DType.float32,
      TensorData(shape: [1, 3], dataType: TensorType.float32, data: data),
    );
    expect(tensor.float32, [1, 2, 3]);
  });

  test('an output with the wrong dtype is rejected', () {
    expect(
      () => toTensor(
        DType.float32,
        TensorData(shape: [1], dataType: TensorType.int64, data: Uint8List(8)),
      ),
      throwsA(isA<ModelPortException>()),
    );
  });

  test('backends', () {
    final adapter = ExecuTorchAdapter(backends: {'xnnpack'});
    Variant variant(String? backend) => Variant(
      id: 'v',
      runtime: Runtimes.executorch,
      backend: backend,
      precision: 'fp32',
      file: const FileRef(
        path: 'm.pte',
        size: 1,
        sha256:
            'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      ),
    );
    expect(adapter.canRun(variant('xnnpack')), isTrue);
    expect(adapter.canRun(variant(null)), isTrue);
    expect(adapter.canRun(variant('vulkan')), isFalse);
  });
}
