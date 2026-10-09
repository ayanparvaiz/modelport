import 'package:flutter_test/flutter_test.dart';
import 'package:modelport/modelport.dart';
import 'package:modelport_onnx/modelport_onnx.dart';

void main() {
  test('float32 values', () {
    final t = toTensor(DType.float32, [1, 2], [0.5, 2]);
    expect(t.float32, [0.5, 2.0]);
    expect(t.shape, [1, 2]);
  });

  test('integers and booleans', () {
    expect(toTensor(DType.int64, [2], [3, -4]).toDoubles(), [3, -4]);
    expect(toTensor(DType.boolean, [2], [true, false]).toDoubles(), [1, 0]);
  });

  test('float16 outputs are stored as half bits', () {
    final t = toTensor(DType.float16, [1], [1.0]);
    expect(t.toDoubles(), [1.0]);
    expect(t.dtype, DType.float16);
  });

  test('adapter metadata', () {
    final adapter = OnnxAdapter();
    expect(adapter.runtime, Runtimes.onnx);
  });
}
