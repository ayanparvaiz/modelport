import 'dart:typed_data';

import 'package:modelport/modelport.dart';
import 'package:test/test.dart';

void main() {
  test('float32 tensor', () {
    final t = Tensor.float32([1, 2], Float32List.fromList([1.5, -2]));
    expect(t.elementCount, 2);
    expect(t.float32, [1.5, -2]);
    expect(t.bytes, hasLength(8));
    expect(t.toString(), 'Tensor(float32, [1, 2])');
  });

  test('shape must match the data length', () {
    expect(
      () => Tensor.float32([2, 2], Float32List(3)),
      throwsA(isA<ModelPortException>()),
    );
  });

  test('fromBytes copies unaligned bytes', () {
    final source = Float32List.fromList([1, 2, 3]).buffer.asUint8List();
    final unaligned = Uint8List(13)..setRange(1, 13, source);
    final view = Uint8List.sublistView(unaligned, 1);
    final t = Tensor.fromBytes(DType.float32, [3], view);
    expect(t.float32, [1, 2, 3]);
  });

  test('float32 getter refuses other dtypes', () {
    final t = Tensor.int64([1], Int64List.fromList([7]));
    expect(() => t.float32, throwsA(isA<ModelPortException>()));
    expect(t.toDoubles(), [7.0]);
  });

  test('float16 decoding', () {
    final t = Tensor.float16([
      5,
    ], Uint16List.fromList([0x3C00, 0xC000, 0x7BFF, 0x0001, 0x7C00]));
    final values = t.toDoubles();
    expect(values[0], 1.0);
    expect(values[1], -2.0);
    expect(values[2], 65504.0);
    expect(values[3], closeTo(5.960464477539063e-8, 1e-20));
    expect(values[4], double.infinity);
  });

  test('uint8 and int8', () {
    expect(Tensor.uint8([2], Uint8List.fromList([0, 255])).toDoubles(), [
      0,
      255,
    ]);
    expect(Tensor.int8([2], Int8List.fromList([-128, 127])).toDoubles(), [
      -128,
      127,
    ]);
  });
}
