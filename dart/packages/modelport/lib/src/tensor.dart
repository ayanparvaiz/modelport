import 'dart:math' as math;
import 'dart:typed_data';

import 'errors.dart';
import 'manifest/tensors.dart';

/// Typed numbers plus a shape: what adapters pass to and from engines.
///
/// Data is stored in host byte order, which is little-endian on every
/// platform Flutter runs on, the same order as golden files.
class Tensor {
  Tensor._(this.dtype, List<int> shape, this.data)
    : shape = List.unmodifiable(shape) {
    final expected = shape.fold<int>(1, (a, b) => a * b);
    if (shape.any((d) => d < 0)) {
      throw ModelPortException('tensor shape $shape has a negative dimension');
    }
    if (expected != data.lengthInBytes ~/ dtype.bytesPerElement) {
      throw ModelPortException(
        'tensor shape $shape needs $expected values, got '
        '${data.lengthInBytes ~/ dtype.bytesPerElement}',
      );
    }
  }

  factory Tensor.float32(List<int> shape, Float32List data) =>
      Tensor._(DType.float32, shape, data);

  /// Half-precision values given as raw IEEE 754 binary16 bit patterns.
  factory Tensor.float16(List<int> shape, Uint16List bits) =>
      Tensor._(DType.float16, shape, bits);

  factory Tensor.int64(List<int> shape, Int64List data) =>
      Tensor._(DType.int64, shape, data);

  factory Tensor.int32(List<int> shape, Int32List data) =>
      Tensor._(DType.int32, shape, data);

  factory Tensor.int8(List<int> shape, Int8List data) =>
      Tensor._(DType.int8, shape, data);

  factory Tensor.uint8(List<int> shape, Uint8List data) =>
      Tensor._(DType.uint8, shape, data);

  /// Booleans stored as one byte each, 0 or 1.
  factory Tensor.boolean(List<int> shape, Uint8List data) =>
      Tensor._(DType.boolean, shape, data);

  /// Builds a tensor from raw little-endian bytes, such as a golden file.
  ///
  /// The bytes are copied so the result is correctly aligned.
  factory Tensor.fromBytes(DType dtype, List<int> shape, Uint8List bytes) {
    if (Endian.host != Endian.little) {
      throw ModelPortException('big-endian hosts are not supported');
    }
    final buffer = Uint8List.fromList(bytes).buffer;
    final TypedData data = switch (dtype) {
      DType.float32 => buffer.asFloat32List(),
      DType.float16 => buffer.asUint16List(),
      DType.int64 => buffer.asInt64List(),
      DType.int32 => buffer.asInt32List(),
      DType.int8 => buffer.asInt8List(),
      DType.uint8 || DType.boolean => buffer.asUint8List(),
    };
    return Tensor._(dtype, shape, data);
  }

  final DType dtype;
  final List<int> shape;
  final TypedData data;

  int get elementCount => data.lengthInBytes ~/ dtype.bytesPerElement;

  /// The raw bytes, without copying.
  Uint8List get bytes =>
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);

  /// The data as float32. Throws for other dtypes; use [toDoubles] to convert.
  Float32List get float32 {
    final values = data;
    if (values is Float32List && dtype == DType.float32) return values;
    throw ModelPortException('tensor is ${dtype.jsonName}, not float32');
  }

  /// Every element as a double, whatever the dtype.
  List<double> toDoubles() => switch (dtype) {
    DType.float16 => [
      for (final bits in data as Uint16List) halfToDouble(bits),
    ],
    DType.float32 => List<double>.of(data as Float32List),
    DType.int64 => [for (final v in data as Int64List) v.toDouble()],
    DType.int32 => [for (final v in data as Int32List) v.toDouble()],
    DType.int8 => [for (final v in data as Int8List) v.toDouble()],
    DType.uint8 ||
    DType.boolean => [for (final v in data as Uint8List) v.toDouble()],
  };

  @override
  String toString() => 'Tensor(${dtype.jsonName}, $shape)';
}

/// Decodes an IEEE 754 binary16 value.
double halfToDouble(int bits) {
  final sign = (bits & 0x8000) != 0 ? -1.0 : 1.0;
  final exponent = (bits >> 10) & 0x1f;
  final fraction = bits & 0x3ff;
  if (exponent == 0) return sign * fraction * math.pow(2, -24).toDouble();
  if (exponent == 0x1f) {
    return fraction == 0 ? sign * double.infinity : double.nan;
  }
  return sign * (1 + fraction / 1024) * math.pow(2, exponent - 15).toDouble();
}
