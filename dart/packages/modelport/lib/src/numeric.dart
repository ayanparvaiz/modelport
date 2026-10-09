import 'dart:typed_data';

/// Rounds to the nearest integer, ties to even, like Python's `round` and NumPy's `rint`.
int roundHalfEven(double value) {
  final floor = value.floorToDouble();
  final diff = value - floor;
  final base = floor.toInt();
  if (diff > 0.5) return base + 1;
  if (diff < 0.5) return base;
  return base.isEven ? base : base + 1;
}

/// IEEE 754 binary16 bits for [value], rounded to nearest even straight from
/// double precision, like NumPy's `astype(float16)`.
int doubleToHalfBits(double value) {
  final data = ByteData(8)..setFloat64(0, value);
  final bits = data.getUint64(0);
  final sign = (bits >> 48) & 0x8000;
  final exponent = (bits >> 52) & 0x7ff;
  final mantissa = bits & 0xFFFFFFFFFFFFF;

  if (exponent == 0x7ff) {
    return sign | 0x7c00 | (mantissa != 0 ? 0x200 : 0);
  }
  final halfExponent = exponent - 1023 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    final full = mantissa | (1 << 52);
    final shift = 1 - halfExponent + 42;
    var half = full >> shift;
    final remainder = full & ((1 << shift) - 1);
    final halfway = 1 << (shift - 1);
    if (remainder > halfway || (remainder == halfway && half.isOdd)) half++;
    return sign | half;
  }
  var result = sign | (halfExponent << 10) | (mantissa >> 42);
  final remainder = mantissa & ((1 << 42) - 1);
  const halfway = 1 << 41;
  if (remainder > halfway || (remainder == halfway && result.isOdd)) result++;
  return result;
}
