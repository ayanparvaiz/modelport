import 'package:modelport/src/numeric.dart';
import 'package:test/test.dart';

void main() {
  test('roundHalfEven matches Python round', () {
    expect([0.5, 1.5, 2.5, -0.5, -1.5, 34.5, 2.4, 2.6].map(roundHalfEven), [
      0,
      2,
      2,
      0,
      -2,
      34,
      2,
      3,
    ]);
  });

  test('doubleToHalfBits known values', () {
    expect(doubleToHalfBits(1.0), 0x3C00);
    expect(doubleToHalfBits(-2.0), 0xC000);
    expect(doubleToHalfBits(65504.0), 0x7BFF);
    expect(doubleToHalfBits(65520.0), 0x7C00); // rounds up to infinity
    expect(doubleToHalfBits(0.0), 0x0000);
    expect(doubleToHalfBits(-0.0), 0x8000);
    expect(
      doubleToHalfBits(5.960464477539063e-8),
      0x0001,
    ); // smallest subnormal
    expect(
      doubleToHalfBits(2.9802322387695312e-8),
      0x0000,
    ); // half of it ties to even
    expect(doubleToHalfBits(double.infinity), 0x7C00);
    expect(doubleToHalfBits(double.nan) & 0x7C00, 0x7C00);
  });

  test('doubleToHalfBits rounds ties to even', () {
    // 1 + 2^-11 is exactly between 1.0 and the next half value.
    expect(doubleToHalfBits(1 + 1 / 2048), 0x3C00);
    expect(doubleToHalfBits(1 + 3 / 2048), 0x3C02);
  });
}
