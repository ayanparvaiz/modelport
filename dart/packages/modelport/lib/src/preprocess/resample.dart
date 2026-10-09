/// Image resizing that matches the Python reference exactly. See docs/spec.md.
library;

import 'dart:typed_data';

import 'rgb_image.dart';

// Pillow's fixed-point precision for 8-bit resampling.
const _precisionBits = 32 - 8 - 2;

double _bilinearKernel(double x) {
  if (x < 0) x = -x;
  return x < 1 ? 1 - x : 0;
}

double _bicubicKernel(double x) {
  const a = -0.5;
  if (x < 0) x = -x;
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

class _Coefficients {
  _Coefficients(this.ksize, this.bounds, this.weights);
  final int ksize;
  final Int32List bounds;
  final Int32List weights;
}

/// Port of Pillow's `precompute_coeffs` and `normalize_coeffs_8bpc`.
_Coefficients _coefficients(
  int inSize,
  int outSize,
  double support,
  double Function(double) kernel,
) {
  final scale = inSize / outSize;
  final filterScale = scale < 1.0 ? 1.0 : scale;
  final scaledSupport = support * filterScale;
  final ksize = scaledSupport.ceil() * 2 + 1;
  final bounds = Int32List(outSize * 2);
  final weights = Int32List(outSize * ksize);
  final k = Float64List(ksize);
  final ss = 1.0 / filterScale;
  for (var xx = 0; xx < outSize; xx++) {
    final center = (xx + 0.5) * scale;
    var xmin = (center - scaledSupport + 0.5).toInt();
    if (xmin < 0) xmin = 0;
    var xmax = (center + scaledSupport + 0.5).toInt();
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    var total = 0.0;
    for (var x = 0; x < xmax; x++) {
      final w = kernel((x + xmin - center + 0.5) * ss);
      k[x] = w;
      total += w;
    }
    for (var x = 0; x < ksize; x++) {
      var value = 0.0;
      if (x < xmax) value = total != 0 ? k[x] / total : k[x];
      weights[xx * ksize + x] = value < 0
          ? (-0.5 + value * (1 << _precisionBits)).toInt()
          : (0.5 + value * (1 << _precisionBits)).toInt();
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return _Coefficients(ksize, bounds, weights);
}

int _clip8(int value) {
  final shifted = value >> _precisionBits;
  return shifted < 0 ? 0 : (shifted > 255 ? 255 : shifted);
}

Uint8List _horizontal(
  Uint8List src,
  int width,
  int height,
  int outWidth,
  _Coefficients c,
) {
  final out = Uint8List(outWidth * height * 3);
  const start = 1 << (_precisionBits - 1);
  for (var y = 0; y < height; y++) {
    final rowIn = y * width * 3;
    final rowOut = y * outWidth * 3;
    for (var xx = 0; xx < outWidth; xx++) {
      final xmin = c.bounds[xx * 2];
      final xmax = c.bounds[xx * 2 + 1];
      final k = xx * c.ksize;
      var s0 = start, s1 = start, s2 = start;
      for (var x = 0; x < xmax; x++) {
        final p = rowIn + (x + xmin) * 3;
        final w = c.weights[k + x];
        s0 += src[p] * w;
        s1 += src[p + 1] * w;
        s2 += src[p + 2] * w;
      }
      final o = rowOut + xx * 3;
      out[o] = _clip8(s0);
      out[o + 1] = _clip8(s1);
      out[o + 2] = _clip8(s2);
    }
  }
  return out;
}

Uint8List _vertical(Uint8List src, int width, int outHeight, _Coefficients c) {
  final out = Uint8List(width * outHeight * 3);
  const start = 1 << (_precisionBits - 1);
  for (var yy = 0; yy < outHeight; yy++) {
    final ymin = c.bounds[yy * 2];
    final ymax = c.bounds[yy * 2 + 1];
    final k = yy * c.ksize;
    for (var x = 0; x < width; x++) {
      var s0 = start, s1 = start, s2 = start;
      for (var y = 0; y < ymax; y++) {
        final p = ((y + ymin) * width + x) * 3;
        final w = c.weights[k + y];
        s0 += src[p] * w;
        s1 += src[p + 1] * w;
        s2 += src[p + 2] * w;
      }
      final o = (yy * width + x) * 3;
      out[o] = _clip8(s0);
      out[o + 1] = _clip8(s1);
      out[o + 2] = _clip8(s2);
    }
  }
  return out;
}

/// Pillow-compatible filtered resize (`Image.resize` with BILINEAR or BICUBIC).
RgbImage resamplePillow(
  RgbImage image,
  int outWidth,
  int outHeight, {
  required bool bicubic,
}) {
  if (outWidth == image.width && outHeight == image.height) return image;
  final support = bicubic ? 2.0 : 1.0;
  final kernel = bicubic ? _bicubicKernel : _bilinearKernel;
  var pixels = image.pixels;
  var width = image.width;
  if (outWidth != image.width) {
    final c = _coefficients(image.width, outWidth, support, kernel);
    pixels = _horizontal(pixels, image.width, image.height, outWidth, c);
    width = outWidth;
  }
  if (outHeight != image.height) {
    final c = _coefficients(image.height, outHeight, support, kernel);
    pixels = _vertical(pixels, width, outHeight, c);
  }
  return RgbImage(outWidth, outHeight, pixels);
}

/// Pillow-compatible nearest neighbour resize, including its floating-point stepping.
RgbImage resampleNearest(RgbImage image, int outWidth, int outHeight) {
  if (outWidth == image.width && outHeight == image.height) return image;
  final xScale = image.width / outWidth;
  final yScale = image.height / outHeight;
  final xin = Int32List(outWidth);
  var xo = xScale * 0.5;
  for (var x = 0; x < outWidth; x++) {
    xin[x] = xo < 0 ? -1 : xo.toInt();
    xo += xScale;
  }
  final out = Uint8List(outWidth * outHeight * 3);
  var yo = yScale * 0.5;
  for (var y = 0; y < outHeight; y++) {
    final yi = yo < 0 ? -1 : yo.toInt();
    yo += yScale;
    if (yi < 0 || yi >= image.height) continue;
    for (var x = 0; x < outWidth; x++) {
      final sx = xin[x];
      if (sx < 0 || sx >= image.width) continue;
      final p = (yi * image.width + sx) * 3;
      final o = (y * outWidth + x) * 3;
      out[o] = image.pixels[p];
      out[o + 1] = image.pixels[p + 1];
      out[o + 2] = image.pixels[p + 2];
    }
  }
  return RgbImage(outWidth, outHeight, out);
}

/// Plain bilinear with half-pixel centers and no filtering, in floating point.
///
/// Returns height x width x 3 values rounded to float32, like the Python reference.
Float32List resampleBilinearPlain(RgbImage image, int outWidth, int outHeight) {
  final (yLow, yHigh, yFrac) = _axis(image.height, outHeight);
  final (xLow, xHigh, xFrac) = _axis(image.width, outWidth);
  final src = image.pixels;
  final w = image.width;
  // Vertical first, then horizontal, in double precision, as NumPy does.
  final rows = Float64List(outHeight * w * 3);
  for (var y = 0; y < outHeight; y++) {
    final f = yFrac[y];
    final a = yLow[y] * w * 3;
    final b = yHigh[y] * w * 3;
    final o = y * w * 3;
    for (var i = 0; i < w * 3; i++) {
      rows[o + i] = src[a + i] * (1 - f) + src[b + i] * f;
    }
  }
  final out = Float32List(outHeight * outWidth * 3);
  for (var y = 0; y < outHeight; y++) {
    final row = y * w * 3;
    for (var x = 0; x < outWidth; x++) {
      final f = xFrac[x];
      final a = row + xLow[x] * 3;
      final b = row + xHigh[x] * 3;
      final o = (y * outWidth + x) * 3;
      for (var c = 0; c < 3; c++) {
        out[o + c] = rows[a + c] * (1 - f) + rows[b + c] * f;
      }
    }
  }
  return out;
}

(Int32List, Int32List, Float64List) _axis(int inSize, int outSize) {
  final scale = inSize / outSize;
  final low = Int32List(outSize);
  final high = Int32List(outSize);
  final frac = Float64List(outSize);
  for (var i = 0; i < outSize; i++) {
    var src = (i + 0.5) * scale - 0.5;
    if (src < 0) src = 0;
    var l = src.floor();
    if (l > inSize - 1) l = inSize - 1;
    low[i] = l;
    high[i] = l + 1 > inSize - 1 ? inSize - 1 : l + 1;
    frac[i] = src - l;
  }
  return (low, high, frac);
}
