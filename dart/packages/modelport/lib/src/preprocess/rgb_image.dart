import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../errors.dart';

/// An 8-bit RGB image: three bytes per pixel, rows top to bottom.
class RgbImage {
  RgbImage(this.width, this.height, this.pixels) {
    if (pixels.length != width * height * 3) {
      throw ModelPortException(
        'RGB image of ${width}x$height needs ${width * height * 3} bytes, got ${pixels.length}',
      );
    }
  }

  /// Converts an image from `package:image`, applying its EXIF orientation.
  factory RgbImage.fromImage(img.Image image) {
    var source = img.bakeOrientation(image);
    if (source.format != img.Format.uint8 ||
        source.numChannels != 3 ||
        source.hasPalette) {
      source = source.convert(format: img.Format.uint8, numChannels: 3);
    }
    final bytes = source.getBytes(order: img.ChannelOrder.rgb);
    return RgbImage(source.width, source.height, Uint8List.fromList(bytes));
  }

  final int width;
  final int height;
  final Uint8List pixels;
}

/// Decodes PNG, JPEG, WebP, and other formats `package:image` supports.
RgbImage decodeRgbImage(Uint8List bytes) {
  final image = img.decodeImage(bytes);
  if (image == null) throw ModelPortException('could not decode the image');
  return RgbImage.fromImage(image);
}
