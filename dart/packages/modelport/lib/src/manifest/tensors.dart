import '../errors.dart';
import 'json_reader.dart';
import 'postprocess.dart';

/// Element type of a tensor.
enum DType {
  float32('float32', 4),
  float16('float16', 2),
  int64('int64', 8),
  int32('int32', 4),
  int8('int8', 1),
  uint8('uint8', 1),
  boolean('bool', 1);

  const DType(this.jsonName, this.bytesPerElement);

  /// The name used in `modelport.json`.
  final String jsonName;
  final int bytesPerElement;
}

/// Dimension order of image tensors.
enum TensorLayout {
  nchw('NCHW'),
  nhwc('NHWC');

  const TensorLayout(this.jsonName);
  final String jsonName;
}

enum ResizeMethod {
  bilinear,
  nearest,
  bicubic;

  String get jsonName => name;
}

enum ColorOrder {
  rgb('RGB'),
  bgr('BGR');

  const ColorOrder(this.jsonName);
  final String jsonName;
}

/// A size in pixels.
typedef ImageSize = ({int height, int width});

ImageSize _size(JsonReader json, String key) {
  final values = json.integers(key);
  if (values.length != 2 || values.any((v) => v <= 0)) {
    throw ManifestException(
      '${json.at(key)} must be two positive integers [height, width]',
    );
  }
  return (height: values[0], width: values[1]);
}

/// How to resize an image. Exactly one of [shorterSide] or [size] is set.
class ResizeSpec {
  const ResizeSpec({
    this.shorterSide,
    this.size,
    this.method = ResizeMethod.bilinear,
    this.antialias = false,
  });

  factory ResizeSpec.fromJson(JsonReader json) {
    final shorterSide = json.optionalInteger('shorter_side');
    final size = json.has('size') ? _size(json, 'size') : null;
    if ((shorterSide == null) == (size == null)) {
      throw ManifestException(
        '${json.path}: set exactly one of "shorter_side" or "size"',
      );
    }
    if (shorterSide != null && shorterSide <= 0) {
      throw ManifestException('${json.at('shorter_side')} must be positive');
    }
    final method = json.oneOf(
      'method',
      ResizeMethod.values,
      (m) => m.jsonName,
      fallback: ResizeMethod.bilinear,
    );
    final antialias = json.boolean('antialias', false);
    if (method == ResizeMethod.bicubic && !antialias) {
      throw ManifestException(
        '${json.path}: bicubic resize needs antialias: true',
      );
    }
    if (method == ResizeMethod.nearest && antialias) {
      throw ManifestException(
        '${json.path}: nearest resize cannot use antialias',
      );
    }
    return ResizeSpec(
      shorterSide: shorterSide,
      size: size,
      method: method,
      antialias: antialias,
    );
  }

  /// Scale so the shorter side has this length, keeping the aspect ratio.
  final int? shorterSide;

  /// Exact output size, ignoring the aspect ratio.
  final ImageSize? size;
  final ResizeMethod method;

  /// true: PIL-compatible filtered resampling. false: plain half-pixel bilinear.
  final bool antialias;
}

/// Rules for turning a decoded image into a tensor. See docs/spec.md.
class ImagePreprocess {
  const ImagePreprocess({
    required this.resize,
    this.centerCrop,
    this.color = ColorOrder.rgb,
    this.scale = 1 / 255,
    this.mean = const [0, 0, 0],
    this.std = const [1, 1, 1],
  });

  factory ImagePreprocess.fromJson(JsonReader json) {
    final type = json.optionalString('type') ?? 'image';
    if (type != 'image') {
      throw ManifestException('${json.at('type')} "$type" is not supported');
    }
    final scale = json.number('scale', 1 / 255);
    if (scale <= 0) {
      throw ManifestException('${json.at('scale')} must be positive');
    }
    final std = json.numbers('std', 3, const [1, 1, 1]);
    if (std.any((v) => v <= 0)) {
      throw ManifestException('${json.at('std')} values must be positive');
    }
    return ImagePreprocess(
      resize: ResizeSpec.fromJson(json.object('resize')),
      centerCrop: json.has('center_crop') ? _size(json, 'center_crop') : null,
      color: json.oneOf(
        'color',
        ColorOrder.values,
        (c) => c.jsonName,
        fallback: ColorOrder.rgb,
      ),
      scale: scale,
      mean: json.numbers('mean', 3, const [0, 0, 0]),
      std: std,
    );
  }

  final ResizeSpec resize;
  final ImageSize? centerCrop;
  final ColorOrder color;
  final double scale;
  final List<double> mean;
  final List<double> std;

  /// Final size if it is fixed, otherwise null.
  ImageSize? get outputSize => centerCrop ?? resize.size;
}

List<int> _shape(JsonReader json) {
  final shape = json.integers('shape');
  if (shape.any((d) => d == 0 || d < -1)) {
    throw ManifestException(
      '${json.at('shape')}: every dimension must be positive or -1',
    );
  }
  return List.unmodifiable(shape);
}

/// Name, element type, and shape of one model input or output.
abstract class TensorSpec {
  const TensorSpec({
    required this.name,
    required this.dtype,
    required this.shape,
  });

  final String name;
  final DType dtype;

  /// Dimensions. -1 means the size is only known at run time.
  final List<int> shape;

  /// Number of elements, or null if any dimension is dynamic.
  int? get elementCount =>
      shape.contains(-1) ? null : shape.fold<int>(1, (a, b) => a * b);
}

/// A model input with optional preprocessing.
class InputSpec extends TensorSpec {
  const InputSpec({
    required super.name,
    required super.dtype,
    required super.shape,
    this.layout,
    this.preprocess,
  });

  factory InputSpec.fromJson(JsonReader json) {
    final spec = InputSpec(
      name: json.string('name'),
      dtype: json.oneOf('dtype', DType.values, (d) => d.jsonName),
      shape: _shape(json),
      layout: json.has('layout')
          ? json.oneOf('layout', TensorLayout.values, (l) => l.jsonName)
          : null,
      preprocess: json.has('preprocess')
          ? ImagePreprocess.fromJson(json.object('preprocess'))
          : null,
    );
    spec._checkImageInput(json.path);
    return spec;
  }

  final TensorLayout? layout;
  final ImagePreprocess? preprocess;

  void _checkImageInput(String where) {
    final pre = preprocess;
    if (pre == null) return;
    final layout = this.layout;
    if (layout == null) {
      throw ManifestException(
        '$where: input "$name" has image preprocessing but no layout',
      );
    }
    if (shape.length != 4) {
      throw ManifestException(
        '$where: image input "$name" must have 4 dimensions',
      );
    }
    if (!const [DType.float32, DType.float16, DType.uint8].contains(dtype)) {
      throw ManifestException(
        '$where: image input "$name" must be float32, float16, or uint8',
      );
    }
    final (channels, height, width) = layout == TensorLayout.nchw
        ? (shape[1], shape[2], shape[3])
        : (shape[3], shape[1], shape[2]);
    if (channels != -1 && channels != 3) {
      throw ManifestException(
        '$where: image input "$name" must have 3 channels',
      );
    }
    final size = pre.outputSize;
    if (size == null) return;
    if ((height != -1 && height != size.height) ||
        (width != -1 && width != size.width)) {
      throw ManifestException(
        '$where: input "$name" is ${height}x$width, '
        'but preprocessing produces ${size.height}x${size.width}',
      );
    }
  }
}

/// A model output with an optional rule for turning it into results.
class OutputSpec extends TensorSpec {
  const OutputSpec({
    required super.name,
    required super.dtype,
    required super.shape,
    this.postprocess,
  });

  factory OutputSpec.fromJson(JsonReader json) => OutputSpec(
    name: json.string('name'),
    dtype: json.oneOf('dtype', DType.values, (d) => d.jsonName),
    shape: _shape(json),
    postprocess: json.has('postprocess')
        ? Postprocess.fromJson(json.object('postprocess'))
        : null,
  );

  final Postprocess? postprocess;
}
