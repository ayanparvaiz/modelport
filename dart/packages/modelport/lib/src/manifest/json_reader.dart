import '../errors.dart';

/// Reads typed values from decoded JSON and reports the path of any problem,
/// such as `variants[0].file.sha256 is required`.
class JsonReader {
  JsonReader(this.map, this.path);

  /// Wraps a decoded JSON value that must be an object.
  factory JsonReader.of(Object? value, String path) {
    if (value is! Map) throw ManifestException('$path must be an object');
    return JsonReader(value.cast<String, Object?>(), path);
  }

  final Map<String, Object?> map;
  final String path;

  String at(String key) => path.isEmpty ? key : '$path.$key';

  bool has(String key) => map[key] != null;

  T _typed<T>(String key, Object? value, String expected) {
    if (value is T) return value;
    throw ManifestException('${at(key)} must be $expected');
  }

  String string(String key) {
    final value = map[key];
    if (value == null) throw ManifestException('${at(key)} is required');
    final text = _typed<String>(key, value, 'a string');
    if (text.isEmpty) throw ManifestException('${at(key)} must not be empty');
    return text;
  }

  String? optionalString(String key) =>
      map[key] == null ? null : _typed<String>(key, map[key], 'a string');

  int integer(String key) {
    final value = map[key];
    if (value == null) throw ManifestException('${at(key)} is required');
    return _asInt(key, value);
  }

  int? optionalInteger(String key) =>
      map[key] == null ? null : _asInt(key, map[key]);

  int _asInt(String key, Object? value) {
    if (value is int) return value;
    if (value is double && value == value.truncateToDouble()) {
      return value.toInt();
    }
    throw ManifestException('${at(key)} must be an integer');
  }

  double number(String key, double fallback) {
    final value = map[key];
    if (value == null) return fallback;
    return _typed<num>(key, value, 'a number').toDouble();
  }

  double? optionalNumber(String key) => map[key] == null
      ? null
      : _typed<num>(key, map[key], 'a number').toDouble();

  bool boolean(String key, bool fallback) {
    final value = map[key];
    return value == null ? fallback : _typed<bool>(key, value, 'true or false');
  }

  List<Object?> list(String key, {bool required = false}) {
    final value = map[key];
    if (value == null) {
      if (required) throw ManifestException('${at(key)} is required');
      return const [];
    }
    return _typed<List<Object?>>(key, value, 'a list');
  }

  List<double> numbers(String key, int length, List<double> fallback) {
    if (map[key] == null) return fallback;
    final values = list(key);
    if (values.length != length) {
      throw ManifestException('${at(key)} must have $length values');
    }
    return [
      for (final (i, v) in values.indexed)
        if (v is num)
          v.toDouble()
        else
          throw ManifestException('${at(key)}[$i] must be a number'),
    ];
  }

  List<int> integers(String key) => [
    for (final (i, v) in list(key, required: true).indexed)
      if (v is int)
        v
      else
        throw ManifestException('${at(key)}[$i] must be an integer'),
  ];

  JsonReader object(String key) {
    if (map[key] == null) throw ManifestException('${at(key)} is required');
    return JsonReader.of(map[key], at(key));
  }

  JsonReader? optionalObject(String key) =>
      map[key] == null ? null : JsonReader.of(map[key], at(key));

  T oneOf<T extends Enum>(
    String key,
    List<T> values,
    String Function(T) name, {
    T? fallback,
  }) {
    final raw = map[key];
    if (raw == null) {
      if (fallback != null) return fallback;
      throw ManifestException('${at(key)} is required');
    }
    final text = _typed<String>(key, raw, 'a string');
    for (final value in values) {
      if (name(value) == text) return value;
    }
    final allowed = values.map(name).join(', ');
    throw ManifestException('${at(key)} is "$text", expected one of: $allowed');
  }
}
