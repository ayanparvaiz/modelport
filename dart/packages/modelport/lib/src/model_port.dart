import 'dart:io';
import 'dart:typed_data';

import 'package:meta/meta.dart';

import 'errors.dart';
import 'manifest/files.dart';
import 'manifest/manifest.dart';
import 'manifest/postprocess.dart';
import 'manifest/tensors.dart';
import 'postprocess/classification.dart';
import 'preprocess/rgb_image.dart';
import 'runtime/adapter.dart';
import 'runtime/select.dart';
import 'store/location.dart';
import 'store/model_store.dart';
import 'tensor.dart';

/// The entry point: register adapters once, then load models by location.
///
/// ```dart
/// ModelPort.configure(
///   store: ModelStore(root: Directory('cache')),
///   adapters: [OnnxAdapter()],
/// );
/// final model = await ModelPort.load('hf://org/model');
/// ```
///
/// In Flutter, `ModelPortFlutter.init()` from `modelport_flutter` does this
/// for you and picks the app's cache folder and the device's RAM.
/// Decodes JPEG, PNG, and other image bytes into RGB pixels.
typedef ImageDecoder = Future<RgbImage> Function(Uint8List bytes);

abstract final class ModelPort {
  static final List<RuntimeAdapter> _adapters = [];
  static ModelStore? _store;

  /// Decoder used by task APIs. Null means the pure Dart decoder from
  /// `package:image`, run in a background isolate. `modelport_flutter` sets
  /// the much faster native decoder of the Flutter engine.
  static ImageDecoder? imageDecoder;

  /// Device memory used to skip variants that need more. Null means unknown.
  static int? deviceRamMb;

  static void configure({
    required ModelStore store,
    Iterable<RuntimeAdapter> adapters = const [],
    int? deviceRamMb,
  }) {
    _store = store;
    adapters.forEach(register);
    ModelPort.deviceRamMb = deviceRamMb;
  }

  /// Adds an adapter, replacing an earlier one of the same type.
  static void register(RuntimeAdapter adapter) {
    _adapters.removeWhere((a) => a.runtimeType == adapter.runtimeType);
    _adapters.add(adapter);
  }

  static List<RuntimeAdapter> get adapters => List.unmodifiable(_adapters);

  static ModelStore get store =>
      _store ??
      (throw ModelPortException(
        'ModelPort is not configured',
        hint:
            'In Flutter, call await ModelPortFlutter.init(). In plain Dart, call '
            'ModelPort.configure(store: ModelStore(root: Directory(...))).',
      ));

  /// Forgets adapters and the store. For tests.
  @visibleForTesting
  static void reset() {
    _adapters.clear();
    _store = null;
    deviceRamMb = null;
    imageDecoder = null;
  }

  /// Loads a tensor model: downloads what is missing, verifies it, and opens it
  /// with the first registered adapter that can run one of its variants.
  static Future<TensorModel> load(
    String location, {
    String? variantId,
    void Function(DownloadProgress)? onProgress,
    CancelToken? cancel,
  }) async {
    final where = ModelLocation.parse(location);
    final manifest = await store.manifest(where);
    if (!manifest.task.usesTensors) {
      throw ModelPortException(
        '${manifest.id} is a ${manifest.task.jsonName} model',
        hint: 'Load it with TextGenerator.load().',
      );
    }
    final (variant, adapter) = selectVariant<TensorAdapter>(
      manifest,
      _adapters,
      deviceRamMb: deviceRamMb,
      variantId: variantId,
    );
    final labels = [for (final output in manifest.outputs) ?_labelsOf(output)];
    final (loaded, files) = await _fetch(
      where,
      manifest,
      variant,
      labels,
      onProgress,
      cancel,
    );
    final session = await adapter.open(loaded);
    return TensorModel._(where, manifest, variant, session, files);
  }

  /// Loads a language model. See [TextModel].
  static Future<TextModel> loadText(
    String location, {
    String? variantId,
    void Function(DownloadProgress)? onProgress,
    CancelToken? cancel,
  }) async {
    final where = ModelLocation.parse(location);
    final manifest = await store.manifest(where);
    if (manifest.task != Task.textGeneration) {
      throw ModelPortException(
        '${manifest.id} is a ${manifest.task.jsonName} model, not text generation',
        hint: 'Load it with ModelPort.load() or the matching task API.',
      );
    }
    final (variant, adapter) = selectVariant<TextGenerationAdapter>(
      manifest,
      _adapters,
      deviceRamMb: deviceRamMb,
      variantId: variantId,
    );
    final (loaded, _) = await _fetch(
      where,
      manifest,
      variant,
      const [],
      onProgress,
      cancel,
    );
    final session = await adapter.open(loaded);
    return TextModel._(manifest, variant, session);
  }

  static Future<(LoadedVariant, Map<FileRef, File>)> _fetch(
    ModelLocation where,
    Manifest manifest,
    Variant variant,
    List<FileRef> extra,
    void Function(DownloadProgress)? onProgress,
    CancelToken? cancel,
  ) async {
    final files = await store.fetch(
      where,
      manifest,
      [variant.file, ...variant.extraFiles, ...extra],
      onProgress: onProgress,
      cancel: cancel,
    );
    final loaded = LoadedVariant(
      manifest: manifest,
      variant: variant,
      modelFile: files[variant.file]!,
      extraFiles: [for (final ref in variant.extraFiles) files[ref]!],
    );
    return (loaded, files);
  }
}

FileRef? _labelsOf(OutputSpec output) => switch (output.postprocess) {
  ClassificationPostprocess(:final labels) => labels,
  DetectionPostprocess(:final labels) => labels,
  _ => null,
};

/// A loaded tensor model, ready to run.
class TensorModel {
  TensorModel._(
    this.location,
    this.manifest,
    this.variant,
    this._session,
    this._files,
  );

  final ModelLocation location;
  final Manifest manifest;

  /// The variant that was picked for this device.
  final Variant variant;
  final TensorSession _session;
  final Map<FileRef, File> _files;
  bool _closed = false;

  /// Runs the model on inputs keyed by manifest name, after checking their
  /// dtype and shape against the manifest.
  Future<Map<String, Tensor>> run(Map<String, Tensor> inputs) async {
    if (_closed) throw ModelPortException('${manifest.id} is closed');
    for (final name in inputs.keys) {
      if (!manifest.inputs.any((i) => i.name == name)) {
        throw ModelPortException('${manifest.id} has no input "$name"');
      }
    }
    for (final spec in manifest.inputs) {
      final tensor = inputs[spec.name];
      if (tensor == null) {
        throw ModelPortException('missing input "${spec.name}"');
      }
      if (tensor.dtype != spec.dtype) {
        throw ModelPortException(
          'input "${spec.name}" must be ${spec.dtype.jsonName}, got ${tensor.dtype.jsonName}',
        );
      }
      final shapeOk =
          tensor.shape.length == spec.shape.length &&
          [
            for (var i = 0; i < spec.shape.length; i++)
              spec.shape[i] == -1 || spec.shape[i] == tensor.shape[i],
          ].every((ok) => ok);
      if (!shapeOk) {
        throw ModelPortException(
          'input "${spec.name}" must have shape ${spec.shape}, got ${tensor.shape}',
        );
      }
    }
    final outputs = await _session.run(inputs);
    for (final spec in manifest.outputs) {
      if (!outputs.containsKey(spec.name)) {
        throw ModelPortException(
          'the ${variant.runtime} adapter returned no output "${spec.name}"',
        );
      }
    }
    return outputs;
  }

  /// Class names for [output], or null if the manifest lists none.
  Future<List<String>?> labelsFor(OutputSpec output) async {
    final ref = _labelsOf(output);
    if (ref == null) return null;
    return parseLabels(await _files[ref]!.readAsString());
  }

  /// Runs the bundle's golden input on this device and compares the result
  /// with the output Python recorded. This proves the whole chain, from
  /// export to engine, gives the same answer on the phone.
  Future<GoldenReport> checkGolden({
    void Function(DownloadProgress)? onProgress,
  }) async {
    final golden = manifest.golden;
    if (golden == null) {
      throw ModelPortException(
        '${manifest.id} has no golden data',
        hint: 'Export it with modelport export.',
      );
    }
    final files = await ModelPort.store.fetch(location, manifest, [
      ...golden.inputs.values,
      ...golden.outputs.values,
    ], onProgress: onProgress);
    Future<Tensor> load(TensorSpec spec, FileRef ref) async => Tensor.fromBytes(
      spec.dtype,
      _fixedShape(spec, ref),
      await files[ref]!.readAsBytes(),
    );

    final inputs = {
      for (final spec in manifest.inputs)
        if (golden.inputs[spec.name] case final ref?)
          spec.name: await load(spec, ref),
    };
    final actual = await run(inputs);
    final tolerance = variant.tolerance ?? golden.tolerance;
    final checks = <GoldenCheck>[];
    for (final spec in manifest.outputs) {
      final ref = golden.outputs[spec.name];
      if (ref == null) continue;
      final want = (await load(spec, ref)).toDoubles();
      final got = actual[spec.name]!.toDoubles();
      if (got.length != want.length) {
        throw ModelPortException(
          'output "${spec.name}" has ${got.length} values, golden has ${want.length}',
        );
      }
      var worst = 0.0;
      var within = true;
      for (var i = 0; i < got.length; i++) {
        final diff = (got[i] - want[i]).abs();
        if (diff > worst) worst = diff;
        if (!tolerance.allows(got[i], want[i])) within = false;
      }
      bool? top1;
      if (spec.postprocess is ClassificationPostprocess) {
        top1 = _argmax(got) == _argmax(want);
      }
      checks.add(
        GoldenCheck(
          output: spec.name,
          maxAbsDiff: worst,
          withinTolerance: within,
          top1Match: top1,
        ),
      );
    }
    return GoldenReport(
      variantId: variant.id,
      tolerance: tolerance,
      outputs: checks,
    );
  }

  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    await _session.close();
  }

  static List<int> _fixedShape(TensorSpec spec, FileRef ref) {
    final dynamic = spec.shape.where((d) => d == -1).length;
    if (dynamic == 0) return spec.shape;
    final known = spec.shape
        .where((d) => d != -1)
        .fold<int>(1, (a, b) => a * b);
    final count = ref.size ~/ spec.dtype.bytesPerElement;
    if (dynamic > 1 || count % known != 0) return [count];
    return [for (final d in spec.shape) d == -1 ? count ~/ known : d];
  }

  static int _argmax(List<double> values) {
    var best = 0;
    for (var i = 1; i < values.length; i++) {
      if (values[i] > values[best]) best = i;
    }
    return best;
  }
}

/// One output compared with its golden value.
class GoldenCheck {
  const GoldenCheck({
    required this.output,
    required this.maxAbsDiff,
    required this.withinTolerance,
    this.top1Match,
  });

  final String output;
  final double maxAbsDiff;
  final bool withinTolerance;

  /// For classifier outputs, whether the best class is the same.
  final bool? top1Match;

  bool get passed => withinTolerance && top1Match != false;
}

/// The result of [TensorModel.checkGolden].
class GoldenReport {
  const GoldenReport({
    required this.variantId,
    required this.tolerance,
    required this.outputs,
  });

  final String variantId;
  final Tolerance tolerance;
  final List<GoldenCheck> outputs;

  bool get passed => outputs.isNotEmpty && outputs.every((o) => o.passed);

  @override
  String toString() => [
    'Golden check for $variantId: ${passed ? 'PASS' : 'FAIL'}',
    for (final o in outputs)
      '  ${o.output}: max diff ${o.maxAbsDiff.toStringAsExponential(2)} '
          '(allowed ${tolerance.atol} + ${tolerance.rtol}·|x|)'
          '${o.top1Match == null ? '' : ', top-1 ${o.top1Match! ? 'same' : 'different'}'}',
  ].join('\n');
}

/// A loaded language model.
class TextModel {
  TextModel._(this.manifest, this.variant, this._session);

  final Manifest manifest;
  final Variant variant;
  final TextGenerationSession _session;

  /// The manifest's default sampling settings.
  GenerationConfig get defaults =>
      GenerationConfig.fromDefaults(manifest.llm!.defaults);

  Stream<String> chat(List<ChatMessage> messages, {GenerationConfig? config}) =>
      _session.chat(messages, config ?? defaults);

  Future<void> cancel() => _session.cancel();
  Future<void> close() => _session.close();
}
