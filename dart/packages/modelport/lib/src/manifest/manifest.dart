import 'dart:convert';

import '../errors.dart';
import 'files.dart';
import 'json_reader.dart';
import 'postprocess.dart';
import 'tensors.dart';

final _idPattern = RegExp(r'^[a-z0-9][a-z0-9._-]*$');
final _semverPattern = RegExp(
  r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$',
);
final _schemaPattern = RegExp(r'^modelport/(\d+)\.(\d+)$');

/// The spec version a manifest declares, such as `modelport/0.1`.
class SchemaVersion {
  const SchemaVersion(this.major, this.minor);

  /// Newest spec this package was written for.
  static const supported = SchemaVersion(0, 1);

  factory SchemaVersion.parse(String text) {
    final match = _schemaPattern.firstMatch(text);
    if (match == null) {
      throw ManifestException('schema "$text" is not a modelport spec version');
    }
    return SchemaVersion(int.parse(match[1]!), int.parse(match[2]!));
  }

  final int major;
  final int minor;

  /// True if this package fully knows this version; newer minors may add fields.
  bool get isFullySupported =>
      major == supported.major && minor <= supported.minor;

  @override
  String toString() => 'modelport/$major.$minor';
}

/// What a model does. Decides which task API can load it.
enum Task {
  imageClassification('image-classification'),
  objectDetection('object-detection'),
  textGeneration('text-generation');

  const Task(this.jsonName);
  final String jsonName;

  bool get usesTensors => this != Task.textGeneration;
}

/// Runtime names used in manifests.
///
/// Kept as strings so a manifest from a newer spec, with runtimes this package
/// does not know, still loads; those variants are simply never selected.
abstract final class Runtimes {
  static const onnx = 'onnx';
  static const executorch = 'executorch';
  static const llamacpp = 'llamacpp';
}

/// How close two outputs must be: |a - b| <= atol + rtol * |b|.
class Tolerance {
  const Tolerance({this.atol = 1e-3, this.rtol = 1e-3});

  factory Tolerance.fromJson(JsonReader json) {
    final atol = json.number('atol', 1e-3);
    final rtol = json.number('rtol', 1e-3);
    if (atol < 0 || rtol < 0) {
      throw ManifestException('${json.path}: tolerances must not be negative');
    }
    return Tolerance(atol: atol, rtol: rtol);
  }

  final double atol;
  final double rtol;

  bool allows(double actual, double expected) =>
      (actual - expected).abs() <= atol + rtol * expected.abs();
}

/// One concrete form of the model, such as ONNX fp32 or GGUF Q4_K_M.
class Variant {
  const Variant({
    required this.id,
    required this.runtime,
    this.backend,
    required this.precision,
    required this.file,
    this.extraFiles = const [],
    this.minRamMb,
    this.tolerance,
  });

  factory Variant.fromJson(JsonReader json) {
    final id = json.string('id');
    if (!_idPattern.hasMatch(id)) {
      throw ManifestException(
        '${json.at('id')} "$id" must be lowercase letters, digits, ".", "_" or "-"',
      );
    }
    final minRam = json.optionalInteger('min_ram_mb');
    if (minRam != null && minRam <= 0) {
      throw ManifestException('${json.at('min_ram_mb')} must be positive');
    }
    final extra = json.list('extra_files');
    return Variant(
      id: id,
      runtime: json.string('runtime'),
      backend: json.optionalString('backend'),
      precision: json.string('precision'),
      file: FileRef.fromJson(json.object('file')),
      extraFiles: [
        for (final (i, item) in extra.indexed)
          FileRef.fromJson(
            JsonReader.of(item, '${json.at('extra_files')}[$i]'),
          ),
      ],
      minRamMb: minRam,
      tolerance: json.has('tolerance')
          ? Tolerance.fromJson(json.object('tolerance'))
          : null,
    );
  }

  final String id;

  /// See [Runtimes].
  final String runtime;
  final String? backend;
  final String precision;
  final FileRef file;
  final List<FileRef> extraFiles;
  final int? minRamMb;

  /// Overrides the golden tolerance, for example for int8 variants.
  final Tolerance? tolerance;

  @override
  String toString() => 'Variant($id, $runtime $precision)';
}

/// Default sampling settings for text generation.
class GenerationDefaults {
  const GenerationDefaults({
    this.temperature = 0.7,
    this.topP = 0.9,
    this.topK,
    this.maxTokens = 512,
    this.repeatPenalty,
  });

  factory GenerationDefaults.fromJson(JsonReader json) => GenerationDefaults(
    temperature: json.number('temperature', 0.7),
    topP: json.number('top_p', 0.9),
    topK: json.optionalInteger('top_k'),
    maxTokens: json.optionalInteger('max_tokens') ?? 512,
    repeatPenalty: json.optionalNumber('repeat_penalty'),
  );

  final double temperature;
  final double topP;
  final int? topK;
  final int maxTokens;
  final double? repeatPenalty;
}

/// Settings for text-generation models.
class LlmConfig {
  const LlmConfig({
    required this.contextLength,
    this.chatTemplate = 'from_gguf',
    this.defaults = const GenerationDefaults(),
  });

  factory LlmConfig.fromJson(JsonReader json) {
    final context = json.integer('context_length');
    if (context <= 0) {
      throw ManifestException('${json.at('context_length')} must be positive');
    }
    return LlmConfig(
      contextLength: context,
      chatTemplate: json.optionalString('chat_template') ?? 'from_gguf',
      defaults: json.has('defaults')
          ? GenerationDefaults.fromJson(json.object('defaults'))
          : const GenerationDefaults(),
    );
  }

  final int contextLength;
  final String chatTemplate;
  final GenerationDefaults defaults;
}

/// Saved inputs and expected outputs, used to prove a device matches Python.
class Golden {
  const Golden({
    required this.inputs,
    required this.outputs,
    this.tolerance = const Tolerance(),
  });

  factory Golden.fromJson(JsonReader json) {
    Map<String, FileRef> files(String key) {
      final group = json.object(key);
      if (group.map.isEmpty) {
        throw ManifestException('${json.at(key)} must not be empty');
      }
      return {
        for (final name in group.map.keys)
          name: FileRef.fromJson(group.object(name)),
      };
    }

    return Golden(
      inputs: files('inputs'),
      outputs: files('outputs'),
      tolerance: json.has('tolerance')
          ? Tolerance.fromJson(json.object('tolerance'))
          : const Tolerance(),
    );
  }

  final Map<String, FileRef> inputs;
  final Map<String, FileRef> outputs;
  final Tolerance tolerance;
}

/// Everything an app needs to download, run, and interpret a model.
class Manifest {
  const Manifest({
    required this.schema,
    required this.id,
    required this.version,
    required this.task,
    required this.license,
    this.name,
    this.description,
    this.source,
    required this.variants,
    this.inputs = const [],
    this.outputs = const [],
    this.llm,
    this.golden,
  });

  /// Parses the text of a `modelport.json` file.
  factory Manifest.parse(String text) {
    final Object? data;
    try {
      data = jsonDecode(text);
    } on FormatException catch (error) {
      throw ManifestException('not valid JSON: ${error.message}');
    }
    return Manifest.fromJson(data);
  }

  factory Manifest.fromJson(Object? data) {
    final json = JsonReader.of(data, '');
    final schema = SchemaVersion.parse(json.string('schema'));
    if (schema.major != SchemaVersion.supported.major) {
      throw ManifestException(
        'this manifest uses $schema, but this package reads ${SchemaVersion.supported.major}.x',
        hint: 'Update the modelport package.',
      );
    }
    final id = json.string('id');
    if (!_idPattern.hasMatch(id)) {
      throw ManifestException(
        'id "$id" must be lowercase letters, digits, ".", "_" or "-"',
      );
    }
    final version = json.string('version');
    if (!_semverPattern.hasMatch(version)) {
      throw ManifestException(
        'version "$version" must be a semantic version like 1.0.0',
      );
    }
    final task = json.oneOf('task', Task.values, (t) => t.jsonName);

    List<T> items<T>(
      String key,
      T Function(JsonReader) parse, {
      bool required = false,
    }) => [
      for (final (i, item) in json.list(key, required: required).indexed)
        parse(JsonReader.of(item, '$key[$i]')),
    ];

    final manifest = Manifest(
      schema: schema,
      id: id,
      version: version,
      task: task,
      license: json.string('license'),
      name: json.optionalString('name'),
      description: json.optionalString('description'),
      source: json.optionalString('source'),
      variants: items('variants', Variant.fromJson, required: true),
      inputs: items('inputs', InputSpec.fromJson),
      outputs: items('outputs', OutputSpec.fromJson),
      llm: json.has('llm') ? LlmConfig.fromJson(json.object('llm')) : null,
      golden: json.has('golden')
          ? Golden.fromJson(json.object('golden'))
          : null,
    );
    manifest._checkConsistency();
    return manifest;
  }

  final SchemaVersion schema;
  final String id;
  final String version;
  final Task task;
  final String license;
  final String? name;
  final String? description;
  final String? source;
  final List<Variant> variants;
  final List<InputSpec> inputs;
  final List<OutputSpec> outputs;
  final LlmConfig? llm;
  final Golden? golden;

  void _checkConsistency() {
    if (variants.isEmpty) throw ManifestException('variants must not be empty');
    _requireUnique('variant id', variants.map((v) => v.id));
    _requireUnique('input name', inputs.map((i) => i.name));
    _requireUnique('output name', outputs.map((o) => o.name));

    if (task.usesTensors) {
      if (inputs.isEmpty || outputs.isEmpty) {
        throw ManifestException(
          'task "${task.jsonName}" needs inputs and outputs',
        );
      }
      if (llm != null) {
        throw ManifestException(
          'task "${task.jsonName}" must not have an "llm" section',
        );
      }
      if (variants.any((v) => v.runtime == Runtimes.llamacpp)) {
        throw ManifestException(
          'runtime "llamacpp" is only for task "text-generation"',
        );
      }
      if (!inputs.any((i) => i.preprocess != null)) {
        throw ManifestException(
          'task "${task.jsonName}" needs an input with image preprocessing',
        );
      }
      final wanted = task == Task.imageClassification
          ? 'classification'
          : 'detection';
      final found = outputs.any(
        (o) => switch (o.postprocess) {
          ClassificationPostprocess() => wanted == 'classification',
          DetectionPostprocess() => wanted == 'detection',
          _ => false,
        },
      );
      if (!found) {
        throw ManifestException(
          'task "${task.jsonName}" needs an output with "$wanted" postprocess',
        );
      }
    } else {
      if (llm == null) {
        throw ManifestException(
          'task "text-generation" needs an "llm" section',
        );
      }
      if (golden != null) {
        throw ManifestException('golden tests are only for tensor tasks');
      }
    }

    final outputNames = {for (final o in outputs) o.name};
    for (final output in outputs) {
      final post = output.postprocess;
      if (post is! DetectionPostprocess) continue;
      final boxes = post.boxesOutput;
      if (boxes != null &&
          (!outputNames.contains(boxes) || boxes == output.name)) {
        throw ManifestException(
          'output "${output.name}" boxes_output "$boxes" must name another output',
        );
      }
    }

    final goldenData = golden;
    if (goldenData != null) {
      final inputNames = {for (final i in inputs) i.name};
      final outputNames = {for (final o in outputs) o.name};
      for (final name in goldenData.inputs.keys) {
        if (!inputNames.contains(name)) {
          throw ManifestException('golden input "$name" is not a model input');
        }
      }
      for (final name in goldenData.outputs.keys) {
        if (!outputNames.contains(name)) {
          throw ManifestException(
            'golden output "$name" is not a model output',
          );
        }
      }
    }
  }

  static void _requireUnique(String label, Iterable<String> values) {
    final seen = <String>{};
    final duplicates = <String>{
      for (final v in values)
        if (!seen.add(v)) v,
    };
    if (duplicates.isNotEmpty) {
      throw ManifestException(
        'duplicate $label: ${(duplicates.toList()..sort()).join(', ')}',
      );
    }
  }

  /// Every file the manifest points to, in the same order as the Python CLI.
  List<FileRef> get files => [
    for (final variant in variants) ...[variant.file, ...variant.extraFiles],
    for (final output in outputs)
      if (output.postprocess case ClassificationPostprocess(:final labels?))
        labels
      else if (output.postprocess case DetectionPostprocess(:final labels?))
        labels,
    ...?golden?.inputs.values,
    ...?golden?.outputs.values,
  ];

  /// Input spec by name.
  InputSpec input(String name) => inputs.firstWhere(
    (i) => i.name == name,
    orElse: () => throw ModelPortException('model $id has no input "$name"'),
  );

  @override
  String toString() => 'Manifest($id $version, ${task.jsonName})';
}
