import '../errors.dart';
import '../manifest/manifest.dart';
import 'adapter.dart';

/// The adapter package that runs each known runtime.
const adapterPackages = {
  Runtimes.onnx: 'modelport_onnx',
  Runtimes.executorch: 'modelport_executorch',
  Runtimes.llamacpp: 'modelport_llamacpp',
};

/// Picks the first variant, in manifest order, that a registered adapter of
/// type [A] can run and that fits in [deviceRamMb].
///
/// Manifest order is the author's preference, so put the variant you would
/// like most devices to use first.
(Variant, A) selectVariant<A extends RuntimeAdapter>(
  Manifest manifest,
  Iterable<RuntimeAdapter> adapters, {
  int? deviceRamMb,
  String? variantId,
}) {
  final usable = adapters.whereType<A>().toList();
  A? adapterFor(Variant v) {
    for (final adapter in usable) {
      if (adapter.runtime == v.runtime && adapter.canRun(v)) return adapter;
    }
    return null;
  }

  if (variantId != null) {
    final variant = manifest.variants
        .where((v) => v.id == variantId)
        .firstOrNull;
    if (variant == null) {
      throw ModelPortException(
        '${manifest.id} has no variant "$variantId"',
        hint: 'Available: ${manifest.variants.map((v) => v.id).join(', ')}.',
      );
    }
    final adapter = adapterFor(variant);
    if (adapter == null) throw _noAdapter(manifest, [variant]);
    return (variant, adapter);
  }

  final tooBig = <Variant>[];
  for (final variant in manifest.variants) {
    final adapter = adapterFor(variant);
    if (adapter == null) continue;
    final minRam = variant.minRamMb;
    if (deviceRamMb != null && minRam != null && deviceRamMb < minRam) {
      tooBig.add(variant);
      continue;
    }
    return (variant, adapter);
  }

  if (tooBig.isNotEmpty) {
    throw ModelPortException(
      '${manifest.id} needs at least ${tooBig.map((v) => v.minRamMb!).reduce((a, b) => a < b ? a : b)} MB '
      'of RAM, this device has $deviceRamMb MB',
      hint: 'Choose a smaller model or a smaller variant.',
    );
  }
  throw _noAdapter(manifest, manifest.variants);
}

ModelPortException _noAdapter(Manifest manifest, List<Variant> variants) {
  final runtimes = {for (final v in variants) v.runtime};
  final packages = [
    for (final runtime in runtimes)
      adapterPackages[runtime] ?? 'an adapter for "$runtime"',
  ];
  return ModelPortException(
    'no registered adapter can run ${manifest.id} '
    '(variants: ${variants.map((v) => '${v.id} [${v.runtime}]').join(', ')})',
    hint:
        'Add ${packages.join(' or ')} to pubspec.yaml and register it at startup.',
  );
}
