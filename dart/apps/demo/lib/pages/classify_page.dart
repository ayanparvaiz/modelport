import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:modelport_flutter/modelport_flutter.dart';

import '../widgets/common.dart';
import '../zoo.dart';

class ClassifyPage extends StatefulWidget {
  const ClassifyPage({super.key});

  @override
  State<ClassifyPage> createState() => _ClassifyPageState();
}

class _ClassifyPageState extends State<ClassifyPage> {
  ZooModel _model = classifiers.first;
  Engine _engine = classifierEngines.first;
  ImageClassifier? _classifier;
  String? _loadedKey;
  Uint8List? _photo;
  List<Classification> _results = const [];
  GoldenReport? _golden;
  DownloadProgress? _progress;
  String _status = 'Pick a photo or use the sample.';
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    samplePhoto().then((bytes) => setState(() => _photo = bytes));
  }

  @override
  void dispose() {
    _classifier?.close();
    super.dispose();
  }

  Future<ImageClassifier> _load() async {
    final key = '${_model.id}/${_engine.variantId}';
    final current = _classifier;
    if (current != null && _loadedKey == key) return current;
    await current?.close();
    _classifier = null;
    final watch = Stopwatch()..start();
    final classifier = await ImageClassifier.load(
      _model.location,
      variantId: _engine.variantId,
      onProgress: (p) => setState(() => _progress = p),
    );
    _classifier = classifier;
    _loadedKey = key;
    setState(
      () => _status =
          'Loaded ${_engine.label} in ${watch.elapsedMilliseconds} ms',
    );
    return classifier;
  }

  Future<void> _run(Future<void> Function() action) async {
    if (_busy) return;
    setState(() => _busy = true);
    try {
      await action();
    } on Object catch (error) {
      if (mounted) showError(context, error);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _classify() => _run(() async {
    final photo = _photo;
    if (photo == null) return;
    final classifier = await _load();
    final watch = Stopwatch()..start();
    final results = await classifier.classify(photo, topK: 5);
    setState(() {
      _results = results;
      _golden = null;
      _status =
          'Classified in ${watch.elapsedMilliseconds} ms with ${_engine.label}';
    });
  });

  Future<void> _check() => _run(() async {
    final classifier = await _load();
    final report = await classifier.model.checkGolden();
    setState(() => _golden = report);
  });

  @override
  Widget build(BuildContext context) {
    final photo = _photo;
    final golden = _golden;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        DropdownButton<ZooModel>(
          value: _model,
          isExpanded: true,
          items: [
            for (final m in classifiers)
              DropdownMenuItem(value: m, child: Text('${m.title} · ${m.note}')),
          ],
          onChanged: _busy ? null : (m) => setState(() => _model = m!),
        ),
        const SizedBox(height: 8),
        SegmentedButton<Engine>(
          segments: [
            for (final e in classifierEngines)
              ButtonSegment(value: e, label: Text(e.label)),
          ],
          selected: {_engine},
          onSelectionChanged: _busy
              ? null
              : (s) => setState(() => _engine = s.first),
        ),
        const SizedBox(height: 12),
        if (photo != null)
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Image.memory(photo, height: 240, fit: BoxFit.cover),
          ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          children: [
            OutlinedButton.icon(
              onPressed: _busy
                  ? null
                  : () async {
                      final sample = await samplePhoto();
                      setState(() => _photo = sample);
                    },
              icon: const Icon(Icons.pets),
              label: const Text('Sample'),
            ),
            OutlinedButton.icon(
              onPressed: _busy
                  ? null
                  : () async {
                      final picked = await pickPhoto();
                      if (picked != null) setState(() => _photo = picked);
                    },
              icon: const Icon(Icons.photo_library),
              label: const Text('Pick photo'),
            ),
            FilledButton.icon(
              onPressed: _busy ? null : _classify,
              icon: const Icon(Icons.auto_awesome),
              label: const Text('Classify'),
            ),
          ],
        ),
        const SizedBox(height: 8),
        StatusLine(text: _busy ? 'Working…' : _status, progress: _progress),
        for (final r in _results)
          ListTile(
            dense: true,
            title: Text(r.label),
            subtitle: LinearProgressIndicator(value: r.score),
            trailing: Text('${(r.score * 100).toStringAsFixed(1)}%'),
          ),
        if (_results.isNotEmpty)
          TextButton.icon(
            onPressed: _busy ? null : _check,
            icon: const Icon(Icons.verified_outlined),
            label: const Text('Check this device against Python'),
          ),
        if (golden != null) GoldenCard(report: golden),
      ],
    );
  }
}
