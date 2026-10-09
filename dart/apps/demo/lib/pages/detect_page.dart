import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:modelport_flutter/modelport_flutter.dart';

import '../widgets/common.dart';
import '../zoo.dart';

class DetectPage extends StatefulWidget {
  const DetectPage({super.key});

  @override
  State<DetectPage> createState() => _DetectPageState();
}

class _DetectPageState extends State<DetectPage> {
  Engine _engine = detectorEngines.first;
  ObjectDetector? _detector;
  String? _loadedVariant;
  Uint8List? _photo;
  Size? _photoSize;
  List<Detection> _found = const [];
  DownloadProgress? _progress;
  String _status = 'Detect objects in the sample or your own photo.';
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    samplePhoto().then(_setPhoto);
  }

  @override
  void dispose() {
    _detector?.close();
    super.dispose();
  }

  Future<void> _setPhoto(Uint8List bytes) async {
    final image = await decodeImageFromList(bytes);
    setState(() {
      _photo = bytes;
      _photoSize = Size(image.width.toDouble(), image.height.toDouble());
      _found = const [];
    });
    image.dispose();
  }

  Future<ObjectDetector> _load() async {
    final current = _detector;
    if (current != null && _loadedVariant == _engine.variantId) return current;
    await current?.close();
    _detector = null;
    final detector = await ObjectDetector.load(
      detector_.location,
      variantId: _engine.variantId,
      onProgress: (p) => setState(() => _progress = p),
    );
    _detector = detector;
    _loadedVariant = _engine.variantId;
    return detector;
  }

  Future<void> _detect() async {
    final photo = _photo;
    if (photo == null || _busy) return;
    setState(() => _busy = true);
    try {
      final detector = await _load();
      final watch = Stopwatch()..start();
      final found = await detector.detect(photo, minScore: 0.5);
      setState(() {
        _found = found;
        _status =
            '${found.length} objects in ${watch.elapsedMilliseconds} ms with ${_engine.label}';
      });
    } on Object catch (error) {
      if (mounted) showError(context, error);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final photo = _photo;
    final size = _photoSize;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text(
          '${detector_.title} · ${detector_.note}',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 8),
        SegmentedButton<Engine>(
          segments: [
            for (final e in detectorEngines)
              ButtonSegment(value: e, label: Text(e.label)),
          ],
          selected: {_engine},
          onSelectionChanged: _busy
              ? null
              : (s) => setState(() => _engine = s.first),
        ),
        const SizedBox(height: 12),
        if (photo != null && size != null)
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: FittedBox(
              child: SizedBox(
                width: size.width,
                height: size.height,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.memory(photo, fit: BoxFit.fill),
                    CustomPaint(painter: _BoxPainter(_found, size.width / 300)),
                  ],
                ),
              ),
            ),
          ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          children: [
            OutlinedButton.icon(
              onPressed: _busy
                  ? null
                  : () async => _setPhoto(await samplePhoto()),
              icon: const Icon(Icons.pets),
              label: const Text('Sample'),
            ),
            OutlinedButton.icon(
              onPressed: _busy
                  ? null
                  : () async {
                      final picked = await pickPhoto();
                      if (picked != null) await _setPhoto(picked);
                    },
              icon: const Icon(Icons.photo_library),
              label: const Text('Pick photo'),
            ),
            FilledButton.icon(
              onPressed: _busy ? null : _detect,
              icon: const Icon(Icons.center_focus_strong),
              label: const Text('Detect'),
            ),
          ],
        ),
        const SizedBox(height: 8),
        StatusLine(text: _busy ? 'Working…' : _status, progress: _progress),
        for (final d in _found)
          ListTile(
            dense: true,
            leading: const Icon(Icons.crop_free),
            title: Text(d.label),
            trailing: Text('${(d.score * 100).toStringAsFixed(0)}%'),
          ),
      ],
    );
  }
}

// Avoid clashing with the ObjectDetector instance field above.
const detector_ = detector;

class _BoxPainter extends CustomPainter {
  _BoxPainter(this.found, this.scale);

  final List<Detection> found;
  final double scale;

  @override
  void paint(Canvas canvas, Size size) {
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3 * scale
      ..color = Colors.amberAccent;
    for (final d in found) {
      final rect = Rect.fromLTRB(
        d.box.left,
        d.box.top,
        d.box.right,
        d.box.bottom,
      );
      canvas.drawRect(rect, stroke);
      final label = TextPainter(
        text: TextSpan(
          text: ' ${d.label} ${(d.score * 100).toStringAsFixed(0)}% ',
          style: TextStyle(
            color: Colors.black,
            backgroundColor: Colors.amberAccent,
            fontSize: 14 * scale,
          ),
        ),
        textDirection: ui.TextDirection.ltr,
      )..layout();
      label.paint(canvas, rect.topLeft);
    }
  }

  @override
  bool shouldRepaint(_BoxPainter old) => old.found != found;
}
