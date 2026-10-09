import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:image/image.dart' as img;
import 'package:modelport/modelport.dart';
import 'package:test/test.dart';

import 'support/fake_bundle.dart';

Tensor t3(int n, int w, List<double> v) =>
    Tensor.float32([1, n, w], Float32List.fromList(v));

void expectBox(BoundingBox box, List<double> expected) {
  expect(
    [box.left, box.top, box.right, box.bottom],
    [for (final e in expected) closeTo(e, 1e-6)],
  );
}

const detrRule = DetectionPostprocess(
  format: DetectionFormat.detr,
  boxesOutput: 'pred_boxes',
  activation: Activation.softmax,
  backgroundClass: true,
  normalized: true,
  scoreThreshold: 0.5,
  iouThreshold: 1,
);
const logitsSpec = OutputSpec(
  name: 'logits',
  dtype: DType.float32,
  shape: [1, 3, 3],
);

void main() {
  group('InputMapping', () {
    test('exact resize scales each axis', () {
      final m = InputMapping.forImage(
        200,
        100,
        const ImagePreprocess(
          resize: ResizeSpec(size: (height: 50, width: 50)),
        ),
      );
      expectBox(m.toImage(10, 10, 20, 20), [40, 20, 80, 40]);
    });

    test('shorter side and center crop are undone', () {
      final m = InputMapping.forImage(
        300,
        200,
        const ImagePreprocess(
          resize: ResizeSpec(shorterSide: 100),
          centerCrop: (height: 100, width: 100),
        ),
      );
      // resized to 150x100, cropped 25 px from the left
      expectBox(m.toImage(0, 0, 100, 100), [50, 0, 250, 200]);
    });

    test('padding for small images is undone and boxes are clamped', () {
      final m = InputMapping.forImage(
        10,
        10,
        const ImagePreprocess(
          resize: ResizeSpec(size: (height: 10, width: 10)),
          centerCrop: (height: 20, width: 20),
        ),
      );
      expectBox(m.toImage(5, 5, 15, 15), [0, 0, 10, 10]);
      expectBox(m.toImage(6, 6, 8, 8), [1, 1, 3, 3]);
    });
  });

  group('detectObjects', () {
    final mapping = InputMapping.forImage(
      100,
      50,
      const ImagePreprocess(resize: ResizeSpec(size: (height: 10, width: 10))),
    );

    test('DETR outputs: softmax, background dropped, normalized boxes', () {
      final outputs = {
        'logits': t3(3, 3, [
          5, 0, 0, // confident class 0
          0, 0, 5, // background
          0, 0.5, 0, // weak class 1, score 0.45
        ]),
        'pred_boxes': t3(3, 4, [
          0.5,
          0.5,
          0.2,
          0.4,
          0,
          0,
          1,
          1,
          0.1,
          0.1,
          0.1,
          0.1,
        ]),
      };
      final found = detectObjects(
        outputs,
        logitsSpec,
        detrRule,
        mapping,
        labels: ['cat', 'dog'],
      );
      expect(found, hasLength(1));
      expect(found.single.label, 'cat');
      expect(found.single.score, greaterThan(0.98));
      expectBox(found.single.box, [40, 15, 60, 35]);
    });

    test('a lower score threshold returns more', () {
      final outputs = {
        'logits': t3(1, 3, [0, 0.5, 0]), // class 1 scores 0.45
        'pred_boxes': t3(1, 4, [0.5, 0.5, 0.2, 0.2]),
      };
      expect(detectObjects(outputs, logitsSpec, detrRule, mapping), isEmpty);
      expect(
        detectObjects(
          outputs,
          logitsSpec,
          detrRule,
          mapping,
          scoreThreshold: 0.3,
        ),
        hasLength(1),
      );
    });

    test('rows with objectness, pixel xyxy boxes, and per-class NMS', () {
      const rule = DetectionPostprocess(
        boxFormat: BoxFormat.xyxy,
        hasObjectness: true,
        scoreThreshold: 0.3,
        iouThreshold: 0.5,
      );
      const spec = OutputSpec(
        name: 'out',
        dtype: DType.float32,
        shape: [1, 3, 7],
      );
      final outputs = {
        'out': t3(3, 7, [
          0, 0, 5, 5, 0.9, 0.9, 0.1, // class 0, score 0.81
          0,
          0,
          5,
          4,
          0.8,
          0.9,
          0.1, // overlaps the first, same class: suppressed
          0, 0, 5, 5, 0.9, 0.1, 0.8, // same place, class 1: kept
        ]),
      };
      final found = detectObjects(outputs, spec, rule, mapping);
      expect(found.map((d) => d.index), [0, 1]);
      expect(found.first.score, closeTo(0.81, 1e-6));
      expectBox(found.first.box, [0, 0, 50, 25]);
    });

    test('maxDetections keeps the best', () {
      final outputs = {
        'logits': t3(3, 3, [5, 0, 0, 6, 0, 0, 7, 0, 0]),
        'pred_boxes': t3(3, 4, List.filled(12, 0.5)),
      };
      final found = detectObjects(
        outputs,
        logitsSpec,
        detrRule,
        mapping,
        maxDetections: 1,
      );
      expect(found, hasLength(1));
      expect(found.single.score, greaterThan(0.99));
    });

    test('iou', () {
      expect(
        const BoundingBox(0, 0, 2, 2).iou(const BoundingBox(1, 0, 3, 2)),
        closeTo(1 / 3, 1e-9),
      );
      expect(
        const BoundingBox(0, 0, 1, 1).iou(const BoundingBox(2, 2, 3, 3)),
        0,
      );
    });
  });

  group('ObjectDetector', () {
    late Directory cache;
    late FakeServer server;
    const base = 'https://models.test/detector';

    setUp(() {
      cache = Directory.systemTemp.createTempSync('modelport_detect_');
      server = FakeServer();
      final weights = Uint8List.fromList(List.generate(64, (i) => i));
      final labels = Uint8List.fromList(utf8.encode('cat\ndog\n'));
      Map<String, Object?> ref(String path, Uint8List bytes) => {
        'path': path,
        'size': bytes.length,
        'sha256': sha256.convert(bytes).toString(),
      };
      server.files['$base/model.onnx'] = weights;
      server.files['$base/labels.txt'] = labels;
      server.files['$base/modelport.json'] = Uint8List.fromList(
        utf8.encode(
          jsonEncode({
            'schema': 'modelport/0.1',
            'id': 'detector',
            'version': '1.0.0',
            'task': 'object-detection',
            'license': 'MIT',
            'variants': [
              {
                'id': 'onnx-fp32',
                'runtime': 'onnx',
                'precision': 'fp32',
                'file': ref('model.onnx', weights),
              },
            ],
            'inputs': [
              {
                'name': 'pixel_values',
                'dtype': 'float32',
                'shape': [1, 3, 8, 8],
                'layout': 'NCHW',
                'preprocess': {
                  'resize': {
                    'size': [8, 8],
                  },
                },
              },
            ],
            'outputs': [
              {
                'name': 'logits',
                'dtype': 'float32',
                'shape': [1, 1, 3],
                'postprocess': {
                  'type': 'detection',
                  'format': 'detr',
                  'boxes_output': 'pred_boxes',
                  'activation': 'softmax',
                  'background_class': true,
                  'normalized': true,
                  'score_threshold': 0.5,
                  'iou_threshold': 1,
                  'labels': ref('labels.txt', labels),
                },
              },
              {
                'name': 'pred_boxes',
                'dtype': 'float32',
                'shape': [1, 1, 4],
              },
            ],
          }),
        ),
      );
      ModelPort.reset();
      ModelPort.configure(
        store: ModelStore(root: cache, client: server.client),
        adapters: [_FixedDetections()],
      );
    });

    tearDown(() {
      ModelPort.reset();
      cache.deleteSync(recursive: true);
    });

    test('detects in original image pixels', () async {
      final detector = await ObjectDetector.load(base);
      final png = img.encodePng(img.Image(width: 64, height: 32));
      final found = await detector.detect(png);
      expect(found.single.label, 'dog');
      expectBox(found.single.box, [16, 8, 48, 24]);
      await detector.close();
    });

    test('classifiers are refused', () async {
      server.serve('https://models.test/toy', FakeBundle());
      expect(
        ObjectDetector.load('https://models.test/toy'),
        throwsA(isA<ModelPortException>()),
      );
    });
  });
}

class _FixedDetections implements TensorAdapter {
  @override
  String get runtime => Runtimes.onnx;

  @override
  bool canRun(Variant variant) => true;

  @override
  Future<TensorSession> open(LoadedVariant model) async => _Session();
}

class _Session implements TensorSession {
  @override
  Future<Map<String, Tensor>> run(Map<String, Tensor> inputs) async => {
    'logits': t3(1, 3, [0, 6, 0]),
    'pred_boxes': t3(1, 4, [0.5, 0.5, 0.5, 0.5]),
  };

  @override
  Future<void> close() async {}
}
