import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:image/image.dart' as img;
import 'package:modelport/modelport.dart';
import 'package:test/test.dart';

import 'support/fake_adapters.dart';
import 'support/fake_bundle.dart';

Uint8List solidPng(int r, int g, int b) => img.encodePng(
  img.Image(width: 8, height: 8)..clear(img.ColorRgb8(r, g, b)),
);

void main() {
  const base = 'https://models.test/toy';
  late Directory cache;
  late FakeServer server;
  late ChannelMeanAdapter adapter;

  setUp(() {
    cache = Directory.systemTemp.createTempSync('modelport_api_');
    server = FakeServer();
    adapter = ChannelMeanAdapter();
    ModelPort.reset();
    ModelPort.configure(
      store: ModelStore(root: cache, client: server.client),
      adapters: [adapter, EchoTextAdapter()],
    );
  });

  tearDown(() {
    ModelPort.reset();
    cache.deleteSync(recursive: true);
  });

  test('not configured gives a helpful error', () {
    ModelPort.reset();
    expect(
      () => ModelPort.store,
      throwsA(
        isA<ModelPortException>().having(
          (e) => e.hint,
          'hint',
          contains('ModelPortFlutter.init'),
        ),
      ),
    );
  });

  group('ImageClassifier', () {
    setUp(() => server.serve(base, FakeBundle()));

    test('classifies an encoded image end to end', () async {
      final classifier = await ImageClassifier.load(base);
      expect(classifier.labels, ['red', 'green', 'blue']);
      final results = await classifier.classify(solidPng(255, 0, 0));
      expect(results.first.label, 'red');
      expect(results.first.score, greaterThan(results[1].score));
      expect(adapter.opened, 1);
      await classifier.close();
    });

    test('topK limits the results', () async {
      final classifier = await ImageClassifier.load(base);
      expect(
        await classifier.classify(solidPng(0, 0, 255), topK: 1),
        hasLength(1),
      );
    });

    test('a closed model refuses to run', () async {
      final classifier = await ImageClassifier.load(base);
      await classifier.close();
      expect(
        classifier.classify(solidPng(1, 2, 3)),
        throwsA(isA<ModelPortException>()),
      );
    });

    test('inputs are checked against the manifest', () async {
      final model = await ModelPort.load(base);
      expect(model.run({}), throwsA(isA<ModelPortException>()));
      expect(
        model.run({
          'pixel_values': Tensor.float32([1, 3, 2, 2], Float32List(12)),
        }),
        throwsA(
          isA<ModelPortException>().having(
            (e) => e.message,
            'message',
            contains('shape'),
          ),
        ),
      );
      expect(
        model.run({
          'pixel_values': Tensor.uint8([1, 3, 4, 4], Uint8List(48)),
        }),
        throwsA(
          isA<ModelPortException>().having(
            (e) => e.message,
            'message',
            contains('float32'),
          ),
        ),
      );
      expect(
        model.run({
          'nope': Tensor.float32([1], Float32List(1)),
        }),
        throwsA(isA<ModelPortException>()),
      );
    });
  });

  group('golden check', () {
    final input = Float32List.fromList(
      List.generate(48, (i) => i < 16 ? 1.0 : 0.0),
    );

    test('passes when the device matches Python', () async {
      server.serve(
        base,
        FakeBundle()..addGolden(input, Float32List.fromList([1, 0, 0])),
      );
      final model = await ModelPort.load(base);
      final report = await model.checkGolden();
      expect(report.passed, isTrue, reason: '$report');
      expect(report.outputs.single.top1Match, isTrue);
      expect(report.toString(), contains('PASS'));
    });

    test('fails when outputs drift beyond the tolerance', () async {
      server.serve(
        base,
        FakeBundle()..addGolden(input, Float32List.fromList([1, 0.5, 0])),
      );
      final model = await ModelPort.load(base);
      final report = await model.checkGolden();
      expect(report.passed, isFalse);
      expect(report.outputs.single.maxAbsDiff, closeTo(0.5, 1e-6));
    });

    test('needs golden data', () async {
      server.serve(base, FakeBundle());
      final model = await ModelPort.load(base);
      expect(model.checkGolden(), throwsA(isA<ModelPortException>()));
    });
  });

  group('TextGenerator', () {
    const llmBase = 'https://models.test/llm';
    final weights = Uint8List.fromList(List.generate(1000, (i) => i % 256));

    setUp(() {
      server.files['$llmBase/model.gguf'] = weights;
      server.files['$llmBase/modelport.json'] = Uint8List.fromList(
        utf8.encode(
          jsonEncode({
            'schema': 'modelport/0.1',
            'id': 'echo',
            'version': '1.0.0',
            'task': 'text-generation',
            'license': 'MIT',
            'variants': [
              {
                'id': 'gguf-q4_k_m',
                'runtime': 'llamacpp',
                'precision': 'q4_k_m',
                'file': {
                  'url': '$llmBase/model.gguf',
                  'size': weights.length,
                  'sha256': sha256.convert(weights).toString(),
                },
              },
            ],
            'llm': {
              'context_length': 2048,
              'defaults': {'max_tokens': 3, 'temperature': 0.1},
            },
          }),
        ),
      );
    });

    test('streams a reply with the manifest defaults', () async {
      final llm = await TextGenerator.load(llmBase);
      final reply = await llm.generate('one two three four five').join();
      expect(reply, 'one two three ');
      await llm.close();
    });

    test('per-call config overrides the defaults', () async {
      final llm = await TextGenerator.load(llmBase);
      final reply = await llm.chat([
        const ChatMessage.user('a b c d'),
      ], config: llm.model.defaults.copyWith(maxTokens: 1)).join();
      expect(reply, 'a ');
    });

    test('tensor models are refused', () async {
      server.serve(base, FakeBundle());
      expect(
        TextGenerator.load(base),
        throwsA(
          isA<ModelPortException>().having(
            (e) => e.message,
            'message',
            contains('not text generation'),
          ),
        ),
      );
    });

    test('language models are refused by ModelPort.load', () async {
      expect(
        ModelPort.load(llmBase),
        throwsA(
          isA<ModelPortException>().having(
            (e) => e.hint,
            'hint',
            contains('TextGenerator'),
          ),
        ),
      );
    });
  });
}
