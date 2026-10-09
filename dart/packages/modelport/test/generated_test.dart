import 'dart:io';
import 'dart:typed_data';

import 'package:modelport/modelport.dart';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';

import 'generated/toy_rgb.dart';
import 'support/fake_adapters.dart';

// toy_rgb.dart is written by `modelport gen-dart`. A Python test keeps it in
// step with the generator; this test proves the generated Dart compiles and works.
void main() {
  late Directory cache;

  setUp(() {
    cache = Directory.systemTemp.createTempSync('modelport_gen_');
    ModelPort.reset();
    ModelPort.configure(
      store: ModelStore(root: cache),
      adapters: [ChannelMeanAdapter()],
    );
  });

  tearDown(() {
    ModelPort.reset();
    cache.deleteSync(recursive: true);
  });

  test('the generated wrapper loads and runs with named tensors', () async {
    final bundle = p.join(
      Directory.current.path,
      'test',
      'fixtures',
      'toy_bundle',
    );
    final model = await ToyRgbModel.load(bundle);
    expect(ToyRgbModel.pixelValuesShape, [1, 3, 4, 4]);
    final red = Float32List(48)..fillRange(0, 16, 1);
    final outputs = await model.run(
      pixelValues: Tensor.float32(ToyRgbModel.pixelValuesShape, red),
    );
    expect(outputs.logits.float32, [1, 0, 0]);
    await model.close();
  });
}
