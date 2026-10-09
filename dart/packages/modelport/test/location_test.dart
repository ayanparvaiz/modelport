import 'package:modelport/modelport.dart';
import 'package:test/test.dart';

const _sha = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';

void main() {
  String manifestOf(String text) =>
      ModelLocation.parse(text).manifestUri.toString();

  test('Hugging Face locations', () {
    expect(
      manifestOf('hf://modelport-dev/mobilenet_v3_small'),
      'https://huggingface.co/modelport-dev/mobilenet_v3_small/resolve/main/modelport.json',
    );
    expect(
      manifestOf('hf://org/repo@v1.2'),
      'https://huggingface.co/org/repo/resolve/v1.2/modelport.json',
    );
    expect(
      manifestOf('hf://org/repo/models/small'),
      'https://huggingface.co/org/repo/resolve/main/models/small/modelport.json',
    );
  });

  test('https folders and manifest files', () {
    expect(
      manifestOf('https://x.dev/models/a'),
      'https://x.dev/models/a/modelport.json',
    );
    expect(
      manifestOf('https://x.dev/models/a/'),
      'https://x.dev/models/a/modelport.json',
    );
    expect(
      manifestOf('https://x.dev/m/custom.json'),
      'https://x.dev/m/custom.json',
    );
  });

  test('local folders', () {
    expect(
      manifestOf('/data/models/a'),
      'file:///data/models/a/modelport.json',
    );
    expect(
      manifestOf('file:///data/models/a/'),
      'file:///data/models/a/modelport.json',
    );
  });

  test('assets', () {
    final location = ModelLocation.parse('asset://assets/models/mobilenet');
    expect(location.isAsset, isTrue);
    expect(location.manifestUri.path, 'assets/models/mobilenet/modelport.json');
    final file = location.uriFor(
      const FileRef(path: 'onnx-fp32/model.onnx', size: 1, sha256: _sha),
    );
    expect(file.path, 'assets/models/mobilenet/onnx-fp32/model.onnx');
  });

  test('file paths resolve next to the manifest; urls are kept', () {
    final location = ModelLocation.parse('hf://org/repo');
    expect(
      location
          .uriFor(
            const FileRef(path: 'onnx-fp32/model.onnx', size: 1, sha256: _sha),
          )
          .toString(),
      'https://huggingface.co/org/repo/resolve/main/onnx-fp32/model.onnx',
    );
    expect(
      location
          .uriFor(
            const FileRef(url: 'https://cdn.dev/m.gguf', size: 1, sha256: _sha),
          )
          .toString(),
      'https://cdn.dev/m.gguf',
    );
  });

  test('plain http is only for localhost', () {
    expect(ModelLocation.parse('http://localhost:8000/m').isRemote, isTrue);
    expect(
      () => ModelLocation.parse('http://example.com/m'),
      throwsA(
        isA<ModelPortException>().having(
          (e) => e.hint,
          'hint',
          contains('https'),
        ),
      ),
    );
  });

  test('bad locations', () {
    for (final text in [
      'mobilenet',
      'hf://onlyorg',
      'ftp://x/y',
      'hf://org/@rev',
    ]) {
      expect(
        () => ModelLocation.parse(text),
        throwsA(isA<ModelPortException>()),
        reason: text,
      );
    }
  });

  test('cache keys are stable and short', () {
    final a = ModelLocation.parse('hf://org/repo');
    expect(a.cacheKey, hasLength(16));
    expect(a.cacheKey, ModelLocation.parse('hf://org/repo/').cacheKey);
  });
}
