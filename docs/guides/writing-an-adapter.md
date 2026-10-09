# Writing an adapter

An adapter connects one inference engine to ModelPort. Implement `TensorAdapter` for tensor models or `TextGenerationAdapter` for language models.

```dart
class MyEngineAdapter implements TensorAdapter {
  @override
  String get runtime => 'my_engine'; // the manifest's variant runtime

  @override
  bool canRun(Variant variant) => variant.runtime == runtime;

  @override
  Future<TensorSession> open(LoadedVariant model) async {
    final engine = await MyEngine.load(model.modelFile.path);
    return MySession(engine, model.manifest);
  }
}

class MySession implements TensorSession {
  MySession(this.engine, this.manifest);
  final MyEngine engine;
  final Manifest manifest;

  @override
  Future<Map<String, Tensor>> run(Map<String, Tensor> inputs) async {
    // Inputs are keyed by manifest name and already checked against the
    // manifest's dtype and shape. Return outputs keyed the same way.
  }

  @override
  Future<void> close() => engine.dispose();
}
```

Rules that keep apps safe:

- Take tensor names, types, and shapes from the manifest, not from the engine.
- Turn engine errors into `ModelPortException` with a `hint` that says how to fix them.
- Free native memory in `close()`, and after every `run()` if the engine allocates per call.
- Prove your adapter with a golden check on a real device.
