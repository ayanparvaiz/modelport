# modelport_llamacpp

[llama.cpp](https://github.com/ggml-org/llama.cpp) engine for [ModelPort](https://pub.dev/packages/modelport). Runs GGUF language models from `modelport.json` bundles through [llm_llamacpp](https://pub.dev/packages/llm_llamacpp), with streaming chat.

```dart
await ModelPortFlutter.init(adapters: [LlamaCppAdapter()]);
final llm = await TextGenerator.load(
  'https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/smollm2-135m-instruct.json',
  onProgress: (p) => print('${(p.fraction * 100).round()}%'),
);
await for (final piece in llm.chat([ChatMessage.user('What is Flutter?')])) {
  stdout.write(piece);
}
```

- **Only a manifest ships with the app.** The GGUF file downloads on first use, is checked against its sha256, and is cached. On the test phone, SmolLM2 135M (105 MB) downloaded and answered in about a minute, with the first words 0.9 s after the question.
- **Works on Android without patches.** `llm_llamacpp` 0.7.0's model loader finds no CPU backend on Android. This adapter loads the backends correctly before every model.
- **The right file for the device.** `modelport import-gguf` gives every quantization a rough minimum RAM, so small phones get the smaller file.
- **Stop any time.** `cancel()` stops the reply that is being generated.
- **Pure Dart.** Works in Dart command-line apps as well as Flutter.

## Android setup

llama.cpp loads its CPU backends from separate libraries, which must be extracted from the APK. Add this to `android/app/build.gradle.kts`:

```kotlin
android {
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}
```

Without it, loading a model fails with an error that names this setting.

## Any GGUF model

```bash
pip install "modelport[hf]"
modelport import-gguf Qwen/Qwen2.5-0.5B-Instruct-GGUF -q q4_k_m,q8_0
```

This writes only a manifest; the files stay in the original Hugging Face repo, pinned to a commit.

## App size

llama.cpp adds about 60 MB to an arm64 APK, 44 MB of which is the Vulkan GPU backend.

## Example

The [example app](https://github.com/ayanparvaiz/modelport/tree/main/dart/packages/modelport_llamacpp/example) is a small chat app that downloads SmolLM2 on first use.

## License

Apache-2.0
