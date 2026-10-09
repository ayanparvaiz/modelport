# modelport_llamacpp

[llama.cpp](https://github.com/ggml-org/llama.cpp) adapter for [ModelPort](https://github.com/ayanparvaiz/modelport). Runs GGUF language models from `modelport.json` bundles through [llm_llamacpp](https://pub.dev/packages/llm_llamacpp), with streaming chat.

> **Status: early development.** Not published yet.

```dart
await ModelPortFlutter.init(adapters: [LlamaCppAdapter()]);
final llm = await TextGenerator.load('hf://org/qwen2.5-0.5b-instruct');
await for (final piece in llm.chat([ChatMessage.user('What is Flutter?')])) {
  stdout.write(piece);
}
```

Make a bundle for any GGUF repo on the Hugging Face Hub with the CLI:

```bash
modelport import-gguf Qwen/Qwen2.5-0.5B-Instruct-GGUF -q q4_k_m,q8_0
```

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

The adapter also works around an `llm_llamacpp` 0.7.0 issue where its model loader finds no backend on Android.

## App size

llama.cpp adds about 60 MB to an arm64 APK, 44 MB of which is the Vulkan GPU backend.
