# Chat with a language model

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

- `chat` streams the reply piece by piece. `cancel()` stops it.
- Sampling defaults come from the manifest. Override them per call with `llm.model.defaults.copyWith(temperature: 0, maxTokens: 64)`.
- The chat template comes from the GGUF file.

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

## Any GGUF model

```bash
modelport import-gguf Qwen/Qwen2.5-0.5B-Instruct-GGUF -q q4_k_m,q8_0
```

This writes only a manifest. The GGUF files stay in the original repo, pinned to a commit, and each variant gets a rough minimum RAM so small phones pick the smaller file.
