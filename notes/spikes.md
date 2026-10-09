# Phase 0 spikes

Spike mane chhoto, fela dewar moto experiment. Lokkho: design er age dekha je engine gulo asholei chole, ar kon jhamela ache.

Machine: MacBook (Apple Silicon), Flutter 3.44.9, Dart 3.12.2, Python 3.12 (uv).

---

## Spike A + B, Python side: MobileNetV3 Small → ONNX ar ExecuTorch

Script: `spikes/python/export_mobilenet.py`
Versions: torch 2.14.1, torchvision 0.29.1, onnx 1.23.2, onnxruntime 1.31.0, executorch 1.5.1

**Result (dog.jpg, PyTorch er sathe tulona):**

| Engine | File size | Export time | max abs diff | top-1 |
|---|---|---|---|---|
| PyTorch | | | | Samoyed |
| ONNX (dynamo) | 10.5 MB | ~2 s | 1.8e-05 | Samoyed ✅ |
| ExecuTorch XNNPACK | 10.2 MB | ~7 s | 4.7e-05 | Samoyed ✅ |

**Shikkha:**

1. **ExecuTorch e input er memory layout export er sathe atke jay.** torchvision transform `channels_last` tensor dey. Oi tensor diye export korle `.pte` sadharon NCHW input e fail kore (`XNNExecutor: Propagating input shapes failed`). Solution: export er age example input e `.contiguous()`. **CLI te eta sobsomoy korte hobe.**
2. ONNX dynamo export e `input_names` ar `output_names` kaj kore. Manifest er naam gulo sorasori ONNX e boshano jay.
3. ExecuTorch Python runtime (`executorch.runtime.Runtime`) diye `verify` command banano jabe. Method metadata theke input shape ar dtype o pawa jay.
4. Golden file size manifest er sathe mile: `pixel_values.bin` 602,112 byte, `logits.bin` 4,000 byte.

---

## Spike A, B, C: Flutter side

App: `spikes/flutter_spike`, test: `integration_test/spike_test.dart`
Packages: flutter_onnxruntime 1.9.0, executorch_flutter 0.8.0, llm_llamacpp 0.7.0

Golden input (Python er preprocess kora tensor) phone e chaliye output Python er sathe milano holo.

### Phone: OPPO CPH1937 (Android 11, Snapdragon 665, 8 GB RAM)

| Spike | Engine | Load | Prothom run | Warm run | Python er sathe diff | Result |
|---|---|---|---|---|---|---|
| A | ONNX Runtime | ~390 ms | ~200 ms | ~90 ms | 1.84e-05 | Samoyed ✅ |
| B | ExecuTorch XNNPACK | ~230 ms | ~85 ms | ~25 ms | 4.67e-05 | Samoyed ✅ |
| C | llama.cpp, Qwen2.5 0.5B Q4_K_M | ~1.0 s | 2.2 s (prothom text) | ~10 piece/s | | Thik uttor ✅ |

### Mac (Apple Silicon, macOS 26)

| Spike | Engine | Load | Warm run | Result |
|---|---|---|---|---|
| A | ONNX Runtime | ~190 ms | ~8 ms | Samoyed ✅ |
| B | ExecuTorch XNNPACK | ~120 ms | ~2 ms | Samoyed ✅ |
| C | llama.cpp | ~0.6 s | ~100 piece/s | Thik uttor ✅ |

LLM er uttor (dui jaygay ek): "Flutter is a cross-platform mobile app development framework."

### Shikkha (ModelPort er design e lagbe)

1. **Golden test er idea kaj kore.** Phone er output Python er exported-model output er sathe hubohu ek (diff ekdom same number). Tai on-device parity test bhorsha jogyo.
2. **Phone e ExecuTorch ONNX er cheye prai 4 gun druto** ei model e. Docs e engine tulona table rakhte hobe.
3. **Tin ta engine eki app e build hoy.** Kono symbol conflict nai (Android ar macOS).
4. **flutter_onnxruntime macOS 14+ chay.** `MACOSX_DEPLOYMENT_TARGET` 14.0 korte hoyeche. Docs e likhte hobe.
5. **executorch_flutter build machine e `cmake` chay** (prebuilt library download kore, kintu FFI wrapper cmake diye build kore). `brew install cmake` lage. Docs er requirements e likhte hobe.
6. **llm_llamacpp 0.7.0 er Android bug:**
   - `ModelLoader` `ggml_backend_load_all()` call kore, jeta Android e kono backend pay na: `no backends are loaded`.
   - Package er bhitore `BackendInitializer` sothik kaj kore, kintu export kora na, ar loader eta use kore na.
   - Tar upor Android native library APK theke extract na hole backend `.so` path diye pawa jay na.
   - **Workaround:** `BackendInitializer.initializeBackend()` age call, ar Gradle e `packaging.jniLibs.useLegacyPackaging = true`.
   - **ModelPort er `modelport_llamacpp` adapter eta nije korbe**, jate user ke jante na hoy. Upstream e issue report kora uchit (repo: brynjen/dart-llm).
7. **`flutter test` Android e shesh e app uninstall kore**, tai app er data folder e rakha model muche jay. Test e `--no-uninstall` lage.
8. **Debug APK 351 MB** (sob ABI, Vulkan validation layer, unstripped library). Release size niche.

### Siddhanto

- LLM engine: **llm_llamacpp** thakbe. Android e kaj kore, active, ar workaround adapter er bhitore lukano jay. `llama_cpp_dart` backup.
- Prothom end-to-end (Phase 4) ONNX diye, plan moto. ExecuTorch Phase 5 e, karon eta druto ar export o kaj kore.
- iPhone e ekhono test hoy nai. Signing (Apple developer team) setup kore Phase 4 e korbo.

### App size: release APK, sudhu arm64

Mot APK: **82.5 MB**, er moddhe test model (ONNX + PTE) 21 MB. Native library (unzip kora size):

| Engine | Library | Size |
|---|---|---|
| Flutter nijei | libflutter + libapp | 14.5 MB |
| ONNX Runtime | libonnxruntime + JNI | 28.7 MB |
| ExecuTorch | libexecutorch_ffi + fbjni | 7.6 MB |
| llama.cpp | libggml-vulkan | **44.1 MB** |
| llama.cpp | 7 ta CPU variant + libllama + ggml + omp | 16.3 MB |

**Shikkha:**

9. **ExecuTorch sobcheye halka engine** (7.6 MB). ONNX Runtime prai 4 gun boro.
10. **llama.cpp er Vulkan GPU backend ekai 44 MB.** Je app GPU chay na, tar jonno eta bad dewar upay docs e dite hobe, ba adapter e option rakhte hobe.
11. Ei karone adapter alada package rakhar design ta thik: user sudhu je engine lagbe setai nibe.
