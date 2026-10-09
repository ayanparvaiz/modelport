# Performance

Measured on an OPPO CPH1937 (Android 11, Snapdragon 665, 8 GB, 2019) in profile mode, and on an Apple Silicon Mac.

## Image classification, MobileNetV3 Small

| Step | Phone | Mac |
|---|---|---|
| Decode a 1546x1213 JPEG (native decoder) | 302 ms | — |
| Preprocess | 157 ms | — |
| Run, ONNX Runtime | 77 ms | 8 ms |
| Run, ExecuTorch (XNNPACK) | 17 ms | 2 ms |
| Whole `classify()` with ExecuTorch | 382 ms | 78 ms |

The pure Dart JPEG decoder took 717 ms on the phone, which is why `modelport_flutter` uses the Flutter engine's decoder.

## Object detection, YOLOS Tiny

| Input size | Phone | Mac |
|---|---|---|
| 512x512, ExecuTorch | 3.6 s | 0.58 s |
| 320x320, ONNX | — | 82 ms |

## Language models, llama.cpp CPU

| Model | First words | Download |
|---|---|---|
| SmolLM2 135M Q4_K_M | 0.9 s on the phone | 105 MB, 63 s on the phone's Wi-Fi |
| Qwen2.5 0.5B Q4_K_M | 2.2 s on the phone, about 10 pieces/s | 491 MB |

## App size, release APK, arm64 only

| Engine | Adds |
|---|---|
| ExecuTorch | about 7.6 MB |
| ONNX Runtime | about 28.7 MB |
| llama.cpp | about 60 MB, 44 MB of which is the Vulkan GPU backend |

## Measuring your own

Integration tests run in debug mode, where Dart code is several times slower. Measure with `flutter drive --profile`; the example apps include a driver for this.
