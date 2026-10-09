# Phase 4: First Light

Prothom bar ModelPort er puro chain asol phone e: Python CLI diye export kora bundle → Flutter asset → `ModelPort.load` → ONNX adapter → golden check ar chobi classify.

Test: `dart/packages/modelport_onnx/example/integration_test/first_light_test.dart`
Model: MobileNetV3 Small (torchvision), bundle e onnx-fp32, onnx-fp16, onnx-int8.

## Golden check (phone e Python er sathe tulona)

| Variant | OPPO CPH1937 (Android 11, ARMv8.0) | macOS (Apple Silicon) | Python `verify` |
|---|---|---|---|
| onnx-fp32 | PASS, max diff 3.34e-5 | PASS, 3.34e-5 | 3.34e-5 |
| onnx-int8 | PASS, 1.27e-1 (allowed 0.26) | PASS, 1.27e-1 | 1.27e-1 |
| onnx-fp16 | PASS, 1.05e-1 (allowed 0.21) | PASS, 1.05e-1 | 1.05e-1 |

Phone er number Python er sathe **hubohu ek**. Mane export, manifest, download/cache, adapter, sob thik.

**Khola proshno mitlo:** fp16 model purono ARMv8.0 CPU teo chole ar thik result dey.

## Asol chobi

`dog.jpg` (1546x1213) → **Samoyed (0.77)**, Pomeranian (0.06), Arctic fox (0.06). Phone ar Mac e same.

## Somoy, phone e (profile mode, release er kachakachi)

| Step | Somoy |
|---|---|
| Model load (cache theke) | 100 – 190 ms |
| JPEG decode (`package:image`) | **717 ms** |
| Preprocess (resize, crop, normalize) | 133 ms |
| Model run (ONNX Runtime) | 70 ms |
| Mot classify | ~950 ms |

Debug mode e (integration test default) decode 2252 ms, preprocess 607 ms, run 119 ms. Tai performance sobsomoy profile mode e mapte hobe: `flutter drive --profile`.

## Shikkha

1. **JPEG decode i sobcheye boro bottleneck.** Pure Dart decoder dhire. Flutter engine er native decoder (`ui.instantiateImageCodec`) use korle onek druto hobar kotha. Porer kaj.
2. Preprocess o isolate e pathano uchit, jate UI atke na jay.
3. Prothom load e asset theke cache e copy ar sha256 hoy (fp32 1.7 s debug e). Porer load cache theke.

## Update: native decoder (2026-10-10)

`modelport_flutter` ekhon Flutter engine er native decoder use kore, ar decode/preprocess isolate e chole.

| Step | Age (`package:image`) | Ekhon (native) |
|---|---|---|
| JPEG decode | 717 ms | **302 ms** |
| Preprocess | 133 ms | 157 ms |
| Model run | 70 ms | 77 ms |
| Mot classify | ~950 ms | **584 ms** |

Native decoder er JPEG pixel samanyo alada, tai Samoyed er score 0.77 theke 0.76. Top-1 ekoi. PNG e pixel hubohu ek (test ache).

Aro druto korar upay (pore): decode er somoy i choto kore pora (`targetWidth`), kintu tate resize er niyom bodlay, tai eta optional "fast mode" hishebe bhabte hobe.

## ExecuTorch adapter (Phase 5)

| | OPPO CPH1937 | macOS |
|---|---|---|
| Golden check | PASS, max diff 3.72e-5 | PASS, 3.34e-5 |
| Model run | **17 ms** (ONNX: 77 ms) | 2 ms |
| Mot classify | 464 ms | 81 ms |
| Result | Samoyed (0.76) | Samoyed (0.76) |

**Shikkha:**
- Ei model e phone e ExecuTorch ONNX Runtime er cheye prai 4.5 gun druto. App size o kom (spike e 7.6 MB vs 28.7 MB).
- macOS e Xcode 27 deployment target 12 er niche nay. Example app er target 14.0 rakhte hobe.
- Prothom macOS build bhul target e hole `executorch_dart` er CMake cache 11.0 e atke thake. Thik korte `dart/.dart_tool/hooks_runner/shared/executorch_dart/build/` er cache folder muchte hoy. Docs er troubleshooting e jabe.

