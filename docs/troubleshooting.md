# Troubleshooting

## "no registered adapter can run …"

Add the adapter package the error names and pass it to `ModelPortFlutter.init(adapters: [...])`.

## "… needs at least N MB of RAM"

Every variant needs more memory than the device reports. Pick a smaller model, or ship a smaller variant first in the manifest.

## "sha256 of … does not match the manifest"

The file on the server changed after the manifest was written, or the download was corrupted. Republish the bundle with `modelport pack` and `modelport publish`.

## Android: llama.cpp finds no CPU backend

Add `packaging { jniLibs { useLegacyPackaging = true } }` to `android/app/build.gradle.kts`. `modelport_llamacpp` also works around an `llm_llamacpp` 0.7.0 issue where its model loader does not load backends on Android.

## Android: release build crashes in ONNX Runtime

Add `-keep class ai.onnxruntime.** { *; }` to `android/app/proguard-rules.pro`.

## macOS: deployment target errors

`flutter_onnxruntime` needs macOS 14. Xcode 27 also rejects targets below 12. Set `MACOSX_DEPLOYMENT_TARGET = 14.0` in `macos/Runner.xcodeproj`.

## macOS: executorch_dart fails with target 11.0

`executorch_dart` passes `CMAKE_OSX_DEPLOYMENT_TARGET=11.0` to CMake, which Xcode 27 rejects, and the value can stick in its build cache. Delete `.dart_tool/hooks_runner/shared/executorch_dart/build/` in your project or workspace and build again.

## ExecuTorch: build fails with "cmake not found"

`executorch_flutter` downloads prebuilt ExecuTorch but builds a small wrapper with CMake. Install it, for example `brew install cmake`.

## ExecuTorch: "Propagating input shapes failed"

The `.pte` was exported from a `channels_last` example input. `modelport export` always makes example inputs contiguous; if you export by hand, call `.contiguous()` first.

## Tests: models disappear between Android integration test runs

`flutter test` uninstalls the app at the end, which deletes its data. Pass `--no-uninstall`.

## Everything is slow in tests

Integration tests run in debug mode. Use `flutter drive --profile` to measure.
