# Changelog

## 0.1.0

First release.

- `export` from torchvision, Hugging Face (image classifiers and DETR-family detectors), and your own code, to ONNX and ExecuTorch, with golden data.
- `quantize` to fp16 and int8 ONNX variants with measured tolerances.
- `verify` every variant against PyTorch.
- `import-gguf` for GGUF language models on the Hugging Face Hub or on disk.
- `publish` to GitHub releases or the Hugging Face Hub.
- `gen-dart` for typed Dart wrappers.
- `inspect`, `validate`, `schema`, `pack`, and `doctor`.
- Manifest spec 0.1 with JSON Schema.
