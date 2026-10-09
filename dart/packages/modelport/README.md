# modelport

Run AI models in Dart and Flutter apps from a `modelport.json` manifest. Part of [ModelPort](https://github.com/ayanparvaiz/modelport).

> **Status: early development.** Not published yet. APIs will change.

This is the pure Dart core. It reads manifests, downloads and verifies model files, prepares inputs, and turns outputs into results. Engines plug in through adapter packages such as `modelport_onnx`.
