## 0.1.1

* Docs: the Python CLI installs with `pip install modelport-cli`.

## 0.1.0

First release.

* Manifests: parse and validate `modelport.json` (spec 0.1), refuse newer major versions, and ignore unknown fields from newer minor versions.
* Downloads: resume with HTTP Range, verify size and sha256, cache verified files, cancel, and load offline from the last manifest.
* Locations: GitHub release and other HTTPS URLs, `hf://`, `asset://`, and local folders.
* Image preprocessing that matches the Python reference byte for byte, including Pillow-compatible resizing.
* Task APIs: `ImageClassifier`, `ObjectDetector` (YOLO and DETR outputs, boxes in image pixels), and `TextGenerator`.
* `TensorModel.checkGolden()` compares a device with Python's recorded output.
* Variant selection by registered engine and device RAM.
