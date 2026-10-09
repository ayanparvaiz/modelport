# modelport_onnx example

Runs MobileNetV3 Small from a ModelPort bundle with the ONNX adapter, shows the top three classes for a sample photo, and runs the bundle's golden check on the device.

## Setup

```bash
cd python
uv run modelport export torchvision:mobilenet_v3_small
uv run modelport quantize ../dist/mobilenet_v3_small --fp16 --int8
../dart/packages/modelport_onnx/example/tool/copy_bundle.sh
```

## Run

```bash
cd dart/packages/modelport_onnx/example
flutter run
flutter test integration_test -d <device-id> --no-uninstall
```
