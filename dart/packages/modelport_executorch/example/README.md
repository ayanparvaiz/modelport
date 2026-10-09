# modelport_executorch example

Runs MobileNetV3 Small from a ModelPort bundle with the ExecuTorch adapter, shows the top three classes for a sample photo, and runs the bundle's golden check on the device.

## Setup

```bash
brew install cmake   # executorch_flutter builds a small native wrapper
cd python
uv run modelport export torchvision:mobilenet_v3_small --target onnx,executorch
../dart/packages/modelport_executorch/example/tool/copy_bundle.sh
```

## Run

```bash
cd dart/packages/modelport_executorch/example
flutter run
flutter test integration_test -d <device-id> --no-uninstall
```
