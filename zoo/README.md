# ModelPort model zoo

Ready, tested bundles. Each one passed `modelport verify` and a golden check on an Android phone.

| Model | Task | License | Location |
|---|---|---|---|
| MobileNetV3 Small | image-classification | BSD-3-Clause | `https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/mobilenet_v3_small.json` |
| DeiT Tiny | image-classification | Apache-2.0 | `https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/deit-tiny-patch16-224.json` |
| YOLOS Tiny (320x320) | object-detection | Apache-2.0 | `https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/yolos-tiny.json` |
| SmolLM2 135M Instruct | text-generation | Apache-2.0 | `https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/smollm2-135m-instruct.json` |
| Qwen2.5 0.5B Instruct | text-generation | Apache-2.0 | `https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/qwen2.5-0.5b-instruct.json` |

```dart
final classifier = await ImageClassifier.load(
  'https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/mobilenet_v3_small.json',
);
```

`index.json` lists the same models for apps that want to show a picker.

## How the zoo is built

```bash
modelport export torchvision:mobilenet_v3_small --target onnx,executorch
modelport quantize dist/mobilenet_v3_small --fp16 --int8
modelport export hf:facebook/deit-tiny-patch16-224 --target onnx,executorch
modelport quantize dist/deit-tiny-patch16-224 --fp16 --int8
modelport export hf:hustvl/yolos-tiny --image-size 320 --target onnx,executorch
modelport import-gguf unsloth/SmolLM2-135M-Instruct-GGUF -q q4_k_m
modelport import-gguf Qwen/Qwen2.5-0.5B-Instruct-GGUF -q q4_k_m,q8_0
for b in dist/*/; do modelport publish "$b" --github ayanparvaiz/modelport --tag zoo-v1; done
```

Language model weights are not copied: their manifests point at the original Hugging Face files, pinned to a commit.

## Rules for new models

- Permissive licenses only: Apache-2.0, MIT, or BSD.
- Every file has a `sha256`, and tensor models have golden data that passes on a real device.
