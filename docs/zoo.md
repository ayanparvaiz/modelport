# Model zoo

Every model here passed `modelport verify` and a golden check on an Android phone (OPPO CPH1937, Android 11).

| Model | Task | Engines | Download | License |
|---|---|---|---|---|
| MobileNetV3 Small | image-classification | ONNX fp32/fp16/int8, ExecuTorch | 5–10 MB | BSD-3-Clause |
| DeiT Tiny | image-classification | ONNX fp32/fp16/int8, ExecuTorch | 7–24 MB | Apache-2.0 |
| YOLOS Tiny, 320x320 | object-detection | ONNX, ExecuTorch | 26–27 MB | Apache-2.0 |
| SmolLM2 135M Instruct | text-generation | llama.cpp (Q4_K_M) | 105 MB | Apache-2.0 |
| Qwen2.5 0.5B Instruct | text-generation | llama.cpp (Q4_K_M, Q8_0) | 491 / 676 MB | Apache-2.0 |

Locations follow one pattern:

```
https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1/<id>.json
```

with these ids: `mobilenet_v3_small`, `deit-tiny-patch16-224`, `yolos-tiny`, `smollm2-135m-instruct`, `qwen2.5-0.5b-instruct`.

`zoo/index.json` in the repository lists the same models for apps that show a picker. A weekly job checks that every manifest loads and every file exists with its listed size.
