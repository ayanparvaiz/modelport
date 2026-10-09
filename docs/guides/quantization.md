# Smaller models

```bash
modelport quantize dist/<bundle> --fp16 --int8
```

| Option | What it does | MobileNetV3 | DeiT Tiny |
|---|---|---|---|
| `--fp16` | Weights in half precision; inputs and outputs stay float32 | 51% of fp32 | 52% |
| `--int8` | MatMul and Gemm weights in int8, per channel | 54% | 31% |

`--int8` leaves convolutions alone on purpose. Dynamic int8 on convolutions changed MobileNetV3's top-1 class in testing, so convolution-heavy models shrink less. Transformers shrink the most.

Each new variant is run on the golden input. Its tolerance is set to twice the measured error, and a variant that changes the top-1 class is refused unless you pass `--allow-top1-change`.

Both fp16 and int8 ONNX variants passed their golden checks on an ARMv8.0 phone.
