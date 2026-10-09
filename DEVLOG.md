# DEVLOG

Protidin er choto note. Shurute "aj ki korbo", sheshe "ki holo, kothay atkalam".

---

## 2026-10-09 · Day 1

**Aj ki korbo**
- Repo banano, GitHub e public kora
- Phase 0 er machine-side setup: uv, Python 3.12
- Spike A, B, C shuru

**Ki holo**
- Repo live: https://github.com/ayanparvaiz/modelport
- README, LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue template add holo
- Python package skeleton, manifest er Pydantic model, JSON Schema, 3 ta example, 7 ta bhul example
- CLI: `modelport --version`, `schema`, `validate`
- GitHub Actions Python CI green (3.11, 3.12, 3.13)
- Spike A, B, C pass: OPPO phone ar macOS duijaygay. Python er sathe output hubohu mile.

**Kothay atkalam**
- ExecuTorch export e channels_last input → `.contiguous()` diye thik
- executorch_flutter er jonno cmake lage, flutter_onnxruntime er jonno macOS 14
- llm_llamacpp Android e backend load kore na → workaround peyechi, upstream e report korte hobe
- `flutter test` Android e app uninstall kore model muche dey → `--no-uninstall`

- Phase 2 shuru: `doctor` ar `inspect` (.onnx, .pte, .gguf) command hoye geche

**Kal ki korbo**
- Phase 2: source loader (`torchvision:`, `hf:`, `file:`) ar ONNX exporter
- Spike er shikkha (`.contiguous()`) exporter e boshano
- Hugging Face org, PyPI account (nije)

---

## 2026-10-09 · Day 1, rat

**Ki holo**
- Phase 2 er Python CLI prai shesh: `export`, `quantize`, `verify`, `pack`, `publish`
- Source: `torchvision:`, `hf:`, `file:`
- Export: ONNX ar ExecuTorch (Phase 5 er Python ongsho age i hoye gelo)
- Preprocessing er Python reference implementation, torchvision er sathe 0.00001 er moddhe mile
- MobileNetV3 ar DeiT-tiny asol model e export → quantize → verify sob pass
- CI fail er karon thik: pyright optional import e error dicchilo. Ekhon warning, ar alada CPU torch job sob test chalay.

**Shikkha**
- Dynamic int8 Conv layer e CNN er accuracy nosto kore (MobileNetV3 er top-1 bodle gechilo). Tai int8 sudhu MatMul/Gemm e. Transformer (DeiT) e eta 31% size e niye ashe.
- torch er dynamo exporter graph e shape annotation (`value_info`) rakhe ja ONNX Runtime quantizer er sathe mele na. Quantize er age egulo muche fela lage.
- torchvision PIL image ke PIL diye resize kore, tai antialias sobsomoy on. Spec e eta spashto kora holo.
- Center crop e torchvision Python er `round()` (half to even) use kore, HF floor use kore. Spec e torchvision er ta nilam.

**Khola proshno**
- fp16 ONNX model purono ARMv8.0 phone e (jemon OPPO CPH1937) chole kina. Phase 4 e phone e test korte hobe.

**Porer kaj**
- Phase 3: Dart core package

---

## 2026-10-10 · Day 2

**Ki holo**
- Phase 3 shesh: Dart core e manifest, Tensor, preprocessing, download/cache, adapter contract, `ImageClassifier`, `TextGenerator`, phone e golden check. 99 test.
- Dart ar Python er preprocessing 12 ta fixture e **byte-for-byte ek**.
- `modelport_flutter`: cache folder, asset bundle, RAM, native image decoder.
- **Phase 4 First Light:** `modelport_onnx` adapter diye OPPO phone e MobileNetV3 chole, golden check pass, chobi te Samoyed.
- Phone e classify 954 ms theke 584 ms, native decoder ar isolate diye.

**Shikkha**
- Flutter SDK `meta` package pin kore. Core package er constraint beshi uchu hole Flutter app e install i hoy na.
- Integration test debug mode e chole, tai somoy mapte `flutter drive --profile` lage.
- Flutter er native decoder EXIF orientation nijei thik kore, Pillow er moto.

**Porer kaj**
- Phase 5: `modelport_executorch` adapter (Python ongsho hoye geche)

