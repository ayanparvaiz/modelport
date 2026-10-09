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
