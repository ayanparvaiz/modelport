# Phase 6: LLM (GGUF)

## Flow

1. `modelport import-gguf unsloth/SmolLM2-135M-Instruct-GGUF -q q4_k_m`
   - Hugging Face API theke size, sha256, license, context length, chat template pore
   - 105 MB file download kore na; URL e repo er commit hash pin kore
2. App e sudhu `modelport.json` asset hishebe (2 KB)
3. Prothom `TextGenerator.load()` e phone Hugging Face theke GGUF download kore, sha256 check kore, cache e rakhe
4. `modelport_llamacpp` adapter llama.cpp diye stream kore uttor dey

## Result

| | OPPO CPH1937 (Android 11) | macOS |
|---|---|---|
| Download + verify + load (105 MB) | 63 s | 56 s |
| Prothom shobdo | 0.94 s | 0.18 s |
| 22 / 9 piece | 1.7 s | 0.2 s |
| Uttor | "The capital of France is Paris, a city known for its historical landmarks..." | "The capital of France is Paris, France." |

Temperature 0 holeo phone ar Mac er uttor alada, karon CPU kernel ar thread alada.

## Shikkha

- Android e app e sudhu Gradle er `useLegacyPackaging = true` lage. `llm_llamacpp` er backend bug er workaround adapter er bhitore lukano.
- Backend na pele adapter error e Gradle fix ta bole dey.
- Majhe net bondho kore boro download resume er test unit test e (fake server) pass. Phone e 400 MB+ diye ekhono kora hoy nai.
