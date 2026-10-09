# modelport_llamacpp example

A small chat app. Only the model's `modelport.json` ships in the app; the 105 MB SmolLM2 GGUF file downloads from the Hugging Face Hub on first use, is checked against its sha256, and is cached.

The manifest was made with:

```bash
modelport import-gguf unsloth/SmolLM2-135M-Instruct-GGUF -q q4_k_m
```

## Run

```bash
cd dart/packages/modelport_llamacpp/example
flutter run
flutter test integration_test -d <device-id> --no-uninstall
```
