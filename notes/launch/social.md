## X / Twitter

I built ModelPort: run PyTorch, Hugging Face, and GGUF models in Flutter with one line of Dart, and prove the phone gives the same answer as Python.

Tested on a 2019 Android phone: ExecuTorch 17 ms vs ONNX 77 ms, offline chat with SmolLM2.

https://github.com/ayanparvaiz/modelport

## LinkedIn

Shipping an AI model in a mobile app is mostly glue work: converting the model, rewriting preprocessing, caching big downloads, and hoping the phone matches Python.

ModelPort, my new open-source project, does that glue:
• A Python CLI exports models to ONNX, ExecuTorch, or GGUF with an exact manifest
• Dart packages load any bundle with one line
• A golden check proves the device matches Python
• Byte-identical preprocessing between Python and Dart

Tested on a real 2019 Android phone, with every model variant matching PyTorch.

Docs and demo app: https://ayanparvaiz.github.io/modelport/
