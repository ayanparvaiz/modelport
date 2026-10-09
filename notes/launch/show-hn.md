**Title:** Show HN: ModelPort – PyTorch models in Flutter apps, verified against Python on-device

**URL:** https://github.com/ayanparvaiz/modelport

**First comment:**

I built this after hand-porting image preprocessing to Dart one too many times. When a mean or a resize filter is slightly off, a mobile model doesn't fail, it just gets a bit worse, and you rarely notice.

ModelPort exports a model (ONNX, ExecuTorch, or GGUF) with a manifest that pins down preprocessing exactly, plus one saved input and PyTorch's output for it. The Dart side reproduces Pillow's resize bit for bit and can rerun that saved input on the device to compare. On a 2019 Snapdragon 665 phone every variant matched, including fp16 and int8.

A few numbers from that phone: ExecuTorch ran MobileNetV3 in 17 ms vs 77 ms for ONNX Runtime; the pure-Dart JPEG decoder was the real bottleneck (717 ms) until switching to Flutter's native decoder; llama.cpp's Vulkan backend alone adds 44 MB to an APK.

Apache-2.0, 0.1.0. iOS isn't verified yet.
