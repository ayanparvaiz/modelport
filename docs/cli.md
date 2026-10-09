# CLI reference

```bash
pip install "modelport[onnx,executorch,torchvision,hf,gguf]"
```

| Extra | Adds | Needed for |
|---|---|---|
| `onnx` | onnx, onnxruntime, onnxscript, torch | ONNX export, quantize, verify |
| `executorch` | executorch, torch | ExecuTorch export and verify |
| `torchvision` | torchvision, torch | `torchvision:` sources |
| `hf` | transformers, huggingface_hub, torch | `hf:` sources, `import-gguf` from the Hub, `publish --hf` |
| `gguf` | gguf | Inspecting GGUF files |

## Commands

| Command | What it does |
|---|---|
| `export SOURCE` | Convert a model and write a bundle with `modelport.json`, labels, and golden data. |
| `quantize BUNDLE` | Add `--fp16` and `--int8` ONNX variants with measured tolerances. |
| `verify BUNDLE` | Run every variant on the golden input and compare with PyTorch. |
| `import-gguf SOURCE` | Describe GGUF language models from a Hugging Face repo or a local file. |
| `pack BUNDLE` | Refresh sizes and hashes after manual edits; list unlisted files. |
| `publish BUNDLE` | Upload with `--github owner/repo --tag tag` or `--hf org/name`. |
| `gen-dart MANIFEST` | Write a typed Dart wrapper with named inputs and outputs. |
| `inspect FILE` | Show inputs, outputs, and metadata of `.onnx`, `.pte`, or `.gguf` files. |
| `validate MANIFEST...` | Check manifests against the spec. |
| `schema` | Print the manifest JSON Schema. |
| `doctor` | Check Python, optional packages, and disk space. |

Run `modelport COMMAND --help` for every option.

## Sources

| Source | Example | Notes |
|---|---|---|
| torchvision | `torchvision:efficientnet_b0` | Classifiers, with torchvision's own preprocessing |
| Hugging Face | `hf:facebook/deit-tiny-patch16-224` | Image classifiers and DETR-family detectors; local folders work too |
| Your code | `file:my_model.py:build` | A function returning `modelport.sources.SourceModel` |

`file:` runs the code in that file, so only use files you trust. Load checkpoints with `torch.load(..., weights_only=True)` or safetensors, never plain pickle from strangers.
