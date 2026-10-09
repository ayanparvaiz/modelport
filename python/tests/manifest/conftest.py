import copy

import pytest

SHA = "c" * 64


def file_ref(path: str, size: int = 1000) -> dict:
    return {"path": path, "size": size, "sha256": SHA}


CLASSIFIER = {
    "schema": "modelport/0.1",
    "id": "mobilenet_v3_small",
    "version": "1.0.0",
    "task": "image-classification",
    "license": "BSD-3-Clause",
    "variants": [
        {
            "id": "onnx-fp32",
            "runtime": "onnx",
            "precision": "fp32",
            "file": file_ref("onnx-fp32/model.onnx"),
        }
    ],
    "inputs": [
        {
            "name": "pixel_values",
            "dtype": "float32",
            "shape": [1, 3, 224, 224],
            "layout": "NCHW",
            "preprocess": {
                "type": "image",
                "resize": {"shorter_side": 256},
                "center_crop": [224, 224],
            },
        }
    ],
    "outputs": [
        {
            "name": "logits",
            "dtype": "float32",
            "shape": [1, 1000],
            "postprocess": {"type": "classification", "labels": file_ref("labels.txt")},
        }
    ],
    "golden": {
        "inputs": {"pixel_values": file_ref("golden/pixel_values.bin", 602112)},
        "outputs": {"logits": file_ref("golden/logits.bin", 4000)},
    },
}

LLM = {
    "schema": "modelport/0.1",
    "id": "qwen2.5-0.5b-instruct",
    "version": "1.0.0",
    "task": "text-generation",
    "license": "Apache-2.0",
    "variants": [
        {
            "id": "gguf-q4_k_m",
            "runtime": "llamacpp",
            "precision": "q4_k_m",
            "file": {
                "url": "https://huggingface.co/o/r/resolve/main/m.gguf",
                "size": 9,
                "sha256": SHA,
            },
            "min_ram_mb": 1024,
        }
    ],
    "llm": {"context_length": 4096},
}


@pytest.fixture
def classifier() -> dict:
    return copy.deepcopy(CLASSIFIER)


@pytest.fixture
def llm() -> dict:
    return copy.deepcopy(LLM)
