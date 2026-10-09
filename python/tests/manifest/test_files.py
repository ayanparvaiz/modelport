import re

import pytest
from pydantic import ValidationError

from modelport.manifest.files import SAFE_PATH_PATTERN, FileRef

SHA = "a" * 64


def test_relative_path_is_accepted():
    ref = FileRef(path="onnx-fp32/model.onnx", size=10, sha256=SHA)
    assert ref.path == "onnx-fp32/model.onnx"
    assert ref.url is None


def test_https_url_is_accepted():
    ref = FileRef(url="https://huggingface.co/org/repo/resolve/main/m.gguf", size=10, sha256=SHA)
    assert ref.url is not None


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../secret", "a/../b", "a//b", "./model.onnx", "a\\b", "C:/model.onnx", ""],
)
def test_unsafe_paths_are_rejected(path):
    with pytest.raises(ValidationError):
        FileRef(path=path, size=10, sha256=SHA)


def test_http_url_is_rejected():
    with pytest.raises(ValidationError):
        FileRef(url="http://example.com/model.onnx", size=10, sha256=SHA)


def test_both_path_and_url_is_rejected():
    with pytest.raises(ValidationError, match="exactly one"):
        FileRef(path="m.onnx", url="https://example.com/m.onnx", size=10, sha256=SHA)


def test_neither_path_nor_url_is_rejected():
    with pytest.raises(ValidationError, match="exactly one"):
        FileRef(size=10, sha256=SHA)


@pytest.mark.parametrize("sha", ["abc", "A" * 64, "g" * 64, "a" * 63])
def test_bad_sha256_is_rejected(sha):
    with pytest.raises(ValidationError):
        FileRef(path="m.onnx", size=10, sha256=sha)


@pytest.mark.parametrize("size", [0, -1])
def test_non_positive_size_is_rejected(size):
    with pytest.raises(ValidationError):
        FileRef(path="m.onnx", size=size, sha256=SHA)


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        FileRef.model_validate({"path": "m.onnx", "size": 1, "sha256": SHA, "sha": SHA})


@pytest.mark.parametrize(
    ("path", "ok"),
    [
        ("model.onnx", True),
        ("onnx-fp32/model.onnx", True),
        (".hidden/model.onnx", True),
        ("a/.b", True),
        ("/etc/passwd", False),
        ("../secret", False),
        ("a/../b", False),
        ("./model.onnx", False),
        ("a//b", False),
        ("a/", False),
        ("a\\b", False),
        ("C:/model.onnx", False),
        ("..", False),
        ("", False),
    ],
)
def test_json_schema_pattern_matches_python_rules(path, ok):
    python_ok = True
    try:
        FileRef(path=path, size=1, sha256=SHA)
    except ValidationError:
        python_ok = False
    schema_ok = re.fullmatch(SAFE_PATH_PATTERN, path) is not None
    assert python_ok == ok
    assert schema_ok == ok
