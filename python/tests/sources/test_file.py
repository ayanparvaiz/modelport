from pathlib import Path

import pytest

from modelport.sources import SourceError, load_source
from modelport.sources.file import split_file_spec

SCRIPT = """
import torch
from modelport.manifest import DType, ImagePreprocess, InputSpec, ResizeSpec, Task
from modelport.sources import SourceModel


def build():
    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2)).eval()
    return SourceModel(
        module=net,
        example_inputs=(torch.zeros(1, 3, 8, 8),),
        id="tiny_net",
        task=Task.IMAGE_CLASSIFICATION,
        license="MIT",
        inputs=[
            InputSpec(
                name="image",
                dtype=DType.FLOAT32,
                shape=[1, 3, 8, 8],
                layout="NCHW",
                preprocess=ImagePreprocess(resize=ResizeSpec(size=(8, 8))),
            )
        ],
        output_names=["scores"],
        labels=["no", "yes"],
    )


def wrong():
    return 42
"""


@pytest.fixture
def script(tmp_path) -> Path:
    pytest.importorskip("torch")
    path = tmp_path / "my_model.py"
    path.write_text(SCRIPT, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("spec", "path", "function"),
    [
        ("net.py", "net.py", "build"),
        ("models/net.py:make", "models/net.py", "make"),
        ("C:/models/net.py", "C:/models/net.py", "build"),
        ("C:/models/net.py:make", "C:/models/net.py", "make"),
    ],
)
def test_split_file_spec(spec, path, function):
    assert split_file_spec(spec) == (Path(path), function)


def test_default_build_function(script):
    model = load_source(f"file:{script}")
    assert model.id == "tiny_net"
    assert model.source == "file:my_model.py:build"
    assert tuple(model.module(*model.example_inputs).shape) == (1, 2)


def test_named_function_must_return_source_model(script):
    with pytest.raises(SourceError, match="must return"):
        load_source(f"file:{script}:wrong")


def test_missing_function(script):
    with pytest.raises(SourceError, match="no function named 'nope'"):
        load_source(f"file:{script}:nope")


def test_missing_file(tmp_path):
    with pytest.raises(SourceError, match="does not exist"):
        load_source(f"file:{tmp_path / 'missing.py'}")
