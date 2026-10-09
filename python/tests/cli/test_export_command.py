import pytest
from typer.testing import CliRunner

from modelport.cli import app
from modelport.manifest import Manifest

runner = CliRunner()

SCRIPT = """
import torch
from modelport.manifest import DType, ImagePreprocess, InputSpec, ResizeSpec, Task
from modelport.sources import SourceModel


def build():
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 2, 3), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()
    ).eval()
    return SourceModel(
        module=net,
        example_inputs=(torch.zeros(1, 3, 12, 12),),
        id="cli_net",
        task=Task.IMAGE_CLASSIFICATION,
        license="MIT",
        inputs=[
            InputSpec(
                name="image",
                dtype=DType.FLOAT32,
                shape=[1, 3, 12, 12],
                layout="NCHW",
                preprocess=ImagePreprocess(resize=ResizeSpec(size=(12, 12))),
            )
        ],
        output_names=["scores"],
        labels=["a", "b"],
    )
"""


@pytest.fixture
def script(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnxscript")
    path = tmp_path / "net.py"
    path.write_text(SCRIPT, encoding="utf-8")
    return path


def test_export_creates_bundle(script, tmp_path):
    out = tmp_path / "dist"
    result = runner.invoke(app, ["export", f"file:{script}", "--out", str(out)])
    assert result.exit_code == 0, result.output
    manifest = Manifest.from_file(out / "cli_net" / "modelport.json")
    assert manifest.variants[0].id == "onnx-fp32"
    assert "modelport verify" in result.output


def test_export_refuses_to_overwrite(script, tmp_path):
    out = tmp_path / "dist"
    assert runner.invoke(app, ["export", f"file:{script}", "-o", str(out)]).exit_code == 0
    result = runner.invoke(app, ["export", f"file:{script}", "-o", str(out)])
    assert result.exit_code == 1
    assert "--force" in result.output
    result = runner.invoke(app, ["export", f"file:{script}", "-o", str(out), "--force"])
    assert result.exit_code == 0, result.output


def test_export_bad_source_fails():
    result = runner.invoke(app, ["export", "nope:thing"])
    assert result.exit_code == 1
