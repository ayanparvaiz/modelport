from typing import ClassVar

import pytest

from modelport.errors import ModelPortError
from modelport.pipeline import export_bundle
from modelport.publish import model_card, publish_bundle


class FakeApi:
    calls: ClassVar[list] = []

    def create_repo(self, repo_id, **kwargs):
        FakeApi.calls.append(("create_repo", repo_id, kwargs))

    def upload_folder(self, **kwargs):
        FakeApi.calls.append(("upload_folder", kwargs))


@pytest.fixture
def bundle(tiny_source, tmp_path):
    pytest.importorskip("onnxscript")
    bundle, _ = export_bundle(tiny_source, tmp_path, ["onnx"])
    return bundle


@pytest.fixture
def fake_hub(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    FakeApi.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    return FakeApi


def test_publish_uploads_only_listed_files(bundle, fake_hub):
    (bundle.root / "scratch.txt").write_text("not for upload", encoding="utf-8")
    result = publish_bundle(bundle.root, "someone/tiny-cnn")
    assert result.location == "hf://someone/tiny-cnn"
    (create, upload) = fake_hub.calls
    assert create[1] == "someone/tiny-cnn" and create[2]["exist_ok"]
    patterns = upload[1]["allow_patterns"]
    assert "modelport.json" in patterns and "README.md" in patterns
    assert "onnx-fp32/model.onnx" in patterns and "golden/logits.bin" in patterns
    assert "scratch.txt" not in patterns
    assert (bundle.root / "README.md").read_text(encoding="utf-8").startswith("---\nlicense: mit")


def test_publish_refuses_inconsistent_bundle(bundle, fake_hub):
    (bundle.root / "labels.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ModelPortError, match="modelport pack"):
        publish_bundle(bundle.root, "someone/tiny-cnn")
    assert fake_hub.calls == []


def test_publish_checks_repo_id(bundle, fake_hub):
    with pytest.raises(ModelPortError, match="not a repo id"):
        publish_bundle(bundle.root, "just-a-name")


def test_model_card_lists_variants(bundle):
    card = model_card(bundle.read_manifest(), "someone/tiny-cnn")
    assert "| `onnx-fp32` | onnx | fp32 |" in card
    assert "hf://someone/tiny-cnn" in card


def test_github_manifest_points_at_release_assets(bundle):
    from modelport.publish import github_manifest

    remote = github_manifest(bundle.read_manifest(), "me/models", "zoo-1")
    urls = [ref.url for ref in remote.files()]
    assert all(ref.path is None for ref in remote.files())
    assert (
        "https://github.com/me/models/releases/download/zoo-1/tiny_cnn--onnx-fp32--model.onnx"
        in urls
    )
    assert remote.files()[0].sha256 == bundle.read_manifest().files()[0].sha256


def test_publish_to_github_creates_release_and_uploads(bundle):
    import subprocess

    from modelport.publish import publish_to_github

    calls = []

    def fake_run(args):
        calls.append(args)
        code = 1 if args[:3] == ["gh", "release", "view"] else 0
        return subprocess.CompletedProcess(args, code, "", "")

    result = publish_to_github(bundle.root, "me/models", "zoo-1", run=fake_run)
    assert [c[2] for c in calls] == ["view", "create", "upload"]
    upload = calls[-1]
    assert "--clobber" in upload
    assert any(a.endswith("tiny_cnn.json") for a in upload)
    assert result.location == "https://github.com/me/models/releases/download/zoo-1/tiny_cnn.json"
    assert "tiny_cnn--golden--logits.bin" in result.files
