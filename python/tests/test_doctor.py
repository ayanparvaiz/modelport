from modelport.doctor import collect_report

ALL = {
    "onnx": "1.23.2",
    "onnxruntime": "1.31.0",
    "onnxscript": "0.7.2",
    "executorch": "1.5.1",
    "torch": "2.14.1",
    "gguf": "0.19.0",
    "torchvision": "0.29.1",
    "transformers": "5.19.0",
    "huggingface-hub": "1.0.0",
}


def test_everything_installed(tmp_path):
    report = collect_report(tmp_path, version_of=ALL.get)
    assert all(p.installed for p in report.packages)
    assert report.hints == []
    assert report.python_version.count(".") == 2


def test_missing_extra_gives_install_hint(tmp_path):
    versions = {k: v for k, v in ALL.items() if k != "gguf"}
    report = collect_report(tmp_path, version_of=versions.get)
    assert report.hints == ["pip install 'modelport-cli[gguf]'"]
    assert not any("gguf" in problem for problem in report.problems)


def test_partly_installed_extra_is_a_problem(tmp_path):
    versions = {k: v for k, v in ALL.items() if k != "onnxscript"}
    report = collect_report(tmp_path, version_of=versions.get)
    assert any("missing: onnxscript" in problem for problem in report.problems)
    assert report.hints == ["pip install 'modelport-cli[onnx]'"]
