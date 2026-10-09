"""The committed cross-language fixtures must match the Python reference."""

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_preprocess_fixtures.py"
COMMITTED = Path(__file__).resolve().parents[2] / "spec" / "fixtures" / "preprocess"


def load_script():
    spec = importlib.util.spec_from_file_location("make_preprocess_fixtures", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixtures_are_up_to_date(tmp_path):
    load_script().write(tmp_path)
    hint = "Regenerate with: uv run python scripts/make_preprocess_fixtures.py"
    assert (tmp_path / "cases.json").read_text() == (COMMITTED / "cases.json").read_text(), hint
    for png in tmp_path.glob("*.png"):
        fresh = np.asarray(Image.open(png))
        committed = np.asarray(Image.open(COMMITTED / png.name))
        assert np.array_equal(fresh, committed), f"{png.name}: {hint}"
    for expected in (tmp_path / "expected").glob("*.bin"):
        assert expected.read_bytes() == (COMMITTED / "expected" / expected.name).read_bytes(), (
            f"{expected.name}: {hint}"
        )
