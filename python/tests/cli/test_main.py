from typer.testing import CliRunner

from modelport import __version__
from modelport.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"modelport {__version__}"


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "Usage" in result.stdout
