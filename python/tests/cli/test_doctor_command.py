from typer.testing import CliRunner

from modelport.cli import app

runner = CliRunner()


def test_doctor_runs():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.stdout
    assert "python" in result.stdout
    assert "Optional packages" in result.stdout
