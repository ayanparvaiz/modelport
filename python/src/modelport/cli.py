"""The `modelport` command line tool."""

from __future__ import annotations

import typer
from rich.console import Console

from . import __version__

app = typer.Typer(
    name="modelport",
    help="Prepare PyTorch, Hugging Face, and GGUF models for Flutter apps.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()
err_console = Console(stderr=True)


def _print_version(value: bool) -> None:
    if value:
        console.print(f"modelport {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_print_version,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Prepare PyTorch, Hugging Face, and GGUF models for Flutter apps."""
