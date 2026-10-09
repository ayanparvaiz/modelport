"""The `modelport` command line tool."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from . import __version__
from .manifest.schema import render_schema

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
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_print_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Prepare PyTorch, Hugging Face, and GGUF models for Flutter apps."""


@app.command()
def schema(
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="File or folder to write manifest.schema.json to. Prints to stdout if omitted.",
        ),
    ] = None,
) -> None:
    """Print or write the JSON Schema for modelport.json."""
    text = render_schema()
    if output is None:
        typer.echo(text, nl=False)
        return
    target = output / "manifest.schema.json" if output.is_dir() else output
    target.write_text(text, encoding="utf-8")
    err_console.print(f"Wrote {target}")
