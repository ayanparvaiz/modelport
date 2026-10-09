"""The `modelport` command line tool."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console

from . import __version__
from .manifest import MANIFEST_FILENAME, Manifest
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


@app.command()
def validate(
    paths: Annotated[
        list[Path],
        typer.Argument(help="modelport.json files, or bundle folders that contain one."),
    ],
) -> None:
    """Check that manifests follow the spec. Exits with code 1 if any is invalid."""
    failed = 0
    for path in paths:
        file = path / MANIFEST_FILENAME if path.is_dir() else path
        try:
            manifest = Manifest.from_file(file)
        except ValidationError as error:
            failed += 1
            console.print(f"[red]✗[/red] {file}")
            for issue in error.errors():
                where = ".".join(str(part) for part in issue["loc"]) or "(root)"
                console.print(f"    {where}: {issue['msg']}", markup=False, highlight=False)
        except (OSError, ValueError) as error:
            failed += 1
            console.print(f"[red]✗[/red] {file}")
            console.print(f"    {error}", markup=False, highlight=False)
        else:
            console.print(
                f"[green]✓[/green] {file}  {manifest.id} {manifest.version}, "
                f"{manifest.task}, {len(manifest.variants)} variant(s)"
            )
    if failed:
        raise typer.Exit(code=1)
