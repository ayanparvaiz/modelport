"""The `modelport` command line tool."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from . import __version__
from .doctor import collect_report
from .errors import ModelPortError
from .inspection import ModelInfo, TensorInfo, inspect_model
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


@app.command()
def doctor() -> None:
    """Check Python, optional packages, and disk space."""
    report = collect_report()
    summary = Table.grid(padding=(0, 2))
    summary.add_row("modelport", report.modelport_version)
    summary.add_row("python", report.python_version)
    summary.add_row("platform", report.platform)
    summary.add_row("free disk", f"{report.free_disk_bytes / 1000**3:.1f} GB")
    console.print(summary)
    console.print()

    table = Table(title="Optional packages", title_justify="left", show_edge=False)
    table.add_column("package")
    table.add_column("extra")
    table.add_column("version")
    for package in report.packages:
        version = package.version or "[yellow]not installed[/yellow]"
        table.add_row(package.name, package.extra, version)
    console.print(table)

    for problem in report.problems:
        console.print(f"[red]![/red] {problem}", highlight=False)
    if report.hints:
        console.print("\nTo enable more formats:")
        for hint in report.hints:
            console.print(f"  {hint}", markup=False, highlight=False)
    if not report.problems and not report.hints:
        console.print("\n[green]Everything is installed.[/green]")


@app.command("inspect")
def inspect_command(
    path: Annotated[Path, typer.Argument(help="A .onnx, .pte, or .gguf model file.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    """Show a model file's inputs, outputs, and metadata."""
    try:
        info = inspect_model(path)
    except ModelPortError as error:
        err_console.print(f"[red]Error:[/red] {error}", highlight=False)
        raise typer.Exit(code=1) from error

    if as_json:
        data = asdict(info)
        data["path"] = str(info.path)
        typer.echo(json.dumps(data, indent=2))
        return
    _print_model_info(info)


def _format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1000
    raise AssertionError("unreachable")


def _tensor_table(title: str, tensors: list[TensorInfo]) -> Table:
    table = Table(title=title, title_justify="left", show_edge=False)
    table.add_column("name")
    table.add_column("dtype")
    table.add_column("shape")
    for tensor in tensors:
        table.add_row(tensor.name, tensor.dtype, str(tensor.shape))
    return table


def _print_model_info(info: ModelInfo) -> None:
    summary = Table.grid(padding=(0, 2))
    summary.add_row("file", str(info.path))
    summary.add_row("format", info.format)
    summary.add_row("size", f"{_format_size(info.size)} ({info.size:,} bytes)")
    summary.add_row("sha256", info.sha256)
    console.print(summary)
    if info.inputs:
        console.print()
        console.print(_tensor_table("Inputs", info.inputs))
    if info.outputs:
        console.print()
        console.print(_tensor_table("Outputs", info.outputs))
    if info.metadata:
        console.print()
        meta = Table(title="Metadata", title_justify="left", show_edge=False)
        meta.add_column("key")
        meta.add_column("value")
        for key, value in info.metadata.items():
            meta.add_row(key, value)
        console.print(meta)
