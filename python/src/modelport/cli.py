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


@app.command()
def export(
    source: Annotated[
        str,
        typer.Argument(
            help="Model to export: torchvision:<name>, hf:<repo or folder>, or file:<script.py>."
        ),
    ],
    target: Annotated[
        list[str],
        typer.Option(
            "--target", "-t", help="Format to export: onnx, executorch. Repeat or comma-separate."
        ),
    ] = ["onnx"],  # noqa: B006 - Typer reads list defaults
    out: Annotated[Path, typer.Option("--out", "-o", help="Folder for bundles.")] = Path("dist"),
    license: Annotated[
        str | None, typer.Option(help="SPDX license of the weights, if the source has none.")
    ] = None,
    version: Annotated[str, typer.Option(help="Version of this bundle.")] = "1.0.0",
    sample_image: Annotated[
        Path | None,
        typer.Option(
            "--sample-image", help="Picture used for golden data. A fixed one by default."
        ),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing bundle.")] = False,
) -> None:
    """Convert a model and write a bundle with modelport.json and golden test data."""
    from .pipeline import export_bundle
    from .preprocess import load_image
    from .sources import load_source

    targets = [t.strip() for item in target for t in item.split(",") if t.strip()]
    try:
        with console.status(f"Loading {source}"):
            model = load_source(source, license=license)
        image = load_image(sample_image) if sample_image else None
        with console.status(f"Exporting {model.id} to {', '.join(targets)}"):
            bundle, manifest = export_bundle(
                model, out, targets, image=image, version=version, overwrite=force
            )
    except ModelPortError as error:
        err_console.print(f"[red]Error:[/red] {error}", highlight=False)
        raise typer.Exit(code=1) from error

    table = Table(title=f"Wrote {bundle.root}", title_justify="left", show_edge=False)
    table.add_column("file")
    table.add_column("size", justify="right")
    for ref in manifest.files():
        if ref.path is not None:
            table.add_row(ref.path, _format_size(ref.size))
    table.add_row("modelport.json", _format_size(bundle.manifest_path.stat().st_size))
    console.print(table)
    console.print(f"\nNext: modelport verify {bundle.root}", highlight=False)


@app.command()
def verify(
    bundle: Annotated[Path, typer.Argument(help="Bundle folder that contains modelport.json.")],
) -> None:
    """Run every variant on the golden input and compare with the expected output."""
    from .verify import verify_bundle

    try:
        with console.status(f"Verifying {bundle}"):
            results = verify_bundle(bundle)
    except ModelPortError as error:
        err_console.print(f"[red]Error:[/red] {error}", highlight=False)
        raise typer.Exit(code=1) from error

    table = Table(show_edge=False)
    table.add_column("variant", no_wrap=True)
    for column in ("output", "max diff", "cosine", "top-1", "result"):
        table.add_column(column, no_wrap=True)
    for result in results:
        if result.skipped:
            table.add_row(result.variant_id, "", "", "", "", "[yellow]skipped[/]")
            continue
        for check in result.outputs:
            ok = check.within_tolerance and check.top1_match is not False
            top1 = {None: "", True: "same", False: "[red]different[/]"}[check.top1_match]
            table.add_row(
                result.variant_id,
                check.name,
                f"{check.max_abs_diff:.2e}",
                f"{check.cosine:.6f}",
                top1,
                "[green]pass[/]" if ok else "[red]fail[/]",
            )
    console.print(table)
    for result in results:
        if result.skipped:
            console.print(f"[yellow]![/] {result.variant_id}: {result.skipped}", highlight=False)

    failed = [r for r in results if not r.passed and r.skipped is None]
    if failed or not any(r.passed for r in results):
        raise typer.Exit(code=1)


@app.command()
def quantize(
    bundle: Annotated[Path, typer.Argument(help="Bundle folder with an onnx fp32 variant.")],
    fp16: Annotated[
        bool, typer.Option("--fp16", help="Add an fp16 variant (about half size).")
    ] = False,
    int8: Annotated[
        bool, typer.Option("--int8", help="Add an int8 variant of MatMul and Gemm weights.")
    ] = False,
    allow_top1_change: Annotated[
        bool, typer.Option(help="Keep a variant even if it changes the golden top-1 class.")
    ] = False,
) -> None:
    """Add smaller ONNX variants and record how far they drift from the original."""
    from .quantize import Kind, quantize_bundle

    kinds: list[Kind] = []
    if fp16:
        kinds.append("fp16")
    if int8:
        kinds.append("int8")
    if not kinds:
        err_console.print("Choose at least one of --fp16 or --int8.")
        raise typer.Exit(code=2)
    try:
        with console.status(f"Quantizing {bundle}"):
            results = quantize_bundle(bundle, kinds, allow_top1_change=allow_top1_change)
    except ModelPortError as error:
        err_console.print(f"[red]Error:[/red] {error}", highlight=False)
        raise typer.Exit(code=1) from error

    table = Table(show_edge=False)
    for column in ("variant", "size", "of fp32", "max |diff|", "tolerance", "top-1"):
        table.add_column(column)
    for result in results:
        variant = result.variant
        top1 = {None: "", True: "same", False: "[red]different[/]"}[result.top1_match]
        tolerance = variant.tolerance.atol if variant.tolerance else ""
        table.add_row(
            variant.id,
            _format_size(variant.file.size),
            f"{result.size_ratio:.0%}",
            f"{result.max_abs_diff:.2e}",
            str(tolerance),
            top1,
        )
    console.print(table)
    console.print(f"\nNext: modelport verify {bundle}", highlight=False)


@app.command()
def pack(
    bundle: Annotated[Path, typer.Argument(help="Bundle folder that contains modelport.json.")],
) -> None:
    """Refresh file sizes and hashes after manual edits, and list unlisted files."""
    from .pack import pack_bundle

    try:
        result = pack_bundle(bundle)
    except ModelPortError as error:
        err_console.print(f"[red]Error:[/red] {error}", highlight=False)
        raise typer.Exit(code=1) from error
    for path in result.changed:
        console.print(f"updated hash: {path}", highlight=False)
    for path in result.stray:
        console.print(
            f"[yellow]![/] not in manifest, will not be published: {path}", highlight=False
        )
    console.print(
        f"[green]✓[/green] {result.manifest.id} {result.manifest.version}, "
        f"{_format_size(result.total_bytes)} in {len(result.manifest.files())} files"
    )


@app.command()
def publish(
    bundle: Annotated[Path, typer.Argument(help="Bundle folder that contains modelport.json.")],
    hf: Annotated[str, typer.Option("--hf", help="Hugging Face repo id, like org/name.")],
    private: Annotated[bool, typer.Option(help="Create the repo as private.")] = False,
) -> None:
    """Upload a bundle to the Hugging Face Hub. Log in first with `hf auth login`."""
    from .publish import publish_bundle

    try:
        with console.status(f"Uploading {bundle} to {hf}"):
            result = publish_bundle(bundle, hf, private=private)
    except ModelPortError as error:
        err_console.print(f"[red]Error:[/red] {error}", highlight=False)
        raise typer.Exit(code=1) from error
    console.print(f"[green]✓[/green] Uploaded {len(result.files)} files to {result.url}")
    console.print(f"Load it in Flutter with: {result.hf_uri}", highlight=False)


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
