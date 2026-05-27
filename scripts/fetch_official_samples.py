#!/usr/bin/env python3
"""Download official US identity document reference samples from the catalog."""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.table import Table

from docpii.data.official_samples import (
    DEFAULT_CATALOG,
    DEFAULT_INDEX_PATH,
    DEFAULT_OUTPUT_DIR,
    download_official_samples,
)

console = Console()


@click.command()
@click.option(
    "--catalog",
    type=click.Path(path_type=str),
    default=str(DEFAULT_CATALOG),
    show_default=True,
    help="YAML catalog of official sample URLs.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=str),
    default=str(DEFAULT_OUTPUT_DIR),
    show_default=True,
    help="Directory for downloaded files.",
)
@click.option(
    "--index",
    "index_path",
    type=click.Path(path_type=str),
    default=str(DEFAULT_INDEX_PATH),
    show_default=True,
    help="JSON provenance index output path.",
)
@click.option("--overwrite", is_flag=True, help="Re-download existing files.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(
    catalog: str,
    output_dir: str,
    index_path: str,
    overwrite: bool,
    verbose: bool,
) -> None:
    """Fetch official .gov ID reference samples listed in the catalog."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    report = download_official_samples(
        catalog_path=catalog,
        output_dir=output_dir,
        index_path=index_path,
        overwrite=overwrite,
    )

    table = Table(title="Official sample downloads")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Path / error")
    for result in report.results:
        status = "OK" if result.success else "FAIL"
        detail = result.local_path or (result.error or "")
        table.add_row(result.sample_id, status, detail)
    console.print(table)
    console.print(
        f"\n[bold]{report.success_count}[/bold] succeeded, "
        f"[bold]{report.failure_count}[/bold] failed"
    )
    console.print(f"Index written to [green]{index_path}[/green]")

    if report.failure_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
