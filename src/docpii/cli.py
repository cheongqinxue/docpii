"""Command-line interface for docpii."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from docpii.config import Settings
from docpii.evaluation.runner import EvaluationRunner
from docpii.inference.client import NuExtractClient
from docpii.templates.nuextract import default_extraction_template, template_to_json

console = Console()


def _configure_logging(verbose: bool) -> None:
    """Configure root logging level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@click.group()
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """PII/PCI/PHI document extraction with NuExtract3."""
    _configure_logging(verbose)


@main.command("extract")
@click.argument("document", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Write JSON result to this file.",
)
def extract_cmd(document: Path, output: Path | None) -> None:
    """Extract entities from a single document."""
    client = NuExtractClient()
    result = client.extract_document(str(document))
    payload = result.model_dump(mode="json")
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        console.print(f"[green]Wrote[/green] {output}")
    else:
        console.print(text)


@main.command("template")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Write template JSON to this file.",
)
def template_cmd(output: Path | None) -> None:
    """Print the default NuExtract PII extraction template."""
    template = default_extraction_template()
    text = template_to_json(template)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        console.print(f"[green]Wrote[/green] {output}")
    else:
        console.print(text)


@main.command("eval")
@click.argument("manifest", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--report",
    "-r",
    type=click.Path(path_type=Path),
    default=Path("outputs/eval_report.json"),
    help="Path for JSON evaluation report.",
)
def eval_cmd(manifest: Path, report: Path) -> None:
    """Run systematic evaluation from a manifest file."""
    runner = EvaluationRunner()
    benchmark = runner.run_manifest(manifest)

    table = Table(title="Evaluation Results")
    table.add_column("Sample")
    table.add_column("Score", justify="right")
    table.add_column("Status")
    for result in benchmark.results:
        status = "FAIL" if result.parse_error else "OK"
        table.add_row(result.document_id, f"{result.average_score:.3f}", status)
    console.print(table)
    console.print(
        f"\nMean score: [bold]{benchmark.mean_score:.3f}[/bold]  "
        f"Failure rate: [bold]{benchmark.failure_rate:.1%}[/bold]"
    )

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(benchmark.to_dict(), indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Report saved to[/green] {report}")


@main.command("train")
@click.argument("train_jsonl", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", default="checkpoints/docpii-qlora", show_default=True)
@click.option("--epochs", default=3, show_default=True, type=int)
def train_cmd(train_jsonl: Path, output_dir: str, epochs: int) -> None:
    """Fine-tune NuExtract3 with QLoRA (requires docpii[train])."""
    from docpii.training.qlora import QLoRAConfig, train_qlora

    config = QLoRAConfig(output_dir=output_dir, num_train_epochs=epochs)
    checkpoint = train_qlora(train_jsonl, config=config)
    console.print(f"[green]Saved adapter to[/green] {checkpoint}")


if __name__ == "__main__":
    main()
