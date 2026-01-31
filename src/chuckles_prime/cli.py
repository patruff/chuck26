"""CLI entry point for chuckles parody title generator.

Provides two subcommands:
- generate: Full pipeline (CSV -> generate -> archive -> dataset -> Hub)
- convert: Dataset-only from previously saved JSONL traces

All heavy imports are lazy -- placed inside command handlers for fast startup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console(stderr=True)


def _record_from_dict(d: dict) -> "GenerationRecord":
    """Reconstruct a GenerationRecord from a dict (as produced by asdict + json).

    Handles optional fields with sensible defaults so that JSONL files
    written by older versions or with missing fields still load correctly.

    Args:
        d: Dictionary from json.loads() of one JSONL line.

    Returns:
        Fully reconstructed GenerationRecord with nested dataclasses.
    """
    from chuckles_prime.types import AgentTrace, GenerationRecord, ParodyCandidate

    candidates = [
        ParodyCandidate(
            text=c["text"],
            phonetic_scores=c.get("phonetic_scores", {}),
            humor_note=c.get("humor_note", ""),
        )
        for c in d.get("candidates", [])
    ]
    trace_d = d.get("trace", {})
    trace = AgentTrace(
        steps=trace_d.get("steps", []),
        final_output=trace_d.get("final_output", ""),
        token_usage=trace_d.get("token_usage"),
        state=trace_d.get("state", "unknown"),
    )
    return GenerationRecord(
        input_title=d["input_title"],
        candidates=candidates,
        trace=trace,
        model_name=d.get("model_name", "unknown"),
        error=d.get("error"),
    )


def load_records(jsonl_path: Path) -> list:
    """Load GenerationRecord objects from a JSONL file.

    Args:
        jsonl_path: Path to JSONL file with one JSON record per line.

    Returns:
        List of GenerationRecord objects.
    """
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(_record_from_dict(json.loads(line)))
    return records


def _print_summary(records: list, con: Console) -> None:
    """Print a summary table of generation results.

    Args:
        records: List of GenerationRecord objects.
        con: Rich Console instance for output.
    """
    successes = sum(1 for r in records if r.error is None)
    failures = len(records) - successes
    total_candidates = sum(len(r.candidates) for r in records)

    table = Table(title="Generation Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Titles processed", str(len(records)))
    table.add_row("Successful", f"[green]{successes}[/green]")
    table.add_row("Failed", f"[red]{failures}[/red]" if failures else "0")
    table.add_row("Total candidates", str(total_candidates))
    con.print(table)


def cmd_generate(args: argparse.Namespace) -> None:
    """Full pipeline: CSV -> generate -> archive -> dataset -> push."""
    from chuckles_prime import config as _config
    from chuckles_prime import dataset as _dataset
    from chuckles_prime import generator as _generator
    from chuckles_prime import model as _model
    from chuckles_prime import tools as _tools
    from chuckles_prime import traces as _traces
    from chuckles_prime.types import AgentTrace, GenerationRecord

    # 1. Load config
    config = _config.load_config(args.settings)

    # 2. Read titles
    titles = _generator.read_input_titles(args.input)
    console.print(f"Loaded [bold]{len(titles)}[/bold] titles from {args.input}")

    # 3. Create model, load tools, create agent
    model = _model.create_model(config)
    parody_tool, phone_tool = _tools.load_parody_tools()
    agent = _generator.create_agent(model, parody_tool, phone_tool)

    # 4. Generate with rich Progress
    records: list[GenerationRecord] = []
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating", total=len(titles))
        for title in titles:
            try:
                record = _generator.generate_single(title, agent, parody_tool, config)
            except Exception as e:
                record = GenerationRecord(
                    input_title=title,
                    candidates=[],
                    trace=AgentTrace(
                        steps=[],
                        final_output="",
                        token_usage=None,
                        state="error",
                    ),
                    model_name=config.model_name,
                    error=str(e),
                )
            status = (
                "[green]OK[/green]"
                if not record.error
                else f"[red]ERR: {record.error}[/red]"
            )
            progress.console.print(f"  {title} ... {status}")
            progress.advance(task)
            records.append(record)

    # 5. Archive traces
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "traces.jsonl"
    count = _traces.archive_traces(records, traces_path)
    console.print(f"Archived [bold]{count}[/bold] traces to {traces_path}")

    # 6. Build GRPO dataset
    grpo_ds = _dataset.records_to_grpo_dataset(records)
    console.print(f"GRPO dataset: [bold]{len(grpo_ds)}[/bold] rows")

    # 7. Build DPO dataset
    model_records = {r.input_title: r for r in records if not r.error}
    dpo_ds = _dataset.build_dpo_dataset(config.human_examples, model_records)
    console.print(f"DPO dataset: [bold]{len(dpo_ds)}[/bold] rows")

    # 8. Push datasets
    if not args.no_push:
        if args.grpo_repo:
            _dataset.push_dataset(grpo_ds, args.grpo_repo)
            console.print(f"Pushed GRPO dataset to [bold]{args.grpo_repo}[/bold]")
        if args.dpo_repo:
            _dataset.push_dataset(dpo_ds, args.dpo_repo)
            console.print(f"Pushed DPO dataset to [bold]{args.dpo_repo}[/bold]")
        if not args.grpo_repo and not args.dpo_repo:
            console.print(
                "No --grpo-repo or --dpo-repo specified; skipping Hub push"
            )

    # 9. Summary
    _print_summary(records, console)


def cmd_convert(args: argparse.Namespace) -> None:
    """Dataset-only: JSONL -> dataset -> push."""
    from chuckles_prime import config as _config
    from chuckles_prime import dataset as _dataset

    # 1. Validate traces file exists
    traces_path = Path(args.traces)
    if not traces_path.exists():
        raise FileNotFoundError(f"Traces file not found: {traces_path}")

    # 2. Load records
    records = load_records(traces_path)
    console.print(f"Loaded [bold]{len(records)}[/bold] records from {traces_path}")

    # 3. Load config (needed for DPO human examples)
    config = _config.load_config(args.settings)

    # 4. Build GRPO dataset
    grpo_ds = _dataset.records_to_grpo_dataset(records)
    console.print(f"GRPO dataset: [bold]{len(grpo_ds)}[/bold] rows")

    # 5. Build DPO dataset
    model_records = {r.input_title: r for r in records if not r.error}
    dpo_ds = _dataset.build_dpo_dataset(config.human_examples, model_records)
    console.print(f"DPO dataset: [bold]{len(dpo_ds)}[/bold] rows")

    # 6. Push datasets
    if not args.no_push:
        if args.grpo_repo:
            _dataset.push_dataset(grpo_ds, args.grpo_repo)
            console.print(f"Pushed GRPO dataset to [bold]{args.grpo_repo}[/bold]")
        if args.dpo_repo:
            _dataset.push_dataset(dpo_ds, args.dpo_repo)
            console.print(f"Pushed DPO dataset to [bold]{args.dpo_repo}[/bold]")
        if not args.grpo_repo and not args.dpo_repo:
            console.print(
                "No --grpo-repo or --dpo-repo specified; skipping Hub push"
            )

    # 7. Summary
    _print_summary(records, console)


def cmd_label(args: argparse.Namespace) -> None:
    """Launch the labeling web app."""
    from chuckles_prime.labeler import _prepare_items, create_app

    traces_path = Path(args.traces)
    if not traces_path.exists():
        raise FileNotFoundError(f"Traces file not found: {traces_path}")

    records = load_records(traces_path)
    console.print(f"Loaded [bold]{len(records)}[/bold] records from {traces_path}")

    items = _prepare_items(records)
    console.print(f"[bold]{len(items)}[/bold] titles with 2+ candidates ready for labeling")

    if not items:
        console.print("[yellow]No labelable items found (need records with 2+ candidates)[/yellow]")
        return

    labels_path = Path(args.labels) if args.labels else traces_path.parent / "labels.json"
    app = create_app(items, labels_path)

    console.print(
        f"Starting labeler at [bold]http://localhost:{args.port}[/bold]  "
        "(Ctrl-C to stop)"
    )
    app.run(host="127.0.0.1", port=args.port, debug=False)


def cmd_export_labels(args: argparse.Namespace) -> None:
    """Export human labels to a DPO dataset."""
    from datasets import Dataset

    from chuckles_prime.dataset import push_dataset
    from chuckles_prime.labeler import build_dpo_from_labels, load_labels

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    data = load_labels(labels_path)
    total = len(data.get("labels", []))
    console.print(f"Loaded [bold]{total}[/bold] labels from {labels_path}")

    rows = build_dpo_from_labels(labels_path)
    console.print(
        f"DPO dataset: [bold]{len(rows)}[/bold] rows "
        f"({total - len(rows)} excluded as both_bad)"
    )

    if not rows:
        console.print("[yellow]No usable labels (all marked both_bad)[/yellow]")
        return

    ds = Dataset.from_list(rows)

    if not args.no_push:
        push_dataset(ds, args.dpo_repo)
        console.print(f"Pushed DPO dataset to [bold]{args.dpo_repo}[/bold]")
    else:
        console.print(f"Dataset built with {len(ds)} rows (--no-push, skipping Hub push)")


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Factored out from main() to enable testing argument parsing
    without invoking sys.exit().

    Returns:
        Configured ArgumentParser with generate and convert subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="chuckles",
        description="Generate phonetically sound parody titles and RLVR datasets",
    )
    sub = parser.add_subparsers(dest="command")

    # -- generate subcommand --
    gen = sub.add_parser(
        "generate", help="Full pipeline: CSV -> parodies -> datasets -> Hub"
    )
    gen.add_argument("input", help="Path to input CSV with 'title' column")
    gen.add_argument(
        "--settings",
        default="settings.json",
        help="Path to settings JSON (default: settings.json)",
    )
    gen.add_argument(
        "--output-dir",
        default="output",
        help="Directory for traces JSONL (default: output/)",
    )
    gen.add_argument(
        "--grpo-repo",
        default=None,
        help="HF Hub repo ID for GRPO dataset (e.g. user/chuckles-grpo)",
    )
    gen.add_argument(
        "--dpo-repo",
        default=None,
        help="HF Hub repo ID for DPO dataset (e.g. user/chuckles-dpo)",
    )
    gen.add_argument(
        "--no-push", action="store_true", help="Skip pushing datasets to Hub"
    )
    gen.set_defaults(func=cmd_generate)

    # -- convert subcommand --
    conv = sub.add_parser(
        "convert", help="Convert saved traces to datasets (no generation)"
    )
    conv.add_argument("traces", help="Path to traces JSONL file")
    conv.add_argument(
        "--settings",
        default="settings.json",
        help="Path to settings JSON (needed for DPO human examples)",
    )
    conv.add_argument(
        "--grpo-repo",
        default=None,
        help="HF Hub repo ID for GRPO dataset",
    )
    conv.add_argument(
        "--dpo-repo",
        default=None,
        help="HF Hub repo ID for DPO dataset",
    )
    conv.add_argument(
        "--no-push", action="store_true", help="Skip pushing datasets to Hub"
    )
    conv.set_defaults(func=cmd_convert)

    # -- label subcommand --
    lbl = sub.add_parser(
        "label", help="Launch web UI for labeling parody preferences"
    )
    lbl.add_argument("traces", help="Path to traces JSONL file")
    lbl.add_argument(
        "--labels",
        default=None,
        help="Path to labels JSON (default: labels.json next to traces)",
    )
    lbl.add_argument(
        "--port",
        type=int,
        default=5117,
        help="Port for the labeler web server (default: 5117)",
    )
    lbl.set_defaults(func=cmd_label)

    # -- export-labels subcommand --
    exp = sub.add_parser(
        "export-labels", help="Export human labels to DPO dataset"
    )
    exp.add_argument("labels", help="Path to labels JSON file")
    exp.add_argument(
        "--dpo-repo",
        required=True,
        help="HF Hub repo ID for DPO dataset",
    )
    exp.add_argument(
        "--no-push",
        action="store_true",
        help="Build dataset but skip pushing to Hub",
    )
    exp.set_defaults(func=cmd_export_labels)

    return parser


def main() -> None:
    """CLI entry point for chuckles."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
