#!/usr/bin/env python3
"""Compute the frequency and percentage of all 54 STM moves in reconstruction solutions.

Parses validation (or any) reconstruction dataset, counts each move in solutions
(excluding scrambles), and exports both CSV and JSON frequency statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from cubesight_ml.datasets import compute_move_frequencies

if TYPE_CHECKING:
    import pandas as pd

console = Console()
logger = logging.getLogger("compute_move_frequencies")


def setup_console_encoding() -> None:
    """Ensure utf-8 encoding for stdout on Windows terminals."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def make_ascii_bar(pct: float, max_pct: float = 20.0, bar_len: int = 15) -> str:
    """Generates a small visual bar representing percentage."""
    filled = round((min(pct, max_pct) / max_pct) * bar_len)
    return "█" * filled + "░" * (bar_len - filled)


def print_move_frequencies_summary(
    df: pd.DataFrame,
    dataset_path: Path,
    sort_by: str = "percentage",
    top_n: int = 20,
) -> None:
    """Prints insightful summary tables of STM move frequencies using Rich."""
    total_moves = int(df["count"].sum())

    # Category Breakdown
    cat_df = (
        df.groupby("category")[["count", "percentage"]]
        .sum()
        .sort_values(by="count", ascending=False)
    )

    cat_table = Table(
        title=f"STM Move Category Breakdown ({dataset_path.name})",
        border_style="magenta",
    )
    cat_table.add_column("Category", style="bold yellow")
    cat_table.add_column("Move Count", style="cyan", justify="right")
    cat_table.add_column("Percentage", style="bold green", justify="right")
    cat_table.add_column("Distribution Bar", style="blue")

    for cat, row in cat_df.iterrows():
        cnt = int(row["count"])
        pct = float(row["percentage"])
        bar = make_ascii_bar(pct, max_pct=100.0, bar_len=20)
        cat_table.add_row(str(cat), f"{cnt:,}", f"{pct:6.2f}%", bar)

    cat_table.add_section()
    cat_table.add_row("Total STM Moves", f"{total_moves:,}", "100.00%", "")
    console.print(cat_table)

    # Moves breakdown table
    if sort_by == "percentage":
        display_df = df.sort_values(by="count", ascending=False).reset_index(
            drop=True
        )
        title_extra = f"Top {top_n} Moves by Frequency"
    else:
        display_df = df.sort_values(by="move_index").reset_index(drop=True)
        title_extra = "All 54 STM Moves (by Index)"

    moves_table = Table(
        title=f"{title_extra} - Total Moves: {total_moves:,}",
        border_style="cyan",
    )
    moves_table.add_column("Rank", style="dim", justify="right")
    moves_table.add_column("Index", style="white", justify="right")
    moves_table.add_column("Move", style="bold white")
    moves_table.add_column("Category", style="yellow")
    moves_table.add_column("Count", style="cyan", justify="right")
    moves_table.add_column("Percentage", style="bold green", justify="right")
    moves_table.add_column("Share", style="blue")

    max_pct = (
        float(display_df["percentage"].max()) if not display_df.empty else 1.0
    )
    shown_rows = (
        display_df.head(top_n) if sort_by == "percentage" else display_df
    )

    for rank, (_, row) in enumerate(shown_rows.iterrows(), start=1):
        idx = int(row["move_index"])
        move = str(row["move"])
        cat = str(row["category"])
        cnt = int(row["count"])
        pct = float(row["percentage"])
        bar = make_ascii_bar(pct, max_pct=max_pct, bar_len=15)
        moves_table.add_row(
            str(rank), str(idx), move, cat, f"{cnt:,}", f"{pct:6.3f}%", bar
        )

    console.print(moves_table)


def main() -> None:
    setup_console_encoding()
    parser = argparse.ArgumentParser(
        description="Compute frequency and percentage of all 54 STM moves in solution reconstructions."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/val/reco_cfop_val.jsonl",
        help="Input dataset path (.jsonl or .csv). Default: data/val/reco_cfop_val.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/val/move_frequencies.csv",
        help="Output CSV file path. Default: data/val/move_frequencies.csv",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="data/val/move_percentages.json",
        help="Output JSON file path (move -> percentage mapping). Default: data/val/move_percentages.json",
    )
    parser.add_argument(
        "--output-full-json",
        type=str,
        default="data/val/move_frequencies.json",
        help="Output full metadata JSON file path. Default: data/val/move_frequencies.json",
    )
    parser.add_argument(
        "--include-inspection",
        action="store_true",
        help="Include rotations/moves made during inspection phase (default: False).",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Only count moves from simulator-verified valid solutions.",
    )
    parser.add_argument(
        "--sort",
        type=str,
        choices=["percentage", "index"],
        default="percentage",
        help="Sorting order for terminal summary display.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of moves to display in top moves summary table (default: 25).",
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset)

    console.print(
        f"[bold cyan]Parsing {dataset_path} for STM move occurrences in solutions...[/bold cyan]"
    )
    if args.include_inspection:
        console.print("[dim]Including inspection phase moves[/dim]")
    if args.valid_only:
        console.print(
            "[dim]Filtering to simulator-verified valid solves only[/dim]"
        )

    df = compute_move_frequencies(
        dataset_path=dataset_path,
        include_inspection=args.include_inspection,
        valid_only=args.valid_only,
    )

    # 1. Save CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # 2. Save JSON clean map {move: percentage}
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    pct_map = {
        row["move"]: round(float(row["percentage"]), 4)
        for _, row in df.iterrows()
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(pct_map, f, indent=2)

    # 3. Save full JSON
    out_full_json = Path(args.output_full_json)
    out_full_json.parent.mkdir(parents=True, exist_ok=True)
    total_moves = int(df["count"].sum())
    full_data = {
        "dataset": str(dataset_path),
        "total_moves": total_moves,
        "include_inspection": args.include_inspection,
        "valid_only": args.valid_only,
        "percentages": pct_map,
        "moves": df.to_dict(orient="records"),
    }
    with out_full_json.open("w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2)

    # Print Summary Tables
    print_move_frequencies_summary(
        df=df,
        dataset_path=dataset_path,
        sort_by=args.sort,
        top_n=args.top_n,
    )

    console.print(
        "\n[bold green]Move frequencies successfully generated:[/bold green]"
    )
    console.print(
        f"  - [bold]CSV Table[/bold]: {out_csv} ({out_csv.stat().st_size:,} bytes)"
    )
    console.print(
        f"  - [bold]Percentages JSON[/bold]: {out_json} ({out_json.stat().st_size:,} bytes)"
    )
    console.print(
        f"  - [bold]Full JSON[/bold]: {out_full_json} ({out_full_json.stat().st_size:,} bytes)"
    )


if __name__ == "__main__":
    main()
