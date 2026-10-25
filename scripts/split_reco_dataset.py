#!/usr/bin/env python3
"""Split reconstructions dataset into validation and test sets.

Shuffles with a deterministic seed (default: 42) and copies 2/3 into val/ and
1/3 into test/, keeping both JSONL and CSV representations perfectly aligned.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
from rich.console import Console
from rich.table import Table

from cubesight_ml.datasets import split_reconstructions

console = Console()
logger = logging.getLogger("split_reco_dataset")


def setup_console_encoding() -> None:
    """Ensure utf-8 encoding for stdout on Windows terminals."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def print_split_summary(
    total_count: int,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> None:
    """Prints a comparison summary table for val and test splits."""
    table = Table(
        title="Dataset Split Summary (Seed: " + str(seed) + ")",
        border_style="cyan",
    )
    table.add_column("Metric", style="bold white")
    table.add_column("Validation (2/3)", style="green")
    table.add_column("Test (1/3)", style="yellow")
    table.add_column("Total", style="cyan")

    n_val = len(val_df)
    n_test = len(test_df)
    table.add_row(
        "Solves Count",
        f"{n_val} ({n_val / total_count * 100:.1f}%)",
        f"{n_test} ({n_test / total_count * 100:.1f}%)",
        str(total_count),
    )

    if (
        "is_valid_solution" in val_df.columns
        and "is_valid_solution" in test_df.columns
    ):
        v_valid = int(
            val_df["is_valid_solution"].fillna(False).astype(bool).sum()
        )
        t_valid = int(
            test_df["is_valid_solution"].fillna(False).astype(bool).sum()
        )
        tot_valid = v_valid + t_valid
        table.add_row(
            "Verified Valid",
            f"{v_valid} ({v_valid / n_val * 100:.1f}%)",
            f"{t_valid} ({t_valid / n_test * 100:.1f}%)",
            f"{tot_valid} ({tot_valid / total_count * 100:.1f}%)",
        )

    if "result_time" in val_df.columns and "result_time" in test_df.columns:
        v_times = val_df["result_time"].dropna()
        t_times = test_df["result_time"].dropna()
        table.add_row(
            "Mean Solve Time",
            f"{v_times.mean():.2f}s",
            f"{t_times.mean():.2f}s",
            f"{pd.concat([v_times, t_times]).mean():.2f}s",
        )

    if "movecount" in val_df.columns and "movecount" in test_df.columns:
        v_moves = val_df["movecount"].dropna()
        t_moves = test_df["movecount"].dropna()
        table.add_row(
            "Mean Move Count (STM)",
            f"{v_moves.mean():.1f} moves",
            f"{t_moves.mean():.1f} moves",
            f"{pd.concat([v_moves, t_moves]).mean():.1f} moves",
        )

    if "tps" in val_df.columns and "tps" in test_df.columns:
        v_tps = val_df["tps"].dropna()
        t_tps = test_df["tps"].dropna()
        table.add_row(
            "Mean TPS",
            f"{v_tps.mean():.2f}",
            f"{t_tps.mean():.2f}",
            f"{pd.concat([v_tps, t_tps]).mean():.2f}",
        )

    console.print(table)

    # Top solvers breakdown
    if "solver" in val_df.columns and "solver" in test_df.columns:
        solvers_table = Table(
            title="Top Solvers Distribution", border_style="magenta"
        )
        solvers_table.add_column("Solver", style="bold yellow")
        solvers_table.add_column("Val Solves", style="green")
        solvers_table.add_column("Test Solves", style="yellow")
        solvers_table.add_column("Val %", style="cyan")

        v_counts = val_df["solver"].value_counts()
        t_counts = test_df["solver"].value_counts()
        top_solvers = (
            pd.concat([v_counts, t_counts], axis=1).fillna(0).sum(axis=1)
        )
        top_5 = top_solvers.sort_values(ascending=False).head(5).index

        for s in top_5:
            vc = int(v_counts.get(s, 0))
            tc = int(t_counts.get(s, 0))
            tot = vc + tc
            pct = (vc / tot * 100) if tot else 0
            solvers_table.add_row(s, str(vc), str(tc), f"{pct:.1f}%")

        console.print(solvers_table)


def main() -> None:
    setup_console_encoding()
    parser = argparse.ArgumentParser(
        description="Split reconstruction dataset into validation (2/3) and test (1/3) sets."
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default="data/reconstructions/reco_cfop_all.jsonl",
        help="Source JSONL file path.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/reconstructions/reco_cfop_all.csv",
        help="Source CSV file path.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="data/val",
        help="Target validation directory.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/test",
        help="Target test directory.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=2.0 / 3.0,
        help="Fraction for validation split (default: 2/3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42).",
    )
    parser.add_argument(
        "--val-prefix",
        type=str,
        default="reco_cfop_val",
        help="Base filename for validation files.",
    )
    parser.add_argument(
        "--test-prefix",
        type=str,
        default="reco_cfop_test",
        help="Base filename for test files.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )

    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    console.print(
        f"[bold cyan]Splitting reconstructions dataset with seed={args.seed}...[/bold cyan]"
    )
    paths, val_df, test_df = split_reconstructions(
        input_jsonl=args.jsonl,
        input_csv=args.csv,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        val_prefix=args.val_prefix,
        test_prefix=args.test_prefix,
    )

    total_count = len(val_df) + len(test_df)
    print_split_summary(total_count, val_df, test_df, seed=args.seed)

    console.print("\n[bold green]Files successfully generated:[/bold green]")
    for name, p in paths.items():
        console.print(
            f"  - [bold]{name}[/bold]: {p} ({p.stat().st_size:,} bytes)"
        )


if __name__ == "__main__":
    main()
