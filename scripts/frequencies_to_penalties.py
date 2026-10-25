#!/usr/bin/env python3
"""Converts empirical move frequencies into ergonomic move penalties.

Uses Negative Log-Likelihood (NLL) with Laplace smoothing to map move
frequencies from human CFOP speedcubing solves to ergonomic penalties
suitable for Beam Search and DAVI training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def frequencies_to_penalties(
    df: pd.DataFrame,
    max_penalty: float = 10.0,
    power: float = 2.0,
    epsilon: float = 1e-4,
    zero_threshold: float = 0.10,
    decimals: int = 2,
) -> dict[str, float]:
    """Converts a frequency DataFrame into a move -> penalty dict."""
    freqs = df["frequency"].to_numpy(dtype=np.float64)
    p_max = freqs.max()

    # Negative log-likelihood relative to the most frequent move
    nll = -np.log((freqs + epsilon) / (p_max + epsilon))

    # Apply power exponent to widen the contrast between common and rare moves
    raw_penalties = nll**power

    # Normalize to [0, max_penalty]
    max_raw = raw_penalties.max()
    if max_raw > 0:
        normalized = max_penalty * (raw_penalties / max_raw)
    else:
        normalized = raw_penalties

    # Set moves above threshold (e.g., U, U', R, R') to zero penalty
    if zero_threshold > 0:
        normalized[freqs >= zero_threshold] = 0.0

    # Ensure non-negative and round
    normalized = np.maximum(normalized, 0.0)
    rounded = np.round(normalized, decimals)

    return {
        str(move): float(pen)
        for move, pen in zip(df["move"], rounded, strict=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert move frequencies CSV to an ergonomic penalties JSON file."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/val/move_frequencies.csv",
        help="Path to move_frequencies.csv input.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="presets/data_driven_penalties.json",
        help="Output path for the generated penalties JSON file.",
    )
    parser.add_argument(
        "--max-penalty",
        type=float,
        default=10.0,
        help="Maximum penalty for least frequent moves (default: 10.0).",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=2.0,
        help="Convexity exponent on NLL to widen contrast for rare moves (default: 2.0).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="Laplace smoothing epsilon (default: 1e-4).",
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=0.10,
        help="Frequency threshold above which penalty is 0.0 (default: 0.10).",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimal places (default: 2).",
    )

    args = parser.parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    penalties = frequencies_to_penalties(
        df=df,
        max_penalty=args.max_penalty,
        power=args.power,
        epsilon=args.epsilon,
        zero_threshold=args.zero_threshold,
        decimals=args.decimals,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(penalties, f, indent=2)

    print(f"Generated ergonomic penalties for {len(penalties)} moves.")
    print(f"Saved to: {output_path}")
    print("\nSample generated penalties:")
    sample_moves = [
        "U",
        "U'",
        "U2",
        "R",
        "R'",
        "F",
        "L",
        "D",
        "M2",
        "B",
        "E",
        "S2",
    ]
    for m in sample_moves:
        if m in penalties:
            pct = float(df.loc[df["move"] == m, "percentage"].iloc[0])
            print(
                f"  {m:>4}: penalty = {penalties[m]:.2f}  (frequency: {pct:.2f}%)"
            )


if __name__ == "__main__":
    main()
