"""CLI script to scrape CFOP 3x3 Rubik's Cube reconstructions from https://reco.nz/
and save them for data analysis and machine learning.

Examples:
    # Scrape 50 CFOP 3x3 solves and export to JSONL and CSV:
    python scripts/scrape_reco.py --limit 50 --csv

    # Scrape specific solve IDs:
    python scripts/scrape_reco.py --solve-id 14241 14134 14000 --csv

    # Scrape first 5 pages (~250 solves):
    python scripts/scrape_reco.py --pages 5 --output data/reconstructions/sample_cfop.jsonl --csv

    # Print summary statistics from an existing scraped dataset:
    python scripts/scrape_reco.py --analyze data/reconstructions/reco_cfop_3x3.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from cubesight_ml.datasets.reco_scraper import (
    RecoScraper,
    RecoSolve,
    load_reconstructions,
)

console = Console()

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def analyze_dataset(filepath: str | Path) -> None:
    """Prints insightful summary statistics from a reconstructions dataset."""
    df = load_reconstructions(filepath)
    if df.empty:
        console.print("[yellow]Dataset is empty.[/yellow]")
        return

    table = Table(
        title=f"Dataset Summary: {Path(filepath).name}", border_style="cyan"
    )
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="green")

    table.add_row("Total Solves", str(len(df)))

    if "is_valid_solution" in df.columns:
        valid_count = df["is_valid_solution"].sum()
        valid_rate = (valid_count / len(df)) * 100 if len(df) else 0
        table.add_row(
            "Simulator Verified Valid",
            f"{valid_count} / {len(df)} ({valid_rate:.1f}%)",
        )

    if "result_time" in df.columns:
        times = df["result_time"].dropna()
        if not times.empty:
            table.add_row(
                "Solve Time (Mean / Min / Max)",
                f"{times.mean():.2f}s  /  {times.min():.2f}s  /  {times.max():.2f}s",
            )

    if "movecount" in df.columns:
        moves = df["movecount"].dropna()
        if not moves.empty:
            table.add_row(
                "Move Count (Mean / Min / Max)",
                f"{moves.mean():.1f}  /  {int(moves.min())}  /  {int(moves.max())}",
            )

    if "tps" in df.columns:
        tps = df["tps"].dropna()
        if not tps.empty:
            table.add_row(
                "TPS (Mean / Min / Max)",
                f"{tps.mean():.2f}  /  {tps.min():.2f}  /  {tps.max():.2f}",
            )

    # Stage movecounts breakdown if present
    for stage_col, label in [
        ("cross_movecount", "Cross"),
        ("f2l_total_movecount", "F2L"),
        ("oll_movecount", "OLL"),
        ("pll_movecount", "PLL"),
    ]:
        if stage_col in df.columns:
            st_moves = df[stage_col].dropna()
            st_moves = st_moves[st_moves > 0]
            if not st_moves.empty:
                table.add_row(
                    f"Avg {label} Moves", f"{st_moves.mean():.1f} moves"
                )

    console.print(table)

    # Top Solvers
    if "solver" in df.columns:
        top_solvers = df["solver"].value_counts().head(5)
        solvers_table = Table(
            title="Top Solvers in Dataset", border_style="magenta"
        )
        solvers_table.add_column("Solver", style="bold yellow")
        solvers_table.add_column("Solves", justify="right", style="cyan")
        for solver_name, cnt in top_solvers.items():
            solvers_table.add_row(str(solver_name), str(cnt))
        console.print(solvers_table)


def get_existing_solve_ids(filepath: Path) -> set[int]:
    """Reads existing solve IDs from an existing output file to support resuming."""
    if not filepath.exists():
        return set()

    ids: set[int] = set()
    try:
        if filepath.suffix.lower() == ".jsonl":
            with filepath.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        if "solve_id" in data:
                            ids.add(int(data["solve_id"]))
        elif filepath.suffix.lower() == ".json":
            with filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    if "solve_id" in item:
                        ids.add(int(item["solve_id"]))
    except (OSError, json.JSONDecodeError, ValueError) as e:
        console.print(
            f"[yellow]Warning: Could not read existing IDs from {filepath}: {e}[/yellow]"
        )
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape CFOP 3x3 Rubik's Cube reconstructions from reco.nz"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of solves to scrape (e.g. 50, 100). Default: all available",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Maximum number of listing pages to scrape (each page contains ~50 solves)",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Page to start scraping from (default: 1)",
    )
    parser.add_argument(
        "--puzzle",
        type=str,
        default="3x3",
        help="Puzzle filter (default: 3x3)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="CFOP",
        help="Method filter (default: CFOP)",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        nargs="+",
        default=None,
        help="Filter by one or more solver names (e.g. --solvers 'Yiheng Wang' 'Tymon Kolasiński')",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="id_desc",
        help="Sort order on reco.nz (default: id_desc)",
    )
    parser.add_argument(
        "--solve-id",
        type=int,
        nargs="+",
        help="Specific solve ID(s) to fetch directly",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/reconstructions/reco_cfop_3x3.jsonl",
        help="Output file path (default: data/reconstructions/reco_cfop_3x3.jsonl)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also export a flattened CSV file alongside JSONL for tabular data analysis",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also export a formatted JSON array file",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Polite delay in seconds between HTTP requests (default: 0.25)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=5,
        help="Number of concurrent worker threads for fetching solve details (default: 5)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable cube simulator verification of solutions",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip existing solve IDs (overwrite output file)",
    )
    parser.add_argument(
        "--analyze",
        type=str,
        default=None,
        help="Analyze and print summary statistics from an existing dataset file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # If --analyze is specified, print summary and exit
    if args.analyze:
        analyze_dataset(args.analyze)
        return

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    console.print(
        Panel.fit(
            f"[bold cyan]reco.nz Reconstructions Scraper[/bold cyan]\n"
            f"[white]Puzzle: [bold yellow]{args.puzzle}[/bold yellow] | "
            f"Method: [bold yellow]{args.method}[/bold yellow] | "
            f"Limit: [bold yellow]{args.limit or 'Unlimited'}[/bold yellow] | "
            f"Simulator Verification: [bold yellow]{not args.no_verify}[/bold yellow][/white]",
            border_style="cyan",
        )
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scraper = RecoScraper(delay=args.delay)
    collected_solves: list[RecoSolve] = []

    # Handle direct solve-id mode
    if args.solve_id:
        console.print(
            f"[green]Fetching {len(args.solve_id)} specified solve(s)...[/green]"
        )
        for sid in args.solve_id:
            s = scraper.fetch_solve_detail(sid, verify=not args.no_verify)
            if s:
                collected_solves.append(s)
                status = (
                    "[green][OK][/green]"
                    if s.is_valid_solution
                    else "[yellow][?][/yellow]"
                )
                console.print(
                    f"Solve [bold cyan]{s.solve_id}[/bold cyan] by [bold]{s.solver}[/bold]: "
                    f"{s.result_time}s, {s.movecount} moves, {s.tps} TPS -> {status}"
                )
            else:
                console.print(f"[red]Failed to fetch solve {sid}[/red]")

        RecoScraper.save_jsonl(collected_solves, out_path, append=False)
        console.print(
            f"[green]Saved {len(collected_solves)} solve(s) to {out_path}[/green]"
        )

        if args.csv:
            csv_path = out_path.with_suffix(".csv")
            RecoScraper.save_csv(collected_solves, csv_path)
            console.print(f"[green]Saved flattened CSV to {csv_path}[/green]")
        if args.json:
            json_path = out_path.with_suffix(".json")
            RecoScraper.save_json(collected_solves, json_path)
            console.print(f"[green]Saved JSON to {json_path}[/green]")

        analyze_dataset(out_path)
        return

    # Check resume state
    existing_ids = set() if args.no_resume else get_existing_solve_ids(out_path)
    if existing_ids:
        console.print(
            f"[cyan]Resuming scrape: found {len(existing_ids)} existing solve(s) in {out_path}[/cyan]"
        )

    # Run scraping progress with streaming writes
    append_mode = not args.no_resume and out_path.exists()
    mode = "a" if append_mode else "w"
    out_file = out_path.open(mode, encoding="utf-8")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Scraping reconstructions...", total=args.limit
            )

            def on_solve_scraped(solve: RecoSolve) -> None:
                collected_solves.append(solve)
                # Stream write immediately to JSONL
                out_file.write(
                    json.dumps(solve.to_dict(), ensure_ascii=False) + "\n"
                )
                out_file.flush()
                progress.advance(task)
                status = "[OK]" if solve.is_valid_solution else "[•]"
                progress.update(
                    task,
                    description=f"[cyan]Scraped #{solve.solve_id} ({solve.solver[:15]} - {solve.result_time}s) {status}[/cyan]",
                )

            try:
                for _ in scraper.scrape(
                    limit=args.limit,
                    max_pages=args.pages,
                    start_page=args.start_page,
                    puzzle=args.puzzle,
                    method=args.method,
                    sort=args.sort,
                    solvers=args.solvers,
                    concurrency=args.concurrency,
                    verify=not args.no_verify,
                    existing_ids=existing_ids,
                    callback=on_solve_scraped,
                ):
                    pass
            except KeyboardInterrupt:
                console.print(
                    "\n[yellow]Scraping paused by user. Flushed collected data.[/yellow]"
                )
    finally:
        out_file.close()

    console.print(
        f"\n[bold green]Successfully scraped {len(collected_solves)} new solve(s)![/bold green]"
    )
    console.print(f"[green]Saved to JSONL: {out_path}[/green]")

    if args.csv and out_path.exists():
        csv_path = out_path.with_suffix(".csv")
        all_df = load_reconstructions(out_path)
        all_df.to_csv(csv_path, index=False)
        console.print(f"[green]Saved flattened CSV: {csv_path}[/green]")

    if args.json and out_path.exists():
        json_path = out_path.with_suffix(".json")
        all_df = load_reconstructions(out_path)
        all_records = all_df.to_dict(orient="records")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        console.print(f"[green]Saved JSON: {json_path}[/green]")

    # Show dataset analysis
    if out_path.exists():
        console.print()
        analyze_dataset(out_path)


if __name__ == "__main__":
    main()
