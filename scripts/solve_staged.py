import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from cubesight_ml.cube.layout import to_flat
from cubesight_ml.envs.rubiks_cube import RubiksCubeEnv
from cubesight_ml.search.staged_solver import StagedSolver
from cubesight_ml.stages.definitions import Stage
from cubesight_ml.training.checkpointing import load_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the full 4-stage sequential CFOP pipeline solver with STM and move penalties."
    )
    # Checkpoints
    parser.add_argument(
        "--cross_checkpoint",
        "--cross_ckpt",
        type=str,
        required=True,
        help="Path to trained Cross model checkpoint.",
    )
    parser.add_argument(
        "--f2l_checkpoint",
        "--f2l_ckpt",
        type=str,
        required=True,
        help="Path to trained F2L model checkpoint.",
    )
    parser.add_argument(
        "--oll_checkpoint",
        "--oll_ckpt",
        type=str,
        required=True,
        help="Path to trained OLL model checkpoint.",
    )
    parser.add_argument(
        "--pll_checkpoint",
        "--pll_ckpt",
        type=str,
        required=True,
        help="Path to trained PLL model checkpoint.",
    )

    # Search parameters
    parser.add_argument(
        "--beam_width",
        type=int,
        default=1000,
        help="Default beam search width for all stages.",
    )
    parser.add_argument(
        "--beam_widths",
        type=int,
        nargs=4,
        default=None,
        metavar=("CROSS", "F2L", "OLL", "PLL"),
        help="Individual beam search widths for [Cross, F2L, OLL, PLL].",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Default max search steps per stage.",
    )
    parser.add_argument(
        "--max_steps_per_stage",
        type=int,
        nargs=4,
        default=None,
        metavar=("CROSS", "F2L", "OLL", "PLL"),
        help="Individual max search steps for [Cross, F2L, OLL, PLL].",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Default batch size for beam search expansion.",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs=4,
        default=None,
        metavar=("CROSS", "F2L", "OLL", "PLL"),
        help="Individual batch sizes for [Cross, F2L, OLL, PLL].",
    )

    # Evaluation setup
    parser.add_argument(
        "--n_scrambles",
        type=int,
        default=100,
        help="Number of full scrambles to evaluate.",
    )
    parser.add_argument(
        "--scramble_depth",
        type=int,
        default=30,
        help="Scramble depth (random moves from solved).",
    )
    parser.add_argument(
        "--move_penalties",
        type=str,
        default=None,
        help="Path to JSON file or inline JSON dictionary mapping move names to execution penalties (e.g. '{\"B\": 2.0}').",
    )
    parser.add_argument(
        "--no_double_moves",
        action="store_true",
        default=False,
        help="Disable 180-degree double face/slice/rotation turns.",
    )
    parser.add_argument(
        "--no_rotations",
        action="store_true",
        default=False,
        help="Disable whole-cube rotations (x, y, z).",
    )
    parser.add_argument(
        "--no_stm",
        action="store_true",
        default=False,
        help="Disable STM (fall back to HTM/QTM face turns only).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for test scrambles."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation artifacts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_model_from_ckpt(
    ckpt_path: str | Path, device: torch.device
) -> torch.nn.Module:
    return load_model_from_checkpoint(ckpt_path, device=device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    use_stm = not args.no_stm
    use_double_moves = not args.no_double_moves
    include_rotations = not args.no_rotations

    # 1. Output directory setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("runs/eval/pipeline") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Full 4-Stage CFOP Pipeline Evaluation | Output: {output_dir}")

    # Load 4 stage models
    logger.info("Loading stage models...")
    models = {
        Stage.CROSS: load_model_from_ckpt(args.cross_checkpoint, device),
        Stage.F2L: load_model_from_ckpt(args.f2l_checkpoint, device),
        Stage.OLL: load_model_from_ckpt(args.oll_checkpoint, device),
        Stage.PLL: load_model_from_ckpt(args.pll_checkpoint, device),
    }

    # Setup beam widths and max steps per stage
    if args.beam_widths:
        bw_dict = {
            Stage.CROSS: args.beam_widths[0],
            Stage.F2L: args.beam_widths[1],
            Stage.OLL: args.beam_widths[2],
            Stage.PLL: args.beam_widths[3],
        }
    else:
        bw_dict = dict.fromkeys(Stage, args.beam_width)

    if args.max_steps_per_stage:
        ms_dict = {
            Stage.CROSS: args.max_steps_per_stage[0],
            Stage.F2L: args.max_steps_per_stage[1],
            Stage.OLL: args.max_steps_per_stage[2],
            Stage.PLL: args.max_steps_per_stage[3],
        }
    else:
        ms_dict = dict.fromkeys(Stage, args.max_steps)

    if args.batch_sizes:
        bs_dict = {
            Stage.CROSS: args.batch_sizes[0],
            Stage.F2L: args.batch_sizes[1],
            Stage.OLL: args.batch_sizes[2],
            Stage.PLL: args.batch_sizes[3],
        }
    else:
        bs_dict = dict.fromkeys(Stage, args.batch_size)

    solver = StagedSolver(
        models=models,
        beam_width=bw_dict,
        max_steps=ms_dict,
        batch_size=bs_dict,
        use_stm=use_stm,
        use_double_moves=use_double_moves,
        include_rotations=include_rotations,
        move_penalties=args.move_penalties,
        device=device,
        verbose=False,
    )

    # Generate test scrambles
    logger.info(
        f"Generating {args.n_scrambles} scrambles of depth {args.scramble_depth} (seed={args.seed})..."
    )
    rng = np.random.default_rng(args.seed)
    scramble_env = RubiksCubeEnv(
        batch_size=args.n_scrambles,
        use_stm=use_stm,
        use_double_moves=use_double_moves,
        include_rotations=include_rotations,
        scramble_face_moves_only=True,
        rng=rng,
    )
    scramble_env.reset()
    scramble_env.scramble(depth=args.scramble_depth)
    test_states = to_flat(*scramble_env.get_state())

    # Evaluation loop
    logger.info("Solving test scrambles through CFOP pipeline...")
    total_lengths = []
    total_stm_lengths = []
    total_costs = []
    solved_scrambles = 0
    failure_counts = dict.fromkeys(Stage, 0)
    nodes_expanded_total = 0
    stage_lengths: dict[Stage, list[int]] = {st: [] for st in Stage}
    stage_stm_lengths: dict[Stage, list[int]] = {st: [] for st in Stage}
    stage_costs: dict[Stage, list[float]] = {st: [] for st in Stage}
    solutions_list = []

    start_time = time.time()

    for i in range(args.n_scrambles):
        state = test_states[i]
        result = solver.solve(state)

        sol_record = {
            "cube_idx": i,
            "solved": result.solved,
            "failed_stage": result.failed_stage.value
            if result.failed_stage
            else None,
            "formatted_solution": result.formatted_solution,
            "total_moves": result.total_length,
            "total_stm_moves": result.total_stm_length,
            "total_cost": result.total_cost,
            "stage_lengths": {
                st.value: result.stage_lengths.get(st, 0) for st in Stage
            },
            "stage_stm_lengths": {
                st.value: result.stage_stm_lengths.get(st, 0) for st in Stage
            },
            "stage_costs": {
                st.value: result.stage_costs.get(st, 0.0) for st in Stage
            },
        }
        solutions_list.append(sol_record)

        # Accumulate search node counts
        for st_stats in result.stage_stats.values():
            nodes_expanded_total += st_stats.get("states_expanded", 0)

        if result.solved:
            solved_scrambles += 1
            total_lengths.append(result.total_length)
            total_stm_lengths.append(result.total_stm_length)
            total_costs.append(result.total_cost)
            for st in Stage:
                stage_lengths[st].append(result.stage_lengths.get(st, 0))
                stage_stm_lengths[st].append(
                    result.stage_stm_lengths.get(st, 0)
                )
                stage_costs[st].append(result.stage_costs.get(st, 0.0))
        else:
            if result.failed_stage:
                failure_counts[result.failed_stage] += 1

        if (i + 1) % max(1, args.n_scrambles // 5) == 0 or (
            i + 1
        ) == args.n_scrambles:
            curr_rate = (solved_scrambles / (i + 1)) * 100
            logger.info(
                f"Progress: {i + 1}/{args.n_scrambles} | Current Solve Rate: {curr_rate:.1f}%"
            )

    elapsed_time = time.time() - start_time
    pipeline_solve_rate = solved_scrambles / args.n_scrambles

    # Compile results
    pipeline_results = {
        "solve_rate": pipeline_solve_rate,
        "solved_count": solved_scrambles,
        "total_scrambles": args.n_scrambles,
        "scramble_depth": args.scramble_depth,
        "mean_total_length": float(np.mean(total_lengths))
        if total_lengths
        else None,
        "mean_stm_length": float(np.mean(total_stm_lengths))
        if total_stm_lengths
        else None,
        "mean_execution_cost": float(np.mean(total_costs))
        if total_costs
        else None,
        "median_total_length": float(np.median(total_lengths))
        if total_lengths
        else None,
        "std_total_length": float(np.std(total_lengths))
        if total_lengths
        else None,
        "min_total_length": int(np.min(total_lengths))
        if total_lengths
        else None,
        "max_total_length": int(np.max(total_lengths))
        if total_lengths
        else None,
        "failure_counts": {st.value: failure_counts[st] for st in Stage},
        "elapsed_time_seconds": elapsed_time,
        "mean_time_per_solve": elapsed_time / args.n_scrambles,
        "total_nodes_expanded": nodes_expanded_total,
    }

    stage_breakdown = {}
    for st in Stage:
        lens = stage_lengths[st]
        stms = stage_stm_lengths[st]
        csts = stage_costs[st]
        stage_breakdown[st.value] = {
            "mean_length": float(np.mean(lens)) if lens else None,
            "mean_stm_length": float(np.mean(stms)) if stms else None,
            "mean_cost": float(np.mean(csts)) if csts else None,
            "median_length": float(np.median(lens)) if lens else None,
            "std_length": float(np.std(lens)) if lens else None,
            "min_length": int(np.min(lens)) if lens else None,
            "max_length": int(np.max(lens)) if lens else None,
        }

    # Save artifacts
    config_dict = {
        "cross_checkpoint": str(args.cross_checkpoint),
        "f2l_checkpoint": str(args.f2l_checkpoint),
        "oll_checkpoint": str(args.oll_checkpoint),
        "pll_checkpoint": str(args.pll_checkpoint),
        "beam_widths": {st.value: bw_dict[st] for st in Stage},
        "max_steps": {st.value: ms_dict[st] for st in Stage},
        "n_scrambles": args.n_scrambles,
        "scramble_depth": args.scramble_depth,
        "use_stm": use_stm,
        "use_double_moves": use_double_moves,
        "include_rotations": include_rotations,
        "move_penalties": args.move_penalties,
        "seed": args.seed,
        "device": str(device),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    with (output_dir / "pipeline_results.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(pipeline_results, f, indent=2)

    with (output_dir / "stage_breakdown.json").open("w", encoding="utf-8") as f:
        json.dump(stage_breakdown, f, indent=2)

    with (output_dir / "solutions.json").open("w", encoding="utf-8") as f:
        json.dump(solutions_list, f, indent=2)

    # Terminal summary table
    print("\n" + "=" * 65)
    print("      END-TO-END 4-STAGE CFOP PIPELINE EVALUATION")
    print("=" * 65)
    print(
        f"  Total Solve Rate:      {pipeline_solve_rate * 100:.2f}% ({solved_scrambles}/{args.n_scrambles})"
    )
    if total_lengths:
        print(
            f"  Mean Total Moves:      {pipeline_results['mean_total_length']:.2f} moves"
        )
        print(
            f"  Mean STM Move Length:  {pipeline_results['mean_stm_length']:.2f} moves"
        )
        print(
            f"  Mean Execution Cost:   {pipeline_results['mean_execution_cost']:.2f}"
        )
        print(
            f"  Median Length:         {pipeline_results['median_total_length']:.1f} moves"
        )
        print(
            f"  Min / Max Length:      {pipeline_results['min_total_length']} / {pipeline_results['max_total_length']} moves"
        )
    print("-" * 65)
    print("  STAGE-BY-STAGE MEAN MOVE LENGTHS (Total / STM):")
    for st in Stage:
        m_len = stage_breakdown[st.value]["mean_length"]
        m_stm = stage_breakdown[st.value]["mean_stm_length"]
        m_c = stage_breakdown[st.value]["mean_cost"]
        m_str = (
            f"{m_len:.2f} (STM: {m_stm:.2f}, Cost: {m_c:.2f})"
            if m_len is not None
            else "N/A"
        )
        print(f"    - {st.value.upper():<7} : {m_str}")
    print("-" * 65)
    print(f"  Total Wall-Clock Time: {elapsed_time:.1f}s")
    print(
        f"  Mean Time per Cube:    {pipeline_results['mean_time_per_solve']:.2f}s"
    )
    print("=" * 65)
    logger.info(
        f"Pipeline evaluation finished. Artifacts saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
