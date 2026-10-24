import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from cubesight_ml.models.value_wrapper import build_model
from cubesight_ml.stages.definitions import STAGES, Stage
from cubesight_ml.training.checkpointing import load_checkpoint
from cubesight_ml.training.evaluation import evaluate_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained stage model in isolation using Beam Search with STM and move penalties."
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["cross", "f2l", "oll", "pll"],
        help="Target stage to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--beam_width", type=int, default=1000, help="Beam search width."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum search steps per instance.",
    )
    parser.add_argument(
        "--n_states",
        type=int,
        default=200,
        help="Number of test states from precondition distribution.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation artifacts.",
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


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
        output_dir = Path("runs/eval/stage") / args.stage / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Evaluating stage: {args.stage} | Checkpoint: {args.checkpoint} | Output: {output_dir}"
    )

    # 2. Load model from checkpoint
    ckpt = load_checkpoint(args.checkpoint, device=device)
    model_cfg = ckpt.get("model_config", {})
    model = build_model(model_cfg).to(device)

    state_dict = ckpt.get(
        "model_state_dict", ckpt.get("target_model_state_dict", ckpt)
    )
    cleaned_sd = {
        k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned_sd)
    model.eval()

    stage_enum = Stage(args.stage)
    spec = STAGES[stage_enum]

    # 3. Run evaluation
    logger.info(
        f"Running Beam Search with width={args.beam_width}, max_steps={args.max_steps} on {args.n_states} states..."
    )
    results = evaluate_stage(
        model=model,
        spec=spec,
        n_states=args.n_states,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        seed=args.seed,
        device=device,
        use_stm=use_stm,
        use_double_moves=use_double_moves,
        include_rotations=include_rotations,
        move_penalties=args.move_penalties,
        verbose=True,
    )

    lengths = results["solution_lengths"]
    stm_lengths = results["stm_lengths"]
    costs = results["solution_costs"]
    solved_count = results["solved_count"]
    total = results["total_states"]
    solve_rate = results["solve_rate"]

    # 4. Compute metrics
    metrics = {
        "stage": args.stage,
        "solve_rate": solve_rate,
        "solved_count": solved_count,
        "total_states": total,
        "mean_length": results["mean_length"],
        "mean_stm_length": results["mean_stm_length"],
        "mean_execution_cost": results["mean_cost"],
        "median_length": float(np.median(lengths)) if lengths else None,
        "median_stm_length": (
            float(np.median(stm_lengths)) if stm_lengths else None
        ),
        "std_length": float(np.std(lengths)) if lengths else None,
        "std_stm_length": float(np.std(stm_lengths)) if stm_lengths else None,
        "median_cost": float(np.median(costs)) if costs else None,
        "min_length": int(np.min(lengths)) if lengths else None,
        "max_length": results["max_length"],
        "mean_nodes_expanded": results["mean_nodes"],
    }

    # Lengths distribution
    counts = Counter(lengths)
    lengths_dist = {str(k): v for k, v in sorted(counts.items())}

    # 5. Save artifacts
    config_dict = {
        "stage": args.stage,
        "checkpoint": str(args.checkpoint),
        "beam_width": args.beam_width,
        "max_steps": args.max_steps,
        "n_states": args.n_states,
        "seed": args.seed,
        "use_stm": use_stm,
        "use_double_moves": use_double_moves,
        "include_rotations": include_rotations,
        "move_penalties": args.move_penalties,
        "device": str(device),
        "model_config": model_cfg,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(
        output_dir / "lengths_distribution.json", "w", encoding="utf-8"
    ) as f:
        json.dump(lengths_dist, f, indent=2)

    # First 10 solutions for qualitative inspection
    with open(output_dir / "solutions_sample.json", "w", encoding="utf-8") as f:
        json.dump(results["move_names"][:10], f, indent=2)

    # 6. Terminal Summary Table
    print("\n" + "=" * 55)
    print(f"  SINGLE-STAGE EVALUATION SUMMARY: {args.stage.upper()}")
    print("=" * 55)
    print(
        f"  Solve Rate:         {solve_rate * 100:.2f}% ({solved_count}/{total})"
    )
    print(f"  Mean Total Moves:   {results['mean_length']:.2f} moves")
    print(f"  Mean STM Length:    {results['mean_stm_length']:.2f} moves")
    print(f"  Mean Exec Cost:     {results['mean_cost']:.2f}")
    if lengths:
        print(f"  Median Length:      {metrics['median_length']:.1f} moves")
        print(
            f"  Min / Max Length:   {metrics['min_length']} / {metrics['max_length']} moves"
        )
    print(f"  Mean Nodes Expanded:{results['mean_nodes']:.1f}")
    print("=" * 55)
    logger.info(f"Evaluation complete. Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
