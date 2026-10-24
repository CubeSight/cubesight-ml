import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis
from loguru import logger
from torch.optim.lr_scheduler import ExponentialLR

from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.datasets.rubiks_cube.stage_davi_dataset import (
    create_stage_davi_dataloader,
)
from cubesight_ml.models.value_wrapper import build_model
from cubesight_ml.stages.definitions import STAGES, Stage
from cubesight_ml.stages.goal_checker import StageGoalChecker
from cubesight_ml.training.checkpointing import load_checkpoint
from cubesight_ml.training.davi import train_davi
from cubesight_ml.training.seeding import (
    resolve_seeds,
    seed_model_and_frameworks,
)
from cubesight_ml.training.target_sync import (
    DeepCubeASyncPolicy,
    PlateauSyncPolicy,
)
from cubesight_ml.training.utils import ConsoleLogger

STAGE_DEFAULT_PRECONDITION_FRACTIONS: dict[str, float] = {
    "cross": 0.0,
    "f2l": 0.35,
    "oll": 0.35,
    "pll": 0.35,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResMLP Value Iteration (DAVI) model for CFOP stage."
    )

    # Stage
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["cross", "f2l", "oll", "pll"],
        help="Target CFOP stage to train.",
    )

    # Model architecture (ResMLP)
    parser.add_argument(
        "--state_representation",
        type=str,
        default="per_sticker",
        choices=["per_sticker", "per_piece"],
        help="Input format (per_sticker: 324-d, per_piece: 292-d).",
    )
    parser.add_argument(
        "--n1", type=int, default=700, help="First linear layer width."
    )
    parser.add_argument(
        "--n2", type=int, default=643, help="Residual blocks width."
    )
    parser.add_argument(
        "--n_r", type=int, default=4, help="Number of residual blocks."
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.0, help="Dropout rate."
    )

    # Optimization
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Initial learning rate."
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.9999993,
        help="Exponential learning rate decay factor.",
    )
    parser.add_argument(
        "--steps", type=int, default=1_000_000, help="Total training steps."
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--clip_norm", type=float, default=1.0, help="Gradient clip norm."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint path.",
    )

    # Target sync policy
    parser.add_argument(
        "--sync_policy",
        type=str,
        default="deepcubea",
        choices=["deepcubea", "plateau"],
        help="Target network update policy.",
    )
    parser.add_argument(
        "--sync_interval",
        type=int,
        default=100,
        help="DeepCubeA sync check interval C in steps.",
    )
    parser.add_argument(
        "--sync_loss_threshold",
        type=float,
        default=0.05,
        help="DeepCubeA target sync loss threshold epsilon.",
    )
    parser.add_argument(
        "--min_steps_between_syncs",
        type=int,
        default=500,
        help="Plateau policy minimum steps between syncs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Plateau policy strike patience.",
    )
    parser.add_argument(
        "--safety_ceiling",
        type=float,
        default=2.0,
        help="Plateau policy loss safety ceiling.",
    )

    # Dataset & Dataloader
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10_000,
        help="Batch size (number of states).",
    )
    parser.add_argument(
        "--scramble_depth",
        type=int,
        default=30,
        help="Maximum scramble depth K.",
    )
    parser.add_argument(
        "--precondition_fraction",
        type=float,
        default=None,
        help="Fraction drawn from precondition set (defaults: 0.0 or 0.35).",
    )
    parser.add_argument(
        "--depth_sampling",
        type=str,
        default="stratified",
        choices=["stratified", "uniform"],
        help="Scramble depth distribution.",
    )
    parser.add_argument(
        "--include_depth_zero",
        action="store_true",
        help="Include depth 0 in scrambles.",
    )
    parser.add_argument(
        "--use_double_moves",
        action="store_true",
        help="Include 180 degree face turns.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes.",
    )

    # Logging & Checkpointing
    parser.add_argument(
        "--loss_interval",
        type=int,
        default=100,
        help="Logging interval in steps.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50_000,
        help="Checkpoint save interval.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=25_000,
        help="Evaluation hook interval in steps.",
    )
    parser.add_argument(
        "--eval_n_states",
        type=int,
        default=20,
        help="Number of held-out validation states.",
    )
    parser.add_argument(
        "--eval_beam_width",
        type=int,
        default=30,
        help="Validation beam search width.",
    )
    parser.add_argument(
        "--eval_max_steps",
        type=int,
        default=30,
        help="Validation max search steps.",
    )

    # Seeding
    parser.add_argument(
        "--seed", type=int, default=42, help="Master random seed."
    )
    parser.add_argument("--model_seed", type=int, default=None)
    parser.add_argument("--data_seed", type=int, default=None)
    parser.add_argument("--dataloader_seed", type=int, default=None)

    # Output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output dir. If None, defaults to runs/davi/<stage>/...",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Optional experiment name tag for runs folder.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Precondition fraction default
    if args.precondition_fraction is None:
        precond_frac = STAGE_DEFAULT_PRECONDITION_FRACTIONS[args.stage]
    else:
        precond_frac = args.precondition_fraction

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        run_name = (
            f"{timestamp}_{args.exp_name}" if args.exp_name else timestamp
        )
        save_dir = Path("runs/davi") / args.stage / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    console_logger = ConsoleLogger(save_dir / "console.log").start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"Target stage: {args.stage} | Output directory: {save_dir} | "
        f"Device: {device}"
    )

    # Resolve seeds
    master_seed, model_seed, data_seed, dataloader_seed = resolve_seeds(
        master_seed=args.seed,
        model_seed=args.model_seed,
        data_seed=args.data_seed,
        dataloader_seed=args.dataloader_seed,
    )
    seed_model_and_frameworks(model_seed)

    seeds_dict = {
        "master_seed": master_seed,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "dataloader_seed": dataloader_seed,
    }
    with open(save_dir / "seeds.json", "w", encoding="utf-8") as f:
        json.dump(seeds_dict, f, indent=2)

    # Save configuration
    config_dict = vars(args).copy()
    config_dict["resolved_precondition_fraction"] = precond_frac
    config_dict["device"] = str(device)
    config_dict.update(seeds_dict)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Build model & target model
    model_cfg = {
        "state_representation": args.state_representation,
        "n1": args.n1,
        "n2": args.n2,
        "n_r": args.n_r,
        "dropout_rate": args.dropout_rate,
        "output_dim": 1,
    }
    active_model = build_model(model_cfg).to(device)
    target_model = deepcopy(active_model).to(device)

    # Flop profiling
    try:
        active_model.eval()
        dummy_state = torch.zeros(2, 46, dtype=torch.long, device=device)
        flops_analysis = FlopCountAnalysis(active_model, dummy_state)
        macs_per_sample = float(flops_analysis.total()) / 2.0
        active_model.train()
        profile_data = {
            "macs_per_sample": macs_per_sample,
            "flops_per_sample": macs_per_sample * 2.0,
            "total_params": sum(p.numel() for p in active_model.parameters()),
            "trainable_params": sum(
                p.numel() for p in active_model.parameters() if p.requires_grad
            ),
        }
        with open(save_dir / "model_profile.json", "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)
        n_params = profile_data["total_params"]
        mflops = profile_data["flops_per_sample"] / 1e6
        logger.info(f"Model Profile: {n_params:,} params, {mflops:.2f} MFLOPs")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not compute FLOP profile: {e}")
        macs_per_sample = None

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(active_model.parameters(), lr=args.lr)
    scheduler = (
        ExponentialLR(optimizer, gamma=args.lr_decay)
        if args.lr_decay < 1.0
        else None
    )

    # Target sync policy
    if args.sync_policy == "deepcubea":
        sync_policy = DeepCubeASyncPolicy(
            sync_interval=args.sync_interval,
            loss_threshold=args.sync_loss_threshold,
        )
    else:
        sync_policy = PlateauSyncPolicy(
            loss_interval=args.loss_interval,
            min_steps_between_syncs=args.min_steps_between_syncs,
            patience=args.patience,
            safety_ceiling=args.safety_ceiling,
        )

    # Restore checkpoint if given
    if args.checkpoint:
        logger.info(f"Loading checkpoint from: {args.checkpoint}")
        ckpt = load_checkpoint(args.checkpoint, device=device)
        active_model.load_state_dict(ckpt["model_state_dict"])
        if (
            "target_model_state_dict" in ckpt
            and ckpt["target_model_state_dict"]
        ):
            target_model.load_state_dict(ckpt["target_model_state_dict"])
        else:
            target_model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if (
            scheduler
            and "scheduler_state_dict" in ckpt
            and ckpt["scheduler_state_dict"]
        ):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "sync_policy_state" in ckpt and ckpt["sync_policy_state"]:
            sync_policy.load_state_dict(ckpt["sync_policy_state"])

    # DataLoader
    dataloader = create_stage_davi_dataloader(
        stage=args.stage,
        batch_size=args.batch_size,
        scramble_depth=args.scramble_depth,
        num_workers=args.num_workers,
        depth_sampling=args.depth_sampling,
        include_depth_zero=args.include_depth_zero,
        precondition_fraction=precond_frac,
        avoid_inverse_moves=True,
        use_double_moves=args.use_double_moves,
        data_seed=data_seed,
        dataloader_seed=dataloader_seed,
    )

    stage_enum = Stage(args.stage)
    spec = STAGES[stage_enum]
    cube_ops = TorchCubeOps(
        use_stm=True,
        use_double_moves=args.use_double_moves,
        include_rotations=True,
        device=device,
    )
    goal_checker = StageGoalChecker(spec, device=device)

    # Run DAVI training
    train_davi(
        active_model=active_model,
        target_model=target_model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        spec=spec,
        cube_ops=cube_ops,
        goal_checker=goal_checker,
        sync_policy=sync_policy,
        steps=args.steps,
        save_dir=save_dir,
        batch_size=args.batch_size,
        stage=args.stage,
        model_config=model_cfg,
        grad_accumulation_steps=args.grad_accumulation_steps,
        clip_norm=args.clip_norm,
        loss_interval=args.loss_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        eval_n_states=args.eval_n_states,
        eval_beam_width=args.eval_beam_width,
        eval_max_steps=args.eval_max_steps,
        device=device,
        macs_per_sample=macs_per_sample,
    )

    console_logger.stop()


if __name__ == "__main__":
    main()
