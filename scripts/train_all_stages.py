import argparse
import gc
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from fvcore.nn import FlopCountAnalysis
from loguru import logger
from torch.optim.lr_scheduler import ExponentialLR

from cubesight_ml.cube.moves import INDEX_TO_MOVE, parse_move_penalties
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

# =============================================================================
# STAGE 1: CROSS HYPERPARAMETERS
# =============================================================================
CROSS_CONFIG: dict[str, Any] = {
    "stage": "cross",
    # Model architecture (ResMLP, ~1.0M params)
    "state_representation": "per_piece",
    "n1": 360,
    "n2": 311,
    "n_r": 4,
    "dropout_rate": 0.0,
    # Optimization
    "lr": 1e-3,
    "lr_decay": 0.99993,
    "steps": 10_000,
    "grad_accumulation_steps": 1,
    "clip_norm": 1.0,
    "checkpoint": None,
    # Target sync policy
    "sync_policy": "plateau",
    "sync_interval": 100,
    "sync_loss_threshold": 0.05,
    "min_steps_between_syncs": 200,
    "patience": 3,
    "safety_ceiling": 5.0,
    # Dataset & DataLoader
    "batch_size": 5_120,
    "scramble_depth": 14,
    "precondition_fraction": 0.0,
    "depth_sampling": "stratified",
    "include_depth_zero": False,
    "use_double_moves": True,
    "num_workers": 4,
    # Logging, Validation & Checkpointing
    "loss_interval": 50,
    "save_interval": 2_500,
    "eval_interval": 1_000,
    "eval_max_depth": 14,
    # Seeding
    "seed": 42,
    "model_seed": None,
    "data_seed": None,
    "dataloader_seed": None,
    # Output & Ergonomic Move Penalties
    "output_dir": None,
    "exp_name": "first_pipeline",
    "move_penalties": "presets/data_driven_penalties.json",
}

# =============================================================================
# STAGE 2: F2L HYPERPARAMETERS
# =============================================================================
F2L_CONFIG: dict[str, Any] = {
    "stage": "f2l",
    # Model architecture (ResMLP, ~4.0M params)
    "state_representation": "per_piece",
    "n1": 700,
    "n2": 643,
    "n_r": 4,
    "dropout_rate": 0.0,
    # Optimization
    "lr": 1e-3,
    "lr_decay": 0.999992,
    "steps": 90_000,
    "grad_accumulation_steps": 1,
    "clip_norm": 1.0,
    "checkpoint": None,
    # Target sync policy
    "sync_policy": "plateau",
    "sync_interval": 500,
    "sync_loss_threshold": 0.05,
    "min_steps_between_syncs": 300,
    "patience": 3,
    "safety_ceiling": 5.0,
    # Dataset & DataLoader
    "batch_size": 5_120,
    "scramble_depth": 26,
    "precondition_fraction": 0.25,
    "depth_sampling": "stratified",
    "include_depth_zero": False,
    "use_double_moves": True,
    "num_workers": 4,
    # Logging, Validation & Checkpointing
    "loss_interval": 100,
    "save_interval": 15_000,
    "eval_interval": 4_500,
    "eval_max_depth": 20,
    # Seeding
    "seed": 42,
    "model_seed": None,
    "data_seed": None,
    "dataloader_seed": None,
    # Output & Ergonomic Move Penalties
    "output_dir": None,
    "exp_name": "first_pipeline",
    "move_penalties": "presets/data_driven_penalties.json",
}

# =============================================================================
# STAGE 3: OLL HYPERPARAMETERS
# =============================================================================
OLL_CONFIG: dict[str, Any] = {
    "stage": "oll",
    # Model architecture (ResMLP, ~2.0M params)
    "state_representation": "per_piece",
    "n1": 512,
    "n2": 448,
    "n_r": 4,
    "dropout_rate": 0.0,
    # Optimization
    "lr": 1e-3,
    "lr_decay": 0.999988,
    "steps": 60_000,
    "grad_accumulation_steps": 1,
    "clip_norm": 1.0,
    "checkpoint": None,
    # Target sync policy
    "sync_policy": "plateau",
    "sync_interval": 500,
    "sync_loss_threshold": 0.05,
    "min_steps_between_syncs": 300,
    "patience": 3,
    "safety_ceiling": 5.0,
    # Dataset & DataLoader
    "batch_size": 5_120,
    "scramble_depth": 16,
    "precondition_fraction": 0.35,
    "depth_sampling": "stratified",
    "include_depth_zero": False,
    "use_double_moves": True,
    "num_workers": 4,
    # Logging, Validation & Checkpointing
    "loss_interval": 100,
    "save_interval": 10_000,
    "eval_interval": 3_000,
    "eval_max_depth": 16,
    # Seeding
    "seed": 42,
    "model_seed": None,
    "data_seed": None,
    "dataloader_seed": None,
    # Output & Ergonomic Move Penalties
    "output_dir": None,
    "exp_name": "first_pipeline",
    "move_penalties": "presets/data_driven_penalties.json",
}

# =============================================================================
# STAGE 4: PLL HYPERPARAMETERS
# =============================================================================
PLL_CONFIG: dict[str, Any] = {
    "stage": "pll",
    # Model architecture (ResMLP, ~2.0M params)
    "state_representation": "per_piece",
    "n1": 512,
    "n2": 448,
    "n_r": 4,
    "dropout_rate": 0.0,
    # Optimization
    "lr": 1e-3,
    "lr_decay": 0.999988,
    "steps": 60_000,
    "grad_accumulation_steps": 1,
    "clip_norm": 1.0,
    "checkpoint": None,
    # Target sync policy
    "sync_policy": "plateau",
    "sync_interval": 500,
    "sync_loss_threshold": 0.05,
    "min_steps_between_syncs": 300,
    "patience": 3,
    "safety_ceiling": 5.0,
    # Dataset & DataLoader
    "batch_size": 5_120,
    "scramble_depth": 20,
    "precondition_fraction": 0.35,
    "depth_sampling": "stratified",
    "include_depth_zero": False,
    "use_double_moves": True,
    "num_workers": 4,
    # Logging, Validation & Checkpointing
    "loss_interval": 100,
    "save_interval": 10_000,
    "eval_interval": 3_000,
    "eval_max_depth": 20,
    # Seeding
    "seed": 42,
    "model_seed": None,
    "data_seed": None,
    "dataloader_seed": None,
    # Output & Ergonomic Move Penalties
    "output_dir": None,
    "exp_name": "first_pipeline",
    "move_penalties": "presets/data_driven_penalties.json",
}

STAGE_CONFIGS: dict[str, dict[str, Any]] = {
    "cross": CROSS_CONFIG,
    "f2l": F2L_CONFIG,
    "oll": OLL_CONFIG,
    "pll": PLL_CONFIG,
}


def train_single_stage(cfg: dict[str, Any], batch_timestamp: str) -> Path:
    """Runs DAVI training for a single stage using its hyperparameter dictionary."""
    stage = cfg["stage"]
    precond_frac = cfg["precondition_fraction"]

    if cfg.get("output_dir"):
        save_dir = Path(cfg["output_dir"])
    else:
        exp_name = cfg.get("exp_name")
        run_name = (
            f"{batch_timestamp}_{exp_name}" if exp_name else batch_timestamp
        )
        save_dir = Path("runs/davi") / stage / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    console_logger = ConsoleLogger(save_dir / "console.log").start()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            f"=== Starting Stage '{stage.upper()}' | Output: {save_dir} | "
            f"Device: {device} ==="
        )

        # Resolve seeds
        master_seed, model_seed, data_seed, dataloader_seed = resolve_seeds(
            master_seed=cfg["seed"],
            model_seed=cfg.get("model_seed"),
            data_seed=cfg.get("data_seed"),
            dataloader_seed=cfg.get("dataloader_seed"),
        )
        seed_model_and_frameworks(model_seed)

        seeds_dict = {
            "master_seed": master_seed,
            "model_seed": model_seed,
            "data_seed": data_seed,
            "dataloader_seed": dataloader_seed,
        }
        with (save_dir / "seeds.json").open("w", encoding="utf-8") as f:
            json.dump(seeds_dict, f, indent=2)

        # Save configuration
        config_dict = deepcopy(cfg)
        config_dict["resolved_precondition_fraction"] = precond_frac
        config_dict["device"] = str(device)
        config_dict.update(seeds_dict)
        with (save_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        if cfg.get("move_penalties"):
            penalties_arr = parse_move_penalties(cfg["move_penalties"])
            penalties_dict = {
                INDEX_TO_MOVE[i]: float(penalties_arr[i])
                for i in range(len(penalties_arr))
                if penalties_arr[i] > 0.0
            }
            with (save_dir / "move_penalties.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(penalties_dict, f, indent=2)

        # Build model & target model
        model_cfg = {
            "state_representation": cfg["state_representation"],
            "n1": cfg["n1"],
            "n2": cfg["n2"],
            "n_r": cfg["n_r"],
            "dropout_rate": cfg["dropout_rate"],
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
                "total_params": sum(
                    p.numel() for p in active_model.parameters()
                ),
                "trainable_params": sum(
                    p.numel()
                    for p in active_model.parameters()
                    if p.requires_grad
                ),
            }
            with (save_dir / "model_profile.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(profile_data, f, indent=2)
            n_params = profile_data["total_params"]
            mflops = profile_data["flops_per_sample"] / 1e6
            logger.info(
                f"Model Profile: {n_params:,} params, {mflops:.2f} MFLOPs"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not compute FLOP profile: {e}")
            macs_per_sample = None

        # Optimizer & Scheduler
        optimizer = torch.optim.Adam(active_model.parameters(), lr=cfg["lr"])
        scheduler = (
            ExponentialLR(optimizer, gamma=cfg["lr_decay"])
            if cfg["lr_decay"] < 1.0
            else None
        )

        # Target sync policy
        if cfg["sync_policy"] == "deepcubea":
            sync_policy = DeepCubeASyncPolicy(
                sync_interval=cfg["sync_interval"],
                loss_threshold=cfg["sync_loss_threshold"],
            )
        else:
            sync_policy = PlateauSyncPolicy(
                loss_interval=cfg["loss_interval"],
                min_steps_between_syncs=cfg["min_steps_between_syncs"],
                patience=cfg["patience"],
                safety_ceiling=cfg["safety_ceiling"],
            )

        # Restore checkpoint if given
        if cfg.get("checkpoint"):
            logger.info(f"Loading checkpoint from: {cfg['checkpoint']}")
            ckpt = load_checkpoint(cfg["checkpoint"], device=device)
            active_model.load_state_dict(ckpt["model_state_dict"])
            if ckpt.get("target_model_state_dict"):
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
            if ckpt.get("sync_policy_state"):
                sync_policy.load_state_dict(ckpt["sync_policy_state"])

        # DataLoader
        dataloader = create_stage_davi_dataloader(
            stage=stage,
            batch_size=cfg["batch_size"],
            scramble_depth=cfg["scramble_depth"],
            num_workers=cfg["num_workers"],
            depth_sampling=cfg["depth_sampling"],
            include_depth_zero=cfg["include_depth_zero"],
            precondition_fraction=precond_frac,
            avoid_inverse_moves=True,
            use_double_moves=cfg["use_double_moves"],
            data_seed=data_seed,
            dataloader_seed=dataloader_seed,
        )

        stage_enum = Stage(stage)
        spec = STAGES[stage_enum]
        cube_ops = TorchCubeOps(
            use_stm=True,
            use_double_moves=cfg["use_double_moves"],
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
            steps=cfg["steps"],
            save_dir=save_dir,
            batch_size=cfg["batch_size"],
            stage=stage,
            model_config=model_cfg,
            move_penalties=cfg.get("move_penalties"),
            grad_accumulation_steps=cfg["grad_accumulation_steps"],
            clip_norm=cfg["clip_norm"],
            loss_interval=cfg["loss_interval"],
            save_interval=cfg["save_interval"],
            eval_interval=cfg["eval_interval"],
            eval_max_depth=cfg["eval_max_depth"],
            val_seed=data_seed,
            device=device,
            macs_per_sample=macs_per_sample,
        )
    finally:
        console_logger.stop()

    return save_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially train all 4 CFOP stage models using per-stage hyperparameter constants."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["cross", "f2l", "oll", "pll"],
        choices=["cross", "f2l", "oll", "pll"],
        help="Stages to train sequentially (default: cross f2l oll pll).",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Optional experiment name tag overriding per-stage exp_name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    completed_dirs: dict[str, Path] = {}
    for stage_name in args.stages:
        cfg = deepcopy(STAGE_CONFIGS[stage_name])
        if args.exp_name is not None:
            cfg["exp_name"] = args.exp_name

        save_dir = train_single_stage(cfg, batch_timestamp=batch_timestamp)
        completed_dirs[stage_name] = save_dir

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("All requested stages finished training:")
    for stage_name, run_dir in completed_dirs.items():
        logger.info(f"  - {stage_name}: {run_dir}")


if __name__ == "__main__":
    main()
