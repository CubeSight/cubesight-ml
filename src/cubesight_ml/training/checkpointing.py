import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import nn

from cubesight_ml.cube.moves import (
    INDEX_TO_MOVE,
    NUM_MOVES,
    parse_move_penalties,
)
from cubesight_ml.models.value_wrapper import build_model


def save_checkpoint_and_stats(
    path: Path,
    step: int,
    batch_size: int,
    model: nn.Module,
    target_model: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_loss: float,
    train_loss: float,
    current_lr: float,
    elapsed_time: float,
    stage: str,
    model_config: dict[str, Any],
    sync_policy_state: dict[str, Any] | None = None,
    num_syncs: int = 0,
    cumulative_flops: float | None = None,
    cumulative_flops_active: float | None = None,
    cumulative_flops_target: float | None = None,
    target_sample_counts: list[int] | None = None,
    eval_metrics: dict[str, Any] | None = None,
    move_penalties: Any | None = None,
) -> None:
    """Saves training checkpoint and updates stats file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = step * batch_size

    # Clean potential _orig_mod prefix from compile
    cleaned_model_sd = {
        k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()
    }
    cleaned_target_sd = (
        {
            k.removeprefix("_orig_mod."): v
            for k, v in target_model.state_dict().items()
        }
        if target_model is not None
        else None
    )

    ckpt_data = {
        "step": step,
        "samples": samples,
        "stage": stage,
        "model_config": model_config,
        "model_state_dict": cleaned_model_sd,
        "target_model_state_dict": cleaned_target_sd,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "sync_policy_state": sync_policy_state,
        "num_syncs": num_syncs,
        "best_loss": best_loss,
        "train_loss": train_loss,
        "lr": current_lr,
        "elapsed_time_seconds": elapsed_time,
        "cumulative_flops": cumulative_flops,
        "cumulative_flops_active": cumulative_flops_active,
        "cumulative_flops_target": cumulative_flops_target,
    }
    if target_sample_counts is not None:
        ckpt_data["target_sample_counts"] = target_sample_counts
    if eval_metrics is not None:
        ckpt_data["eval_metrics"] = eval_metrics
    if move_penalties is not None:
        ckpt_data["move_penalties"] = move_penalties

    torch.save(ckpt_data, path)

    stats_data = {
        "step": step,
        "samples": samples,
        "stage": stage,
        "num_syncs": num_syncs,
        "target_sample_counts": target_sample_counts,
        "train_loss": train_loss,
        "best_loss": best_loss,
        "lr": current_lr,
        "elapsed_time_seconds": elapsed_time,
        "cumulative_flops": cumulative_flops,
        "cumulative_flops_active": cumulative_flops_active,
        "cumulative_flops_target": cumulative_flops_target,
        "cumulative_pflops": cumulative_flops / 1e15
        if cumulative_flops is not None
        else None,
        "cumulative_pflops_active": cumulative_flops_active / 1e15
        if cumulative_flops_active is not None
        else None,
        "cumulative_pflops_target": cumulative_flops_target / 1e15
        if cumulative_flops_target is not None
        else None,
    }
    if eval_metrics is not None:
        stats_data["eval_metrics"] = eval_metrics

    stats_path = path.parent / f"{path.stem}_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)


def load_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device | str = "cpu",
    map_location: torch.device | str | None = None,
) -> dict[str, Any]:
    """Loads a saved checkpoint dict with CPU safety."""
    target_device = map_location if map_location is not None else device
    return torch.load(
        checkpoint_path, map_location=target_device, weights_only=False
    )


def extract_model_penalties(
    model_or_path: Any,
) -> tuple[np.ndarray, str, dict[str, float]]:
    """Extracts trained ergonomic move penalties from model file or directory.

    Checks only:
    1. Model attribute: model.move_penalties or model.config['move_penalties']
    2. Model file (.pt): 'move_penalties' key in checkpoint dictionary
    3. Model directory: move_penalties.json or config.json in the model folder

    Returns:
        tuple of (penalties_array, preset_or_source_name, non_zero_dict)
    """
    zero_arr = np.zeros(NUM_MOVES, dtype=np.float32)
    zero_res = (zero_arr, "zero_penalties.json", {})
    if model_or_path is None:
        return zero_res

    # 1. Direct attribute on model or wrapper
    raw_penalties = getattr(model_or_path, "move_penalties", None)
    if raw_penalties is None:
        backbone = getattr(model_or_path, "backbone", None)
        raw_penalties = getattr(backbone, "move_penalties", None)

    # 2. Inside model.config dict
    if raw_penalties is None:
        cfg = getattr(model_or_path, "config", None)
        if isinstance(cfg, dict):
            raw_penalties = cfg.get("move_penalties")

    if raw_penalties is not None:
        penalties_arr = parse_move_penalties(raw_penalties)
        if np.any(penalties_arr > 0.0):
            preset_name = (
                Path(raw_penalties).name
                if isinstance(raw_penalties, str)
                else "Trained Penalties"
            )
            non_zero = {
                INDEX_TO_MOVE[i]: float(penalties_arr[i])
                for i in range(len(penalties_arr))
                if penalties_arr[i] > 0.0
            }
            return penalties_arr, preset_name, non_zero

    # 3. Checkpoint path (either passed directly or attached to model)
    ckpt_path = model_or_path if isinstance(model_or_path, str | Path) else None
    if ckpt_path is None:
        ckpt_path = getattr(
            model_or_path,
            "checkpoint_path",
            getattr(model_or_path, "checkpoint", None),
        )

    if not ckpt_path:
        return zero_res

    p = Path(ckpt_path)
    model_dir = p.parent if p.is_file() else p
    penalties_arr = None
    preset_name = None

    # Load from model file (.pt)
    if p.is_file():
        try:
            ckpt_data = load_checkpoint(p, device="cpu")
            if ckpt_data.get("move_penalties"):
                raw = ckpt_data["move_penalties"]
                penalties_arr = parse_move_penalties(raw)
                if isinstance(raw, str):
                    preset_name = Path(raw).name
                else:
                    preset_name = "Trained Penalties"
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not load penalties from checkpoint {p}: {e}")

    # Fallback to model directory: move_penalties.json
    if (
        penalties_arr is None or not np.any(penalties_arr > 0.0)
    ) and model_dir.is_dir():
        pen_file = model_dir / "move_penalties.json"
        if pen_file.is_file():
            try:
                with pen_file.open(encoding="utf-8") as f:
                    data = json.load(f)
                penalties_arr = parse_move_penalties(data)
                preset_name = "move_penalties.json"
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Could not read {pen_file}: {e}")

    # Fallback to model directory: config.json
    if (
        penalties_arr is None or not np.any(penalties_arr > 0.0)
    ) and model_dir.is_dir():
        cfg_file = model_dir / "config.json"
        if cfg_file.is_file():
            try:
                with cfg_file.open(encoding="utf-8") as f:
                    cfg = json.load(f)
                if cfg.get("move_penalties"):
                    raw_cfg = cfg["move_penalties"]
                    penalties_arr = parse_move_penalties(raw_cfg)
                    if isinstance(raw_cfg, str):
                        preset_name = Path(raw_cfg).name
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Could not read {cfg_file}: {e}")

    if penalties_arr is not None and np.any(penalties_arr > 0.0):
        non_zero = {
            INDEX_TO_MOVE[i]: float(penalties_arr[i])
            for i in range(len(penalties_arr))
            if penalties_arr[i] > 0.0
        }
        display_name = preset_name or f"Trained ({len(non_zero)} penalties)"
        return penalties_arr, display_name, non_zero

    return zero_res


def load_model_from_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device | str = "cpu",
    map_location: torch.device | str | None = None,
) -> nn.Module:
    """Loads model from checkpoint, attaching trained move_penalties."""
    ckpt = load_checkpoint(
        checkpoint_path, device=device, map_location=map_location
    )
    cfg = ckpt.get("model_config", {})
    target_dev = torch.device(device) if isinstance(device, str) else device
    model = build_model(cfg).to(target_dev)
    sd = ckpt.get("model_state_dict", ckpt.get("target_model_state_dict", ckpt))
    if isinstance(sd, dict):
        cleaned_sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        model.load_state_dict(cleaned_sd)
    model.eval()

    penalties = ckpt.get("move_penalties")
    model.move_penalties = penalties
    model.checkpoint_path = Path(checkpoint_path)
    if hasattr(model, "config") and isinstance(model.config, dict):
        model.config["move_penalties"] = penalties
    return model
