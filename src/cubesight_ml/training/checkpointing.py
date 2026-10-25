import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


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
    with open(stats_path, "w", encoding="utf-8") as f:
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
