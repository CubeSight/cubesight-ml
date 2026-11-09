import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from cubesight_ml.cube.moves import (
    BASE_MOVE_COSTS_STM,
    NUM_MOVES,
    parse_move_penalties,
)
from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.stages.definitions import StageSpec
from cubesight_ml.stages.goal_checker import StageGoalChecker
from cubesight_ml.training.checkpointing import save_checkpoint_and_stats
from cubesight_ml.training.davi_targets import compute_davi_targets
from cubesight_ml.training.evaluation import (
    evaluate_validation_loss,
    generate_validation_dataset,
)
from cubesight_ml.training.target_sync import TargetSyncPolicy


def train_davi(
    active_model: nn.Module,
    target_model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    spec: StageSpec,
    cube_ops: TorchCubeOps,
    goal_checker: StageGoalChecker,
    sync_policy: TargetSyncPolicy,
    steps: int,
    save_dir: Path,
    batch_size: int,
    stage: str,
    model_config: dict[str, Any],
    move_penalties: dict[str, float]
    | np.ndarray
    | torch.Tensor
    | str
    | Path
    | None = None,
    grad_accumulation_steps: int = 1,
    clip_norm: float | None = 1.0,
    loss_interval: int = 100,
    save_interval: int = 1000,
    eval_interval: int | None = 500,
    eval_max_depth: int = 26,
    eval_states_per_depth: int = 1000,
    val_seed: int = 42,
    device: torch.device | str = "cpu",
    macs_per_sample: float | None = None,
    save_sample_steps: dict[int, int] | None = None,
    save_steps: list[int] | None = None,
    additional_save_steps: set[int] | None = None,
    warmup_steps: int = 1,
) -> None:
    """DeepCubeA Value Iteration (DAVI) training loop for CFOP stages."""
    device = torch.device(device)
    active_model = active_model.to(device)
    target_model = target_model.to(device)
    target_model.eval()

    # Parse execution penalties & compute step costs for actions
    penalties_np = parse_move_penalties(move_penalties)
    base_costs_np = (
        BASE_MOVE_COSTS_STM.copy()
        if cube_ops.use_stm
        else np.ones(NUM_MOVES, dtype=np.float32)
    )
    total_costs_np = base_costs_np + penalties_np
    action_indices = cube_ops.moves.cpu().numpy()
    move_costs_t = torch.tensor(
        total_costs_np[action_indices], dtype=torch.float32, device=device
    )
    has_penalties = bool(np.any(penalties_np > 0.0))
    if has_penalties:
        logger.info(
            f"Applying ergonomic move penalties during training: "
            f"min_cost={move_costs_t.min().item():.2f}, "
            f"max_cost={move_costs_t.max().item():.2f}, "
            f"mean_cost={move_costs_t.mean().item():.2f}"
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))

    val_states: torch.Tensor | None = None
    val_depths: torch.Tensor | None = None
    if eval_max_depth > 0 and eval_states_per_depth > 0:
        val_json_path = save_dir / "validation_dataset.json"
        logger.info(
            f"Generating validation dataset (depths 1..{eval_max_depth}, "
            f"{eval_states_per_depth} states/depth) -> {val_json_path}"
        )
        val_states, val_depths, _ = generate_validation_dataset(
            spec=spec,
            max_depth=eval_max_depth,
            states_per_depth=eval_states_per_depth,
            use_double_moves=cube_ops.use_double_moves,
            seed=val_seed,
            save_path=val_json_path,
        )

    save_sample_steps = save_sample_steps or {}
    save_steps = save_steps or []
    additional_save_steps = additional_save_steps or set()

    data_iter = iter(dataloader)

    # 1. Warm-up
    if warmup_steps > 0:
        logger.info("Warming up DAVI models, dataloader, and kernels...")
        active_model.train()
        for _ in range(warmup_steps):
            parents_w, _ = next(data_iter)
            parents_w = parents_w.to(device, non_blocking=True)
            with torch.no_grad():
                _ = compute_davi_targets(
                    target_model,
                    parents_w,
                    cube_ops,
                    goal_checker,
                    move_costs=move_costs_t,
                )
                _ = active_model(parents_w)["value"]
        logger.info("Warm-up complete.")

    # FLOP accounting: 1 MAC = 2 FLOPs.
    # Active: 6 * MACs. Target forward: 2 * MACs * num_actions.
    n_actions = len(cube_ops.moves)
    flops_per_active_sample = 6.0 * macs_per_sample if macs_per_sample else None
    flops_per_target_sample = (
        2.0 * macs_per_sample * n_actions if macs_per_sample else None
    )

    cumulative_flops_active = 0.0
    cumulative_flops_target = 0.0
    cumulative_flops = 0.0

    running_loss = 0.0
    last_mean_loss = float("inf")
    best_train_loss = float("inf")
    best_eval_loss = float("inf")

    start_time = time.perf_counter()

    logger.info(
        f"Starting DAVI training for stage '{stage}', steps={steps}, "
        f"device={device}"
    )

    try:
        active_model.train()
        optimizer.zero_grad()

        for step in range(1, steps + 1):
            parents, depths = next(data_iter)
            parents = parents.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)

            # 1. Compute targets using target network
            with torch.no_grad():
                targets = compute_davi_targets(
                    target_model=target_model,
                    parents=parents,
                    cube_ops=cube_ops,
                    goal_checker=goal_checker,
                    move_costs=move_costs_t,
                )

            # 2. Compute active model predictions
            active_preds = active_model(parents)["value"].squeeze(-1)
            loss = F.mse_loss(active_preds, targets)

            # FLOP accounting update
            if macs_per_sample is not None:
                step_active_flops = batch_size * flops_per_active_sample
                step_target_flops = batch_size * flops_per_target_sample
                cumulative_flops_active += step_active_flops
                cumulative_flops_target += step_target_flops
                cumulative_flops += step_active_flops + step_target_flops

            running_loss += loss.item()

            # 3. Backprop with gradient accumulation
            scaled_loss = loss / grad_accumulation_steps
            scaled_loss.backward()

            if step % grad_accumulation_steps == 0 or step == steps:
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        active_model.parameters(), max_norm=clip_norm
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            current_lr = optimizer.param_groups[0]["lr"]
            elapsed_time = time.perf_counter() - start_time

            synced_this_step = False

            # 4. Logging & Diagnostics
            if step % loss_interval == 0:
                mean_loss = running_loss / loss_interval
                last_mean_loss = mean_loss
                running_loss = 0.0

                writer.add_scalar("Time/Elapsed_Seconds", elapsed_time, step)
                writer.add_scalar("Training/LR", current_lr, step)
                writer.add_scalar("Loss_Mean/total", mean_loss, step)
                writer.add_scalar("Loss_Instant/total", loss.item(), step)

                # Goal fraction in batch
                goal_frac = (targets == 0.0).float().mean().item()
                writer.add_scalar("Targets/goal_fraction", goal_frac, step)

                # Mean targets and preds by depth
                for k in range(11):
                    k_mask = depths == k
                    if k_mask.any():
                        writer.add_scalar(
                            f"Targets/mean_by_depth/{k}",
                            targets[k_mask].mean().item(),
                            step,
                        )
                        writer.add_scalar(
                            f"Preds/mean_by_depth/{k}",
                            active_preds[k_mask].mean().item(),
                            step,
                        )

                if macs_per_sample is not None:
                    writer.add_scalar(
                        "Compute/Cumulative_PFLOPs_Total",
                        cumulative_flops / 1e15,
                        step,
                    )

                writer.flush()

                logger.info(
                    f"Step {step:05d}/{steps} | Stage: {stage} | "
                    f"Mean Loss: {mean_loss:.4f} | LR: {current_lr:.2e} | "
                    f"Time: {elapsed_time:.1f}s"
                )

                # Save best train loss model for the current sync period
                if mean_loss < best_train_loss:
                    best_train_loss = mean_loss
                    sync_count = getattr(sync_policy, "num_syncs", 0)
                    logger.info(
                        f"Step {step:05d}: New best loss in period "
                        f"#{sync_count}: {best_train_loss:.4f} "
                        "(saving best_model.pt)"
                    )
                    save_checkpoint_and_stats(
                        path=save_dir / "best_model.pt",
                        step=step,
                        batch_size=batch_size,
                        model=active_model,
                        target_model=target_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        best_loss=best_train_loss,
                        train_loss=mean_loss,
                        current_lr=current_lr,
                        elapsed_time=elapsed_time,
                        stage=stage,
                        model_config=model_config,
                        sync_policy_state=sync_policy.state_dict(),
                        num_syncs=sync_count,
                        cumulative_flops=cumulative_flops,
                        cumulative_flops_active=cumulative_flops_active,
                        cumulative_flops_target=cumulative_flops_target,
                        move_penalties=move_penalties,
                    )

                # 5. Target Network Sync Policy Check
                if sync_policy.should_sync(
                    step=step, loss=loss.item(), interval_loss=mean_loss
                ):
                    target_model.load_state_dict(active_model.state_dict())
                    sync_policy.on_sync(step)
                    synced_this_step = True
                    sync_count = getattr(sync_policy, "num_syncs", 0)
                    logger.info(
                        f"Step {step:05d}: Target network SYNCED "
                        f"(sync #{sync_count})."
                    )
                    writer.add_scalar(
                        "Sync/count", getattr(sync_policy, "num_syncs", 0), step
                    )
                    writer.add_scalar("Sync/step", step, step)

                    # Reset best losses for the new post-sync period
                    best_train_loss = float("inf")
                    best_eval_loss = float("inf")

            # Run loss validation on sync or at periodic eval_interval
            should_eval = (
                eval_interval is not None
                and val_states is not None
                and val_depths is not None
                and (synced_this_step or step % eval_interval == 0)
            )
            if should_eval:
                eval_res = evaluate_validation_loss(
                    model=active_model,
                    val_states=val_states,
                    val_depths=val_depths,
                    batch_size=batch_size,
                    device=device,
                )
                val_loss = eval_res["val_loss"]
                writer.add_scalar("Eval/val_loss", val_loss, step)
                writer.flush()

                logger.info(
                    f"Eval @ Step {step:05d}: Val Loss = {val_loss:.4f}"
                )

                if val_loss < best_eval_loss:
                    best_eval_loss = val_loss
                    save_checkpoint_and_stats(
                        path=save_dir / "best_eval_model.pt",
                        step=step,
                        batch_size=batch_size,
                        model=active_model,
                        target_model=target_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        best_loss=best_train_loss,
                        train_loss=(
                            last_mean_loss
                            if last_mean_loss != float("inf")
                            else loss.item()
                        ),
                        current_lr=current_lr,
                        elapsed_time=elapsed_time,
                        stage=stage,
                        model_config=model_config,
                        sync_policy_state=sync_policy.state_dict(),
                        num_syncs=getattr(sync_policy, "num_syncs", 0),
                        cumulative_flops=cumulative_flops,
                        cumulative_flops_active=cumulative_flops_active,
                        cumulative_flops_target=cumulative_flops_target,
                        eval_metrics=eval_res,
                        move_penalties=move_penalties,
                    )

            # 6. Checkpoint saving
            should_save = (
                (save_interval > 0 and step % save_interval == 0)
                or step in additional_save_steps
                or step == steps
            )
            if should_save:
                ckpt_p = save_dir / f"checkpoint_{step:06d}.pt"
                save_checkpoint_and_stats(
                    path=ckpt_p,
                    step=step,
                    batch_size=batch_size,
                    model=active_model,
                    target_model=target_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_loss=best_train_loss,
                    train_loss=loss.item(),
                    current_lr=current_lr,
                    elapsed_time=elapsed_time,
                    stage=stage,
                    model_config=model_config,
                    sync_policy_state=sync_policy.state_dict(),
                    num_syncs=getattr(sync_policy, "num_syncs", 0),
                    cumulative_flops=cumulative_flops,
                    cumulative_flops_active=cumulative_flops_active,
                    cumulative_flops_target=cumulative_flops_target,
                    move_penalties=move_penalties,
                )

        logger.info(f"Training successfully completed {steps} steps.")

        # Save final model
        save_checkpoint_and_stats(
            path=save_dir / "final_model.pt",
            step=steps,
            batch_size=batch_size,
            model=active_model,
            target_model=target_model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_loss=best_train_loss,
            train_loss=loss.item(),
            current_lr=current_lr,
            elapsed_time=time.perf_counter() - start_time,
            stage=stage,
            model_config=model_config,
            sync_policy_state=sync_policy.state_dict(),
            num_syncs=getattr(sync_policy, "num_syncs", 0),
            move_penalties=move_penalties,
        )

    except KeyboardInterrupt:
        logger.warning(
            f"Training interrupted at step {step}. "
            "Saving recovery checkpoint..."
        )
        save_checkpoint_and_stats(
            path=save_dir / "interrupted_checkpoint.pt",
            step=step,
            batch_size=batch_size,
            model=active_model,
            target_model=target_model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_loss=best_train_loss,
            train_loss=loss.item(),
            current_lr=current_lr,
            elapsed_time=time.perf_counter() - start_time,
            stage=stage,
            model_config=model_config,
            sync_policy_state=sync_policy.state_dict(),
            num_syncs=getattr(sync_policy, "num_syncs", 0),
            move_penalties=move_penalties,
        )
    finally:
        writer.flush()
        writer.close()
