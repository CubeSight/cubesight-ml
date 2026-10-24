import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.stages.definitions import StageSpec
from cubesight_ml.stages.goal_checker import StageGoalChecker
from cubesight_ml.training.checkpointing import save_checkpoint_and_stats
from cubesight_ml.training.davi_targets import compute_davi_targets
from cubesight_ml.training.evaluation import evaluate_stage
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
    grad_accumulation_steps: int = 1,
    clip_norm: float | None = 1.0,
    loss_interval: int = 100,
    save_interval: int = 1000,
    eval_interval: int | None = 500,
    eval_n_states: int = 20,
    eval_beam_width: int = 30,
    eval_max_steps: int = 20,
    device: torch.device | str = "cpu",
    macs_per_sample: float | None = None,
    save_sample_steps: dict[int, int] | None = None,
    save_steps: list[int] | None = None,
    additional_save_steps: set[int] | None = None,
    warmup_steps: int = 1,
) -> None:
    """
    Main DeepCubeA Value Iteration (DAVI) training loop for sequential CFOP stages.
    """
    device = torch.device(device)
    active_model = active_model.to(device)
    target_model = target_model.to(device)
    target_model.eval()

    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))

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
                    target_model, parents_w, cube_ops, goal_checker
                )
                _ = active_model(parents_w)["value"]
        logger.info("Warm-up complete.")

    # FLOP accounting: 1 MAC = 2 FLOPs.
    # Active forward + backward: 3 * 2 * MACs = 6 * MACs. Target forward: 2 * MACs * num_actions.
    n_actions = len(cube_ops.moves)
    flops_per_active_sample = 6.0 * macs_per_sample if macs_per_sample else None
    flops_per_target_sample = (
        2.0 * macs_per_sample * n_actions if macs_per_sample else None
    )

    cumulative_flops_active = 0.0
    cumulative_flops_target = 0.0
    cumulative_flops = 0.0

    running_loss = 0.0
    best_train_loss = float("inf")
    best_eval_solve_rate = -1.0
    best_eval_length = float("inf")

    start_time = time.perf_counter()

    logger.info(
        f"Starting DAVI training for stage '{stage}', steps={steps}, device={device}"
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

            # 4. Logging & Diagnostics
            if step % loss_interval == 0:
                mean_loss = running_loss / loss_interval
                running_loss = 0.0

                writer.add_scalar("Time/Elapsed_Seconds", elapsed_time, step)
                writer.add_scalar("Training/LR", current_lr, step)
                writer.add_scalar("Loss_Mean/total", mean_loss, step)
                writer.add_scalar("Loss_Instant/total", loss.item(), step)

                # Goal fraction in batch
                goal_frac = (targets == 0.0).float().mean().item()
                writer.add_scalar("Targets/goal_fraction", goal_frac, step)

                # Mean targets and preds by depth
                for k in range(0, 11):
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
                    f"Step {step:05d}/{steps} | Stage: {stage} | Mean Loss: {mean_loss:.4f} | LR: {current_lr:.2e} | Time: {elapsed_time:.1f}s"
                )

                # Save best train loss model
                if mean_loss < best_train_loss:
                    best_train_loss = mean_loss
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
                        num_syncs=getattr(sync_policy, "num_syncs", 0),
                        cumulative_flops=cumulative_flops,
                        cumulative_flops_active=cumulative_flops_active,
                        cumulative_flops_target=cumulative_flops_target,
                    )

                # 5. Target Network Sync Policy Check
                if sync_policy.should_sync(
                    step=step, loss=loss.item(), interval_loss=mean_loss
                ):
                    target_model.load_state_dict(active_model.state_dict())
                    sync_policy.on_sync(step)
                    logger.info(
                        f"Step {step:05d}: Target network SYNCED (sync #{getattr(sync_policy, 'num_syncs', 0)})."
                    )
                    writer.add_scalar(
                        "Sync/count", getattr(sync_policy, "num_syncs", 0), step
                    )
                    writer.add_scalar("Sync/step", step, step)

                    # Run evaluation on sync
                    if eval_interval is not None:
                        active_model.eval()
                        eval_res = evaluate_stage(
                            model=active_model,
                            spec=spec,
                            n_states=eval_n_states,
                            beam_width=eval_beam_width,
                            max_steps=eval_max_steps,
                            device=device,
                        )
                        active_model.train()

                        sr = eval_res["solve_rate"]
                        ml = eval_res["mean_length"]
                        writer.add_scalar("Eval/solve_rate", sr, step)
                        writer.add_scalar("Eval/mean_solution_length", ml, step)
                        writer.add_scalar(
                            "Eval/mean_nodes", eval_res["mean_nodes"], step
                        )
                        writer.flush()

                        logger.info(
                            f"Eval @ Step {step:05d}: Solve Rate = {sr * 100:.1f}%, Mean Len = {ml:.2f}"
                        )

                        # Save best eval model
                        if (sr > best_eval_solve_rate) or (
                            sr == best_eval_solve_rate and ml < best_eval_length
                        ):
                            best_eval_solve_rate = sr
                            best_eval_length = ml
                            save_checkpoint_and_stats(
                                path=save_dir / "best_eval_model.pt",
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
                                num_syncs=getattr(sync_policy, "num_syncs", 0),
                                eval_metrics=eval_res,
                            )

            # Periodic standalone evaluation (if not synced)
            if (
                eval_interval is not None
                and step % eval_interval == 0
                and getattr(sync_policy, "last_sync_step", -1) != step
            ):
                active_model.eval()
                eval_res = evaluate_stage(
                    model=active_model,
                    spec=spec,
                    n_states=eval_n_states,
                    beam_width=eval_beam_width,
                    max_steps=eval_max_steps,
                    device=device,
                )
                active_model.train()
                writer.add_scalar(
                    "Eval/solve_rate", eval_res["solve_rate"], step
                )
                writer.add_scalar(
                    "Eval/mean_solution_length", eval_res["mean_length"], step
                )
                writer.flush()

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
        )

    except KeyboardInterrupt:
        logger.warning(
            f"Training interrupted at step {step}. Saving recovery checkpoint..."
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
        )
    finally:
        writer.flush()
        writer.close()
