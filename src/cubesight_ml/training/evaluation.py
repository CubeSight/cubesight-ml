import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cubesight_ml.cube.layout import from_flat, to_flat
from cubesight_ml.cube.moves import INDEX_TO_MOVE
from cubesight_ml.envs.rubiks_cube import RubiksCubeEnv
from cubesight_ml.search.beam_search.beam_search import BeamSearch
from cubesight_ml.search.beam_search.utils import BeamSearchRubiksCubeHandler
from cubesight_ml.stages.definitions import STAGES, StageSpec
from cubesight_ml.stages.goal_checker import StageGoalChecker
from cubesight_ml.stages.goal_sampler import sample_goal_states


def generate_validation_dataset(
    spec: StageSpec,
    max_depth: int = 26,
    states_per_depth: int = 1000,
    use_double_moves: bool = False,
    seed: int = 42,
    save_path: Path | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Generates a stratified validation dataset from depth 1 to max_depth.

    Samples `states_per_depth` goal states for each depth d in [1, max_depth],
    scrambles them to depth d, and optionally saves the scramble move sequences
    to `save_path` as a JSON list.

    Returns:
        tuple of (states_tensor (N, 46) int8, depths_tensor (N,) int16, scramble_strings)
    """
    if max_depth <= 0 or states_per_depth <= 0:
        raise ValueError(
            f"max_depth ({max_depth}) and states_per_depth ({states_per_depth}) must be > 0"
        )

    rng = np.random.default_rng(seed)
    total_states = max_depth * states_per_depth
    depths = np.repeat(
        np.arange(1, max_depth + 1, dtype=np.int16), states_per_depth
    )

    goal_states = sample_goal_states(spec, total_states, rng=rng)
    cp, co, ep, eo = from_flat(goal_states)
    centers = goal_states[:, 40:46] if goal_states.shape[-1] >= 46 else None

    env = RubiksCubeEnv(
        batch_size=total_states,
        use_double_moves=use_double_moves,
        rng=rng,
    )
    env.load_state(cp, co, ep, eo, centers=centers)
    env.scramble(depth=max_depth, depths=depths)

    s_cp, s_co, s_ep, s_eo, s_c = env.get_state_with_centers()
    val_states = np.concatenate([to_flat(s_cp, s_co, s_ep, s_eo), s_c], axis=-1)

    actions = env.last_scramble_actions
    scramble_strings: list[str] = []
    if actions is not None:
        for i in range(total_states):
            d = int(depths[i])
            moves = [INDEX_TO_MOVE[int(a)] for a in actions[:d, i]]
            scramble_strings.append(" ".join(moves))

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(scramble_strings, f, indent=4)

    return (
        torch.from_numpy(val_states.copy()).to(torch.int8),
        torch.from_numpy(depths.copy()).to(torch.int16),
        scramble_strings,
    )


@torch.no_grad()
def evaluate_validation_loss(
    model: nn.Module,
    val_states: torch.Tensor,
    val_depths: torch.Tensor,
    batch_size: int = 10000,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Computes validation MSE loss between model predictions and scramble depths."""
    device = torch.device(device)
    was_training = model.training
    model.eval()
    try:
        preds_list = []
        for chunk in val_states.split(batch_size):
            chunk_dev = chunk.to(device, non_blocking=True)
            chunk_pred = model(chunk_dev)["value"].squeeze(-1).float()
            preds_list.append(chunk_pred)

        preds = torch.cat(preds_list, dim=0)
        targets = val_depths.to(device=device, dtype=torch.float32)
        val_loss = float(F.mse_loss(preds, targets).item())

        return {
            "val_loss": val_loss,
            "total_states": int(val_states.size(0)),
        }
    finally:
        if was_training:
            model.train()


def evaluate_stage(
    model: nn.Module,
    spec: StageSpec,
    n_states: int = 20,
    beam_width: int = 30,
    max_steps: int = 20,
    seed: int = 42,
    device: torch.device | str = "cpu",
    use_stm: bool = True,
    use_double_moves: bool = True,
    include_rotations: bool = True,
    move_penalties: dict[str, float]
    | np.ndarray
    | torch.Tensor
    | str
    | Path
    | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Evaluates stage solver on a fixed deterministic set of states.

    Returns:
        Dictionary containing solve_rate, mean_length, mean_stm_length,
        mean_cost, max_length, mean_nodes, and per-state lengths.
    """
    rng = np.random.default_rng(seed)

    precond_stage = spec.precondition
    precond_spec = None if precond_stage is None else STAGES[precond_stage]

    val_states = sample_goal_states(
        precond_spec,
        n_states,
        rng=rng,
    )

    checker = StageGoalChecker(spec, device=device)
    handler = BeamSearchRubiksCubeHandler(
        goal_checker=checker,
        use_stm=use_stm,
        use_double_moves=use_double_moves,
        include_rotations=include_rotations,
        device=device,
    )
    searcher = BeamSearch(
        model=model,
        puzzle_handler=handler,
        batch_size=512,
        move_penalties=move_penalties,
        use_stm=use_stm,
        verbose=verbose,
        device=device,
    )

    solved_count = 0
    solution_lengths: list[int] = []
    stm_lengths: list[int] = []
    solution_costs: list[float] = []
    nodes_expanded: list[int] = []
    all_solutions: list[list[int] | None] = []
    all_move_names: list[list[str]] = []

    for i in range(n_states):
        state = val_states[i]
        solution, _steps, stats = searcher.find_solution(
            state,
            beam_width=beam_width,
            num_steps=max_steps,
        )
        all_solutions.append(solution)
        all_move_names.append(stats.get("move_names", []))
        nodes_expanded.append(stats.get("states_expanded", 0))

        if solution is not None:
            solved_count += 1
            solution_lengths.append(len(solution))
            stm_lengths.append(stats.get("stm_length", len(solution)))
            solution_costs.append(stats.get("total_cost", float(len(solution))))

    solve_rate = solved_count / n_states
    mean_length = (
        float(np.mean(solution_lengths)) if solution_lengths else float("inf")
    )
    mean_stm_length = (
        float(np.mean(stm_lengths)) if stm_lengths else float("inf")
    )
    mean_cost = (
        float(np.mean(solution_costs)) if solution_costs else float("inf")
    )
    max_length = int(np.max(solution_lengths)) if solution_lengths else 0
    mean_nodes = float(np.mean(nodes_expanded)) if nodes_expanded else 0.0

    return {
        "solve_rate": solve_rate,
        "solved_count": solved_count,
        "total_states": n_states,
        "mean_length": mean_length,
        "mean_stm_length": mean_stm_length,
        "mean_cost": mean_cost,
        "max_length": max_length,
        "mean_nodes": mean_nodes,
        "solution_lengths": solution_lengths,
        "stm_lengths": stm_lengths,
        "solution_costs": solution_costs,
        "solutions": all_solutions,
        "move_names": all_move_names,
    }
