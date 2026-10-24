from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from cubesight_ml.search.beam_search.beam_search import BeamSearch
from cubesight_ml.search.beam_search.utils import BeamSearchRubiksCubeHandler
from cubesight_ml.stages.definitions import STAGES, Stage, StageSpec
from cubesight_ml.stages.goal_checker import StageGoalChecker
from cubesight_ml.stages.goal_sampler import sample_goal_states


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
    """
    Evaluates stage solver on a fixed deterministic set of states drawn from the stage's precondition set.

    Returns:
        Dictionary containing solve_rate, mean_length, mean_stm_length, mean_cost, max_length, mean_nodes, and per-state lengths.
    """
    rng = np.random.default_rng(seed)

    precond_stage = spec.precondition
    precond_spec = None if precond_stage is None else STAGES[precond_stage]

    fixed_cross_face = "D" if spec.stage == Stage.F2L else None
    val_states = sample_goal_states(
        precond_spec,
        n_states,
        rng=rng,
        fixed_cross_face=fixed_cross_face,
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
