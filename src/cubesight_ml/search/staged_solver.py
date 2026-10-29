from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from cubesight_ml.cube.moves import (
    CENTER_FORWARD_MAP,
    CENTER_RESTORE_MAP,
    INDEX_TO_MOVE,
    MOVE_TO_INDEX,
    get_restore_rotations,
)
from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.search.beam_search.beam_search import BeamSearch
from cubesight_ml.search.beam_search.utils import BeamSearchRubiksCubeHandler
from cubesight_ml.stages.definitions import (
    STAGES,
    Stage,
)
from cubesight_ml.stages.goal_checker import (
    StageGoalChecker,
    is_goal_np,
)


@dataclass
class StagedSolution:
    """Stores full metrics, moves, and results from a multi-stage solve."""

    solved: bool
    moves: list[int | str]
    formatted_solution: str = ""
    inspection_moves: list[int | str] = field(default_factory=list)
    chosen_color: int | None = None
    stage_moves: dict[Stage, list[int]] = field(default_factory=dict)
    stage_lengths: dict[Stage, int] = field(default_factory=dict)
    stage_stm_lengths: dict[Stage, int] = field(default_factory=dict)
    stage_costs: dict[Stage, float] = field(default_factory=dict)
    stage_stats: dict[Stage, dict[str, Any]] = field(default_factory=dict)
    failed_stage: Stage | None = None
    final_state: np.ndarray | None = None
    total_length: int = 0
    total_stm_length: int = 0
    total_cost: float = 0.0


def apply_action_sequence(
    state: np.ndarray,
    moves: list[int],
    cube_ops: TorchCubeOps,
) -> np.ndarray:
    """Sequentially applies move indices to a flat numpy state using
    TorchCubeOps.
    """
    if not moves:
        return state.copy()
    dev = cube_ops.device
    curr = torch.tensor(state, dtype=torch.long, device=dev)
    if curr.ndim == 1:
        curr = curr.unsqueeze(0)
    for m in moves:
        m_t = torch.tensor([m], dtype=torch.long, device=dev)
        curr = cube_ops.apply_moves(curr, m_t)
    return curr.squeeze(0).cpu().numpy().astype(np.int8)


class StagedSolver:
    """Coordinates sequential 4-stage Beam Search solving:
    Color-Neutral Cross -> Reorient to D -> F2L -> OLL -> PLL.
    Supports STM, rotations, and move execution penalties.
    """

    def __init__(
        self,
        models: dict[Stage | str, nn.Module | str | Path],
        beam_width: int | dict[Stage | str, int] = 30,
        max_steps: int | dict[Stage | str, int] = 30,
        batch_size: int | dict[Stage | str, int] = 512,
        use_stm: bool = True,
        use_double_moves: bool = True,
        include_rotations: bool = True,
        move_penalties: dict[str, float]
        | np.ndarray
        | torch.Tensor
        | str
        | Path
        | dict[Stage | str, Any]
        | None = None,
        device: torch.device | str = "cpu",
        verbose: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.use_stm = use_stm
        self.use_double_moves = use_double_moves
        self.include_rotations = include_rotations
        self.move_penalties = move_penalties
        self.verbose = verbose
        self.cube_ops = TorchCubeOps(
            use_stm=use_stm,
            use_double_moves=use_double_moves,
            include_rotations=include_rotations,
            device=self.device,
        )

        # Standardize configs per stage
        self.beam_widths: dict[Stage, int] = {}
        self.max_steps_dict: dict[Stage, int] = {}
        self.batch_sizes: dict[Stage, int] = {}
        self.searchers: dict[Stage, BeamSearch] = {}

        is_stage_penalties = isinstance(move_penalties, dict) and any(
            k in move_penalties
            for k in [
                Stage.CROSS,
                Stage.F2L,
                Stage.OLL,
                Stage.PLL,
                "cross",
                "f2l",
                "oll",
                "pll",
            ]
        )

        for st in [Stage.CROSS, Stage.F2L, Stage.OLL, Stage.PLL]:
            bw = beam_width[st] if isinstance(beam_width, dict) else beam_width
            ms = max_steps[st] if isinstance(max_steps, dict) else max_steps
            bs = batch_size[st] if isinstance(batch_size, dict) else batch_size
            self.beam_widths[st] = bw
            self.max_steps_dict[st] = ms
            self.batch_sizes[st] = bs

            spec = STAGES[st]
            checker = StageGoalChecker(spec, device=self.device)
            handler = BeamSearchRubiksCubeHandler(
                goal_checker=checker,
                use_stm=use_stm,
                use_double_moves=use_double_moves,
                include_rotations=include_rotations,
                device=self.device,
            )
            model_key = st if st in models else st.value
            model = models[model_key]

            st_penalties = (
                move_penalties.get(st, move_penalties.get(st.value, None))
                if is_stage_penalties
                else move_penalties
            )

            self.searchers[st] = BeamSearch(
                model=model,
                puzzle_handler=handler,
                batch_size=bs,
                move_penalties=st_penalties if use_stm else None,
                use_stm=use_stm,
                verbose=verbose,
                device=self.device,
            )

    def solve(
        self,
        state: np.ndarray | torch.Tensor,
        centers: np.ndarray | None = None,
    ) -> StagedSolution:
        """Solves a cube state through the 4 sequential CFOP stages."""
        if isinstance(state, torch.Tensor):
            curr_state = state.cpu().numpy().astype(np.int8)
        else:
            curr_state = state.copy()
        if curr_state.ndim == 2:
            curr_state = curr_state[0]

        if centers is not None:
            centers = np.asarray(centers, dtype=np.int8)
            if centers.ndim == 2:
                centers = centers[0]

        # Ensure 46-element state with centers appended
        if curr_state.shape[-1] >= 46:
            centers = curr_state[40:46]
        elif centers is not None:
            curr_state = np.concatenate([curr_state[:40], centers])
        else:
            centers = np.arange(6, dtype=np.int8)
            curr_state = np.concatenate([curr_state, centers])

        all_moves: list[int | str] = []
        stage_moves: dict[Stage, list[int]] = {}
        stage_lengths: dict[Stage, int] = {}
        stage_stm_lengths: dict[Stage, int] = {}
        stage_costs: dict[Stage, float] = {}
        stage_stats: dict[Stage, dict[str, Any]] = {}

        # ------------------------------------------------------------------
        # Stage 0: Reorient cube to canonical centers if non-canonical
        # ------------------------------------------------------------------
        if centers is not None:
            c_tuple = tuple(int(c) for c in centers)
            if c_tuple != (0, 1, 2, 3, 4, 5):
                restore_rotations = get_restore_rotations(c_tuple)
                for rot_name in restore_rotations:
                    rot_idx = MOVE_TO_INDEX[rot_name]
                    curr_state = apply_action_sequence(
                        curr_state, [rot_idx], self.cube_ops
                    )
                    all_moves.append(rot_idx)

        # ------------------------------------------------------------------
        # Inspection Phase: Rate all 24 orientations using Cross value heuristic
        # ------------------------------------------------------------------
        centers_list = list(CENTER_RESTORE_MAP.keys())
        candidate_states: list[np.ndarray] = []
        for c in centers_list:
            fwd_names = CENTER_FORWARD_MAP[c]
            fwd_indices = [MOVE_TO_INDEX[m] for m in fwd_names]
            rotated = apply_action_sequence(
                curr_state, fwd_indices, self.cube_ops
            )
            candidate_states.append(rotated)

        candidate_tensor = torch.tensor(
            np.stack(candidate_states), dtype=torch.long, device=self.device
        )

        cross_searcher = self.searchers[Stage.CROSS]
        values_list: list[torch.Tensor] = []
        eval_bs = max(1, min(cross_searcher.batch_size, 24))
        for b_idx in range(0, 24, eval_bs):
            chunk = candidate_tensor[b_idx : b_idx + eval_bs]
            act_size = chunk.size(0)
            if act_size < eval_bs:
                pad = chunk[0:1].expand(eval_bs - act_size, -1)
                padded = torch.cat([chunk, pad], dim=0)
                pred = cross_searcher._get_model_predictions(padded).view(-1)[
                    :act_size
                ]
            else:
                pred = cross_searcher._get_model_predictions(chunk).view(-1)
            values_list.append(pred)
        cross_values = torch.cat(values_list, dim=0).cpu().numpy()

        # Select best orientation minimizing Cross heuristic, breaking ties preferring fewer moves
        best_orientation_id = min(
            range(24),
            key=lambda idx: (
                round(float(cross_values[idx]), 5),
                len(CENTER_FORWARD_MAP[centers_list[idx]]),
                idx,
            ),
        )
        inspection_rot_names = CENTER_FORWARD_MAP[
            centers_list[best_orientation_id]
        ]
        inspection_rot_indices = [
            MOVE_TO_INDEX[m] for m in inspection_rot_names
        ]

        # Apply inspection rotation to best position (for free: 0 cost/length penalty)
        if inspection_rot_indices:
            curr_state = apply_action_sequence(
                curr_state, inspection_rot_indices, self.cube_ops
            )
            all_moves.extend(inspection_rot_indices)

        # Determine chosen reference color on D after inspection
        chosen_color = int(curr_state[43])

        # Lock all 4 stages (Cross, F2L, OLL, PLL) to chosen reference color
        for st in [Stage.CROSS, Stage.F2L, Stage.OLL, Stage.PLL]:
            self.searchers[st].puzzle_handler.goal_checker.set_target_color(
                chosen_color
            )

        # ------------------------------------------------------------------
        # Stage 1: Cross (forced on chosen reference color)
        # ------------------------------------------------------------------
        cross_bw = self.beam_widths[Stage.CROSS]
        cross_ms = self.max_steps_dict[Stage.CROSS]
        cross_sol, _, cross_stats = self.searchers[Stage.CROSS].find_solution(
            curr_state, beam_width=cross_bw, num_steps=cross_ms
        )
        stage_stats[Stage.CROSS] = cross_stats

        if cross_sol is None:
            return StagedSolution(
                solved=False,
                moves=all_moves,
                inspection_moves=inspection_rot_names,
                chosen_color=chosen_color,
                stage_moves=stage_moves,
                stage_lengths=stage_lengths,
                stage_stm_lengths=stage_stm_lengths,
                stage_costs=stage_costs,
                stage_stats=stage_stats,
                failed_stage=Stage.CROSS,
                final_state=curr_state,
            )

        curr_state = apply_action_sequence(curr_state, cross_sol, self.cube_ops)
        all_moves.extend(cross_sol)
        stage_moves[Stage.CROSS] = cross_sol
        stage_lengths[Stage.CROSS] = len(cross_sol)
        stage_stm_lengths[Stage.CROSS] = cross_stats.get(
            "stm_length", len(cross_sol)
        )
        stage_costs[Stage.CROSS] = cross_stats.get(
            "total_cost", float(len(cross_sol))
        )
        assert is_goal_np(
            curr_state,
            STAGES[Stage.CROSS],
            target_color=chosen_color,
        )

        # ------------------------------------------------------------------
        # Stages 2..4: F2L -> OLL -> PLL (locked to chosen reference color)
        # ------------------------------------------------------------------
        for st in [Stage.F2L, Stage.OLL, Stage.PLL]:
            bw = self.beam_widths[st]
            ms = self.max_steps_dict[st]
            st_sol, _, st_stats = self.searchers[st].find_solution(
                curr_state, beam_width=bw, num_steps=ms
            )
            stage_stats[st] = st_stats

            if st_sol is None:
                return StagedSolution(
                    solved=False,
                    moves=all_moves,
                    inspection_moves=inspection_rot_names,
                    chosen_color=chosen_color,
                    stage_moves=stage_moves,
                    stage_lengths=stage_lengths,
                    stage_stm_lengths=stage_stm_lengths,
                    stage_costs=stage_costs,
                    stage_stats=stage_stats,
                    failed_stage=st,
                    final_state=curr_state,
                )

            curr_state = apply_action_sequence(
                curr_state, st_sol, self.cube_ops
            )
            all_moves.extend(st_sol)
            stage_moves[st] = st_sol
            stage_lengths[st] = len(st_sol)
            stage_stm_lengths[st] = st_stats.get("stm_length", len(st_sol))
            stage_costs[st] = st_stats.get("total_cost", float(len(st_sol)))
            assert is_goal_np(curr_state, STAGES[st], target_color=chosen_color)

        # Verify final state is completely solved
        assert is_goal_np(
            curr_state,
            STAGES[Stage.PLL],
            target_color=chosen_color,
        )

        # Format human-readable solution string
        formatted_parts = [
            INDEX_TO_MOVE.get(m, str(m)) if isinstance(m, int) else m
            for m in all_moves
        ]
        formatted_str = " ".join(formatted_parts)

        total_len = len(all_moves)
        total_stm = sum(stage_stm_lengths.values())
        total_c = sum(stage_costs.values())

        return StagedSolution(
            solved=True,
            moves=all_moves,
            formatted_solution=formatted_str,
            inspection_moves=inspection_rot_names,
            chosen_color=chosen_color,
            stage_moves=stage_moves,
            stage_lengths=stage_lengths,
            stage_stm_lengths=stage_stm_lengths,
            stage_costs=stage_costs,
            stage_stats=stage_stats,
            failed_stage=None,
            final_state=curr_state,
            total_length=total_len,
            total_stm_length=total_stm,
            total_cost=total_c,
        )
