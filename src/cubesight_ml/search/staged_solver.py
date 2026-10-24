from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from cubesight_ml.cube.layout import (
    FACE_TO_D_ROTATIONS,
    apply_cube_rotation,
)
from cubesight_ml.cube.moves import (
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
    detect_solved_cross_face,
    is_goal_np,
)


@dataclass
class StagedSolution:
    """Stores full metrics, moves, and results from a multi-stage solve."""

    solved: bool
    moves: list[int | str]
    formatted_solution: str = ""
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
    """Sequentially applies move indices to a flat numpy state using TorchCubeOps."""
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
    """
    Coordinates sequential 4-stage Beam Search solving (Color-Neutral Cross -> Reorient to D -> F2L -> OLL -> PLL).
    Supports STM, rotations, and move execution penalties.
    """

    def __init__(
        self,
        models: dict[Stage | str, nn.Module],
        beam_width: int | dict[Stage | str, int] = 30,
        max_steps: int | dict[Stage | str, int] = 30,
        use_stm: bool = True,
        use_double_moves: bool = True,
        include_rotations: bool = True,
        move_penalties: dict[str, float]
        | np.ndarray
        | torch.Tensor
        | str
        | Path
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
        self.searchers: dict[Stage, BeamSearch] = {}

        for st in [Stage.CROSS, Stage.F2L, Stage.OLL, Stage.PLL]:
            bw = beam_width[st] if isinstance(beam_width, dict) else beam_width
            ms = max_steps[st] if isinstance(max_steps, dict) else max_steps
            self.beam_widths[st] = bw
            self.max_steps_dict[st] = ms

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
            self.searchers[st] = BeamSearch(
                model=model,
                puzzle_handler=handler,
                batch_size=512,
                move_penalties=move_penalties if use_stm else None,
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
        # Stage 1: Color-Neutral Cross
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

        # ------------------------------------------------------------------
        # Stage 2: Detect Cross Face & Reorient to D
        # ------------------------------------------------------------------
        solved_face = detect_solved_cross_face(curr_state)
        rot_cmd = FACE_TO_D_ROTATIONS[solved_face]

        if rot_cmd != "id":
            curr_pieces = apply_cube_rotation(curr_state[:40], rot_cmd)
            curr_state = np.concatenate(
                [curr_pieces, np.arange(6, dtype=np.int8)]
            )
            rot_idx = MOVE_TO_INDEX.get(rot_cmd, rot_cmd)
            all_moves.append(rot_idx)

        # Verify that D-cross is indeed now solved
        curr_t = torch.tensor(curr_state, dtype=torch.long, device=self.device)
        if curr_t.ndim == 1:
            curr_t = curr_t.unsqueeze(0)
        assert (
            self.searchers[Stage.CROSS]
            .puzzle_handler.is_solved(curr_t)[0]
            .item()
        )

        # ------------------------------------------------------------------
        # Stages 3..5: F2L -> OLL -> PLL (in standard D/U-frame)
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
            assert is_goal_np(curr_state, STAGES[st])

        # Verify final state is completely solved
        assert is_goal_np(curr_state, STAGES[Stage.PLL])

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
