import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from cubesight_ml.cube.layout import (
    from_flat,
    to_flat,
)
from cubesight_ml.cube.moves import (
    INDEX_TO_MOVE,
    MOVE_TO_INDEX,
    get_restore_rotations,
    parse_move_penalties,
)
from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.envs.rubiks_cube import RubiksCubeEnv
from cubesight_ml.models.value_wrapper import build_model
from cubesight_ml.search.beam_search.beam_search import BeamSearch
from cubesight_ml.search.beam_search.utils import BeamSearchRubiksCubeHandler
from cubesight_ml.search.staged_solver import (
    StagedSolver,
    apply_action_sequence,
)
from cubesight_ml.stages.definitions import (
    CROSS_SPEC,
    F2L_SPEC,
    OLL_SPEC,
    PLL_SPEC,
    STAGES,
    Stage,
)
from cubesight_ml.stages.goal_checker import StageGoalChecker
from cubesight_ml.stages.goal_sampler import sample_goal_states
from cubesight_ml.training.checkpointing import load_checkpoint
from cubesight_ml.visualization.cube_state import (
    get_cubies_representation,
    state_to_stickers,
    stickers_to_face_dict,
)


class DistanceMockModel(nn.Module):
    """Fallback admissible heuristic model when trained checkpoints are not yet available."""

    def __init__(self, spec) -> None:
        super().__init__()
        self.checker = StageGoalChecker(spec, device="cpu")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        solved = self.checker.solved_flat
        masks = self.checker.masks
        x_expanded = x[..., :40].unsqueeze(1)
        diffs = (x_expanded != solved) & masks
        mismatch_counts = diffs.sum(dim=-1)
        min_mismatches = mismatch_counts.min(dim=-1).values
        return {"value": min_mismatches.unsqueeze(-1).float()}


class SolverAPI:
    """
    Core backend controller for the Rubik's Cube 3D visualizer.
    Manages cube simulation, checkpoint loading, and Beam Search solving.
    """

    def __init__(
        self,
        checkpoints: dict[str, str | Path | None] | None = None,
        move_penalties: str | dict | None = None,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.env = RubiksCubeEnv(
            batch_size=1,
            use_stm=True,
            use_double_moves=True,
            include_rotations=True,
        )

        self.checkpoints: dict[str, Path | None] = {
            "cross": None,
            "f2l": None,
            "oll": None,
            "pll": None,
        }
        self.models: dict[str, nn.Module] = {}
        self.is_heuristic: dict[str, bool] = {}

        self.move_penalties_cfg = move_penalties
        self.default_penalties = parse_move_penalties(move_penalties)

        self.goal_checkers = {
            Stage.CROSS: StageGoalChecker(CROSS_SPEC, device=self.device),
            Stage.F2L: StageGoalChecker(F2L_SPEC, device=self.device),
            Stage.OLL: StageGoalChecker(OLL_SPEC, device=self.device),
            Stage.PLL: StageGoalChecker(PLL_SPEC, device=self.device),
        }

        # Auto-detect or load initial checkpoints
        self._init_checkpoints(checkpoints)

    def _init_checkpoints(
        self, checkpoints: dict[str, str | Path | None] | None
    ) -> None:
        # Check user-supplied
        if checkpoints:
            for s, path in checkpoints.items():
                if path and Path(path).exists():
                    self.checkpoints[s] = Path(path)

        # Auto-detect from runs/ if not provided
        self.auto_discover_checkpoints()

        # Load or initialize models
        for s in ["cross", "f2l", "oll", "pll"]:
            self._load_or_create_model(s)

    def auto_discover_checkpoints(self) -> dict[str, str | None]:
        """Scans runs/ for the best or most recent checkpoint for each stage."""
        runs_dir = Path("runs")
        discovered: dict[str, str | None] = {}
        for s in ["cross", "f2l", "oll", "pll"]:
            if self.checkpoints.get(s) is not None:
                discovered[s] = str(self.checkpoints[s])
                continue

            stage_runs = runs_dir / "davi" / s
            best_ckpt: Path | None = None
            if stage_runs.exists():
                # Search for best_model.pt or final_model.pt or latest checkpoint
                candidates = list(stage_runs.glob("**/best_model.pt"))
                if not candidates:
                    candidates = list(stage_runs.glob("**/final_model.pt"))
                if not candidates:
                    candidates = list(stage_runs.glob("**/*.pt"))

                if candidates:
                    # Pick most recent by modification time
                    candidates.sort(
                        key=lambda p: p.stat().st_mtime, reverse=True
                    )
                    best_ckpt = candidates[0]

            if best_ckpt:
                self.checkpoints[s] = best_ckpt
                discovered[s] = str(best_ckpt)
            else:
                discovered[s] = None

        return discovered

    def _load_or_create_model(self, stage_name: str) -> None:
        ckpt_path = self.checkpoints.get(stage_name)
        if ckpt_path and ckpt_path.exists():
            try:
                data = load_checkpoint(ckpt_path, device=self.device)
                cfg = data.get("model_config", {})
                if not cfg:
                    cfg = {
                        "type": "res_mlp",
                        "state_dim": 40,
                        "hidden_dim": 256,
                        "num_layers": 4,
                        "representation": "per_piece",
                    }
                model = build_model(cfg)
                if data.get("model_state_dict"):
                    model.load_state_dict(data["model_state_dict"])
                model.to(self.device).eval()
                self.models[stage_name] = model
                self.is_heuristic[stage_name] = False
                logger.info(
                    f"Loaded trained model for {stage_name} from {ckpt_path}"
                )
                return
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Failed loading checkpoint for {stage_name}: {e}. Falling back to heuristic."
                )

        # Fallback to DistanceMockModel
        spec = STAGES[Stage(stage_name)]
        mock = DistanceMockModel(spec).to(self.device).eval()
        self.models[stage_name] = mock
        self.is_heuristic[stage_name] = True
        logger.info(f"Using DistanceMockModel heuristic for {stage_name}")

    def get_state(self) -> dict[str, Any]:
        """Returns the current cube state formatted for 3D visualization."""
        cp, co, ep, eo, centers = self.env.get_state_with_centers()
        stickers = state_to_stickers(cp[0], co[0], ep[0], eo[0], centers[0])
        cubies = get_cubies_representation(stickers)
        faces = stickers_to_face_dict(stickers)

        # Stage goal status
        flat_40 = torch.tensor(
            to_flat(cp, co, ep, eo), dtype=torch.long, device=self.device
        )
        cross_solved = bool(
            self.goal_checkers[Stage.CROSS].is_goal(flat_40)[0].item()
        )
        f2l_solved = bool(
            self.goal_checkers[Stage.F2L].is_goal(flat_40)[0].item()
        )
        oll_solved = bool(
            self.goal_checkers[Stage.OLL].is_goal(flat_40)[0].item()
        )
        pll_solved = bool(
            self.goal_checkers[Stage.PLL].is_goal(flat_40)[0].item()
        )

        return {
            "stickers": stickers,
            "faces": faces,
            "cubies": cubies,
            "centers": centers[0].tolist(),
            "status": {
                "cross_solved": cross_solved,
                "f2l_solved": f2l_solved,
                "oll_solved": oll_solved,
                "pll_solved": pll_solved,
                "is_solved": pll_solved,
            },
            "models": {
                s: {
                    "checkpoint": str(self.checkpoints[s])
                    if self.checkpoints[s]
                    else None,
                    "is_heuristic": self.is_heuristic.get(s, True),
                }
                for s in ["cross", "f2l", "oll", "pll"]
            },
        }

    def reset(self) -> dict[str, Any]:
        """Resets cube to solved identity state."""
        self.env.reset()
        return self.get_state()

    def apply_move(self, move_name: str) -> dict[str, Any]:
        """Applies a single move (e.g. 'R', 'M', 'x') to the cube."""
        if move_name not in MOVE_TO_INDEX:
            raise ValueError(f"Unknown move: {move_name}")
        act = np.array([MOVE_TO_INDEX[move_name]], dtype=np.int32)
        self.env.step(act)
        return self.get_state()

    def scramble(
        self, depth: int = 30, custom_moves: str | list[str] | None = None
    ) -> dict[str, Any]:
        """Scrambles cube with random moves or a specific move sequence."""
        self.env.reset()
        if custom_moves:
            if isinstance(custom_moves, str):
                moves_list = [
                    m.strip() for m in custom_moves.split() if m.strip()
                ]
            else:
                moves_list = custom_moves

            for m in moves_list:
                if m in MOVE_TO_INDEX:
                    self.env.step(np.array([MOVE_TO_INDEX[m]], dtype=np.int32))
        else:
            self.env.scramble(depth=depth)

        return self.get_state()

    def precondition(self, stage: str) -> dict[str, Any]:
        """Loads a state with the given stage completed (e.g. 'cross' -> Cross Done,
        'f2l' -> F2L Done, 'oll' -> OLL Done, 'pll' -> Fully Solved).
        """
        stage_enum = Stage(stage)
        spec = STAGES[stage_enum]
        # Keep Cross on standard D face
        fixed_face = "D" if stage_enum == Stage.CROSS else None

        states = sample_goal_states(
            spec=spec,
            n=1,
            fixed_cross_face=fixed_face,
        )
        cp, co, ep, eo = from_flat(states)
        self.env.reset()
        self.env.load_state(cp, co, ep, eo)
        return self.get_state()

    def solve(
        self,
        stage: str = "all",
        beam_width: int = 100,
        max_steps: int = 40,
        move_penalties: str | dict | None = None,
    ) -> dict[str, Any]:
        """
        Runs Beam Search inference for the requested stage or the whole 4-stage CFOP pipeline.
        Returns solution moves, metrics, and intermediate states after each move for 3D animation.
        """
        penalties = (
            parse_move_penalties(move_penalties)
            if move_penalties is not None
            else self.default_penalties
        )

        cp, co, ep, eo, centers = self.env.get_state_with_centers()
        initial_pieces = to_flat(cp, co, ep, eo)

        start_time = time.perf_counter()

        # 1. Single stage or full pipeline
        if stage == "all":
            models_dict = {
                Stage.CROSS: self.models["cross"],
                Stage.F2L: self.models["f2l"],
                Stage.OLL: self.models["oll"],
                Stage.PLL: self.models["pll"],
            }
            solver = StagedSolver(
                models=models_dict,
                beam_width=beam_width,
                max_steps=max_steps,
                use_stm=True,
                use_double_moves=True,
                include_rotations=True,
                move_penalties=penalties,
                device=self.device,
            )
            sol = solver.solve(initial_pieces, centers=centers[0])
            elapsed = time.perf_counter() - start_time

            success = sol.solved
            moves = [
                INDEX_TO_MOVE[m]
                if isinstance(m, int)
                else str(m).replace("_prime", "'")
                for m in sol.moves
            ]
            stm_len = sol.total_stm_length
            cost = sol.total_cost
            breakdown = [
                {
                    "stage": s.value,
                    "solved": (
                        sol.failed_stage != s
                        and (s in sol.stage_moves or sol.solved)
                    ),
                    "moves": [
                        INDEX_TO_MOVE[m]
                        if isinstance(m, int)
                        else str(m).replace("_prime", "'")
                        for m in sol.stage_moves.get(s, [])
                    ],
                    "length": sol.stage_lengths.get(s, 0),
                    "stm_length": sol.stage_stm_lengths.get(s, 0),
                    "cost": sol.stage_costs.get(s, 0.0),
                }
                for s in [Stage.CROSS, Stage.F2L, Stage.OLL, Stage.PLL]
            ]
        else:
            stage_enum = Stage(stage)
            spec = STAGES[stage_enum]
            checker = StageGoalChecker(spec, device=self.device)

            # Reorient cube to canonical centers if non-canonical
            prefix_rotations: list[str] = []
            curr_pieces = initial_pieces.copy()
            c_tuple = tuple(int(c) for c in centers[0])
            if c_tuple != (0, 1, 2, 3, 4, 5):
                prefix_rotations = get_restore_rotations(c_tuple)
                if prefix_rotations:
                    rot_ops = TorchCubeOps(use_stm=True, device=self.device)
                    for r_name in prefix_rotations:
                        r_idx = MOVE_TO_INDEX[r_name]
                        curr_pieces = apply_action_sequence(
                            curr_pieces, [r_idx], rot_ops
                        )

            handler = BeamSearchRubiksCubeHandler(
                goal_checker=checker,
                use_stm=True,
                use_double_moves=True,
                include_rotations=True,
                device=self.device,
            )
            searcher = BeamSearch(
                model=self.models[stage],
                puzzle_handler=handler,
                move_penalties=penalties,
                use_stm=True,
                device=self.device,
            )
            initial_centers = (
                np.arange(6, dtype=np.int8) if prefix_rotations else centers[0]
            )
            curr_flat = curr_pieces[0] if curr_pieces.ndim == 2 else curr_pieces
            state_46 = np.concatenate([curr_flat[:40], initial_centers])
            state_tensor = torch.tensor(
                state_46, dtype=torch.long, device=self.device
            )
            sol_indices, steps, stats = searcher.find_solution(
                state_tensor, beam_width=beam_width, num_steps=max_steps
            )
            elapsed = time.perf_counter() - start_time

            success = sol_indices is not None
            stage_sol_moves = (
                [INDEX_TO_MOVE[i] for i in sol_indices]
                if sol_indices is not None
                else []
            )
            moves = prefix_rotations + stage_sol_moves
            stm_len = stats.get("stm_length", 0)
            cost = stats.get("total_cost", 0.0)
            breakdown = [
                {
                    "stage": stage,
                    "solved": success,
                    "moves": moves,
                    "length": len(moves),
                    "stm_length": stm_len,
                    "cost": cost,
                }
            ]

        # 2. Simulate solution step-by-step to capture intermediate 3D states
        sim_env = RubiksCubeEnv(
            batch_size=1,
            use_stm=True,
            use_double_moves=True,
            include_rotations=True,
        )
        sim_env.load_state(cp, co, ep, eo, centers)

        intermediate_states = []
        # Initial state (step 0)
        c_cp, c_co, c_ep, c_eo, c_cnt = sim_env.get_state_with_centers()
        stk_0 = state_to_stickers(c_cp[0], c_co[0], c_ep[0], c_eo[0], c_cnt[0])
        intermediate_states.append(
            {
                "step": 0,
                "move": None,
                "stickers": stk_0,
                "cubies": get_cubies_representation(stk_0),
                "faces": stickers_to_face_dict(stk_0),
            }
        )

        for step_idx, m_name in enumerate(moves, start=1):
            m_key = m_name.replace("_prime", "'")
            act = np.array([MOVE_TO_INDEX[m_key]], dtype=np.int32)
            sim_env.step(act)
            s_cp, s_co, s_ep, s_eo, s_cnt = sim_env.get_state_with_centers()
            stk = state_to_stickers(
                s_cp[0], s_co[0], s_ep[0], s_eo[0], s_cnt[0]
            )
            intermediate_states.append(
                {
                    "step": step_idx,
                    "move": m_name,
                    "stickers": stk,
                    "cubies": get_cubies_representation(stk),
                    "faces": stickers_to_face_dict(stk),
                }
            )

        return {
            "success": success,
            "moves": moves,
            "total_length": len(moves),
            "stm_length": stm_len,
            "total_cost": cost,
            "time_seconds": round(elapsed, 4),
            "stage_breakdown": breakdown,
            "intermediate_states": intermediate_states,
        }
