import numpy as np
import torch

from cubesight_ml.cube.layout import to_flat
from cubesight_ml.cube.moves import (
    INDEX_TO_MOVE,
    MOVE_TO_INDEX,
)
from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.stages.goal_checker import StageGoalChecker


class BeamSearchRubiksCubeHandler:
    """
    Handles state conversion, transitions, and goal checking for BeamSearch on a specific CFOP stage.
    Supports STM (54 moves with double turns and rotations) as well as HTM/QTM.
    """

    ACTION_TO_STR = INDEX_TO_MOVE
    STR_TO_ACTION = MOVE_TO_INDEX

    def __init__(
        self,
        goal_checker: StageGoalChecker,
        use_stm: bool = True,
        use_double_moves: bool = True,
        include_rotations: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.goal_checker = goal_checker
        self.use_stm = use_stm
        self.use_double_moves = use_double_moves
        self.include_rotations = include_rotations
        self.cube_ops = TorchCubeOps(
            use_stm=use_stm,
            use_double_moves=use_double_moves,
            include_rotations=include_rotations,
            device=self.device,
        )

    def get_state_size(self) -> int:
        return 40

    def get_moves(self) -> torch.Tensor:
        return self.cube_ops.moves

    def get_neighbors(self, states: torch.Tensor) -> torch.Tensor:
        children = self.cube_ops.expand(states)
        return children.reshape(-1, states.size(-1))

    def apply_moves(
        self, states: torch.Tensor, moves: torch.Tensor
    ) -> torch.Tensor:
        return self.cube_ops.apply_moves(states, moves)

    def is_solved(self, states: torch.Tensor) -> torch.Tensor:
        return self.goal_checker.is_goal(states)

    def state_to_model_input(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return (states,)

    def state_from_env(
        self, state: np.ndarray | torch.Tensor | tuple[np.ndarray, ...]
    ) -> torch.Tensor:
        if isinstance(state, tuple):
            if len(state) == 5:
                cp, co, ep, eo, centers = state
                flat_pieces = to_flat(cp, co, ep, eo)
                flat = np.concatenate([flat_pieces, centers], axis=-1)
            else:
                cp, co, ep, eo = state[:4]
                flat = to_flat(cp, co, ep, eo)
            return torch.tensor(
                flat, dtype=torch.long, device=self.device
            ).squeeze()
        if isinstance(state, np.ndarray):
            return torch.tensor(
                state, dtype=torch.long, device=self.device
            ).squeeze()
        return state.to(device=self.device, dtype=torch.long).squeeze()

    def verify_solution(
        self,
        initial_state: np.ndarray | torch.Tensor | tuple[np.ndarray, ...],
        solution: list[int] | None,
    ) -> bool:
        if solution is None:
            return False
        curr = self.state_from_env(initial_state)
        if curr.ndim == 1:
            curr = curr.unsqueeze(0)
        for move in solution:
            move_t = torch.tensor([move], dtype=torch.long, device=self.device)
            curr = self.cube_ops.apply_moves(curr, move_t)
        return bool(self.goal_checker.is_goal(curr).item())

    def format_solution(self, solution: list[int] | None) -> str:
        if solution is None:
            return "No solution found"
        return " ".join(self.ACTION_TO_STR.get(m, str(m)) for m in solution)
