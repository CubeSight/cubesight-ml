import torch
import torch.nn as nn

from cubesight_ml.cube.layout import SOLVED_FLAT
from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.search.beam_search.beam_search import BeamSearch
from cubesight_ml.search.beam_search.utils import BeamSearchRubiksCubeHandler
from cubesight_ml.stages.definitions import CROSS_SPEC, PLL_SPEC
from cubesight_ml.stages.goal_checker import StageGoalChecker


class DistanceMockModel(nn.Module):
    """
    Mock heuristic model that computes distance to the goal set by counting mismatched elements.
    Provides an admissible, highly informative gradient for testing Beam Search without training.
    """

    def __init__(self, spec) -> None:
        super().__init__()
        self.checker = StageGoalChecker(spec, device="cpu")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Count elements that don't match SOLVED_FLAT where mask is True
        solved = self.checker.solved_flat
        # x is (B, 40) or (B, 46)
        x_expanded = x[..., :40].unsqueeze(1)  # (B, 1, 40)
        masks = self.checker.masks
        diffs = (x_expanded != solved) & masks  # (B, M, 40)
        mismatch_counts = diffs.sum(dim=-1)  # (B, M)
        min_mismatches = mismatch_counts.min(dim=-1).values  # (B,)
        return {"value": min_mismatches.unsqueeze(-1).float()}


def test_beam_search_pll_immediate_skip():
    checker = StageGoalChecker(PLL_SPEC, device="cpu")
    handler = BeamSearchRubiksCubeHandler(
        checker, use_double_moves=False, device="cpu"
    )
    model = DistanceMockModel(PLL_SPEC)
    searcher = BeamSearch(model, handler, batch_size=64, device="cpu")

    # Solved cube is already PLL goal
    solution, steps, _stats = searcher.find_solution(
        SOLVED_FLAT, beam_width=10, num_steps=10
    )
    assert solution == []
    assert steps == 0


def test_beam_search_finds_single_move_solution():
    ops = TorchCubeOps(use_double_moves=False, device="cpu")
    checker = StageGoalChecker(PLL_SPEC, device="cpu")
    handler = BeamSearchRubiksCubeHandler(
        checker, use_double_moves=False, device="cpu"
    )
    model = DistanceMockModel(PLL_SPEC)
    searcher = BeamSearch(model, handler, batch_size=64, device="cpu")

    # Scramble with 1 move: R (action 9)
    solved_t = torch.tensor(SOLVED_FLAT, dtype=torch.long).unsqueeze(0)
    scramble_move = torch.tensor([9], dtype=torch.long)
    state = ops.apply_moves(solved_t, scramble_move).squeeze(0).numpy()

    solution, _steps, _stats = searcher.find_solution(
        state, beam_width=10, num_steps=5
    )

    assert solution is not None
    assert len(solution) == 1
    # Move 11 is R' (inverse of R)
    assert solution[0] == 11
    assert handler.verify_solution(state, solution)


def test_beam_search_finds_multi_move_cross_solution():
    ops = TorchCubeOps(use_double_moves=False, device="cpu")
    checker = StageGoalChecker(CROSS_SPEC, device="cpu")
    handler = BeamSearchRubiksCubeHandler(
        checker, use_double_moves=False, device="cpu"
    )
    model = DistanceMockModel(CROSS_SPEC)
    searcher = BeamSearch(model, handler, batch_size=64, device="cpu")

    # Scramble D-cross with 2 moves: D (3), R (9)
    solved_t = torch.tensor(SOLVED_FLAT, dtype=torch.long).unsqueeze(0)
    s1 = ops.apply_moves(solved_t, torch.tensor([3], dtype=torch.long))
    state = (
        ops.apply_moves(s1, torch.tensor([9], dtype=torch.long))
        .squeeze(0)
        .numpy()
    )

    solution, _steps, _stats = searcher.find_solution(
        state, beam_width=20, num_steps=6
    )

    assert solution is not None
    assert len(solution) <= 2
    assert handler.verify_solution(state, solution)
