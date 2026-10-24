import numpy as np
import pytest
import torch
import torch.nn as nn

from cubesight_ml.cube.layout import SOLVED_FLAT
from cubesight_ml.cube.moves import (
    MOVE_TO_INDEX,
    parse_move_penalties,
)
from cubesight_ml.search.beam_search.beam_search import BeamSearch
from cubesight_ml.search.beam_search.utils import BeamSearchRubiksCubeHandler
from cubesight_ml.stages.definitions import PLL_SPEC
from cubesight_ml.stages.goal_checker import StageGoalChecker


class DistanceMockModel(nn.Module):
    """
    Mock heuristic model that computes distance to the goal set by counting mismatched elements.
    """

    def __init__(self, spec) -> None:
        super().__init__()
        self.checker = StageGoalChecker(spec, device="cpu")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        solved = self.checker.solved_flat
        masks = self.checker.masks  # (M, 40)
        x_expanded = x[..., :40].unsqueeze(1)  # (B, 1, 40)
        diffs = (x_expanded != solved) & masks  # (B, M, 40)
        mismatch_counts = diffs.sum(dim=-1)  # (B, M)
        min_mismatches = mismatch_counts.min(dim=-1).values  # (B,)
        return {"value": min_mismatches.unsqueeze(-1).float()}


def test_parse_move_penalties():
    # None defaults to 0.0
    penalties = parse_move_penalties(None)
    assert penalties.shape == (54,)
    assert np.all(penalties == 0.0)

    # Dict input
    d = {"R": 0.5, "B": 2.0, "x": 0.3}
    p = parse_move_penalties(d)
    assert p[MOVE_TO_INDEX["R"]] == pytest.approx(0.5)
    assert p[MOVE_TO_INDEX["B"]] == pytest.approx(2.0)
    assert p[MOVE_TO_INDEX["x"]] == pytest.approx(0.3)
    assert p[MOVE_TO_INDEX["U"]] == 0.0

    # JSON string input
    json_str = '{"U": 1.2, "M": 0.4}'
    p_str = parse_move_penalties(json_str)
    assert p_str[MOVE_TO_INDEX["U"]] == pytest.approx(1.2)
    assert p_str[MOVE_TO_INDEX["M"]] == pytest.approx(0.4)

    # Invalid penalty value raises
    with pytest.raises(ValueError):
        parse_move_penalties({"R": -1.0})

    # Unknown move name raises
    with pytest.raises(KeyError):
        parse_move_penalties({"INVALID_MOVE": 1.0})


def test_beam_search_with_penalties():
    device = torch.device("cpu")
    model = DistanceMockModel(PLL_SPEC)
    checker = StageGoalChecker(PLL_SPEC, device=device)
    handler = BeamSearchRubiksCubeHandler(
        goal_checker=checker,
        use_stm=True,
        use_double_moves=True,
        include_rotations=True,
        device=device,
    )

    # Scramble 1 move away using R
    r_idx = MOVE_TO_INDEX["R"]
    r_prime_idx = MOVE_TO_INDEX["R'"]

    state = torch.tensor(
        SOLVED_FLAT, dtype=torch.long, device=device
    ).unsqueeze(0)
    scrambled = handler.apply_moves(state, torch.tensor([r_idx], device=device))

    # Without penalties, search should find R' (or equivalent 1-move solution)
    searcher = BeamSearch(
        model=model,
        puzzle_handler=handler,
        use_stm=True,
        device=device,
    )
    sol, steps, stats = searcher.find_solution(
        scrambled, beam_width=10, num_steps=5
    )
    assert sol is not None
    assert len(sol) == 1
    assert sol[0] == r_prime_idx
    assert stats["stm_length"] == 1
    assert stats["total_length"] == 1
    assert stats["total_cost"] == pytest.approx(1.0)


def test_beam_search_favors_lower_penalty_moves():
    device = torch.device("cpu")
    model = DistanceMockModel(PLL_SPEC)
    checker = StageGoalChecker(PLL_SPEC, device=device)
    handler = BeamSearchRubiksCubeHandler(
        goal_checker=checker,
        use_stm=True,
        use_double_moves=True,
        include_rotations=True,
        device=device,
    )

    # Suppose state is solved: goal check is 0 steps
    state = torch.tensor(
        SOLVED_FLAT, dtype=torch.long, device=device
    ).unsqueeze(0)
    searcher = BeamSearch(
        model=model,
        puzzle_handler=handler,
        move_penalties={"U": 5.0},
        use_stm=True,
        device=device,
    )
    sol, steps, stats = searcher.find_solution(state, beam_width=5, num_steps=5)
    assert sol == []
    assert steps == 0
    assert stats["total_cost"] == 0.0

    # Check move costs in searcher
    u_idx = MOVE_TO_INDEX["U"]
    assert searcher.move_costs[u_idx].item() == pytest.approx(1.0 + 5.0)

    # Rotations count as moves in Beam Search (base cost 1.0)
    x_idx = MOVE_TO_INDEX["x"]
    assert searcher.move_costs[x_idx].item() == pytest.approx(1.0)
