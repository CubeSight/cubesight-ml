import numpy as np
import torch
import torch.nn as nn

from cubesight_ml.cube.layout import SOLVED_FLAT
from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.stages.definitions import CROSS_SPEC, PLL_SPEC
from cubesight_ml.stages.goal_checker import StageGoalChecker
from cubesight_ml.training.davi_targets import compute_davi_targets


class DummyValueModel(nn.Module):
    """Mock model that returns a constant value."""

    def __init__(self, constant_val: float) -> None:
        super().__init__()
        self.val = constant_val

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "value": torch.full((x.shape[0], 1), self.val, dtype=torch.float32)
        }


def test_davi_targets_goal_parent_masking():
    """Goal parents must receive target 0.0 regardless of target model value."""
    ops = TorchCubeOps(use_double_moves=False, device="cpu")
    checker = StageGoalChecker(PLL_SPEC, device="cpu")
    model = DummyValueModel(constant_val=5.0)

    # State 0 is solved (goal), State 1 is not solved
    parents = torch.tensor(
        np.stack([SOLVED_FLAT, SOLVED_FLAT + 1]),
        dtype=torch.long,
    )

    targets = compute_davi_targets(
        target_model=model,
        parents=parents,
        cube_ops=ops,
        goal_checker=checker,
        chunk_size=100,
    )

    # Goal parent must be 0.0
    assert targets[0].item() == 0.0
    # Non-goal parent must be 1.0 + min_a(5.0) = 6.0
    assert targets[1].item() == 6.0


def test_davi_targets_goal_children_masking():
    """If child of a parent is in goal set, its value is masked to 0.0,

    so parent target should be 1.0 + 0.0 = 1.0 even if model predicts 99.0.
    """
    ops = TorchCubeOps(use_double_moves=False, device="cpu")
    checker = StageGoalChecker(PLL_SPEC, device="cpu")
    model = DummyValueModel(constant_val=99.0)

    # Create a 1-move-away state from solved (e.g. U move applied to solved)
    solved_t = torch.tensor(SOLVED_FLAT, dtype=torch.long).unsqueeze(0)
    # Apply U move (action 0)
    move_u = torch.tensor([0], dtype=torch.long)
    state_depth_1 = ops.apply_moves(solved_t, move_u)

    # This state is not solved itself
    assert not checker.is_goal(state_depth_1).item()

    targets = compute_davi_targets(
        target_model=model,
        parents=state_depth_1,
        cube_ops=ops,
        goal_checker=checker,
        chunk_size=100,
    )

    # Because U' is a valid move that returns to solved, one child is solved!
    # With goal children masking, best child value is 0.0.
    # So target is 1.0 + 0.0 = 1.0!
    assert targets[0].item() == 1.0


def test_davi_targets_clamp_min():
    """Verify clamp_min is respected when target network outputs negatives."""
    ops = TorchCubeOps(use_double_moves=False, device="cpu")
    checker = StageGoalChecker(CROSS_SPEC, device="cpu")
    model = DummyValueModel(constant_val=-10.0)

    # Random non-goal state
    parents = torch.randint(10, 20, size=(4, 40), dtype=torch.long)

    targets = compute_davi_targets(
        target_model=model,
        parents=parents,
        cube_ops=ops,
        goal_checker=checker,
        clamp_min=0.0,
    )

    # 1.0 + (-10.0) = -9.0 clamped to 0.0
    assert (targets == 0.0).all()


def test_davi_targets_with_action_move_costs():
    """Verify compute_davi_targets respects custom action move costs."""
    ops = TorchCubeOps(use_double_moves=False, device="cpu")
    checker = StageGoalChecker(PLL_SPEC, device="cpu")
    model = DummyValueModel(constant_val=2.0)

    # Non-goal state
    parents = torch.tensor(
        np.stack([SOLVED_FLAT + 1]),
        dtype=torch.long,
    )

    # Custom costs of shape (A,)
    action_costs = torch.full((ops.num_moves,), 5.0, dtype=torch.float32)
    action_costs[3] = 2.5  # Cheapest action has cost 2.5

    targets = compute_davi_targets(
        target_model=model,
        parents=parents,
        cube_ops=ops,
        goal_checker=checker,
        move_costs=action_costs,
    )

    # Target should be min_a [ c(a) + J(s') ] = 2.5 + 2.0 = 4.5
    assert torch.isclose(targets[0], torch.tensor(4.5))


def test_davi_targets_with_54_move_costs_and_goal():
    """Verify 54-move costs tensor indexing and goal masking interaction."""
    ops = TorchCubeOps(use_double_moves=False, device="cpu")
    checker = StageGoalChecker(PLL_SPEC, device="cpu")
    model = DummyValueModel(constant_val=10.0)

    # Create 1-move-away state from solved (apply U move)
    solved_t = torch.tensor(SOLVED_FLAT, dtype=torch.long).unsqueeze(0)
    move_u = torch.tensor([0], dtype=torch.long)
    state_depth_1 = ops.apply_moves(solved_t, move_u)

    # Move costs for all 54 moves: base 1.0, but U' (index 2) has cost 3.5
    move_costs_54 = torch.ones(54, dtype=torch.float32)
    move_costs_54[2] = 3.5

    targets = compute_davi_targets(
        target_model=model,
        parents=state_depth_1,
        cube_ops=ops,
        goal_checker=checker,
        move_costs=move_costs_54,
    )

    # For U', child is solved (J=0), so candidate cost is 3.5 + 0.0 = 3.5.
    # For other moves, candidate cost is 1.0 + 10.0 = 11.0.
    # Minimum is 3.5!
    assert torch.isclose(targets[0], torch.tensor(3.5))
