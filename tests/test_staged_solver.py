import torch
import torch.nn as nn

from cubesight_ml.cube.layout import (
    SOLVED_FLAT,
    apply_cube_rotation,
)
from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.search.staged_solver import StagedSolver
from cubesight_ml.stages.definitions import STAGES, Stage
from cubesight_ml.stages.goal_checker import (
    StageGoalChecker,
    detect_solved_cross_face,
)


class StageDistanceMock(nn.Module):
    """Counts mismatched elements under stage goal masks."""

    def __init__(self, spec) -> None:
        super().__init__()
        self.checker = StageGoalChecker(spec, device="cpu")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        solved = self.checker.solved_flat
        masks = self.checker.masks
        x_expanded = x.unsqueeze(1)
        diffs = (x_expanded != solved) & masks
        mismatch_counts = diffs.sum(dim=-1)
        min_mismatches = mismatch_counts.min(dim=-1).values
        return {"value": min_mismatches.unsqueeze(-1).float()}


def test_staged_solver_on_solved_cube():
    """On an already solved cube, StagedSolver completes with 0 moves."""
    models = {st: StageDistanceMock(STAGES[st]) for st in Stage}
    solver = StagedSolver(
        models=models, beam_width=10, max_steps=5, device="cpu"
    )

    result = solver.solve(SOLVED_FLAT)
    assert result.solved is True
    assert result.moves == []
    assert result.failed_stage is None


def test_staged_solver_with_cross_reorientation(monkeypatch):
    """
    Test when Cross is solved on face F:
    Solver should detect F, reorient to D via x_prime, then solve F2L/OLL/PLL.
    """
    models = {st: StageDistanceMock(STAGES[st]) for st in Stage}
    solver = StagedSolver(
        models=models, beam_width=15, max_steps=10, device="cpu"
    )

    # State with F-cross solved, others broken
    from cubesight_ml.cube.layout import EO

    f_cross_cube = apply_cube_rotation(SOLVED_FLAT, "x")
    for e in [1, 2, 3, 5, 6, 7]:
        f_cross_cube[EO.start + e] ^= 1
    assert detect_solved_cross_face(f_cross_cube) == "F"

    monkeypatch.setattr(
        solver.searchers[Stage.CROSS],
        "find_solution",
        lambda s, **kw: ([], 0, {}),
    )
    monkeypatch.setattr(
        solver.searchers[Stage.F2L],
        "find_solution",
        lambda s, **kw: ([], 0, {}),
    )
    monkeypatch.setattr(
        solver.searchers[Stage.OLL],
        "find_solution",
        lambda s, **kw: ([], 0, {}),
    )
    monkeypatch.setattr(
        solver.searchers[Stage.PLL],
        "find_solution",
        lambda s, **kw: ([], 0, {}),
    )
    monkeypatch.setattr(
        "cubesight_ml.search.staged_solver.is_goal_np", lambda s, spec: True
    )

    result = solver.solve(f_cross_cube)
    assert result.solved is True
    assert "x_prime" in result.moves
    assert result.failed_stage is None


def test_staged_solver_failure_handling():
    """If a stage cannot find a solution within max_steps, reports failed_stage."""
    ops = TorchCubeOps(use_double_moves=False, device="cpu")

    # Scramble PLL only using U moves (preserves Cross, F2L, and OLL)
    solved_t = torch.tensor(SOLVED_FLAT, dtype=torch.long).unsqueeze(0)
    s = solved_t
    for m in [0, 1]:  # U, U2 (net: U_PRIME)
        s = ops.apply_moves(s, torch.tensor([m], dtype=torch.long))
    scrambled = s.squeeze(0).numpy()

    # Model with zero max_steps for PLL so it fails
    models = {st: StageDistanceMock(STAGES[st]) for st in Stage}
    solver = StagedSolver(
        models=models,
        beam_width=5,
        max_steps={Stage.CROSS: 10, Stage.F2L: 10, Stage.OLL: 10, Stage.PLL: 0},
        device="cpu",
    )

    result = solver.solve(scrambled)
    assert result.solved is False
    assert result.failed_stage == Stage.PLL
