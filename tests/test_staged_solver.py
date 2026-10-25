import numpy as np
import torch
import torch.nn as nn

from cubesight_ml.cube.layout import (
    EO,
    SOLVED_FLAT,
    apply_cube_rotation,
)
from cubesight_ml.cube.moves import (
    BASE_MOVE_COSTS_STM,
    CENTER_RESTORE_MAP,
    MOVE_CENTERS_54,
    MOVE_TO_INDEX,
    ROTATION_MOVES,
    get_restore_rotations,
)
from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.search.staged_solver import StagedSolver
from cubesight_ml.stages.definitions import STAGES, Stage
from cubesight_ml.stages.goal_checker import (
    StageGoalChecker,
    detect_solved_cross_face,
    is_goal_np,
)


class StageDistanceMock(nn.Module):
    """Counts mismatched elements under stage goal masks."""

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


def test_get_restore_rotations_all_24_orientations():
    """Verify that get_restore_rotations restores all 24 reachable center orientations."""
    canonical = (0, 1, 2, 3, 4, 5)
    assert len(CENTER_RESTORE_MAP) == 24

    for target_center in CENTER_RESTORE_MAP:
        rot_moves = get_restore_rotations(target_center)
        curr = np.array(target_center, dtype=np.int8)
        for m in rot_moves:
            curr = curr[MOVE_CENTERS_54[MOVE_TO_INDEX[m]]]
        assert tuple(curr) == canonical


def test_staged_solver_with_non_canonical_centers():
    """Verify StagedSolver automatically reorients cubes with rotated centers."""
    models = {st: StageDistanceMock(STAGES[st]) for st in Stage}
    solver = StagedSolver(
        models=models, beam_width=10, max_steps=5, device="cpu"
    )

    # Rotate solved cube with 'x' (center U becomes Green 2, etc.)
    x_idx = MOVE_TO_INDEX["x"]
    x_prime_idx = MOVE_TO_INDEX["x'"]
    rotated_cube = (
        solver.cube_ops.apply_moves(
            torch.tensor(SOLVED_FLAT, dtype=torch.long).unsqueeze(0),
            torch.tensor([x_idx], dtype=torch.long),
        )
        .squeeze(0)
        .numpy()
    )

    rotated_centers = np.arange(6, dtype=np.int8)[MOVE_CENTERS_54[x_idx]]

    result = solver.solve(rotated_cube, centers=rotated_centers)
    assert result.solved is True
    # The first move should be the restoring rotation x'
    assert result.moves[0] == x_prime_idx


def test_full_stm_rotations_counted_and_executed():
    """Verify that full STM is used, rotations count as 1.0, and centers match."""
    # 1. Rotations must count as 1.0 move in BASE_MOVE_COSTS_STM
    for rot in ROTATION_MOVES:
        idx = MOVE_TO_INDEX[rot]
        assert BASE_MOVE_COSTS_STM[idx] == 1.0

    # 2. Solver configured with full STM and rotations
    models = {st: StageDistanceMock(STAGES[st]) for st in Stage}
    solver = StagedSolver(
        models=models,
        beam_width=10,
        max_steps=5,
        use_stm=True,
        use_double_moves=True,
        include_rotations=True,
        device="cpu",
    )

    # 3. All 54 moves must be present in searchers
    assert solver.cube_ops.num_moves == 54
    for st in Stage:
        assert solver.searchers[st].n_moves == 54
        # Verify rotation move cost in searcher is 1.0
        x_idx = MOVE_TO_INDEX["x"]
        assert solver.searchers[st].move_costs[x_idx].item() == 1.0

    # 4. Solves with full STM
    scramble_ops = TorchCubeOps(use_stm=True, device="cpu")
    solved_46 = torch.tensor(
        [*list(SOLVED_FLAT), 0, 1, 2, 3, 4, 5], dtype=torch.long
    ).unsqueeze(0)
    # Scramble with a rotation move 'y' and face move 'U'
    moves = torch.tensor(
        [MOVE_TO_INDEX["y"], MOVE_TO_INDEX["U"]], dtype=torch.long
    )
    curr = solved_46
    for m in moves:
        curr = scramble_ops.apply_moves(curr, m.unsqueeze(0))

    scrambled_46 = curr.squeeze(0).numpy()
    res = solver.solve(scrambled_46)
    assert res.solved is True
    assert is_goal_np(res.final_state, STAGES[Stage.PLL])
