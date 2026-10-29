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
from cubesight_ml.search.staged_solver import (
    StagedSolver,
    apply_action_sequence,
)
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
        masks = self.checker.masks.to(x.device)
        if x.size(-1) >= 46:
            c = x[..., 40:46].long()
            h = (c * self.checker.weights.to(x.device)).sum(dim=-1)
            c_ids = self.checker.lut.to(x.device)[h]
            target = self.checker.solved_table.to(x.device)[c_ids]
            x_pieces = x[..., :40]
        else:
            target = self.checker.solved_flat.to(x.device)
            x_pieces = x

        x_expanded = x_pieces.unsqueeze(-2)
        target_expanded = (
            target.unsqueeze(-2) if target.ndim == x_pieces.ndim else target
        )
        diffs = (x_expanded != target_expanded) & masks
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
    Solver should inspect all 24 orientations, pick the orientation that places F on D (x'),
    apply x' for free, then solve F2L/OLL/PLL locked to that frame of reference.
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
        "cubesight_ml.search.staged_solver.is_goal_np",
        lambda s, spec, **kw: True,
    )

    result = solver.solve(f_cross_cube)
    assert result.solved is True
    assert (
        "x'" in result.inspection_moves
        or MOVE_TO_INDEX["x'"] in result.moves
        or "x_prime" in result.moves
    )
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


def test_staged_solver_configurable_batch_size():
    models = {st: StageDistanceMock(STAGES[st]) for st in Stage}

    # 1. Default batch size is 512
    default_solver = StagedSolver(models=models, device="cpu")
    for st in Stage:
        assert default_solver.batch_sizes[st] == 512
        assert default_solver.searchers[st].batch_size == 512

    # 2. Integer batch size
    int_solver = StagedSolver(models=models, batch_size=256, device="cpu")
    for st in Stage:
        assert int_solver.batch_sizes[st] == 256
        assert int_solver.searchers[st].batch_size == 256

    # 3. Dict of batch sizes per stage
    custom_bs = {
        Stage.CROSS: 64,
        Stage.F2L: 128,
        Stage.OLL: 256,
        Stage.PLL: 1024,
    }
    dict_solver = StagedSolver(
        models=models, batch_size=custom_bs, device="cpu"
    )
    for st, expected_bs in custom_bs.items():
        assert dict_solver.batch_sizes[st] == expected_bs
        assert dict_solver.searchers[st].batch_size == expected_bs


def test_cross_inspection_orientation_selection():
    """
    Verify that inspection evaluates all 24 orientations and selects the easiest cross,
    applying the inspection rotation for free (0 cost) and recording it in inspection_moves.
    """
    ops = TorchCubeOps(use_stm=True, device="cpu")
    solved_46 = np.concatenate([SOLVED_FLAT, np.arange(6, dtype=np.int8)])
    # Scramble D layer with 'D': breaks D cross, leaves U cross completely solved
    d_scrambled = apply_action_sequence(solved_46, [MOVE_TO_INDEX["D"]], ops)

    models = {st: StageDistanceMock(STAGES[st]) for st in Stage}
    solver = StagedSolver(
        models=models, beam_width=10, max_steps=10, device="cpu"
    )

    result = solver.solve(d_scrambled)
    assert result.solved is True
    # Inspection should choose x2 to put the solved U cross onto D
    assert result.inspection_moves == ["x2"]
    assert result.moves[0] == MOVE_TO_INDEX["x2"]
    # Inspection rotation cost should be 0.0 in stage_costs
    assert result.stage_costs[Stage.CROSS] == 0.0
    assert result.stage_lengths[Stage.CROSS] == 0
    # Final state must be fully solved in the chosen orientation frame
    assert result.final_state is not None
    assert result.chosen_color == 0
    assert is_goal_np(
        result.final_state, STAGES[Stage.PLL], target_color=result.chosen_color
    )


def test_all_stages_locked_to_inspection_frame():
    """
    Verify that once inspection picks an orientation, all 4 goal checkers
    have target_color set to the chosen reference color.
    """
    ops = TorchCubeOps(use_stm=True, device="cpu")
    solved_46 = np.concatenate([SOLVED_FLAT, np.arange(6, dtype=np.int8)])
    # Scramble F layer with 'F': breaks F cross, leaves B cross completely solved
    f_scrambled = apply_action_sequence(solved_46, [MOVE_TO_INDEX["F"]], ops)

    models = {st: StageDistanceMock(STAGES[st]) for st in Stage}
    solver = StagedSolver(
        models=models, beam_width=10, max_steps=10, device="cpu"
    )

    result = solver.solve(f_scrambled)
    assert result.solved is True
    # Inspection should rotate B to D via x
    assert result.inspection_moves == ["x"]
    assert result.moves[0] == MOVE_TO_INDEX["x"]
    # In canonical cube, B is face 5 (Blue). Rotated to D via x, chosen color is 5.
    expected_chosen_color = 5
    assert result.chosen_color == expected_chosen_color

    for st in Stage:
        assert (
            solver.searchers[st].puzzle_handler.goal_checker.target_color
            == expected_chosen_color
        )
    assert is_goal_np(
        result.final_state,
        STAGES[Stage.PLL],
        target_color=expected_chosen_color,
    )


def test_color_locked_stages_rotation_invariance():
    """
    Verify that when a stage goal checker is locked to target_color,
    the goal condition is satisfied regardless of which face target_color physically lands on
    due to mid-solve rotations (x, y, z, etc.).
    """
    ops = TorchCubeOps(use_stm=True, device="cpu")
    canonical_centers = np.arange(6, dtype=np.int8)
    solved_46 = np.concatenate([SOLVED_FLAT, canonical_centers])

    # Put White (color 0) on D via x2
    white_on_d = apply_action_sequence(solved_46, [MOVE_TO_INDEX["x2"]], ops)
    assert white_on_d[43] == 0

    # Scramble the last layer (Yellow is on U, face 0) with a U turn
    # This leaves Cross and F2L on White completely solved!
    f2l_solved_on_white = apply_action_sequence(
        white_on_d, [MOVE_TO_INDEX["U"]], ops
    )

    f2l_checker = StageGoalChecker(STAGES[Stage.F2L])
    f2l_checker.set_target_color(0)

    # 1. When White is on D:
    assert (
        f2l_checker.is_goal(
            torch.tensor(f2l_solved_on_white).unsqueeze(0)
        ).item()
        is True
    )
    assert (
        is_goal_np(
            f2l_solved_on_white, STAGES[Stage.F2L], target_color=0
        ).item()
        is True
    )

    # 2. When White is rotated to other faces via cube rotations:
    for rot in ["x", "x'", "x2", "y", "y'", "y2", "z", "z'", "z2"]:
        rot_idx = MOVE_TO_INDEX[rot]
        rotated = apply_action_sequence(f2l_solved_on_white, [rot_idx], ops)
        rot_t = torch.tensor(rotated).unsqueeze(0)
        # F2L on White must still be recognized as solved!
        assert f2l_checker.is_goal(rot_t).item() is True, (
            f"Failed on rotation {rot}"
        )
        assert (
            is_goal_np(rotated, STAGES[Stage.F2L], target_color=0).item()
            is True
        )

    # 3. But a turn affecting the White layer (e.g. D turn when White is on D) must break F2L
    broken = apply_action_sequence(
        f2l_solved_on_white, [MOVE_TO_INDEX["D"]], ops
    )
    assert (
        f2l_checker.is_goal(torch.tensor(broken).unsqueeze(0)).item() is False
    )
    assert is_goal_np(broken, STAGES[Stage.F2L], target_color=0).item() is False
