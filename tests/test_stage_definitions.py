import numpy as np
import pytest
import torch

from cubesight_ml.cube.layout import (
    CP,
    EP,
    FACE_TO_D_ROTATIONS,
    SOLVED_FLAT,
    STATE_SIZE,
    apply_cube_rotation,
    from_flat,
)
from cubesight_ml.cube.moves import (
    CENTER_FORWARD_MAP,
    CENTER_RESTORE_MAP,
    MOVE_TO_INDEX,
    ROTATION_MOVES,
)
from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.stages.definitions import (
    CROSS_FACE_EDGES,
    CROSS_FACE_MASKS,
    CROSS_SPEC,
    F2L_SPEC,
    OLL_SPEC,
    PLL_SPEC,
    STAGES,
)
from cubesight_ml.stages.goal_checker import (
    StageGoalChecker,
    detect_solved_cross_face,
    is_goal_np,
)
from cubesight_ml.stages.goal_sampler import sample_goal_states
from cubesight_ml.visualization.cube_state import (
    state_to_stickers,
    stickers_to_face_dict,
)


def test_stage_hierarchy():
    """Verify goal(PLL) ⊂ goal(OLL) ⊂ goal(F2L) ⊂ goal(CROSS_D) ⊂ goal(CROSS)."""
    solved = SOLVED_FLAT.copy()

    # Solved cube satisfies all goals
    assert is_goal_np(solved, PLL_SPEC)
    assert is_goal_np(solved, OLL_SPEC)
    assert is_goal_np(solved, F2L_SPEC)
    assert is_goal_np(solved, CROSS_SPEC)

    # State with OLL satisfied but PLL not (swap U-corners 0 and 1, and U-edges 0 and 1 for parity)
    oll_state = solved.copy()
    oll_state[CP.start], oll_state[CP.start + 1] = (
        oll_state[CP.start + 1],
        oll_state[CP.start],
    )
    oll_state[EP.start], oll_state[EP.start + 1] = (
        oll_state[EP.start + 1],
        oll_state[EP.start],
    )

    assert not is_goal_np(oll_state, PLL_SPEC)
    assert is_goal_np(oll_state, OLL_SPEC)
    assert is_goal_np(oll_state, F2L_SPEC)
    assert is_goal_np(oll_state, CROSS_SPEC)

    # State with F2L satisfied but OLL not (twist U-corners 0 and 1)
    f2l_state = solved.copy()
    f2l_state[8] = 1  # co[0] = 1
    f2l_state[9] = 2  # co[1] = 2 (sum = 3 = 0 mod 3)

    assert not is_goal_np(f2l_state, PLL_SPEC)
    assert not is_goal_np(f2l_state, OLL_SPEC)
    assert is_goal_np(f2l_state, F2L_SPEC)
    assert is_goal_np(f2l_state, CROSS_SPEC)

    # State with only D-cross satisfied (break F2L corner 4 and edge 8)
    cross_state = solved.copy()
    cross_state[CP.start + 4] = 0  # break corner 4
    cross_state[EP.start + 8] = 0  # break edge 8

    assert not is_goal_np(cross_state, PLL_SPEC)
    assert not is_goal_np(cross_state, OLL_SPEC)
    assert not is_goal_np(cross_state, F2L_SPEC)
    assert is_goal_np(cross_state, CROSS_SPEC)


def test_cross_faces_and_detection():
    """Verify that each of the 6 cross faces is recognized correctly."""
    solved = SOLVED_FLAT.copy()

    # On completely solved cube, detect_solved_cross_face prefers D
    assert detect_solved_cross_face(solved) == "D"

    rng = np.random.default_rng(42)

    # For each face, create a state where that face's cross is solved
    for face in CROSS_FACE_EDGES:
        state = sample_goal_states(
            CROSS_SPEC, 1, rng=rng, fixed_cross_face=face
        )[0]

        # Verify detect_solved_cross_face returns this face
        detected = detect_solved_cross_face(state)
        assert detected == face

        # Reorienting this state using FACE_TO_D_ROTATIONS should make D-cross solved
        rot = FACE_TO_D_ROTATIONS[face]
        reoriented = apply_cube_rotation(state[:40], rot)
        assert ((reoriented == SOLVED_FLAT) | ~CROSS_FACE_MASKS["D"]).all()


def test_unsolved_cross_detection_raises():
    """Verify ValueError is raised if no cross is solved."""
    scrambled = np.full(STATE_SIZE, 99, dtype=np.int8)
    with pytest.raises(ValueError, match="No cross face is solved"):
        detect_solved_cross_face(scrambled)


def test_stage_goal_checker_pytorch():
    """Verify StageGoalChecker matches is_goal_np across batch shapes."""
    rng = np.random.default_rng(42)

    for _stage_enum, spec in STAGES.items():
        checker = StageGoalChecker(spec, device="cpu")

        # Test on solved
        solved_t = torch.tensor(SOLVED_FLAT, dtype=torch.long)
        assert checker.is_goal(solved_t).item() is True

        # Test batched (B, 40)
        batch_np = rng.integers(0, 10, size=(10, STATE_SIZE), dtype=np.int8)
        batch_np[0] = SOLVED_FLAT  # at least one solved
        batch_t = torch.tensor(batch_np, dtype=torch.long)

        np_res = is_goal_np(batch_np, spec)
        torch_res = checker.is_goal(batch_t).numpy()
        np.testing.assert_array_equal(np_res, torch_res)

        # Test (B, A, 40)
        batch_3d_np = rng.integers(
            0, 10, size=(4, 6, STATE_SIZE), dtype=np.int8
        )
        batch_3d_np[1, 2] = SOLVED_FLAT
        batch_3d_t = torch.tensor(batch_3d_np, dtype=torch.long)

        np_3d_res = is_goal_np(batch_3d_np, spec)
        torch_3d_res = checker.is_goal(batch_3d_t).numpy()
        np.testing.assert_array_equal(np_3d_res, torch_3d_res)


def test_cross_on_d_and_rotations():
    """Verify that CROSS_SPEC requires cross on D and respects rotations."""
    checker = StageGoalChecker(CROSS_SPEC, device="cpu")
    ops = TorchCubeOps(use_stm=True, include_rotations=True, device="cpu")

    # 46-element solved state
    state_46 = torch.tensor(
        np.concatenate([SOLVED_FLAT, np.arange(6, dtype=np.int8)]),
        dtype=torch.long,
    ).unsqueeze(0)

    # 1. Solved state satisfies D-cross
    assert checker.is_goal(state_46).item() is True

    # 2. All 24 whole-cube rotations of solved state remain valid D-crosses
    for rot in ROTATION_MOVES:
        m = torch.tensor([MOVE_TO_INDEX[rot]], dtype=torch.long)
        rot_state = ops.apply_moves(state_46, m)
        assert checker.is_goal(rot_state).item() is True, (
            f"Failed for rotation {rot}"
        )

    # 3. State with cross on U but broken on D must NOT satisfy CROSS_SPEC
    u_cross_state = state_46.clone()
    u_cross_state[0, EP.start + 4] = 9  # break DF edge on D
    assert checker.is_goal(u_cross_state).item() is False

    # 4. Rotating u_cross_state via x2 places the solved U-cross on D -> True
    m_x2 = torch.tensor([MOVE_TO_INDEX["x2"]], dtype=torch.long)
    rotated_to_d = ops.apply_moves(u_cross_state, m_x2)
    assert checker.is_goal(rotated_to_d).item() is True


def test_wide_d_misaligned_cross_rejected():
    """Verify a wide d move followed by D' (edges solved relative to old centers, misaligned by 90 deg) is rejected."""
    ops = TorchCubeOps(use_stm=True, include_rotations=True, device="cpu")
    state_46 = torch.tensor(
        np.concatenate([SOLVED_FLAT, np.arange(6, dtype=np.int8)]),
        dtype=torch.long,
    ).unsqueeze(0)

    # Apply 'd' (rotates D and E layers) then "D'" (puts Yellow D-edges back into canonical slots 4..7,
    # while F/R/B/L centers remain rotated by 90 degrees -> cross is misaligned with centers!)
    after_d = ops.apply_moves(
        state_46, torch.tensor([MOVE_TO_INDEX["d"]], dtype=torch.long)
    )
    misaligned = ops.apply_moves(
        after_d, torch.tensor([MOVE_TO_INDEX["D'"]], dtype=torch.long)
    )

    # Must be False under all StageGoalChecker modes (default, target_color=3, target_orientation=0)
    chk_default = StageGoalChecker(CROSS_SPEC, device="cpu")
    assert chk_default.is_goal(misaligned).item() is False
    assert is_goal_np(misaligned.numpy(), CROSS_SPEC).item() is False

    chk_color = StageGoalChecker(CROSS_SPEC, device="cpu")
    chk_color.set_target_color(3)
    assert chk_color.is_goal(misaligned).item() is False
    assert (
        is_goal_np(misaligned.numpy(), CROSS_SPEC, target_color=3).item()
        is False
    )

    chk_orient = StageGoalChecker(CROSS_SPEC, device="cpu")
    chk_orient.set_target_orientation(0)
    assert chk_orient.is_goal(misaligned).item() is False
    assert (
        is_goal_np(misaligned.numpy(), CROSS_SPEC, target_orientation=0).item()
        is False
    )


def test_oll_goal_checker_and_sampler_all_24_orientations():
    """Verify OLL_SPEC goal checking and sampling across all 24 cube orientations."""
    ops = TorchCubeOps(use_stm=True, include_rotations=True, device="cpu")
    oll_checker = StageGoalChecker(OLL_SPEC, device="cpu")
    solved_46 = torch.tensor(
        np.concatenate([SOLVED_FLAT, np.arange(6, dtype=np.int8)]),
        dtype=torch.long,
    ).unsqueeze(0)

    # 1. For every one of the 24 orientations, rotating a solved cube and applying U / U2 / U'
    # preserves OLL and must be recognized as an OLL goal!
    for c in CENTER_RESTORE_MAP:
        curr = solved_46.clone()
        for r_name in CENTER_FORWARD_MAP[c]:
            curr = ops.apply_moves(
                curr, torch.tensor([MOVE_TO_INDEX[r_name]], dtype=torch.long)
            )
        for u_move in ["U", "U2", "U'"]:
            u_state = ops.apply_moves(
                curr, torch.tensor([MOVE_TO_INDEX[u_move]], dtype=torch.long)
            )
            assert oll_checker.is_goal(u_state).item() is True, (
                f"OLL goal check failed for orientation {c} + {u_move}"
            )
            assert is_goal_np(u_state.numpy(), OLL_SPEC).item() is True

    # 2. Sampled OLL goal states across all 24 orientations must have a uniform U face
    rng = np.random.default_rng(123)
    sampled_oll = sample_goal_states(OLL_SPEC, 100, rng=rng)
    assert oll_checker.is_goal(torch.tensor(sampled_oll)).all().item() is True
    assert is_goal_np(sampled_oll, OLL_SPEC).all()

    for i in range(100):
        s = sampled_oll[i]
        cp, co, ep, eo = from_flat(s[:40])
        stk = state_to_stickers(cp, co, ep, eo, s[40:46])
        u_face = stickers_to_face_dict(stk)["U"]
        assert len(set(u_face)) == 1, (
            f"Sampled OLL state {i} does not have uniform U face: {u_face}"
        )
