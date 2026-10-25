import numpy as np
import pytest
import torch

from cubesight_ml.cube.layout import (
    CUBE_ROTATION_TABLES,
    FACE_TO_D_ROTATIONS,
    SOLVED_FLAT,
    apply_cube_rotation,
    to_flat,
)
from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.envs.rubiks_cube import RubiksCubeEnv


def test_noop_move():
    # Test STM default
    env_stm = RubiksCubeEnv(batch_size=4, use_stm=True)
    assert env_stm.NOOP == 54
    assert env_stm.NOOP not in env_stm.get_possible_moves()

    # Stepping with NOOP should not alter the state
    cp_before, co_before, ep_before, eo_before = env_stm.get_state()
    noop_actions = np.full(4, env_stm.NOOP, dtype=np.int32)
    cp_after, co_after, ep_after, eo_after = env_stm.step(noop_actions)

    assert np.array_equal(cp_before, cp_after)
    assert np.array_equal(co_before, co_after)
    assert np.array_equal(ep_before, ep_after)
    assert np.array_equal(eo_before, eo_after)

    # Test HTM mode
    env_htm = RubiksCubeEnv(batch_size=4, use_stm=False, use_double_moves=False)
    assert env_htm.NOOP == 18
    assert env_htm.NOOP not in env_htm.get_possible_moves()


def test_load_state():
    env = RubiksCubeEnv(batch_size=4)
    cp, co, ep, eo = env.scramble(depth=15)

    saved_cp = cp.copy()
    saved_co = co.copy()
    saved_ep = ep.copy()
    saved_eo = eo.copy()

    env.reset()
    assert not np.array_equal(env.cp, saved_cp)

    env.load_state(saved_cp, saved_co, saved_ep, saved_eo)
    assert np.array_equal(env.cp, saved_cp)
    assert np.array_equal(env.co, saved_co)
    assert np.array_equal(env.ep, saved_ep)
    assert np.array_equal(env.eo, saved_eo)

    # Verify modifying input does not mutate internal buffer (no shared reference)
    saved_cp[0, 0] = 99
    assert env.cp[0, 0] != 99


def test_scramble_per_sample_depths():
    batch_size = 4
    env = RubiksCubeEnv(batch_size=batch_size, use_double_moves=False)
    depths = np.array([0, 1, 5, 10], dtype=np.int32)

    cp, co, ep, eo = env.scramble(depth=10, depths=depths)

    # Cube 0 had depth 0, so it must remain strictly solved
    flat = to_flat(cp, co, ep, eo)
    assert np.array_equal(flat[0], SOLVED_FLAT)

    # Other cubes should be valid cube states
    for i in range(1, batch_size):
        assert np.sum(co[i]) % 3 == 0
        assert np.sum(eo[i]) % 2 == 0


@pytest.mark.parametrize(
    ("use_stm", "use_double_moves", "expected_moves"),
    [
        (False, False, 12),
        (False, True, 18),
        (True, False, 36),
        (True, True, 54),
    ],
)
def test_torch_ops_expand_matches_numpy(
    use_stm: bool, use_double_moves: bool, expected_moves: int
):
    batch_size = 6
    env = RubiksCubeEnv(
        batch_size=batch_size,
        use_stm=use_stm,
        use_double_moves=use_double_moves,
    )
    cp, co, ep, eo = env.scramble(depth=12)

    # NumPy child generation
    np_child_cp, np_child_co, np_child_ep, np_child_eo = (
        env.get_all_child_states()
    )
    np_children = np.concatenate(
        [np_child_cp, np_child_co, np_child_ep, np_child_eo], axis=-1
    )

    # PyTorch child generation
    parents_flat = to_flat(cp, co, ep, eo)
    parents_tensor = torch.tensor(parents_flat, dtype=torch.long)

    ops = TorchCubeOps(
        use_stm=use_stm, use_double_moves=use_double_moves, device="cpu"
    )
    torch_children = ops.expand(parents_tensor)

    assert torch_children.shape == (batch_size, expected_moves, 40)
    assert np.array_equal(torch_children.numpy(), np_children)


def test_torch_ops_apply_moves():
    batch_size = 8
    env = RubiksCubeEnv(batch_size=batch_size, use_double_moves=False)
    cp, co, ep, eo = env.scramble(depth=10)
    parents_flat = to_flat(cp.copy(), co.copy(), ep.copy(), eo.copy())

    actions = np.array([0, 2, 3, 5, 6, 8, 9, 11], dtype=np.int32)
    next_cp, next_co, next_ep, next_eo = env.step(actions)
    expected_flat = to_flat(next_cp, next_co, next_ep, next_eo)

    ops = TorchCubeOps(use_double_moves=False, device="cpu")
    parents_tensor = torch.tensor(parents_flat, dtype=torch.long)
    actions_tensor = torch.tensor(actions, dtype=torch.long)

    actual_tensor = ops.apply_moves(parents_tensor, actions_tensor)
    assert np.array_equal(actual_tensor.numpy(), expected_flat)


def test_cube_rotations_invariants():
    state = SOLVED_FLAT.copy()

    # Identity rotation preserves state
    assert np.array_equal(apply_cube_rotation(state, "id"), state)

    # 4 applications of x rotation = identity
    s = state.copy()
    for _ in range(4):
        s = apply_cube_rotation(s, "x")
    assert np.array_equal(s, state)

    # 4 applications of z rotation = identity
    s = state.copy()
    for _ in range(4):
        s = apply_cube_rotation(s, "z")
    assert np.array_equal(s, state)

    # x followed by x' = identity
    s_x = apply_cube_rotation(state, "x")
    s_back = apply_cube_rotation(s_x, "x_prime")
    assert np.array_equal(s_back, state)

    # z followed by z' = identity
    s_z = apply_cube_rotation(state, "z")
    s_back_z = apply_cube_rotation(s_z, "z_prime")
    assert np.array_equal(s_back_z, state)

    # Test PyTorch tensor support
    state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(
        0
    )  # shape (1, 40)
    t_rot = apply_cube_rotation(state_tensor, "x")
    t_back = apply_cube_rotation(t_rot, "x_prime")
    assert torch.equal(t_back, state_tensor)

    # All face_to_d rotations map to defined rotation tables
    for _face, rot_name in FACE_TO_D_ROTATIONS.items():
        assert rot_name in CUBE_ROTATION_TABLES
