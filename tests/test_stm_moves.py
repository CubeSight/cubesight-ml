import numpy as np
import torch

from cubesight_ml.cube.moves import (
    ALL_MOVES,
    BASE_MOVE_COSTS_STM,
    INDEX_TO_MOVE,
    INVERSE_MOVES_54,
    MOVE_TO_INDEX,
    NOOP_INDEX,
    ROTATION_MOVES,
    SLICE_MOVES,
    STICKER_MOVE_PERMS_54,
    WIDE_MOVES,
)
from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.envs.rubiks_cube import RubiksCubeEnv


def test_stm_move_definitions():
    assert len(ALL_MOVES) == 54
    assert len(MOVE_TO_INDEX) == 55  # 54 + NOOP
    assert len(INDEX_TO_MOVE) == 55
    assert NOOP_INDEX == 54

    # Verify slices, wide, rotations count
    assert len(SLICE_MOVES) == 9
    assert len(WIDE_MOVES) == 18
    assert len(ROTATION_MOVES) == 9

    # Verify base costs: rotations cost 0.0, turns cost 1.0
    for name in ROTATION_MOVES:
        idx = MOVE_TO_INDEX[name]
        assert BASE_MOVE_COSTS_STM[idx] == 0.0

    for name in ALL_MOVES:
        if name not in ROTATION_MOVES:
            idx = MOVE_TO_INDEX[name]
            assert BASE_MOVE_COSTS_STM[idx] == 1.0


def test_stm_inverses():
    # Every move composed with its inverse must equal identity
    perms = STICKER_MOVE_PERMS_54
    identity = np.arange(54, dtype=np.int32)

    for i in range(54):
        inv_i = INVERSE_MOVES_54[i]
        composed = perms[i][perms[inv_i]]
        assert np.array_equal(composed, identity), (
            f"Inverse failed for move {INDEX_TO_MOVE[i]}"
        )

        # Inverse of inverse is self
        assert INVERSE_MOVES_54[inv_i] == i


def test_stm_cycle_orders():
    perms = STICKER_MOVE_PERMS_54
    identity = np.arange(54, dtype=np.int32)

    for i in range(54):
        name = INDEX_TO_MOVE[i]
        p = perms[i]
        if name.endswith("2"):
            # Double move has order 2
            assert np.array_equal(p[p], identity), (
                f"{name} does not have order 2"
            )
        else:
            # Quarter turns and rotations have order 4
            p4 = p[p][p][p]
            assert np.array_equal(p4, identity), f"{name} does not have order 4"


def test_rubiks_cube_env_stm():
    env = RubiksCubeEnv(
        batch_size=4,
        use_stm=True,
        use_double_moves=True,
        include_rotations=True,
    )
    assert len(env.action_space) == 54
    assert len(env.get_possible_moves()) == 54

    env.reset()
    assert env.centers.shape == (4, 6)
    assert np.all(env.centers == np.arange(6))

    # Test single slice move M
    m_idx = MOVE_TO_INDEX["M"]
    cp, co, ep, eo = env.step(np.full(4, m_idx, dtype=np.int32))
    # Corners do not move under M
    assert np.all(cp == np.arange(8))
    assert np.all(co == 0)

    # Test inverse move M' brings cube back to solved
    m_prime_idx = MOVE_TO_INDEX["M'"]
    env.step(np.full(4, m_prime_idx, dtype=np.int32))
    assert np.all(env.centers == np.arange(6))
    assert np.all(env.ep == np.arange(12))
    assert np.all(env.eo == 0)


def test_rubiks_cube_env_scramble_stm():
    env = RubiksCubeEnv(batch_size=16, use_stm=True)
    env.reset()
    env.scramble(depth=15)
    # Scrambled states should be non-trivial
    assert not np.all(env.ep == np.arange(12))


def test_torch_cube_ops_stm():
    ops = TorchCubeOps(
        use_stm=True,
        use_double_moves=True,
        include_rotations=True,
        device="cpu",
    )
    assert ops.num_moves == 54

    # Test expanding 40-element state
    from cubesight_ml.cube.layout import SOLVED_FLAT

    states = (
        torch.tensor(SOLVED_FLAT, dtype=torch.long).unsqueeze(0).expand(2, -1)
    )
    children = ops.expand(states)
    assert children.shape == (2, 54, 40)

    # Test expanding 46-element state (with centers)
    centers = torch.arange(6, dtype=torch.long).unsqueeze(0).expand(2, -1)
    states_46 = torch.cat([states, centers], dim=1)
    children_46 = ops.expand(states_46)
    assert children_46.shape == (2, 54, 46)
