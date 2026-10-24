import numpy as np
import pytest
import torch

from cubesight_ml.envs.rubiks_cube import RubiksCubeEnv
from cubesight_ml.models.res_mlp import ResMLP


def permutation_parity(arr: np.ndarray) -> int:
    """Computes parity of 1D permutation array: 0 for even, 1 for odd."""
    inversions = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inversions += 1
    return inversions % 2


def test_env_moves_and_inverses():
    batch_size = 1
    env = RubiksCubeEnv(batch_size=batch_size, use_double_moves=True)

    solved_cp = np.arange(8, dtype=np.int8)
    solved_ep = np.arange(12, dtype=np.int8)
    solved_co = np.zeros(8, dtype=np.int8)
    solved_eo = np.zeros(12, dtype=np.int8)

    for move in range(18):
        env.reset()
        action = np.array([move], dtype=np.int32)
        inv_action = np.array([env.inverse_moves[move]], dtype=np.int32)

        # Move followed by its inverse should return to solved state
        env.step(action)
        cp, co, ep, eo = env.step(inv_action)
        assert np.array_equal(cp[0], solved_cp)
        assert np.array_equal(co[0], solved_co)
        assert np.array_equal(ep[0], solved_ep)
        assert np.array_equal(eo[0], solved_eo)

        # Applying a 90-degree move 4 times should return to solved state
        if move % 3 == 0:  # quarter turn moves (U, D, L, R, F, B)
            env.reset()
            for _ in range(4):
                cp, co, ep, eo = env.step(action)
            assert np.array_equal(cp[0], solved_cp)
            assert np.array_equal(co[0], solved_co)
            assert np.array_equal(ep[0], solved_ep)
            assert np.array_equal(eo[0], solved_eo)


def test_env_orientation_and_parity_invariants():
    batch_size = 16
    # In HTM, centers are fixed so corner parity equals edge parity
    env_htm = RubiksCubeEnv(
        batch_size=batch_size, use_stm=False, use_double_moves=False
    )
    cp, co, ep, eo = env_htm.scramble(depth=30)

    for i in range(batch_size):
        # Corner orientations sum to 0 mod 3
        assert np.sum(co[i]) % 3 == 0
        # Edge orientations sum to 0 mod 2
        assert np.sum(eo[i]) % 2 == 0
        # Parity of corner permutation equals parity of edge permutation
        assert permutation_parity(cp[i]) == permutation_parity(ep[i])

    # Default scramble preserves centers
    env_default = RubiksCubeEnv(batch_size=batch_size, use_stm=True)
    _, _, _, _ = env_default.scramble(depth=30)
    for i in range(batch_size):
        assert np.array_equal(env_default.centers[i], np.arange(6))

    # In STM with face_moves_only=False, slices and rotations also move centers;
    # total group parity sums to 0 mod 2
    env_stm = RubiksCubeEnv(
        batch_size=batch_size, use_stm=True, scramble_face_moves_only=False
    )
    cp_s, co_s, ep_s, eo_s = env_stm.scramble(depth=30)
    for i in range(batch_size):
        assert np.sum(co_s[i]) % 3 == 0
        assert np.sum(eo_s[i]) % 2 == 0
        total_parity = (
            permutation_parity(cp_s[i])
            ^ permutation_parity(ep_s[i])
            ^ permutation_parity(env_stm.centers[i])
        )
        assert total_parity == 0


@pytest.mark.parametrize("rep", ["per_piece", "per_sticker"])
def test_res_mlp_shapes(rep: str):
    batch_size = 4
    model = ResMLP(
        state_representation=rep,
        output_dim=1,
        n1=64,
        n2=64,
        n_r=2,
        dropout_rate=0.0,
    )

    env = RubiksCubeEnv(batch_size=batch_size)
    cp, co, ep, eo = env.scramble(depth=10)
    flat_state = np.concatenate([cp, co, ep, eo], axis=1)
    tensor_input = torch.tensor(flat_state, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        out = model(tensor_input)

    assert out.shape == (batch_size, 1)
    assert not torch.isnan(out).any()
