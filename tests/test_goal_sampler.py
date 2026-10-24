import numpy as np

from cubesight_ml.cube.layout import (
    CP,
    EO,
    EP,
    SOLVED_FLAT,
)
from cubesight_ml.stages.definitions import (
    CROSS_FACE_MASKS,
    CROSS_SPEC,
    F2L_SPEC,
    OLL_SPEC,
    PLL_SPEC,
    STAGES,
)
from cubesight_ml.stages.goal_checker import is_goal_np
from cubesight_ml.stages.goal_sampler import (
    _permutation_parities,
    sample_goal_states,
)


def _check_legal_cube_invariants(states: np.ndarray) -> None:
    """Verifies that every state in states satisfies the Rubik's cube group invariants."""
    n = len(states)
    cp = states[:, CP]
    co = states[:, 8:16]
    ep = states[:, EP]
    eo = states[:, EO]

    # 1. Valid permutations
    for i in range(n):
        assert set(cp[i]) == set(range(8)), (
            f"Invalid corner permutation at index {i}"
        )
        assert set(ep[i]) == set(range(12)), (
            f"Invalid edge permutation at index {i}"
        )

    # 2. Matching parity
    cp_parities = _permutation_parities(cp)
    ep_parities = _permutation_parities(ep)
    np.testing.assert_array_equal(
        cp_parities,
        ep_parities,
        err_msg="Parity mismatch between corners and edges",
    )

    # 3. Orientation conservation
    np.testing.assert_array_equal(
        co.sum(axis=1) % 3,
        np.zeros(n),
        err_msg="Corner orientation sum not 0 mod 3",
    )
    np.testing.assert_array_equal(
        eo.sum(axis=1) % 2,
        np.zeros(n),
        err_msg="Edge orientation sum not 0 mod 2",
    )


def test_sample_goal_states_empty():
    states = sample_goal_states(CROSS_SPEC, 0)
    assert states.shape == (0, 40)


def test_sample_goal_states_none_precondition():
    """Testing spec=None samples uniform legal cubes."""
    rng = np.random.default_rng(123)
    states = sample_goal_states(None, 50, rng=rng)
    assert states.shape == (50, 40)
    _check_legal_cube_invariants(states)


def test_sample_goal_states_cross():
    rng = np.random.default_rng(456)
    states = sample_goal_states(CROSS_SPEC, 60, rng=rng)
    assert states.shape == (60, 40)
    _check_legal_cube_invariants(states)
    assert is_goal_np(states, CROSS_SPEC).all()

    # Test fixed_cross_face="D"
    d_states = sample_goal_states(CROSS_SPEC, 30, rng=rng, fixed_cross_face="D")
    _check_legal_cube_invariants(d_states)
    assert ((d_states == SOLVED_FLAT) | ~CROSS_FACE_MASKS["D"]).all()


def test_sample_goal_states_f2l():
    rng = np.random.default_rng(789)
    states = sample_goal_states(F2L_SPEC, 50, rng=rng)
    assert states.shape == (50, 40)
    _check_legal_cube_invariants(states)
    assert is_goal_np(states, F2L_SPEC).all()

    # D-cross and F2L pairs should be solved
    assert (states[:, CP.start + 4 : CP.start + 8] == np.arange(4, 8)).all()
    assert (states[:, 8 + 4 : 8 + 8] == 0).all()
    assert (states[:, EP.start + 4 : EP.start + 12] == np.arange(4, 12)).all()
    assert (states[:, EO.start + 4 : EO.start + 12] == 0).all()


def test_sample_goal_states_oll():
    rng = np.random.default_rng(101)
    states = sample_goal_states(OLL_SPEC, 50, rng=rng)
    assert states.shape == (50, 40)
    _check_legal_cube_invariants(states)
    assert is_goal_np(states, OLL_SPEC).all()

    # All orientations must be 0
    assert (states[:, 8:16] == 0).all()
    assert (states[:, EO] == 0).all()


def test_sample_goal_states_pll():
    rng = np.random.default_rng(202)
    states = sample_goal_states(PLL_SPEC, 20, rng=rng)
    assert states.shape == (20, 40)
    _check_legal_cube_invariants(states)
    assert is_goal_np(states, PLL_SPEC).all()
    assert (states == SOLVED_FLAT).all()


def test_sample_diversity():
    """Verify that sampled states for non-PLL stages have high diversity."""
    rng = np.random.default_rng(303)
    n = 100
    for stage_enum, spec in STAGES.items():
        states = sample_goal_states(spec, n, rng=rng)
        # Count unique states
        unique_count = len(np.unique(states, axis=0))
        if stage_enum == "pll":
            assert unique_count == 1
        elif stage_enum == "oll":
            # For OLL (total set size 288), expected unique count for 100 draws is ~84.5
            assert unique_count >= 70
        else:
            # CROSS (size ~2.28e14) and F2L (size 62,208) should have >= 90 unique items
            assert unique_count >= 90
