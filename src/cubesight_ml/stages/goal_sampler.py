import numpy as np

from cubesight_ml.cube.moves import CENTER_RESTORE_MAP
from cubesight_ml.stages.definitions import (
    CROSS_FACE_EDGES,
    Stage,
    StageSpec,
)
from cubesight_ml.stages.goal_checker import SOLVED_TABLE_24_NP

CENTERS_TABLE_24_NP = np.array(list(CENTER_RESTORE_MAP.keys()), dtype=np.int8)


def _permutation_parities(p: np.ndarray) -> np.ndarray:
    """Computes permutation parity (0 for even, 1 for odd) for a batch."""
    n, k = p.shape
    if k <= 1:
        return np.zeros(n, dtype=int)
    diffs = p[:, :, None] > p[:, None, :]
    mask = np.triu(np.ones((k, k), dtype=bool), k=1)
    inv_counts = (diffs & mask).sum(axis=(1, 2))
    return inv_counts % 2


def sample_goal_states(
    spec: StageSpec | None,
    n: int,
    rng: np.random.Generator | None = None,
    fixed_cross_face: str | None = None,
    fixed_orientation: int | None = None,
) -> np.ndarray:
    """Uniformly samples valid Rubik's cube states for the stage goal.

    Supports all 24 orientations (any center on any face). If spec is None,
    samples uniform random legal cubes across all 24 orientations.

    Args:
        spec: Target StageSpec (CROSS, F2L, OLL, PLL) or None for any cube.
        n: Number of states to sample.
        rng: Optional NumPy random Generator.
        fixed_cross_face: If given for CROSS, restricts to canonical centers
            with that face's cross solved.
        fixed_orientation: Optional orientation index in [0, 23].

    Returns:
        np.ndarray of shape (n, 46) with dtype int8 containing 40 piece features
        and 6 center face IDs.
    """
    if n <= 0:
        return np.empty((0, 46), dtype=np.int8)

    if rng is None:
        rng = np.random.default_rng()

    if fixed_cross_face is not None and (
        spec is not None and spec.stage == Stage.CROSS
    ):
        orient_ids = np.zeros(n, dtype=int)
        cross_edges = CROSS_FACE_EDGES[fixed_cross_face]
        free_edges = tuple(e for e in range(12) if e not in cross_edges)
        free_corners = tuple(range(8))
        orient_c_zero, orient_e_zero = False, False
    elif fixed_orientation is not None:
        orient_ids = np.full(n, fixed_orientation, dtype=int)
        if spec is None:
            free_corners = tuple(range(8))
            free_edges = tuple(range(12))
            orient_c_zero, orient_e_zero = False, False
        elif spec.stage == Stage.CROSS:
            free_corners = tuple(range(8))
            free_edges = (0, 1, 2, 3, 8, 9, 10, 11)
            orient_c_zero, orient_e_zero = False, False
        elif spec.stage == Stage.F2L:
            free_corners = (0, 1, 2, 3)
            free_edges = (0, 1, 2, 3)
            orient_c_zero, orient_e_zero = False, False
        elif spec.stage == Stage.OLL:
            free_corners = (0, 1, 2, 3)
            free_edges = (0, 1, 2, 3)
            orient_c_zero, orient_e_zero = True, True
        elif spec.stage == Stage.PLL:
            free_corners = ()
            free_edges = ()
            orient_c_zero, orient_e_zero = True, True
        else:
            raise ValueError(f"Unknown stage: {spec.stage}")
    elif spec is None:
        orient_ids = rng.integers(0, 24, size=n)
        free_corners = tuple(range(8))
        free_edges = tuple(range(12))
        orient_c_zero, orient_e_zero = False, False
    elif spec.stage == Stage.CROSS:
        orient_ids = rng.integers(0, 24, size=n)
        free_corners = tuple(range(8))
        free_edges = (0, 1, 2, 3, 8, 9, 10, 11)
        orient_c_zero, orient_e_zero = False, False
    elif spec.stage == Stage.F2L:
        orient_ids = rng.integers(0, 24, size=n)
        free_corners = (0, 1, 2, 3)
        free_edges = (0, 1, 2, 3)
        orient_c_zero, orient_e_zero = False, False
    elif spec.stage == Stage.OLL:
        orient_ids = rng.integers(0, 24, size=n)
        free_corners = (0, 1, 2, 3)
        free_edges = (0, 1, 2, 3)
        orient_c_zero, orient_e_zero = True, True
    elif spec.stage == Stage.PLL:
        orient_ids = rng.integers(0, 24, size=n)
        free_corners = ()
        free_edges = ()
        orient_c_zero, orient_e_zero = True, True
    else:
        raise ValueError(f"Unknown stage: {spec.stage}")

    target = SOLVED_TABLE_24_NP[orient_ids]
    target_c = CENTERS_TABLE_24_NP[orient_ids]

    t_cp = target[:, 0:8]
    t_co = target[:, 8:16]
    t_ep = target[:, 16:28]
    t_eo = target[:, 28:40]

    cp = t_cp.copy()
    co = t_co.copy()
    ep = t_ep.copy()
    eo = t_eo.copy()

    k_c = len(free_corners)
    k_e = len(free_edges)

    if k_c > 0:
        c_keys = rng.random(size=(n, k_c))
        c_perm = np.argsort(c_keys, axis=1)
        c_parities = _permutation_parities(c_perm)
    else:
        c_perm = np.zeros((n, 0), dtype=int)
        c_parities = np.zeros(n, dtype=int)

    if k_e > 0:
        e_keys = rng.random(size=(n, k_e))
        e_perm = np.argsort(e_keys, axis=1)
        e_parities = _permutation_parities(e_perm)
    else:
        e_perm = np.zeros((n, 0), dtype=int)
        e_parities = np.zeros(n, dtype=int)

    if k_e >= 2 and k_c > 0:
        mismatch = c_parities != e_parities
        e_perm[mismatch, 0], e_perm[mismatch, 1] = (
            e_perm[mismatch, 1],
            e_perm[mismatch, 0],
        )

    rows = np.arange(n)[:, None]
    if k_c > 0:
        free_c_arr = np.array(free_corners, dtype=np.int8)
        orig_free_c = t_cp[:, free_c_arr]
        cp[:, free_c_arr] = orig_free_c[rows, c_perm]

    if k_e > 0:
        free_e_arr = np.array(free_edges, dtype=np.int8)
        orig_free_e = t_ep[:, free_e_arr]
        ep[:, free_e_arr] = orig_free_e[rows, e_perm]

    if k_c > 0 and not orient_c_zero:
        free_c_arr = np.array(free_corners, dtype=np.int8)
        co_free = rng.integers(0, 3, size=(n, k_c), dtype=np.int8)
        target_co_sum = t_co[:, free_c_arr].sum(axis=1) % 3
        co_free[:, -1] = (target_co_sum - co_free[:, :-1].sum(axis=1)) % 3
        co[:, free_c_arr] = co_free

    if k_e > 0 and not orient_e_zero:
        free_e_arr = np.array(free_edges, dtype=np.int8)
        eo_free = rng.integers(0, 2, size=(n, k_e), dtype=np.int8)
        target_eo_sum = t_eo[:, free_e_arr].sum(axis=1) % 2
        eo_free[:, -1] = (target_eo_sum - eo_free[:, :-1].sum(axis=1)) % 2
        eo[:, free_e_arr] = eo_free

    states = np.empty((n, 46), dtype=np.int8)
    states[:, 0:8] = cp
    states[:, 8:16] = co
    states[:, 16:28] = ep
    states[:, 28:40] = eo
    states[:, 40:46] = target_c
    return states
