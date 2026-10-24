import numpy as np

from cubesight_ml.cube.layout import (
    CO,
    CP,
    EO,
    EP,
    SOLVED_FLAT,
    STATE_SIZE,
)
from cubesight_ml.stages.definitions import (
    CROSS_FACE_EDGES,
    Stage,
    StageSpec,
)


def _permutation_parities(p: np.ndarray) -> np.ndarray:
    """
    Computes permutation parity (0 for even, 1 for odd) for a batch of permutations.
    p: (n, k) array of permutation indices.
    """
    n, k = p.shape
    if k <= 1:
        return np.zeros(n, dtype=int)
    diffs = p[:, :, None] > p[:, None, :]
    mask = np.triu(np.ones((k, k), dtype=bool), k=1)
    inv_counts = (diffs & mask).sum(axis=(1, 2))
    return inv_counts % 2


def _sample_permutations_and_orientations(
    n: int,
    free_corners: tuple[int, ...],
    free_edges: tuple[int, ...],
    rng: np.random.Generator,
    orient_corners_zero: bool = False,
    orient_edges_zero: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Samples valid permutations and orientations for free corners and free edges.
    Returns: (cp, co, ep, eo) each of shape (n, 8) or (n, 12).
    """
    # Start with identity solved pieces
    cp = np.tile(np.arange(8, dtype=np.int8), (n, 1))
    co = np.zeros((n, 8), dtype=np.int8)
    ep = np.tile(np.arange(12, dtype=np.int8), (n, 1))
    eo = np.zeros((n, 12), dtype=np.int8)

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

    # Enforce matching parity between corner and edge permutations
    if k_e >= 2 and k_c > 0:
        mismatch = c_parities != e_parities
        e_perm[mismatch, 0], e_perm[mismatch, 1] = (
            e_perm[mismatch, 1],
            e_perm[mismatch, 0],
        )

    # Assign permuted piece labels to the free positions
    if k_c > 0:
        free_c_arr = np.array(free_corners, dtype=np.int8)
        cp[:, free_c_arr] = free_c_arr[c_perm]
    if k_e > 0:
        free_e_arr = np.array(free_edges, dtype=np.int8)
        ep[:, free_e_arr] = free_e_arr[e_perm]

    # Sample orientations satisfying conservation laws
    if k_c > 0 and not orient_corners_zero:
        free_c_arr = np.array(free_corners, dtype=np.int8)
        co_free = rng.integers(0, 3, size=(n, k_c), dtype=np.int8)
        co_free[:, -1] = (-co_free[:, :-1].sum(axis=1)) % 3
        co[:, free_c_arr] = co_free

    if k_e > 0 and not orient_edges_zero:
        free_e_arr = np.array(free_edges, dtype=np.int8)
        eo_free = rng.integers(0, 2, size=(n, k_e), dtype=np.int8)
        eo_free[:, -1] = (-eo_free[:, :-1].sum(axis=1)) % 2
        eo[:, free_e_arr] = eo_free

    return cp, co, ep, eo


def sample_goal_states(
    spec: StageSpec | None,
    n: int,
    rng: np.random.Generator | None = None,
    fixed_cross_face: str | None = None,
) -> np.ndarray:
    """
    Uniformly samples valid, legal Rubik's cube states satisfying the given stage goal.
    If spec is None, samples uniform random legal cubes (precondition for CROSS).

    Args:
        spec: Target StageSpec (CROSS, F2L, OLL, PLL) or None for any legal cube.
        n: Number of states to sample.
        rng: Optional NumPy random Generator.
        fixed_cross_face: If spec is CROSS, optionally fix the cross face ('D', 'U', etc.).
                          If None, randomly chooses among the 6 faces.

    Returns:
        np.ndarray of shape (n, 40) with dtype int8.
    """
    if n <= 0:
        return np.empty((0, STATE_SIZE), dtype=np.int8)

    if rng is None:
        rng = np.random.default_rng()

    # Precondition for CROSS: uniform random legal cubes
    if spec is None:
        free_corners = tuple(range(8))
        free_edges = tuple(range(12))
        cp, co, ep, eo = _sample_permutations_and_orientations(
            n, free_corners, free_edges, rng
        )
        states = np.empty((n, STATE_SIZE), dtype=np.int8)
        states[:, CP] = cp
        states[:, CO] = co
        states[:, EP] = ep
        states[:, EO] = eo
        return states

    if spec.stage == Stage.PLL:
        return np.tile(SOLVED_FLAT, (n, 1)).astype(np.int8)

    if spec.stage == Stage.CROSS:
        if fixed_cross_face is not None:
            faces = [fixed_cross_face] * n
        else:
            face_choices = ("D", "U", "F", "B", "L", "R")
            faces = [face_choices[i] for i in rng.integers(0, 6, size=n)]

        states = np.empty((n, STATE_SIZE), dtype=np.int8)
        unique_faces = set(faces)
        for face in unique_faces:
            idx = np.where(np.array(faces) == face)[0]
            count = len(idx)
            if count == 0:
                continue
            cross_edges = CROSS_FACE_EDGES[face]
            free_edges = tuple(e for e in range(12) if e not in cross_edges)
            free_corners = tuple(range(8))

            cp, co, ep, eo = _sample_permutations_and_orientations(
                count, free_corners, free_edges, rng
            )
            states[idx, CP] = cp
            states[idx, CO] = co
            states[idx, EP] = ep
            states[idx, EO] = eo
        return states

    if spec.stage == Stage.F2L:
        # D-cross + 4 F2L pairs solved (corners 4..7 and edges 4..11)
        free_corners = (0, 1, 2, 3)
        free_edges = (0, 1, 2, 3)
        cp, co, ep, eo = _sample_permutations_and_orientations(
            n, free_corners, free_edges, rng
        )
        states = np.empty((n, STATE_SIZE), dtype=np.int8)
        states[:, CP] = cp
        states[:, CO] = co
        states[:, EP] = ep
        states[:, EO] = eo
        return states

    if spec.stage == Stage.OLL:
        # F2L solved and all pieces oriented (co=0, eo=0)
        free_corners = (0, 1, 2, 3)
        free_edges = (0, 1, 2, 3)
        cp, co, ep, eo = _sample_permutations_and_orientations(
            n,
            free_corners,
            free_edges,
            rng,
            orient_corners_zero=True,
            orient_edges_zero=True,
        )
        states = np.empty((n, STATE_SIZE), dtype=np.int8)
        states[:, CP] = cp
        states[:, CO] = co
        states[:, EP] = ep
        states[:, EO] = eo
        return states

    raise ValueError(f"Unknown stage: {spec.stage}")
