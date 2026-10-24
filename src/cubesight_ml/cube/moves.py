import json
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Move Nomenclature: 54 moves in STM with double turns and rotations
# ---------------------------------------------------------------------------

FACE_MOVES: list[str] = [
    "U",
    "U2",
    "U'",
    "D",
    "D2",
    "D'",
    "L",
    "L2",
    "L'",
    "R",
    "R2",
    "R'",
    "F",
    "F2",
    "F'",
    "B",
    "B2",
    "B'",
]

SLICE_MOVES: list[str] = [
    "M",
    "M2",
    "M'",
    "E",
    "E2",
    "E'",
    "S",
    "S2",
    "S'",
]

WIDE_MOVES: list[str] = [
    "u",
    "u2",
    "u'",
    "d",
    "d2",
    "d'",
    "l",
    "l2",
    "l'",
    "r",
    "r2",
    "r'",
    "f",
    "f2",
    "f'",
    "b",
    "b2",
    "b'",
]

ROTATION_MOVES: list[str] = [
    "x",
    "x2",
    "x'",
    "y",
    "y2",
    "y'",
    "z",
    "z2",
    "z'",
]

ALL_MOVES: list[str] = FACE_MOVES + SLICE_MOVES + WIDE_MOVES + ROTATION_MOVES
NUM_MOVES: int = len(ALL_MOVES)  # 54
NOOP_NAME: str = "NOOP"
NOOP_INDEX: int = NUM_MOVES  # 54

MOVE_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(ALL_MOVES)}
MOVE_TO_INDEX[NOOP_NAME] = NOOP_INDEX

INDEX_TO_MOVE: dict[int, str] = dict(enumerate(ALL_MOVES))
INDEX_TO_MOVE[NOOP_INDEX] = NOOP_NAME

# Sets for quick categorization
FACE_MOVE_SET: set[str] = set(FACE_MOVES)
SLICE_MOVE_SET: set[str] = set(SLICE_MOVES)
WIDE_MOVE_SET: set[str] = set(WIDE_MOVES)
ROTATION_MOVE_SET: set[str] = set(ROTATION_MOVES)

# In STM for beam search, all moves (including rotations) count as 1 move
BASE_MOVE_COSTS_STM: np.ndarray = np.ones(NUM_MOVES, dtype=np.float32)


def get_possible_moves(
    use_stm: bool = True,
    use_double_moves: bool = True,
    include_rotations: bool = True,
) -> list[int]:
    """Returns valid action indices for the requested move space."""
    if use_stm:
        moves = []
        for idx, name in enumerate(ALL_MOVES):
            is_rotation = name in ROTATION_MOVE_SET
            is_double = "2" in name
            if is_rotation and not include_rotations:
                continue
            if is_double and not use_double_moves:
                continue
            moves.append(idx)
        return moves
    if use_double_moves:
        return list(range(18))
    return [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17]


def _compute_sticker_perms() -> tuple[np.ndarray, np.ndarray]:
    """
    Computes exact 54-sticker permutation arrays for all 54 STM moves plus NOOP.
    Returns:
        perms: ndarray of shape (55, 54), where perms[m] is the permutation for move m.
        inverses: ndarray of shape (55,), mapping each move index to its inverse index.
    """
    from cubesight_ml.cube.layout import (
        C_ROT_NP,
        E_ROT_NP,
    )

    base_moves = {
        "U": {
            "cp": np.array([3, 0, 1, 2, 4, 5, 6, 7], dtype=np.int32),
            "co": np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
            "ep": np.array(
                [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32
            ),
            "eo": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        },
        "D": {
            "cp": np.array([0, 1, 2, 3, 5, 6, 7, 4], dtype=np.int32),
            "co": np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
            "ep": np.array(
                [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11], dtype=np.int32
            ),
            "eo": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        },
        "L": {
            "cp": np.array([0, 2, 6, 3, 4, 1, 5, 7], dtype=np.int32),
            "co": np.array([0, 1, 2, 0, 0, 2, 1, 0], dtype=np.int8),
            "ep": np.array(
                [0, 10, 2, 3, 4, 9, 6, 7, 8, 1, 5, 11], dtype=np.int32
            ),
            "eo": np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0], dtype=np.int8),
        },
        "R": {
            "cp": np.array([4, 1, 2, 0, 7, 5, 6, 3], dtype=np.int32),
            "co": np.array([2, 0, 0, 1, 1, 0, 0, 2], dtype=np.int8),
            "ep": np.array(
                [0, 1, 2, 8, 4, 5, 6, 11, 7, 9, 10, 3], dtype=np.int32
            ),
            "eo": np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1], dtype=np.int8),
        },
        "F": {
            "cp": np.array([1, 5, 2, 3, 0, 4, 6, 7], dtype=np.int32),
            "co": np.array([1, 2, 0, 0, 2, 1, 0, 0], dtype=np.int8),
            "ep": np.array(
                [9, 1, 2, 3, 8, 5, 6, 7, 0, 4, 10, 11], dtype=np.int32
            ),
            "eo": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        },
        "B": {
            "cp": np.array([0, 1, 3, 7, 4, 5, 2, 6], dtype=np.int32),
            "co": np.array([0, 0, 1, 2, 0, 0, 2, 1], dtype=np.int8),
            "ep": np.array(
                [0, 1, 11, 3, 4, 5, 10, 7, 8, 9, 2, 6], dtype=np.int32
            ),
            "eo": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        },
    }

    generated_moves = []
    for face in ["U", "D", "L", "R", "F", "B"]:
        m1 = base_moves[face]
        m2 = {
            "cp": m1["cp"][m1["cp"]],
            "ep": m1["ep"][m1["ep"]],
            "co": (m1["co"][m1["cp"]] + m1["co"]) % 3,
            "eo": m1["eo"][m1["ep"]] ^ m1["eo"],
        }
        m3 = {
            "cp": m1["cp"][m2["cp"]],
            "ep": m1["ep"][m2["ep"]],
            "co": (m1["co"][m2["cp"]] + m2["co"]) % 3,
            "eo": m1["eo"][m2["ep"]] ^ m2["eo"],
        }
        generated_moves.extend([m1, m2, m3])

    def get_face_turn_sticker_perm(m: dict[str, np.ndarray]) -> np.ndarray:
        perm = np.arange(54, dtype=np.int32)
        cp = m["cp"]
        co = m["co"]
        for s in range(8):
            src_slot = cp[s]
            src_co = co[s]
            for j in range(3):
                dst_idx = 6 + 3 * s + j
                src_idx = 6 + 3 * src_slot + C_ROT_NP[src_co][j]
                perm[dst_idx] = src_idx

        ep = m["ep"]
        eo = m["eo"]
        for s in range(12):
            src_slot = ep[s]
            src_eo = eo[s]
            for j in range(2):
                dst_idx = 30 + 2 * s + j
                src_idx = 30 + 2 * src_slot + E_ROT_NP[src_eo][j]
                perm[dst_idx] = src_idx

        return perm

    # 18 Face moves
    face_perms = [get_face_turn_sticker_perm(m) for m in generated_moves]

    # Slices M, E, S
    perm_m = np.arange(54, dtype=np.int32)
    perm_m[2] = 0
    perm_m[3] = 2
    perm_m[5] = 3
    perm_m[0] = 5
    perm_m[31] = 34
    perm_m[38] = 31
    perm_m[43] = 38
    perm_m[34] = 43
    perm_m[30] = 35
    perm_m[39] = 30
    perm_m[42] = 39
    perm_m[35] = 42

    perm_e = np.arange(54, dtype=np.int32)
    perm_e[1] = 2
    perm_e[5] = 1
    perm_e[4] = 5
    perm_e[2] = 4
    perm_e[47] = 48
    perm_e[52] = 47
    perm_e[51] = 52
    perm_e[48] = 51
    perm_e[46] = 49
    perm_e[53] = 46
    perm_e[50] = 53
    perm_e[49] = 50

    perm_s = np.arange(54, dtype=np.int32)
    perm_s[1] = 0
    perm_s[3] = 1
    perm_s[4] = 3
    perm_s[0] = 4
    perm_s[36] = 33
    perm_s[45] = 36
    perm_s[40] = 45
    perm_s[33] = 40
    perm_s[37] = 32
    perm_s[44] = 37
    perm_s[41] = 44
    perm_s[32] = 41

    def dbl(p: np.ndarray) -> np.ndarray:
        return p[p]

    def inv(p: np.ndarray) -> np.ndarray:
        return np.argsort(p).astype(np.int32)

    move_dict: dict[str, np.ndarray] = {}
    for i, name in enumerate(FACE_MOVES):
        move_dict[name] = face_perms[i]

    # Slices
    move_dict["M"] = perm_m
    move_dict["M2"] = dbl(perm_m)
    move_dict["M'"] = inv(perm_m)

    move_dict["E"] = perm_e
    move_dict["E2"] = dbl(perm_e)
    move_dict["E'"] = inv(perm_e)

    move_dict["S"] = perm_s
    move_dict["S2"] = dbl(perm_s)
    move_dict["S'"] = inv(perm_s)

    # Wide moves
    p_u = move_dict["U"][move_dict["E'"]]
    move_dict["u"] = p_u
    move_dict["u2"] = dbl(p_u)
    move_dict["u'"] = inv(p_u)

    p_d = move_dict["D"][move_dict["E"]]
    move_dict["d"] = p_d
    move_dict["d2"] = dbl(p_d)
    move_dict["d'"] = inv(p_d)

    p_l = move_dict["L"][move_dict["M"]]
    move_dict["l"] = p_l
    move_dict["l2"] = dbl(p_l)
    move_dict["l'"] = inv(p_l)

    p_r = move_dict["R"][move_dict["M'"]]
    move_dict["r"] = p_r
    move_dict["r2"] = dbl(p_r)
    move_dict["r'"] = inv(p_r)

    p_f = move_dict["F"][move_dict["S"]]
    move_dict["f"] = p_f
    move_dict["f2"] = dbl(p_f)
    move_dict["f'"] = inv(p_f)

    p_b = move_dict["B"][move_dict["S'"]]
    move_dict["b"] = p_b
    move_dict["b2"] = dbl(p_b)
    move_dict["b'"] = inv(p_b)

    # Rotations
    p_x = move_dict["r"][move_dict["L'"]]
    move_dict["x"] = p_x
    move_dict["x2"] = dbl(p_x)
    move_dict["x'"] = inv(p_x)

    p_y = move_dict["u"][move_dict["D'"]]
    move_dict["y"] = p_y
    move_dict["y2"] = dbl(p_y)
    move_dict["y'"] = inv(p_y)

    p_z = move_dict["f"][move_dict["B'"]]
    move_dict["z"] = p_z
    move_dict["z2"] = dbl(p_z)
    move_dict["z'"] = inv(p_z)

    # Assemble perms array (55, 54)
    perms = np.empty((55, 54), dtype=np.int32)
    for i, name in enumerate(ALL_MOVES):
        perms[i] = move_dict[name]
    perms[NOOP_INDEX] = np.arange(54, dtype=np.int32)

    # Calculate inverses array
    inverses = np.empty(55, dtype=np.int32)
    for i, name in enumerate(ALL_MOVES):
        if name.endswith("2"):
            inverses[i] = i
        elif name.endswith("'"):
            base = name[:-1]
            inverses[i] = MOVE_TO_INDEX[base]
        else:
            prime = f"{name}'"
            inverses[i] = MOVE_TO_INDEX[prime]
    inverses[NOOP_INDEX] = NOOP_INDEX

    return perms, inverses


STICKER_MOVE_PERMS_54, INVERSE_MOVES_54 = _compute_sticker_perms()


def _compute_piece_tables(
    perms_54: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from cubesight_ml.cube.layout import (
        CORNER_STICKER_LUT_NP,
        CORNERS_FACES,
        EDGE_STICKER_LUT_NP,
        EDGES_FACES,
    )

    stickers_solved = np.empty(54, dtype=np.int8)
    stickers_solved[:6] = np.arange(6)
    for s in range(8):
        for j in range(3):
            stickers_solved[6 + 3 * s + j] = CORNERS_FACES[s][j]
    for s in range(12):
        for j in range(2):
            stickers_solved[30 + 2 * s + j] = EDGES_FACES[s][j]

    move_cp = np.empty((55, 8), dtype=np.int32)
    move_co = np.empty((55, 8), dtype=np.int8)
    move_ep = np.empty((55, 12), dtype=np.int32)
    move_eo = np.empty((55, 12), dtype=np.int8)
    move_centers = np.empty((55, 6), dtype=np.int8)

    for i in range(55):
        perm = perms_54[i]
        stk_new = stickers_solved[perm]
        move_centers[i] = stk_new[:6]
        for s in range(8):
            c_cols = stk_new[6 + 3 * s : 6 + 3 * s + 3]
            p, o = CORNER_STICKER_LUT_NP[c_cols[0], c_cols[1], c_cols[2]]
            move_cp[i, s] = p
            move_co[i, s] = o
        for s in range(12):
            e_cols = stk_new[30 + 2 * s : 30 + 2 * s + 2]
            p, o = EDGE_STICKER_LUT_NP[e_cols[0], e_cols[1]]
            move_ep[i, s] = p
            move_eo[i, s] = o

    return move_cp, move_co, move_ep, move_eo, move_centers


(
    MOVE_CP_54,
    MOVE_CO_54,
    MOVE_EP_54,
    MOVE_EO_54,
    MOVE_CENTERS_54,
) = _compute_piece_tables(STICKER_MOVE_PERMS_54)


def parse_move_penalties(
    penalties: dict[str, float] | str | Path | None,
) -> np.ndarray:
    """
    Parses a configurable move penalty mapping into an array of shape (54,).

    Args:
        penalties: Either:
            - None: defaults to 0.0 for all moves.
            - dict: mapping move name string -> float penalty.
            - Path or str pointing to a JSON file.
            - str containing JSON dictionary.

    Returns:
        np.ndarray of shape (54,) with non-negative float32 penalties.
    """
    out = np.zeros(NUM_MOVES, dtype=np.float32)
    if penalties is None:
        return out
    if isinstance(penalties, np.ndarray):
        if len(penalties) != NUM_MOVES:
            raise ValueError(
                f"Expected {NUM_MOVES} penalties, got {len(penalties)}"
            )
        return penalties.astype(np.float32)
    if hasattr(penalties, "detach"):
        return penalties.detach().cpu().numpy().astype(np.float32)

    data: dict[str, Any]
    if isinstance(penalties, str | Path):
        p_path = Path(penalties)
        if not p_path.is_file():
            preset_candidate = Path("presets") / penalties
            if preset_candidate.is_file():
                p_path = preset_candidate
        if p_path.is_file():
            with open(p_path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            # Try parsing as JSON string
            try:
                data = json.loads(str(penalties))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Could not parse move penalties from '{penalties}': {e}"
                ) from e
    elif isinstance(penalties, dict):
        data = penalties
    else:
        raise TypeError(f"Unsupported penalties type: {type(penalties)}")

    for m_name, val in data.items():
        if m_name in MOVE_TO_INDEX:
            idx = MOVE_TO_INDEX[m_name]
            if idx < NUM_MOVES:
                float_val = float(val)
                if float_val < 0.0:
                    raise ValueError(
                        f"Move penalty for '{m_name}' must be non-negative, got {float_val}"
                    )
                out[idx] = float_val
        else:
            raise KeyError(
                f"Unknown move name '{m_name}' in move penalties. Valid moves: {ALL_MOVES}"
            )

    return out


def _build_center_restore_map() -> dict[tuple[int, ...], list[str]]:
    """Precomputes BFS rotation sequences to restore canonical centers."""
    canonical = (0, 1, 2, 3, 4, 5)
    parent: dict[tuple[int, ...], tuple[tuple[int, ...] | None, str | None]] = {
        canonical: (None, None)
    }
    queue: list[tuple[int, ...]] = [canonical]

    while queue:
        curr = queue.pop(0)
        curr_arr = np.array(curr, dtype=np.int8)
        for rot in ROTATION_MOVES:
            m_idx = MOVE_TO_INDEX[rot]
            perm = MOVE_CENTERS_54[m_idx]
            nxt = tuple(int(x) for x in curr_arr[perm])
            if nxt not in parent:
                parent[nxt] = (curr, rot)
                queue.append(nxt)

    restore_map: dict[tuple[int, ...], list[str]] = {}
    for state in parent:
        moves: list[str] = []
        c: tuple[int, ...] | None = state
        while c is not None and c != canonical:
            p, m = parent[c]
            if m is not None:
                inv_idx = int(INVERSE_MOVES_54[MOVE_TO_INDEX[m]])
                moves.append(INDEX_TO_MOVE[inv_idx])
            c = p
        restore_map[state] = moves

    return restore_map


CENTER_RESTORE_MAP: dict[tuple[int, ...], list[str]] = (
    _build_center_restore_map()
)


def get_restore_rotations(
    centers: tuple[int, ...] | list[int] | np.ndarray,
) -> list[str]:
    """Returns whole-cube rotation moves restoring centers to canonical."""
    key = tuple(int(c) for c in centers)
    if key not in CENTER_RESTORE_MAP:
        raise ValueError(f"Invalid or unreachable center configuration: {key}")
    return list(CENTER_RESTORE_MAP[key])
