import numpy as np
import torch

CP = slice(0, 8)
CO = slice(8, 16)
EP = slice(16, 28)
EO = slice(28, 40)

STATE_SIZE = 40

SOLVED_FLAT = np.array(
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,  # cp
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  # co
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,  # ep
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  # eo
    ],
    dtype=np.int8,
)

# Reference definitions of the 8 corners and 12 edges
# Faces: U=0, R=1, F=2, D=3, L=4, B=5
CORNERS_FACES = [
    [0, 1, 2],  # 0: URF
    [0, 2, 4],  # 1: UFL
    [0, 4, 5],  # 2: ULB
    [0, 5, 1],  # 3: UBR
    [3, 2, 1],  # 4: DFR
    [3, 4, 2],  # 5: DLF
    [3, 5, 4],  # 6: DBL
    [3, 1, 5],  # 7: DRB
]

EDGES_FACES = [
    [0, 2],  # 0: UF
    [0, 4],  # 1: UL
    [0, 5],  # 2: UB
    [0, 1],  # 3: UR
    [3, 2],  # 4: DF
    [3, 4],  # 5: DL
    [3, 5],  # 6: DB
    [3, 1],  # 7: DR
    [1, 2],  # 8: FR
    [4, 2],  # 9: FL
    [4, 5],  # 10: BL
    [1, 5],  # 11: BR
]

C_COLORS_NP = np.array(CORNERS_FACES, dtype=np.int8)  # (8, 3)
E_COLORS_NP = np.array(EDGES_FACES, dtype=np.int8)  # (12, 2)
C_ROT_NP = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=np.int8)
E_ROT_NP = np.array([[0, 1], [1, 0]], dtype=np.int8)

CORNER_STICKER_LUT_NP = np.full((6, 6, 6, 2), -1, dtype=np.int8)
for p in range(8):
    for co in range(3):
        cols = C_COLORS_NP[p][C_ROT_NP[co]]
        CORNER_STICKER_LUT_NP[cols[0], cols[1], cols[2]] = [p, co]

EDGE_STICKER_LUT_NP = np.full((6, 6, 2), -1, dtype=np.int8)
for p in range(12):
    for eo in range(2):
        cols = E_COLORS_NP[p][E_ROT_NP[eo]]
        EDGE_STICKER_LUT_NP[cols[0], cols[1]] = [p, eo]

# Face maps: U=0, R=1, F=2, D=3, L=4, B=5
_ROTATION_MAPS: dict[str, dict[int, int]] = {
    "id": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    "x": {0: 5, 3: 2, 2: 0, 5: 3, 1: 1, 4: 4},
    "x2": {0: 3, 3: 0, 2: 5, 5: 2, 1: 1, 4: 4},
    "x_prime": {0: 2, 3: 5, 2: 3, 5: 0, 1: 1, 4: 4},
    "x'": {0: 2, 3: 5, 2: 3, 5: 0, 1: 1, 4: 4},
    "y": {2: 4, 4: 5, 5: 1, 1: 2, 0: 0, 3: 3},
    "y2": {2: 5, 5: 2, 4: 1, 1: 4, 0: 0, 3: 3},
    "y_prime": {2: 1, 1: 5, 5: 4, 4: 2, 0: 0, 3: 3},
    "y'": {2: 1, 1: 5, 5: 4, 4: 2, 0: 0, 3: 3},
    "z": {0: 1, 1: 3, 3: 4, 4: 0, 2: 2, 5: 5},
    "z2": {0: 3, 3: 0, 1: 4, 4: 1, 2: 2, 5: 5},
    "z_prime": {0: 4, 4: 3, 3: 1, 1: 0, 2: 2, 5: 5},
    "z'": {0: 4, 4: 3, 3: 1, 1: 0, 2: 2, 5: 5},
}

STICKER_ROTATION_PERMS: dict[str, np.ndarray] = {}
COLOR_ROTATION_LUTS: dict[str, np.ndarray] = {}

for rot_name, face_map in _ROTATION_MAPS.items():
    stk_perm = np.zeros(54, dtype=int)
    for c in range(6):
        stk_perm[c] = face_map[c]
    for s in range(8):
        new_s_faces = [face_map[f] for f in CORNERS_FACES[s]]
        target_s = [
            i for i, f in enumerate(CORNERS_FACES) if set(f) == set(new_s_faces)
        ][0]
        for j, f in enumerate(CORNERS_FACES[s]):
            new_f = face_map[f]
            target_j = CORNERS_FACES[target_s].index(new_f)
            stk_perm[6 + 3 * s + j] = 6 + 3 * target_s + target_j
    for s in range(12):
        new_s_faces = [face_map[f] for f in EDGES_FACES[s]]
        target_s = [
            i for i, f in enumerate(EDGES_FACES) if set(f) == set(new_s_faces)
        ][0]
        for j, f in enumerate(EDGES_FACES[s]):
            new_f = face_map[f]
            target_j = EDGES_FACES[target_s].index(new_f)
            stk_perm[30 + 2 * s + j] = 30 + 2 * target_s + target_j

    STICKER_ROTATION_PERMS[rot_name] = stk_perm
    COLOR_ROTATION_LUTS[rot_name] = np.array(
        [face_map[c] for c in range(6)], dtype=np.int8
    )

CUBE_ROTATION_TABLES = _ROTATION_MAPS

# Maps which rotation places a given cross face onto D
FACE_TO_D_ROTATIONS: dict[str, str] = {
    "D": "id",
    "U": "x2",
    "F": "x_prime",
    "B": "x",
    "R": "z",
    "L": "z_prime",
}


def to_flat(
    cp: np.ndarray | torch.Tensor,
    co: np.ndarray | torch.Tensor,
    ep: np.ndarray | torch.Tensor,
    eo: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Concatenates (cp, co, ep, eo) into a single 40-element flat state (1D or 2D batched)."""
    if isinstance(cp, torch.Tensor):
        return torch.cat([cp, co, ep, eo], dim=-1)
    return np.concatenate([cp, co, ep, eo], axis=-1)


def from_flat(
    x: np.ndarray | torch.Tensor,
) -> tuple[
    np.ndarray | torch.Tensor,
    np.ndarray | torch.Tensor,
    np.ndarray | torch.Tensor,
    np.ndarray | torch.Tensor,
]:
    """Splits flat 40-element state into (cp, co, ep, eo)."""
    return x[..., CP], x[..., CO], x[..., EP], x[..., EO]


def apply_cube_rotation(
    state: np.ndarray | torch.Tensor, rot: str
) -> np.ndarray | torch.Tensor:
    """
    Applies a rigid 3D whole-cube rotation to a flat 40-element cube state (1D or 2D batched).
    Rotates both piece positions and orientations consistently with center reorientation.
    """
    if rot == "id":
        return state.copy() if isinstance(state, np.ndarray) else state.clone()

    if rot not in _ROTATION_MAPS:
        raise ValueError(
            f"Unknown rotation '{rot}'. Valid options: {list(_ROTATION_MAPS.keys())}"
        )

    is_torch = isinstance(state, torch.Tensor)
    if is_torch:
        orig_device = state.device
        orig_dtype = state.dtype
        state_np = state.detach().cpu().numpy()
    else:
        state_np = state

    is_1d = state_np.ndim == 1
    if is_1d:
        state_np = state_np[None, :]

    b = state_np.shape[0]
    cp, co, ep, eo = from_flat(state_np)

    # 1. State to stickers
    c_colors = C_COLORS_NP[cp]
    c_rot = C_ROT_NP[co]
    stk_c = np.take_along_axis(c_colors, c_rot, axis=2).reshape(b, 24)

    e_colors = E_COLORS_NP[ep]
    e_rot = E_ROT_NP[eo]
    stk_e = np.take_along_axis(e_colors, e_rot, axis=2).reshape(b, 24)

    centers = np.tile(np.arange(6, dtype=np.int8), (b, 1))
    stickers = np.concatenate([centers, stk_c, stk_e], axis=-1)

    # 2. Permute stickers and map colors
    stk_perm = STICKER_ROTATION_PERMS[rot]
    color_lut = COLOR_ROTATION_LUTS[rot]
    new_stickers = np.empty_like(stickers)
    new_stickers[:, stk_perm] = color_lut[stickers]

    # 3. Stickers to state via precomputed LUT
    new_c_stk = new_stickers[:, 6:30].reshape(b, 8, 3)
    c_res = CORNER_STICKER_LUT_NP[
        new_c_stk[..., 0], new_c_stk[..., 1], new_c_stk[..., 2]
    ]
    new_cp = c_res[..., 0]
    new_co = c_res[..., 1]

    new_e_stk = new_stickers[:, 30:54].reshape(b, 12, 2)
    e_res = EDGE_STICKER_LUT_NP[new_e_stk[..., 0], new_e_stk[..., 1]]
    new_ep = e_res[..., 0]
    new_eo = e_res[..., 1]

    res_np = to_flat(new_cp, new_co, new_ep, new_eo)
    if is_1d:
        res_np = res_np[0]

    if is_torch:
        return torch.from_numpy(res_np).to(device=orig_device, dtype=orig_dtype)
    return res_np
