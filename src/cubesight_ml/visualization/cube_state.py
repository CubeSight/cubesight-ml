from typing import Any

import numpy as np
import torch

from cubesight_ml.models.res_mlp import RubiksStateConverter

# Standard WCA Rubik's Cube color palette (Hex codes)
COLOR_HEX_MAP: dict[int, str] = {
    0: "#FFFFFF",  # U: White
    1: "#DC2626",  # R: Red
    2: "#16A34A",  # F: Green
    3: "#FACC15",  # D: Yellow
    4: "#EA580C",  # L: Orange
    5: "#2563EB",  # B: Blue
}

COLOR_NAMES: list[str] = ["White", "Red", "Green", "Yellow", "Orange", "Blue"]

# 3x3 grid of sticker indices (row-major: top-left to bottom-right) for each face
FACE_GRIDS: dict[str, list[int]] = {
    "U": [12, 34, 15, 32, 0, 36, 9, 30, 6],
    "D": [21, 38, 18, 40, 3, 44, 24, 42, 27],
    "F": [10, 31, 8, 49, 2, 47, 23, 39, 19],
    "B": [16, 35, 14, 53, 5, 51, 29, 43, 25],
    "L": [13, 33, 11, 50, 4, 48, 26, 41, 22],
    "R": [7, 37, 17, 46, 1, 52, 20, 45, 28],
}

_CONVERTER = RubiksStateConverter()


def state_to_stickers(
    cp: np.ndarray,
    co: np.ndarray,
    ep: np.ndarray,
    eo: np.ndarray,
    centers: np.ndarray | None = None,
) -> list[int]:
    """
    Converts 40-d or 46-d Rubik's cube state into a list of 54 sticker color IDs (0..5).
    """
    if cp.ndim == 1:
        cp = cp[None, :]
        co = co[None, :]
        ep = ep[None, :]
        eo = eo[None, :]
        if centers is not None:
            centers = centers[None, :]

    if centers is not None:
        flat = np.concatenate([cp, co, ep, eo, centers], axis=-1)
    else:
        flat = np.concatenate([cp, co, ep, eo], axis=-1)

    with torch.no_grad():
        stk_t = _CONVERTER(torch.tensor(flat, dtype=torch.long))
        return stk_t.squeeze(0).tolist()


def stickers_to_face_dict(stickers: list[int]) -> dict[str, list[int]]:
    """Groups 54 stickers into 3x3 grids for faces U, R, F, D, L, B."""
    return {
        face: [stickers[idx] for idx in grid]
        for face, grid in FACE_GRIDS.items()
    }


def get_cubies_representation(stickers: list[int]) -> list[dict[str, Any]]:
    """
    Constructs the 3D cubie definition for Three.js rendering.
    Returns 27 cubies at (x, y, z) in {-1, 0, 1}^3 with face colors:
      - right (+X, R)
      - left (-X, L)
      - top (+Y, U)
      - bottom (-Y, D)
      - front (+Z, F)
      - back (-Z, B)
    Interior faces are None (rendered as black plastic).
    """
    # Inverse map from (face, row, col) to sticker color
    cubies: list[dict[str, Any]] = []

    # Map face grid index to row, col
    # Face U: y = +1, row = z+1 (0..2), col = x+1 (0..2)
    # Face D: y = -1, row = 1-z (0..2), col = x+1 (0..2)
    # Face F: z = +1, row = 1-y (0..2), col = x+1 (0..2)
    # Face B: z = -1, row = 1-y (0..2), col = 1-x (0..2)
    # Face L: x = -1, row = 1-y (0..2), col = z+1 (0..2)
    # Face R: x = +1, row = 1-y (0..2), col = 1-z (0..2)

    face_dict = stickers_to_face_dict(stickers)

    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                # Calculate face sticker colors for exterior faces
                r_color = None
                l_color = None
                u_color = None
                d_color = None
                f_color = None
                b_color = None

                if x == 1:
                    row = 1 - y
                    col = 1 - z
                    r_color = face_dict["R"][row * 3 + col]
                elif x == -1:
                    row = 1 - y
                    col = z + 1
                    l_color = face_dict["L"][row * 3 + col]

                if y == 1:
                    row = z + 1
                    col = x + 1
                    u_color = face_dict["U"][row * 3 + col]
                elif y == -1:
                    row = 1 - z
                    col = x + 1
                    d_color = face_dict["D"][row * 3 + col]

                if z == 1:
                    row = 1 - y
                    col = x + 1
                    f_color = face_dict["F"][row * 3 + col]
                elif z == -1:
                    row = 1 - y
                    col = 1 - x
                    b_color = face_dict["B"][row * 3 + col]

                cubies.append(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "faces": {
                            "right": r_color,
                            "left": l_color,
                            "top": u_color,
                            "bottom": d_color,
                            "front": f_color,
                            "back": b_color,
                        },
                    }
                )

    return cubies
