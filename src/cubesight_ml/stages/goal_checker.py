import numpy as np
import torch
import torch.nn as nn

from cubesight_ml.cube.layout import SOLVED_FLAT
from cubesight_ml.cube.moves import (
    CENTER_RESTORE_MAP,
    MOVE_CENTERS_54,
    MOVE_TO_INDEX,
    ROTATION_MOVES,
)
from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.stages.definitions import CROSS_FACE_MASKS, StageSpec


def _build_center_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precomputes 24 solved piece targets and base-6 center hash LUT."""
    ops = TorchCubeOps(use_stm=True, device="cpu")
    centers_list = list(CENTER_RESTORE_MAP.keys())
    assert len(centers_list) == 24

    weights = np.array([7776, 1296, 216, 36, 6, 1], dtype=np.int64)
    lut = np.full(46656, -1, dtype=np.int64)
    for idx, c in enumerate(centers_list):
        h = int(np.sum(np.array(c, dtype=np.int64) * weights))
        lut[h] = idx

    canonical = (0, 1, 2, 3, 4, 5)
    fwd: dict[tuple[int, ...], list[int]] = {canonical: []}
    q: list[tuple[int, ...]] = [canonical]
    while q:
        curr = q.pop(0)
        curr_arr = np.array(curr, dtype=np.int8)
        for rot in ROTATION_MOVES:
            m_idx = MOVE_TO_INDEX[rot]
            perm = MOVE_CENTERS_54[m_idx]
            nxt = tuple(int(x) for x in curr_arr[perm])
            if nxt not in fwd:
                fwd[nxt] = fwd[curr] + [m_idx]
                q.append(nxt)

    solved_table = np.empty((24, 40), dtype=np.int8)
    solved_flat_t = torch.tensor(SOLVED_FLAT, dtype=torch.long).unsqueeze(0)
    for idx, c in enumerate(centers_list):
        moves = fwd[c]
        curr_t = solved_flat_t
        for m in moves:
            m_t = torch.tensor([m], dtype=torch.long)
            curr_t = ops.apply_moves(curr_t, m_t)
        solved_table[idx] = curr_t.squeeze(0).numpy().astype(np.int8)

    return solved_table, lut, weights


SOLVED_TABLE_24_NP, CENTER_ID_LUT, CENTER_BASE6_WEIGHTS = _build_center_tables()


def is_goal_np(
    states: np.ndarray,
    spec: StageSpec,
    centers: np.ndarray | None = None,
) -> np.ndarray:
    """
    Checks if flat cube state(s) satisfy the stage goal.
    Supports 1D (40,)/(46,) or batched (B, 40)/(B, 46) or (B, A, 40)/(B, A, 46).
    """
    if states.shape[-1] >= 46:
        c = states[..., 40:46]
        pieces = states[..., :40]
        h = np.sum(c * CENTER_BASE6_WEIGHTS, axis=-1)
        c_ids = CENTER_ID_LUT[h]
        target = SOLVED_TABLE_24_NP[c_ids]
    elif centers is not None:
        c = np.asarray(centers)
        h = np.sum(c * CENTER_BASE6_WEIGHTS, axis=-1)
        c_ids = CENTER_ID_LUT[h]
        target = SOLVED_TABLE_24_NP[c_ids]
        pieces = states
    else:
        target = SOLVED_FLAT
        pieces = states

    if len(spec.masks) == 1:
        mask = spec.masks[0]
        return ((pieces == target) | ~mask).all(axis=-1)

    matches = [((pieces == target) | ~m).all(axis=-1) for m in spec.masks]
    return np.any(matches, axis=0)


def detect_solved_cross_face(
    state: np.ndarray,
    centers: np.ndarray | None = None,
) -> str:
    """
    Returns which cross face ('D', 'U', 'F', 'B', 'L', or 'R') is solved.
    If 'D' is among the solved faces, prefers 'D' to avoid unnecessary reorientation.
    """
    if state.shape[-1] >= 46:
        c = state[..., 40:46]
        pieces = state[..., :40]
        h = np.sum(c * CENTER_BASE6_WEIGHTS, axis=-1)
        c_ids = CENTER_ID_LUT[h]
        target = SOLVED_TABLE_24_NP[c_ids]
    elif centers is not None:
        c = np.asarray(centers)
        h = np.sum(c * CENTER_BASE6_WEIGHTS, axis=-1)
        c_ids = CENTER_ID_LUT[h]
        target = SOLVED_TABLE_24_NP[c_ids]
        pieces = state
    else:
        target = SOLVED_FLAT
        pieces = state

    # If D is solved, prefer D to avoid rotating
    if ((pieces == target) | ~CROSS_FACE_MASKS["D"]).all():
        return "D"

    for face in ("U", "F", "B", "L", "R"):
        if ((pieces == target) | ~CROSS_FACE_MASKS[face]).all():
            return face

    raise ValueError("No cross face is solved in the given state.")


class StageGoalChecker(nn.Module):
    """
    PyTorch goal checker module for a stage, compatible with arbitrary leading batch dimensions.
    Center-aware: supports both 40-d (canonical) and 46-d (with centers) states.
    """

    def __init__(
        self, spec: StageSpec, device: torch.device | str = "cpu"
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.spec = spec
        self.num_masks = len(spec.masks)

        masks_tensor = torch.tensor(
            np.stack(spec.masks), dtype=torch.bool, device=self.device
        )
        self.register_buffer("masks", masks_tensor)
        self.register_buffer(
            "solved_table",
            torch.tensor(
                SOLVED_TABLE_24_NP, dtype=torch.long, device=self.device
            ),
        )
        self.register_buffer(
            "lut",
            torch.tensor(CENTER_ID_LUT, dtype=torch.long, device=self.device),
        )
        self.register_buffer(
            "solved_flat",
            torch.tensor(SOLVED_FLAT, dtype=torch.long, device=self.device),
        )
        self.register_buffer(
            "weights",
            torch.tensor(
                CENTER_BASE6_WEIGHTS, dtype=torch.long, device=self.device
            ),
        )

    def is_goal(self, states: torch.Tensor) -> torch.Tensor:
        """
        Returns boolean tensor indicating which states satisfy the stage goal.
        Works for states of shape (B, 40)/(B, 46) or (B, A, 40)/(B, A, 46).
        """
        states = states.to(device=self.device, dtype=torch.long)

        if states.size(-1) >= 46:
            c = states[..., 40:46]
            h = (c * self.weights).sum(dim=-1)
            c_ids = self.lut[h]
            target = self.solved_table[c_ids]
            pieces = states[..., :40]
        else:
            target = self.solved_table[0]
            pieces = states

        if self.num_masks == 1:
            return ((pieces == target) | ~self.masks[0]).all(dim=-1)

        # Broadcast states (..., 1, 40) against masks (num_masks, 40)
        expanded_pieces = pieces.unsqueeze(-2)
        expanded_target = (
            target.unsqueeze(-2) if target.ndim == pieces.ndim else target
        )
        matches = ((expanded_pieces == expanded_target) | ~self.masks).all(
            dim=-1
        )
        return matches.any(dim=-1)
