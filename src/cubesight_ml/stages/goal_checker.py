import numpy as np
import torch
import torch.nn as nn

from cubesight_ml.cube.layout import SOLVED_FLAT
from cubesight_ml.cube.moves import (
    CENTER_RESTORE_MAP,
    MOVE_CENTERS_54,
    MOVE_TO_INDEX,
    NOOP_INDEX,
    ROTATION_MOVES,
)
from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.stages.definitions import CROSS_FACE_MASKS, StageSpec

FACE_TO_D_ROTATIONS: dict[int, str | None] = {
    0: "x2",
    1: "z",
    2: "x'",
    3: None,
    4: "z'",
    5: "x",
}
FACE_TO_D_MOVE_INDICES: list[int] = [
    MOVE_TO_INDEX["x2"],
    MOVE_TO_INDEX["z"],
    MOVE_TO_INDEX["x'"],
    NOOP_INDEX,
    MOVE_TO_INDEX["z'"],
    MOVE_TO_INDEX["x"],
]


def _build_center_tables() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Precomputes 24 solved piece targets, U-orientation LUTs, and base-6 center hash LUT."""
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
    u_co_by_piece = np.full((24, 8), -1, dtype=np.int8)
    u_eo_by_piece = np.full((24, 12), -1, dtype=np.int8)
    solved_flat_t = torch.tensor(SOLVED_FLAT, dtype=torch.long).unsqueeze(0)
    for idx, c in enumerate(centers_list):
        moves = fwd[c]
        curr_t = solved_flat_t
        for m in moves:
            m_t = torch.tensor([m], dtype=torch.long)
            curr_t = ops.apply_moves(curr_t, m_t)
        row = curr_t.squeeze(0).numpy().astype(np.int8)
        solved_table[idx] = row
        u_co_by_piece[idx, row[0:4]] = row[8:12]
        u_eo_by_piece[idx, row[16:20]] = row[28:32]

    return solved_table, lut, weights, u_co_by_piece, u_eo_by_piece


(
    SOLVED_TABLE_24_NP,
    CENTER_ID_LUT,
    CENTER_BASE6_WEIGHTS,
    U_CO_BY_PIECE_24_NP,
    U_EO_BY_PIECE_24_NP,
) = _build_center_tables()


def _apply_oriented_piece_targets_np(
    pieces: np.ndarray,
    target: np.ndarray,
    c_ids: np.ndarray | int,
    spec: StageSpec,
) -> np.ndarray:
    """Replaces U-layer target orientations using piece-ID lookup when spec has oriented pieces."""
    if not (spec.oriented_corners or spec.oriented_edges):
        return target
    target = np.broadcast_to(target, pieces.shape).copy()
    c_ids_arr = np.clip(np.asarray(c_ids), 0, 23)[..., None]
    if spec.oriented_corners:
        slots_c = list(spec.oriented_corners)
        cp_u = np.clip(pieces[..., slots_c], 0, 7)
        target[..., [8 + s for s in slots_c]] = U_CO_BY_PIECE_24_NP[
            c_ids_arr, cp_u
        ]
    if spec.oriented_edges:
        slots_e = list(spec.oriented_edges)
        ep_u = np.clip(pieces[..., [16 + s for s in slots_e]], 0, 11)
        target[..., [28 + s for s in slots_e]] = U_EO_BY_PIECE_24_NP[
            c_ids_arr, ep_u
        ]
    return target


def is_goal_np(
    states: np.ndarray,
    spec: StageSpec,
    centers: np.ndarray | None = None,
    target_orientation: int | None = None,
    target_color: int | None = None,
) -> np.ndarray:
    """
    Checks if flat cube state(s) satisfy the stage goal.
    Supports 1D (40,)/(46,) or batched (B, 40)/(B, 46) or (B, A, 40)/(B, A, 46).
    """
    orient_match: np.ndarray | bool = True
    if target_color is not None:
        states_arr = np.asarray(states)
        if states_arr.shape[-1] == 40 and centers is not None:
            c_arr = np.asarray(centers)
            if states_arr.ndim == 1:
                states_arr = np.concatenate([states_arr, c_arr], axis=-1)
            else:
                c_expanded = np.broadcast_to(c_arr, (*states_arr.shape[:-1], 6))
                states_arr = np.concatenate([states_arr, c_expanded], axis=-1)

        ops = TorchCubeOps(use_stm=True, device="cpu")
        states_t = torch.tensor(states_arr, dtype=torch.long)
        orig_shape = states_t.shape
        states_2d = states_t.reshape(-1, orig_shape[-1])
        c_2d = states_2d[:, 40:46]
        face_indices = (c_2d == target_color).nonzero(as_tuple=True)[1]
        face_to_d = torch.tensor(FACE_TO_D_MOVE_INDICES, dtype=torch.long)
        if not (face_indices == 3).all():
            moves_to_d = face_to_d[face_indices]
            states_2d = ops.apply_moves(states_2d, moves_to_d)
        aligned = states_2d.reshape(orig_shape).numpy().astype(np.int8)
        c = aligned[..., 40:46]
        pieces = aligned[..., :40]
        h = np.sum(c * CENTER_BASE6_WEIGHTS, axis=-1)
        c_ids = CENTER_ID_LUT[h]
        target = SOLVED_TABLE_24_NP[c_ids]
    elif target_orientation is not None:
        c_ids = target_orientation
        target = SOLVED_TABLE_24_NP[target_orientation]
        pieces = states[..., :40] if states.shape[-1] >= 40 else states
        if states.shape[-1] >= 46:
            c = states[..., 40:46]
            h = np.sum(c * CENTER_BASE6_WEIGHTS, axis=-1)
            orient_match = CENTER_ID_LUT[h] == target_orientation
        elif centers is not None:
            c = np.asarray(centers)
            h = np.sum(c * CENTER_BASE6_WEIGHTS, axis=-1)
            orient_match = CENTER_ID_LUT[h] == target_orientation
    elif states.shape[-1] >= 46:
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
        c_ids = 0
        target = SOLVED_FLAT
        pieces = states

    target = _apply_oriented_piece_targets_np(pieces, target, c_ids, spec)

    if len(spec.masks) == 1:
        mask = spec.masks[0]
        res = ((pieces == target) | ~mask).all(axis=-1)
        return res & orient_match

    matches = [((pieces == target) | ~m).all(axis=-1) for m in spec.masks]
    return np.any(matches, axis=0) & orient_match


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
        self.register_buffer(
            "u_co_by_piece",
            torch.tensor(
                U_CO_BY_PIECE_24_NP, dtype=torch.long, device=self.device
            ),
        )
        self.register_buffer(
            "u_eo_by_piece",
            torch.tensor(
                U_EO_BY_PIECE_24_NP, dtype=torch.long, device=self.device
            ),
        )
        self.target_orientation: int | None = None
        self.target_color: int | None = None
        self.cube_ops = TorchCubeOps(use_stm=True, device=self.device)
        self.register_buffer(
            "face_to_d_moves",
            torch.tensor(
                FACE_TO_D_MOVE_INDICES, dtype=torch.long, device=self.device
            ),
        )

    def set_target_orientation(self, orientation_id: int | None) -> None:
        """Locks the target orientation to a fixed orientation ID in [0, 23]."""
        self.target_orientation = orientation_id

    def set_target_color(self, color: int | None) -> None:
        """Locks the frame of reference to a specific reference color on D."""
        self.target_color = color

    def align_states_to_colors(
        self, states: torch.Tensor, target_colors: torch.Tensor | int
    ) -> torch.Tensor:
        """Vectorized alignment of states so that target_colors is rotated to face D (index 3).

        Args:
            states: Tensor of shape (..., 46) or (B, A, 46).
            target_colors: Scalar int or Tensor of shape (B,) / (...,) matching leading dims.
        """
        if states.size(-1) < 46:
            return states

        orig_shape = states.shape
        states_2d = states.reshape(-1, orig_shape[-1])
        centers = states_2d[:, 40:46]

        if isinstance(target_colors, int):
            colors_2d = target_colors
        else:
            tc = target_colors.to(device=states.device, dtype=centers.dtype)
            if (
                tc.ndim == 1
                and states.ndim == 3
                and tc.size(0) == orig_shape[0]
            ):
                tc = tc.unsqueeze(1).expand(orig_shape[0], orig_shape[1])
            colors_2d = tc.reshape(-1, 1)

        if (centers[:, 3:4] == colors_2d).all():
            return states

        face_indices = (centers == colors_2d).to(torch.int8).argmax(dim=1)
        if (face_indices == 3).all():
            return states

        if self.cube_ops.device != states.device:
            self.cube_ops.to(states.device)
        moves_to_d = self.face_to_d_moves.to(states.device)[face_indices]
        aligned_2d = self.cube_ops.apply_moves(states_2d, moves_to_d)
        return aligned_2d.reshape(orig_shape)

    def align_to_d(self, states: torch.Tensor) -> torch.Tensor:
        """Vectorized alignment of states so that target_color is rotated to face D (index 3)."""
        if self.target_color is None or states.size(-1) < 46:
            return states
        return self.align_states_to_colors(states, self.target_color)

    def _apply_oriented_piece_targets(
        self,
        pieces: torch.Tensor,
        target: torch.Tensor,
        c_ids: torch.Tensor | int,
    ) -> torch.Tensor:
        if not (self.spec.oriented_corners or self.spec.oriented_edges):
            return target
        target = target.expand_as(pieces).clone()
        c_ids_t = (
            (
                torch.tensor(c_ids, dtype=torch.long, device=pieces.device)
                if isinstance(c_ids, int)
                else c_ids
            )
            .clamp(0, 23)
            .unsqueeze(-1)
        )
        if self.spec.oriented_corners:
            slots_c = list(self.spec.oriented_corners)
            cp_u = pieces[..., slots_c].clamp(0, 7)
            target[..., [8 + s for s in slots_c]] = self.u_co_by_piece[
                c_ids_t, cp_u
            ]
        if self.spec.oriented_edges:
            slots_e = list(self.spec.oriented_edges)
            ep_u = pieces[..., [16 + s for s in slots_e]].clamp(0, 11)
            target[..., [28 + s for s in slots_e]] = self.u_eo_by_piece[
                c_ids_t, ep_u
            ]
        return target

    def is_goal(self, states: torch.Tensor) -> torch.Tensor:
        """
        Returns boolean tensor indicating which states satisfy the stage goal.
        Works for states of shape (B, 40)/(B, 46) or (B, A, 40)/(B, A, 46).
        """
        states = states.to(device=self.device, dtype=torch.long)
        orient_match: torch.Tensor | bool = True

        if self.target_color is not None and states.size(-1) >= 46:
            aligned = self.align_to_d(states)
            c = aligned[..., 40:46]
            h = (c * self.weights).sum(dim=-1)
            c_ids = self.lut[h]
            target = self.solved_table[c_ids]
            pieces = aligned[..., :40]
        elif self.target_orientation is not None:
            c_ids = self.target_orientation
            target = self.solved_table[self.target_orientation]
            pieces = states[..., :40]
            if states.size(-1) >= 46:
                c = states[..., 40:46]
                h = (c * self.weights).sum(dim=-1)
                orient_match = self.lut[h] == self.target_orientation
        elif states.size(-1) >= 46:
            c = states[..., 40:46]
            h = (c * self.weights).sum(dim=-1)
            c_ids = self.lut[h]
            target = self.solved_table[c_ids]
            pieces = states[..., :40]
        else:
            c_ids = 0
            target = self.solved_table[0]
            pieces = states

        target = self._apply_oriented_piece_targets(pieces, target, c_ids)

        if self.num_masks == 1:
            res = ((pieces == target) | ~self.masks[0]).all(dim=-1)
            return res & orient_match

        # Broadcast states (..., 1, 40) against masks (num_masks, 40)
        expanded_pieces = pieces.unsqueeze(-2)
        expanded_target = (
            target.unsqueeze(-2) if target.ndim == pieces.ndim else target
        )
        matches = ((expanded_pieces == expanded_target) | ~self.masks).all(
            dim=-1
        )
        return matches.any(dim=-1) & orient_match
