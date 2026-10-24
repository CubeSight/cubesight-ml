import numpy as np
import torch
import torch.nn as nn

from cubesight_ml.cube.layout import SOLVED_FLAT
from cubesight_ml.stages.definitions import CROSS_FACE_MASKS, StageSpec


def is_goal_np(states: np.ndarray, spec: StageSpec) -> np.ndarray:
    """
    Checks if flat cube state(s) satisfy the stage goal.
    Supports 1D state (40,) or batched 2D state (B, 40) or (B, A, 40).
    """
    if len(spec.masks) == 1:
        mask = spec.masks[0]
        return ((states == SOLVED_FLAT) | ~mask).all(axis=-1)

    matches = [((states == SOLVED_FLAT) | ~m).all(axis=-1) for m in spec.masks]
    return np.any(matches, axis=0)


def detect_solved_cross_face(state: np.ndarray) -> str:
    """
    Returns which cross face ('D', 'U', 'F', 'B', 'L', or 'R') is solved.
    If 'D' is among the solved faces, prefers 'D' to avoid unnecessary reorientation.
    """
    # If D is solved, prefer D to avoid rotating
    if ((state == SOLVED_FLAT) | ~CROSS_FACE_MASKS["D"]).all():
        return "D"

    for face in ("U", "F", "B", "L", "R"):
        if ((state == SOLVED_FLAT) | ~CROSS_FACE_MASKS[face]).all():
            return face

    raise ValueError("No cross face is solved in the given state.")


class StageGoalChecker(nn.Module):
    """PyTorch goal checker module for a stage, compatible with arbitrary leading batch dimensions."""

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
        solved_tensor = torch.tensor(
            SOLVED_FLAT, dtype=torch.long, device=self.device
        )

        self.register_buffer("masks", masks_tensor)
        self.register_buffer("solved_flat", solved_tensor)

    def is_goal(self, states: torch.Tensor) -> torch.Tensor:
        """
        Returns boolean tensor indicating which states satisfy the stage goal.
        Works for states of shape (B, 40) or (B, A, 40).
        """
        states = states.to(device=self.device, dtype=torch.long)

        if self.num_masks == 1:
            return ((states == self.solved_flat) | ~self.masks[0]).all(dim=-1)

        # Broadcast states (..., 1, 40) against masks (num_masks, 40)
        expanded_states = states.unsqueeze(-2)
        matches = ((expanded_states == self.solved_flat) | ~self.masks).all(
            dim=-1
        )
        return matches.any(dim=-1)
