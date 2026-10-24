from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

import numpy as np

from cubesight_ml.cube.layout import CP, EO, EP, STATE_SIZE

# Faces: U=0, R=1, F=2, D=3, L=4, B=5
# Edge indices:
# 0: UF, 1: UL, 2: UB, 3: UR
# 4: DF, 5: DL, 6: DB, 7: DR
# 8: FR, 9: FL, 10: BL, 11: BR
CROSS_FACE_EDGES: dict[str, tuple[int, ...]] = {
    "D": (4, 5, 6, 7),
    "U": (0, 1, 2, 3),
    "F": (0, 4, 8, 9),
    "B": (2, 6, 10, 11),
    "L": (1, 5, 9, 10),
    "R": (3, 7, 8, 11),
}


def _make_cross_mask(edges: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(STATE_SIZE, dtype=bool)
    for e in edges:
        mask[EP.start + e] = True
        mask[EO.start + e] = True
    return mask


CROSS_FACE_MASKS: dict[str, np.ndarray] = {
    face: _make_cross_mask(edges) for face, edges in CROSS_FACE_EDGES.items()
}


class Stage(StrEnum):
    CROSS = "cross"  # Nothing -> Color-Neutral Cross (any of 6 faces)
    F2L = "f2l"  # Cross   -> F2L (D-cross + 4 F2L slots)
    OLL = "oll"  # F2L     -> OLL (F2L + U face oriented)
    PLL = "pll"  # OLL     -> Solved


@dataclass(frozen=True)
class StageSpec:
    stage: Stage
    solved_corners: tuple[int, ...] = ()
    solved_edges: tuple[int, ...] = ()
    oriented_corners: tuple[int, ...] = ()
    oriented_edges: tuple[int, ...] = ()
    precondition: Stage | None = None
    custom_masks: tuple[np.ndarray, ...] | None = None

    @cached_property
    def masks(self) -> tuple[np.ndarray, ...]:
        if self.custom_masks is not None:
            return self.custom_masks

        mask = np.zeros(STATE_SIZE, dtype=bool)
        for c in self.solved_corners:
            mask[CP.start + c] = True
            mask[8 + c] = True  # CO
        for e in self.solved_edges:
            mask[EP.start + e] = True
            mask[EO.start + e] = True

        for c in self.oriented_corners:
            mask[8 + c] = True
        for e in self.oriented_edges:
            mask[EO.start + e] = True

        return (mask,)


# Specifications for each stage
CROSS_SPEC = StageSpec(
    stage=Stage.CROSS,
    custom_masks=tuple(
        CROSS_FACE_MASKS[f] for f in ("D", "U", "F", "B", "L", "R")
    ),
    precondition=None,
)

F2L_SPEC = StageSpec(
    stage=Stage.F2L,
    solved_corners=(4, 5, 6, 7),
    solved_edges=(4, 5, 6, 7, 8, 9, 10, 11),
    precondition=Stage.CROSS,
)

OLL_SPEC = StageSpec(
    stage=Stage.OLL,
    solved_corners=(4, 5, 6, 7),
    solved_edges=(4, 5, 6, 7, 8, 9, 10, 11),
    oriented_corners=(0, 1, 2, 3),
    oriented_edges=(0, 1, 2, 3),
    precondition=Stage.F2L,
)

PLL_SPEC = StageSpec(
    stage=Stage.PLL,
    solved_corners=(0, 1, 2, 3, 4, 5, 6, 7),
    solved_edges=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    precondition=Stage.OLL,
)

STAGES: dict[Stage, StageSpec] = {
    Stage.CROSS: CROSS_SPEC,
    Stage.F2L: F2L_SPEC,
    Stage.OLL: OLL_SPEC,
    Stage.PLL: PLL_SPEC,
}

STAGE_ORDER: list[Stage] = [Stage.CROSS, Stage.F2L, Stage.OLL, Stage.PLL]
