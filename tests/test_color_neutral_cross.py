import numpy as np
import torch

from cubesight_ml.cube.layout import SOLVED_FLAT
from cubesight_ml.cube.moves import MOVE_TO_INDEX
from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.search.beam_search.beam_search import BeamSearch
from cubesight_ml.search.beam_search.utils import BeamSearchRubiksCubeHandler
from cubesight_ml.stages.definitions import CROSS_SPEC
from cubesight_ml.stages.goal_checker import StageGoalChecker
from cubesight_ml.visualization.backend import DistanceMockModel


def test_color_neutral_cross_solving_non_yellow():
    """Verify that Beam Search solves whichever cross is easiest on D, including non-yellow."""
    device = "cpu"
    checker = StageGoalChecker(CROSS_SPEC, device=device)
    ops = TorchCubeOps(use_stm=True, device=device)
    handler = BeamSearchRubiksCubeHandler(checker, use_stm=True, device=device)
    model = DistanceMockModel(CROSS_SPEC)
    searcher = BeamSearch(model, handler, use_stm=True, device=device)

    # Scramble all 6 faces on canonical cube
    cube = torch.tensor(
        np.concatenate([SOLVED_FLAT, np.arange(6, dtype=np.int8)]),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    for m in ["R", "U", "B", "L", "F", "D"]:
        cube = ops.apply_moves(
            cube, torch.tensor([MOVE_TO_INDEX[m]], dtype=torch.long)
        )

    sol, _, _ = searcher.find_solution(cube, beam_width=50, num_steps=10)
    assert sol is not None

    final_state = cube
    for i in sol:
        final_state = ops.apply_moves(
            final_state, torch.tensor([i], dtype=torch.long)
        )

    d_color = final_state[0, 43].item()
    # Cross solved on D is verified by goal checker and is non-yellow (Blue=5)
    assert handler.is_solved(final_state).item() is True
    assert d_color != 3, f"Expected non-yellow cross, but got {d_color}"
