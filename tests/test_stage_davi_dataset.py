import pytest
import torch

from cubesight_ml.datasets.rubiks_cube.stage_davi_dataset import (
    StageDAVIDataset,
    create_stage_davi_dataloader,
)
from cubesight_ml.stages.definitions import STAGES, Stage
from cubesight_ml.stages.goal_checker import is_goal_np


@pytest.mark.parametrize("stage", ["cross", "f2l", "oll", "pll"])
def test_stage_davi_dataset_shapes_and_dtypes(stage: str):
    batch_size = 20
    scramble_depth = 5
    dataset = StageDAVIDataset(
        stage=stage,
        batch_size=batch_size,
        scramble_depth=scramble_depth,
        depth_sampling="stratified",
        precondition_fraction=0.0,
        data_seed=42,
    )
    iterator = iter(dataset)
    parents, depths = next(iterator)

    assert isinstance(parents, torch.Tensor)
    assert isinstance(depths, torch.Tensor)
    assert parents.shape == (batch_size, 46)
    assert depths.shape == (batch_size,)
    assert parents.dtype == torch.int8
    assert depths.dtype == torch.int16

    # All depths should be between 1 and scramble_depth
    assert (depths >= 1).all()
    assert (depths <= scramble_depth).all()


def test_stage_davi_dataset_include_depth_zero():
    batch_size = 30
    scramble_depth = 4
    dataset = StageDAVIDataset(
        stage="cross",
        batch_size=batch_size,
        scramble_depth=scramble_depth,
        include_depth_zero=True,
        depth_sampling="stratified",
        data_seed=123,
    )
    _parents, depths = next(iter(dataset))
    assert (depths >= 0).all()
    # With depth 0 included and stratified, at least some depths should be 0
    assert (depths == 0).any()


def test_stage_davi_dataset_precondition_mixing():
    batch_size = 100
    scramble_depth = 6
    precond_fraction = 0.35
    dataset = StageDAVIDataset(
        stage="f2l",
        batch_size=batch_size,
        scramble_depth=scramble_depth,
        precondition_fraction=precond_fraction,
        data_seed=456,
    )
    parents, depths = next(iter(dataset))
    precond_mask = depths == -1
    num_precond = precond_mask.sum().item()

    # Expected count is round(0.35 * 100) = 35
    assert num_precond == 35

    # For F2L, precondition states must have Cross solved on their D face
    precond_parents_np = parents[precond_mask].numpy()
    assert is_goal_np(precond_parents_np, STAGES[Stage.CROSS]).all()


def test_stage_davi_dataloader_num_workers():
    batch_size = 16
    scramble_depth = 4
    loader = create_stage_davi_dataloader(
        stage="oll",
        batch_size=batch_size,
        scramble_depth=scramble_depth,
        num_workers=2,
        precondition_fraction=0.25,
        data_seed=789,
        dataloader_seed=101,
    )
    loader_iter = iter(loader)
    for _ in range(3):
        parents, depths = next(loader_iter)
        assert parents.shape == (batch_size, 46)
        assert depths.shape == (batch_size,)
