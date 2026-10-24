from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.models.value_wrapper import build_model
from cubesight_ml.stages.definitions import CROSS_SPEC
from cubesight_ml.stages.goal_checker import StageGoalChecker
from cubesight_ml.training.davi import train_davi
from cubesight_ml.training.target_sync import DeepCubeASyncPolicy


class DummyDataset(torch.utils.data.IterableDataset):
    """Infinite dataset yielding dummy (parents, depths) batches."""

    def __iter__(self):
        while True:
            parents = torch.zeros(4, 46, dtype=torch.long)
            depths = torch.zeros(4, dtype=torch.long)
            yield parents, depths


def test_train_davi_with_move_penalties(tmp_path: Path):
    """Tests that train_davi runs end-to-end with ergonomic penalties."""
    device = "cpu"
    model_cfg = {
        "state_representation": "per_piece",
        "n1": 32,
        "n2": 32,
        "n_r": 1,
        "dropout_rate": 0.0,
        "output_dim": 1,
    }
    active_model = build_model(model_cfg).to(device)
    target_model = deepcopy(active_model).to(device)

    dataloader = DataLoader(DummyDataset(), batch_size=None)
    optimizer = torch.optim.Adam(active_model.parameters(), lr=1e-3)
    spec = CROSS_SPEC
    cube_ops = TorchCubeOps(use_double_moves=False, device=device)
    goal_checker = StageGoalChecker(spec, device=device)
    sync_policy = DeepCubeASyncPolicy(sync_interval=1, loss_threshold=100.0)

    train_davi(
        active_model=active_model,
        target_model=target_model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=None,
        spec=spec,
        cube_ops=cube_ops,
        goal_checker=goal_checker,
        sync_policy=sync_policy,
        steps=2,
        save_dir=tmp_path / "run",
        batch_size=4,
        stage="cross",
        model_config=model_cfg,
        move_penalties={"B": 5.0, "z": 10.0},
        loss_interval=1,
        save_interval=1,
        eval_interval=1,
        eval_n_states=2,
        eval_beam_width=2,
        eval_max_steps=2,
        device=device,
        warmup_steps=1,
    )

    # Check checkpoints created
    run_dir = tmp_path / "run"
    assert (run_dir / "best_model.pt").is_file()
    assert (run_dir / "best_eval_model.pt").is_file()
    assert (run_dir / "final_model.pt").is_file()
