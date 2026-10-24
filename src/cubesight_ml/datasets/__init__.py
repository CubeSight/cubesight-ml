from cubesight_ml.datasets.rubiks_cube.stage_davi_dataset import (
    StageDAVIDataset,
    create_stage_davi_dataloader,
)

STAGE_DATALOADER_REGISTRY = {
    "RubiksCubeStage": {
        "loader_davi": create_stage_davi_dataloader,
        "run_dir": "rubiks_cube",
    }
}

__all__ = [
    "STAGE_DATALOADER_REGISTRY",
    "StageDAVIDataset",
    "create_stage_davi_dataloader",
]
