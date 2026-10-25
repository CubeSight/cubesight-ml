from cubesight_ml.datasets.reco_scraper import (
    RecoScraper,
    RecoSolve,
    RecoStep,
    classify_cfop_stage,
    load_reconstructions,
    normalize_cube_move,
    verify_solve_reconstruction,
)
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
    "RecoScraper",
    "RecoSolve",
    "RecoStep",
    "StageDAVIDataset",
    "classify_cfop_stage",
    "create_stage_davi_dataloader",
    "load_reconstructions",
    "normalize_cube_move",
    "verify_solve_reconstruction",
]
