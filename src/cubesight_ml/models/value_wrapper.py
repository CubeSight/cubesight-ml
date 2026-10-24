from typing import Any

import torch
import torch.nn as nn

from cubesight_ml.models.res_mlp import ResMLP


class ResMLPValueWrapper(nn.Module):
    """
    Wraps ResMLP backbone to return a dictionary with key 'value'
    matching the expected interface for DAVI and Beam Search.
    """

    def __init__(
        self, backbone: nn.Module, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config or {}

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        return {"value": self.backbone(x)}


def build_model(cfg: dict[str, Any] | None = None) -> ResMLPValueWrapper:
    """Factory function to construct ResMLPValueWrapper from config dict.

    Defaults to ResMLP: per_sticker, n1=700, n2=643, n_r=4, dropout=0.0.
    """
    if cfg is None:
        cfg = {}

    state_representation = cfg.get("state_representation", "per_sticker")
    n1 = int(cfg.get("n1", 700))
    n2 = int(cfg.get("n2", 643))
    n_r = int(cfg.get("n_r", 4))
    dropout_rate = float(cfg.get("dropout_rate", 0.0))
    output_dim = int(cfg.get("output_dim", 1))

    backbone = ResMLP(
        state_representation=state_representation,
        output_dim=output_dim,
        n1=n1,
        n2=n2,
        n_r=n_r,
        dropout_rate=dropout_rate,
    )

    clean_cfg = {
        "state_representation": state_representation,
        "n1": n1,
        "n2": n2,
        "n_r": n_r,
        "dropout_rate": dropout_rate,
        "output_dim": output_dim,
    }

    return ResMLPValueWrapper(backbone=backbone, config=clean_cfg)
