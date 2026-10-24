import torch
import torch.nn as nn
import torch.nn.functional as F


class RubiksStateConverter(nn.Module):
    """
    Transforms the [batch, 40] piece-oriented tensor into a [batch, 54]
    sticker-oriented tensor.
    Colors map to standard U=0, R=1, F=2, D=3, L=4, B=5 notation.
    """

    def __init__(self):
        super().__init__()
        # 8 corners x 3 faces
        self.register_buffer(
            "corner_colors",
            torch.tensor(
                [
                    [0, 1, 2],
                    [0, 2, 4],
                    [0, 4, 5],
                    [0, 5, 1],
                    [3, 2, 1],
                    [3, 4, 2],
                    [3, 5, 4],
                    [3, 1, 5],
                ],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "corner_rot",
            torch.tensor([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=torch.long),
        )
        self.register_buffer(
            "edge_colors",
            torch.tensor(
                [
                    [0, 2],
                    [0, 4],
                    [0, 5],
                    [0, 1],
                    [3, 2],
                    [3, 4],
                    [3, 5],
                    [3, 1],
                    [1, 2],
                    [4, 2],
                    [4, 5],
                    [1, 5],
                ],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "edge_rot", torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        )
        self.register_buffer(
            "centers", torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 54:
            return x.long()

        B = x.size(0)
        stickers = torch.empty(B, 54, dtype=torch.long, device=x.device)

        cp = x[:, 0:8].long()
        co = x[:, 8:16].long()
        ep = x[:, 16:28].long()
        eo = x[:, 28:40].long()

        if x.size(1) >= 46:
            stickers[:, 0:6] = x[:, 40:46].long()
        else:
            stickers[:, 0:6] = self.centers.unsqueeze(0).expand(B, -1)

        c_colors = self.corner_colors[cp]
        c_rot = self.corner_rot[co]
        stickers[:, 6:30] = torch.gather(c_colors, 2, c_rot).view(B, 24)

        e_colors = self.edge_colors[ep]
        e_rot = self.edge_rot[eo]
        stickers[:, 30:54] = torch.gather(e_colors, 2, e_rot).view(B, 24)

        return stickers


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual

        return self.relu(out)


class ResMLP(nn.Module):
    def __init__(
        self,
        state_representation: str,
        output_dim: int,
        n1: int,
        n2: int,
        n_r: int,
        dropout_rate: float,
    ):
        super().__init__()

        self.state_representation = state_representation

        if self.state_representation == "per_piece":
            self.input_dim = 8 * 8 + 8 * 3 + 12 * 12 + 12 * 2 + 6 * 6  # 292
        elif self.state_representation == "per_sticker":
            self.input_dim = 54 * 6  # 324
            self.converter = RubiksStateConverter()
        else:
            raise ValueError(
                "Invalid state_representation. "
                "Choose 'per_piece' or 'per_sticker'."
            )

        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, n1),
            nn.BatchNorm1d(n1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n1, n2),
            nn.BatchNorm1d(n2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(n2, dropout_rate) for _ in range(n_r)]
        )
        self.output_layer = nn.Linear(n2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.state_representation == "per_piece":
            cp = (
                F.one_hot(x[:, 0:8].long(), num_classes=8)
                .to(torch.float32)
                .flatten(start_dim=1)
            )
            co = (
                F.one_hot(x[:, 8:16].long(), num_classes=3)
                .to(torch.float32)
                .flatten(start_dim=1)
            )
            ep = (
                F.one_hot(x[:, 16:28].long(), num_classes=12)
                .to(torch.float32)
                .flatten(start_dim=1)
            )
            eo = (
                F.one_hot(x[:, 28:40].long(), num_classes=2)
                .to(torch.float32)
                .flatten(start_dim=1)
            )
            if x.size(1) >= 46:
                centers = (
                    F.one_hot(x[:, 40:46].long(), num_classes=6)
                    .to(torch.float32)
                    .flatten(start_dim=1)
                )
            else:
                can = (
                    torch.arange(6, dtype=torch.long, device=x.device)
                    .unsqueeze(0)
                    .expand(x.size(0), -1)
                )
                centers = (
                    F.one_hot(can, num_classes=6)
                    .to(torch.float32)
                    .flatten(start_dim=1)
                )

            x = torch.cat([cp, co, ep, eo, centers], dim=1)

        elif self.state_representation == "per_sticker":
            stickers = self.converter(x)
            x = (
                F.one_hot(stickers, num_classes=6)
                .to(torch.float32)
                .flatten(start_dim=1)
            )

        x = self.layer1(x)
        x = self.layer2(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)
