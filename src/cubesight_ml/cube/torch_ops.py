import numpy as np
import torch

from cubesight_ml.cube.moves import (
    MOVE_CENTERS_54,
    MOVE_CO_54,
    MOVE_CP_54,
    MOVE_EO_54,
    MOVE_EP_54,
    get_possible_moves,
)


class TorchCubeOps:
    """Vectorized PyTorch operations for expanding cube transitions directly on device."""

    def __init__(
        self,
        use_stm: bool = True,
        use_double_moves: bool = True,
        include_rotations: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.use_stm = use_stm
        self.use_double_moves = use_double_moves
        self.include_rotations = include_rotations

        if self.use_stm:
            raw_cp = MOVE_CP_54
            raw_co = MOVE_CO_54
            raw_ep = MOVE_EP_54
            raw_eo = MOVE_EO_54
            raw_centers = MOVE_CENTERS_54
        else:
            raw_cp = np.vstack([MOVE_CP_54[:18], np.arange(8, dtype=np.int32)])
            raw_co = np.vstack([MOVE_CO_54[:18], np.zeros(8, dtype=np.int8)])
            raw_ep = np.vstack([MOVE_EP_54[:18], np.arange(12, dtype=np.int32)])
            raw_eo = np.vstack([MOVE_EO_54[:18], np.zeros(12, dtype=np.int8)])
            raw_centers = np.vstack(
                [MOVE_CENTERS_54[:18], np.arange(6, dtype=np.int8)]
            )

        self.move_cp = torch.tensor(
            raw_cp, device=self.device, dtype=torch.int64
        )
        self.move_co = torch.tensor(
            raw_co, device=self.device, dtype=torch.int64
        )
        self.move_ep = torch.tensor(
            raw_ep, device=self.device, dtype=torch.int64
        )
        self.move_eo = torch.tensor(
            raw_eo, device=self.device, dtype=torch.int64
        )
        self.move_centers = torch.tensor(
            raw_centers, device=self.device, dtype=torch.int64
        )

        possible_moves = get_possible_moves(
            use_stm=use_stm,
            use_double_moves=use_double_moves,
            include_rotations=include_rotations,
        )
        self.moves = torch.tensor(
            possible_moves,
            dtype=torch.long,
            device=self.device,
        )

        self.num_moves = len(self.moves)
        self._cached_m_cp = self.move_cp[self.moves].unsqueeze(0)
        self._cached_m_co = self.move_co[self.moves].unsqueeze(0)
        self._cached_m_ep = self.move_ep[self.moves].unsqueeze(0)
        self._cached_m_eo = self.move_eo[self.moves].unsqueeze(0)
        self._cached_m_c = self.move_centers[self.moves].unsqueeze(0)

    def to(self, device: torch.device | str) -> "TorchCubeOps":
        """Moves operations to the target device."""
        self.device = torch.device(device)
        self.move_cp = self.move_cp.to(self.device)
        self.move_co = self.move_co.to(self.device)
        self.move_ep = self.move_ep.to(self.device)
        self.move_eo = self.move_eo.to(self.device)
        self.move_centers = self.move_centers.to(self.device)
        self.moves = self.moves.to(self.device)
        self._cached_m_cp = self._cached_m_cp.to(self.device)
        self._cached_m_co = self._cached_m_co.to(self.device)
        self._cached_m_ep = self._cached_m_ep.to(self.device)
        self._cached_m_eo = self._cached_m_eo.to(self.device)
        self._cached_m_c = self._cached_m_c.to(self.device)
        return self

    def expand(self, states: torch.Tensor) -> torch.Tensor:
        """
        Expands all valid 1-step child transitions for a batch of flat states.

        Args:
            states: Tensor of shape (batch_size, 40) or (batch_size, 46).

        Returns:
            Tensor of shape (batch_size, num_moves, state_size).
        """
        batch_size = states.size(0)
        num_moves = self.num_moves
        has_centers = states.size(1) >= 46

        cp = states[:, 0:8].unsqueeze(1).expand(-1, num_moves, -1)
        co = states[:, 8:16].unsqueeze(1).expand(-1, num_moves, -1)
        ep = states[:, 16:28].unsqueeze(1).expand(-1, num_moves, -1)
        eo = states[:, 28:40].unsqueeze(1).expand(-1, num_moves, -1)

        m_cp = self._cached_m_cp.expand(batch_size, -1, -1)
        m_co = self._cached_m_co.expand(batch_size, -1, -1)
        m_ep = self._cached_m_ep.expand(batch_size, -1, -1)
        m_eo = self._cached_m_eo.expand(batch_size, -1, -1)

        new_cp = torch.gather(cp, 2, m_cp)
        new_ep = torch.gather(ep, 2, m_ep)
        new_co = (torch.gather(co, 2, m_cp) + m_co) % 3
        new_eo = torch.bitwise_xor(torch.gather(eo, 2, m_ep), m_eo)

        if has_centers:
            centers = states[:, 40:46].unsqueeze(1).expand(-1, num_moves, -1)
            m_c = self._cached_m_c.expand(batch_size, -1, -1)
            new_centers = torch.gather(centers, 2, m_c)
            return torch.cat(
                (new_cp, new_co, new_ep, new_eo, new_centers), dim=2
            )

        return torch.cat((new_cp, new_co, new_ep, new_eo), dim=2)

    def apply_moves(
        self, states: torch.Tensor, moves: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies a batch of specific moves to a batch of flat states.

        Args:
            states: Tensor of shape (batch_size, 40) or (batch_size, 46).
            moves: 1D Tensor of shape (batch_size,) containing action indices.

        Returns:
            Tensor of shape (batch_size, state_size).
        """
        has_centers = states.size(1) >= 46
        cp, co = states[:, 0:8], states[:, 8:16]
        ep, eo = states[:, 16:28], states[:, 28:40]

        m_cp = self.move_cp[moves]
        m_co = self.move_co[moves]
        m_ep = self.move_ep[moves]
        m_eo = self.move_eo[moves]

        new_cp = torch.gather(cp, 1, m_cp)
        new_ep = torch.gather(ep, 1, m_ep)
        new_co = (torch.gather(co, 1, m_cp) + m_co) % 3
        new_eo = torch.bitwise_xor(torch.gather(eo, 1, m_ep), m_eo)

        if has_centers:
            centers = states[:, 40:46]
            m_c = self.move_centers[moves]
            new_centers = torch.gather(centers, 1, m_c)
            return torch.cat(
                (new_cp, new_co, new_ep, new_eo, new_centers), dim=1
            )

        return torch.cat((new_cp, new_co, new_ep, new_eo), dim=1)
