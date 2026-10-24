from collections.abc import Generator
from typing import override

import numpy as np

from cubesight_ml.cube.moves import (
    INVERSE_MOVES_54,
    MOVE_CENTERS_54,
    MOVE_CO_54,
    MOVE_CP_54,
    MOVE_EO_54,
    MOVE_EP_54,
    NOOP_INDEX,
)
from cubesight_ml.envs.base import Env


class RubiksCubeEnv(Env):
    # Face moves (0..17)
    U, U2, U_PRIME = 0, 1, 2
    D, D2, D_PRIME = 3, 4, 5
    L, L2, L_PRIME = 6, 7, 8
    R, R2, R_PRIME = 9, 10, 11
    F, F2, F_PRIME = 12, 13, 14
    B, B2, B_PRIME = 15, 16, 17

    # Slices (18..26)
    M, M2, M_PRIME = 18, 19, 20
    E, E2, E_PRIME = 21, 22, 23
    S, S2, S_PRIME = 24, 25, 26

    # Wide moves (27..44)
    U_WIDE, U_WIDE2, U_WIDE_PRIME = 27, 28, 29
    D_WIDE, D_WIDE2, D_WIDE_PRIME = 30, 31, 32
    L_WIDE, L_WIDE2, L_WIDE_PRIME = 33, 34, 35
    R_WIDE, R_WIDE2, R_WIDE_PRIME = 36, 37, 38
    F_WIDE, F_WIDE2, F_WIDE_PRIME = 39, 40, 41
    B_WIDE, B_WIDE2, B_WIDE_PRIME = 42, 43, 44

    # Rotations (45..53)
    X, X2, X_PRIME = 45, 46, 47
    Y, Y2, Y_PRIME = 48, 49, 50
    Z, Z2, Z_PRIME = 51, 52, 53

    NOOP = 54

    def __init__(
        self,
        batch_size: int,
        use_stm: bool = True,
        use_double_moves: bool = True,
        include_rotations: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.num_faces = 6
        self.use_stm = use_stm
        self.use_double_moves = use_double_moves
        self.include_rotations = include_rotations
        self.rng = rng

        if self.use_stm:
            self.action_space = list(
                range(54 if self.include_rotations else 45)
            )
            self.NOOP = NOOP_INDEX
        else:
            self.action_space = list(range(18 if self.use_double_moves else 12))
            self.NOOP = 18

        # Allocate state tensors
        self.cp = np.empty((batch_size, 8), dtype=np.int8)
        self.co = np.empty((batch_size, 8), dtype=np.int8)
        self.ep = np.empty((batch_size, 12), dtype=np.int8)
        self.eo = np.empty((batch_size, 12), dtype=np.int8)
        self.centers = np.empty((batch_size, 6), dtype=np.int8)

        # Allocate temporary state buffers
        self._temp_cp = np.empty((batch_size, 8), dtype=np.int8)
        self._temp_co = np.empty((batch_size, 8), dtype=np.int8)
        self._temp_ep = np.empty((batch_size, 12), dtype=np.int8)
        self._temp_eo = np.empty((batch_size, 12), dtype=np.int8)
        self._temp_centers = np.empty((batch_size, 6), dtype=np.int8)

        self._idx_buf_8 = np.empty((batch_size, 8), dtype=np.int32)
        self._idx_buf_12 = np.empty((batch_size, 12), dtype=np.int32)
        self._idx_buf_6 = np.empty((batch_size, 6), dtype=np.int32)

        # Calculate memory offsets, as np.take only supports 1D indexing
        self._batch_offsets_8 = (np.arange(batch_size, dtype=np.int32) * 8)[
            :, None
        ]
        self._batch_offsets_12 = (np.arange(batch_size, dtype=np.int32) * 12)[
            :, None
        ]
        self._batch_offsets_6 = (np.arange(batch_size, dtype=np.int32) * 6)[
            :, None
        ]

        self._init_move_tables()
        self._init_scramble_tables()
        self.reset()

    def _init_move_tables(self) -> None:
        if self.use_stm:
            self._move_cp = MOVE_CP_54
            self._move_co = MOVE_CO_54
            self._move_ep = MOVE_EP_54
            self._move_eo = MOVE_EO_54
            self._move_centers = MOVE_CENTERS_54
        else:
            # 18 face moves + NOOP 18
            self._move_cp = np.vstack(
                [MOVE_CP_54[:18], np.arange(8, dtype=np.int32)]
            )
            self._move_co = np.vstack(
                [MOVE_CO_54[:18], np.zeros(8, dtype=np.int8)]
            )
            self._move_ep = np.vstack(
                [MOVE_EP_54[:18], np.arange(12, dtype=np.int32)]
            )
            self._move_eo = np.vstack(
                [MOVE_EO_54[:18], np.zeros(12, dtype=np.int8)]
            )
            self._move_centers = np.vstack(
                [MOVE_CENTERS_54[:18], np.arange(6, dtype=np.int8)]
            )

    def _init_scramble_tables(self) -> None:
        self._possible_moves = np.array(
            self.get_possible_moves(), dtype=np.int32
        )
        self.n_gens = len(self._possible_moves)

        num_actions = len(self._move_cp)
        self.inverse_moves = self.get_inverse_actions(
            np.arange(num_actions, dtype=np.int32)
        )

        self._action_to_inv_idx = np.empty(num_actions, dtype=np.int32)
        move_to_idx = {
            int(move): idx for idx, move in enumerate(self._possible_moves)
        }
        for move in self._possible_moves:
            inv_move = int(self.inverse_moves[move])
            self._action_to_inv_idx[move] = move_to_idx[inv_move]

        self._generator_inv_idx = np.array(
            [self._action_to_inv_idx[m] for m in self._possible_moves],
            dtype=np.int32,
        )

    def _randint(
        self, low: int, high: int, size: int | tuple[int, ...]
    ) -> np.ndarray:
        if self.rng is not None:
            return self.rng.integers(low, high, size=size, dtype=np.int32)
        return np.random.randint(low, high, size=size, dtype=np.int32)

    def reset(self) -> None:
        """Reset all cubes to the solved state."""
        self.cp[:] = np.arange(8, dtype=np.int8)
        self.ep[:] = np.arange(12, dtype=np.int8)
        self.co.fill(0)
        self.eo.fill(0)
        self.centers[:] = np.arange(6, dtype=np.int8)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute a batch of actions.
        actions: 1D array of shape (batch_size,) containing action indices.
        """
        local_cp = self._move_cp[actions]
        local_ep = self._move_ep[actions]
        local_co = self._move_co[actions]
        local_eo = self._move_eo[actions]
        local_centers = self._move_centers[actions]

        # Map the index to the correct cube's memory block
        np.add(self._batch_offsets_8, local_cp, out=self._idx_buf_8)
        np.add(self._batch_offsets_12, local_ep, out=self._idx_buf_12)
        np.add(self._batch_offsets_6, local_centers, out=self._idx_buf_6)

        flat_idx_8 = self._idx_buf_8.ravel()
        flat_idx_12 = self._idx_buf_12.ravel()
        flat_idx_6 = self._idx_buf_6.ravel()

        # Update permutations
        np.take(self.cp.ravel(), flat_idx_8, out=self._temp_cp.ravel())
        np.take(self.ep.ravel(), flat_idx_12, out=self._temp_ep.ravel())
        np.take(
            self.centers.ravel(), flat_idx_6, out=self._temp_centers.ravel()
        )
        self.cp[:] = self._temp_cp
        self.ep[:] = self._temp_ep
        self.centers[:] = self._temp_centers

        # Update orientations
        np.take(self.co.ravel(), flat_idx_8, out=self._temp_co.ravel())
        np.take(self.eo.ravel(), flat_idx_12, out=self._temp_eo.ravel())

        np.add(self._temp_co, local_co, out=self.co)
        np.remainder(self.co, 3, out=self.co)

        np.bitwise_xor(self._temp_eo, local_eo, out=self.eo)

        return self.cp, self.co, self.ep, self.eo

    def get_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.cp, self.co, self.ep, self.eo

    def get_state_with_centers(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.cp, self.co, self.ep, self.eo, self.centers

    def set_state(
        self,
        cp: np.ndarray,
        co: np.ndarray,
        ep: np.ndarray,
        eo: np.ndarray,
        centers: np.ndarray | None = None,
    ) -> None:
        self.cp = cp
        self.co = co
        self.ep = ep
        self.eo = eo
        if centers is not None:
            self.centers = centers

    def load_state(
        self,
        cp: np.ndarray,
        co: np.ndarray,
        ep: np.ndarray,
        eo: np.ndarray,
        centers: np.ndarray | None = None,
    ) -> None:
        """Copies external state arrays into internal buffers."""
        self.cp[:] = cp
        self.co[:] = co
        self.ep[:] = ep
        self.eo[:] = eo
        if centers is not None:
            self.centers[:] = centers
        else:
            self.centers[:] = np.arange(6, dtype=np.int8)

    def scramble(
        self, depth: int = 26, depths: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Scramble the batch of cubes by performing random steps while
        avoiding immediate inverse moves.
        If depths (shape: batch_size) is supplied, each cube is scrambled
        to its own specified depth using NOOP steps.
        """
        actions_seq = self._get_scramble_actions(depth, depths=depths)

        for d in range(depth):
            self.step(actions_seq[d])

        return self.get_state()

    def scramble_yield(
        self, depth: int = 26
    ) -> Generator[
        tuple[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            np.ndarray | None,
        ]
    ]:
        """
        Yield states step-by-step while scrambling, avoiding immediate
        inverse moves.
        """
        actions_seq = self._get_scramble_actions(depth)

        yield self.get_state(), None
        for d in range(depth):
            self.step(actions_seq[d])
            yield self.get_state(), actions_seq[d]

    def do_random_step(
        self,
        states: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        | np.ndarray
        | None = None,
        last_moves: np.ndarray | None = None,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]:
        """Perform a random step while avoiding immediate inverse moves."""
        if (
            last_moves is None
            and isinstance(states, np.ndarray)
            and states.ndim == 1
        ):
            last_moves = states
            states = None
        elif states is not None:
            self.set_state(*states)

        if last_moves is not None and hasattr(last_moves, "cpu"):
            last_moves = last_moves.detach().cpu().numpy()

        if last_moves is None:
            next_idx = self._randint(0, self.n_gens, size=self.batch_size)
        else:
            last_moves = np.asarray(last_moves, dtype=np.int32)
            if np.all(np.isin(last_moves, self._possible_moves)):
                inv_idx = self._action_to_inv_idx[last_moves]
            elif np.all((last_moves >= 0) & (last_moves < self.n_gens)):
                inv_idx = self._generator_inv_idx[last_moves]
            else:
                inv_idx = self._action_to_inv_idx[last_moves]

            k = self._randint(1, self.n_gens, size=self.batch_size)
            next_idx = (inv_idx + k) % self.n_gens

        next_moves = self._possible_moves[next_idx]
        new_states = self.step(next_moves)
        return new_states, next_moves

    def get_inverse_actions(self, action_ids: np.ndarray) -> np.ndarray:
        if self.use_stm:
            return INVERSE_MOVES_54[action_ids]
        faces = action_ids // 3
        turns = action_ids % 3
        return faces * 3 + (2 - turns)

    def map_12_to_18(self, action_12: int) -> int:
        face = action_12 // 2
        turn = (action_12 % 2) * 2
        return face * 3 + turn

    @override
    def get_possible_moves(self) -> list[int]:
        if self.use_stm:
            moves = []
            for idx in range(54):
                is_rot = idx >= 45
                is_double = (idx % 3) == 1
                if is_rot and not self.include_rotations:
                    continue
                if is_double and not self.use_double_moves:
                    continue
                moves.append(idx)
            return moves
        if self.use_double_moves:
            return list(range(18))
        return [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17]

    def get_all_child_states(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies all possible valid moves to the current batch of cubes to generate
        all child states, then reverts the environment back to the parent states.

        Returns:
            tuple of (child_cp, child_co, child_ep, child_eo).
            (batch_size, num_actions, num_pieces).
        """
        possible_moves = self.get_possible_moves()
        num_moves = len(possible_moves)

        child_cp = np.empty((num_moves, self.batch_size, 8), dtype=np.int8)
        child_co = np.empty((num_moves, self.batch_size, 8), dtype=np.int8)
        child_ep = np.empty((num_moves, self.batch_size, 12), dtype=np.int8)
        child_eo = np.empty((num_moves, self.batch_size, 12), dtype=np.int8)

        for i, action in enumerate(possible_moves):
            action_array = np.full(self.batch_size, action, dtype=np.int32)

            c_cp, c_co, c_ep, c_eo = self.step(action_array)

            child_cp[i] = c_cp
            child_co[i] = c_co
            child_ep[i] = c_ep
            child_eo[i] = c_eo

            inverse_action_array = self.get_inverse_actions(action_array)
            self.step(inverse_action_array)

        return (
            np.swapaxes(child_cp, 0, 1),
            np.swapaxes(child_co, 0, 1),
            np.swapaxes(child_ep, 0, 1),
            np.swapaxes(child_eo, 0, 1),
        )

    def set_rng(self, rng: np.random.Generator | None) -> None:
        """Set a dedicated NumPy Generator for deterministic scramble generation."""
        self.rng = rng

    def _get_scramble_actions(
        self, depth: int, depths: np.ndarray | None = None
    ) -> np.ndarray:
        if depth <= 0:
            return np.empty((0, self.batch_size), dtype=np.int32)

        actions_seq = np.empty((depth, self.batch_size), dtype=np.int32)

        init_idx = self._randint(0, self.n_gens, size=self.batch_size)
        actions_seq[0] = self._possible_moves[init_idx]
        if depths is not None:
            actions_seq[0, depths <= 0] = self.NOOP

        for d in range(1, depth):
            prev = actions_seq[d - 1]
            is_noop = prev == self.NOOP
            safe_prev = np.where(is_noop, 0, prev)

            inv_idx = self._action_to_inv_idx[safe_prev]
            k = self._randint(1, self.n_gens, size=self.batch_size)
            next_idx = np.where(
                is_noop,
                self._randint(0, self.n_gens, size=self.batch_size),
                (inv_idx + k) % self.n_gens,
            )
            actions_seq[d] = self._possible_moves[next_idx]

            if depths is not None:
                actions_seq[d, depths <= d] = self.NOOP

        return actions_seq
