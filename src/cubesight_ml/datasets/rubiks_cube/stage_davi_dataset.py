import math
from collections.abc import Iterator
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from cubesight_ml.cube.layout import from_flat, to_flat
from cubesight_ml.envs.rubiks_cube import RubiksCubeEnv
from cubesight_ml.stages.definitions import STAGES, Stage
from cubesight_ml.stages.goal_sampler import sample_goal_states
from cubesight_ml.training.seeding import make_worker_init_fn


class StageDAVIDataset(IterableDataset):
    """
    Iterable PyTorch dataset for DeepCubeA Value Iteration (DAVI) on sequential CFOP stages.
    Yields batches of parents: Tensor (B, 40) int8, and depths: Tensor (B,) int16.
    """

    def __init__(
        self,
        stage: str | Stage,
        batch_size: int,
        scramble_depth: int,
        depth_sampling: Literal["uniform", "stratified"] = "stratified",
        include_depth_zero: bool = False,
        precondition_fraction: float = 0.0,
        avoid_inverse_moves: bool = True,
        use_double_moves: bool = False,
        data_seed: int | None = None,
        dataloader_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.stage_enum = Stage(stage)
        self.spec = STAGES[self.stage_enum]
        self.batch_size = batch_size
        self.scramble_depth = scramble_depth
        self.depth_sampling = depth_sampling
        self.include_depth_zero = include_depth_zero
        self.precondition_fraction = precondition_fraction
        self.avoid_inverse_moves = avoid_inverse_moves
        self.use_double_moves = use_double_moves
        self.data_seed = data_seed
        self.dataloader_seed = dataloader_seed

        if not (0.0 <= precondition_fraction <= 1.0):
            raise ValueError(
                f"precondition_fraction must be in [0.0, 1.0], got {precondition_fraction}"
            )

        self.n_precond = round(self.precondition_fraction * self.batch_size)
        self.n_scramble = self.batch_size - self.n_precond

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if self.data_seed is not None:
            worker_seed = (self.data_seed + worker_id * 10007) % (2**31)
            worker_rng = np.random.default_rng(worker_seed)
        elif worker_info is not None:
            worker_rng = np.random.default_rng(worker_info.seed % (2**32))
        else:
            worker_rng = np.random.default_rng()

        min_k = 0 if self.include_depth_zero else 1
        num_k = self.scramble_depth - min_k + 1

        env = (
            RubiksCubeEnv(
                batch_size=self.n_scramble,
                use_double_moves=self.use_double_moves,
                rng=worker_rng,
            )
            if self.n_scramble > 0
            else None
        )

        while True:
            # 1. Scrambled goal states
            if self.n_scramble > 0 and env is not None:
                # Sample goal states for the active stage
                goal_states = sample_goal_states(
                    self.spec, self.n_scramble, rng=worker_rng
                )
                cp, co, ep, eo = from_flat(goal_states)
                env.load_state(cp, co, ep, eo)

                # Generate per-sample scramble depths
                if self.depth_sampling == "stratified":
                    reps = math.ceil(self.n_scramble / num_k)
                    tile = np.tile(
                        np.arange(min_k, self.scramble_depth + 1), reps
                    )
                    scramble_depths = worker_rng.permutation(
                        tile[: self.n_scramble]
                    ).astype(np.int16)
                else:
                    scramble_depths = worker_rng.integers(
                        min_k, self.scramble_depth + 1, size=self.n_scramble
                    ).astype(np.int16)

                env.scramble(depth=self.scramble_depth, depths=scramble_depths)
                s_cp, s_co, s_ep, s_eo = env.get_state()
                scrambled_parents = to_flat(s_cp, s_co, s_ep, s_eo)
            else:
                scrambled_parents = np.empty((0, 40), dtype=np.int8)
                scramble_depths = np.empty(0, dtype=np.int16)

            # 2. Precondition set states (if mixing enabled)
            if self.n_precond > 0:
                precond_spec = (
                    STAGES[self.spec.precondition]
                    if self.spec.precondition is not None
                    else None
                )
                precond_fixed_face = (
                    "D" if self.stage_enum == Stage.F2L else None
                )
                precond_parents = sample_goal_states(
                    precond_spec,
                    self.n_precond,
                    rng=worker_rng,
                    fixed_cross_face=precond_fixed_face,
                )
                precond_depths = np.full(self.n_precond, -1, dtype=np.int16)

                # Concatenate and shuffle
                parents = np.concatenate(
                    [scrambled_parents, precond_parents], axis=0
                )
                depths = np.concatenate(
                    [scramble_depths, precond_depths], axis=0
                )
                perm = worker_rng.permutation(self.batch_size)
                parents = parents[perm]
                depths = depths[perm]
            else:
                parents = scrambled_parents
                depths = scramble_depths

            yield (
                torch.from_numpy(parents.copy()).to(torch.int8),
                torch.from_numpy(depths.copy()).to(torch.int16),
            )


def create_stage_davi_dataloader(
    stage: str | Stage,
    batch_size: int,
    scramble_depth: int,
    num_workers: int = 4,
    depth_sampling: Literal["uniform", "stratified"] = "stratified",
    include_depth_zero: bool = False,
    precondition_fraction: float = 0.0,
    avoid_inverse_moves: bool = True,
    use_double_moves: bool = False,
    pin_memory: bool = True,
    data_seed: int | None = None,
    dataloader_seed: int | None = None,
) -> DataLoader:
    """Creates a PyTorch DataLoader yielding batches from StageDAVIDataset."""
    dataset = StageDAVIDataset(
        stage=stage,
        batch_size=batch_size,
        scramble_depth=scramble_depth,
        depth_sampling=depth_sampling,
        include_depth_zero=include_depth_zero,
        precondition_fraction=precondition_fraction,
        avoid_inverse_moves=avoid_inverse_moves,
        use_double_moves=use_double_moves,
        data_seed=data_seed,
        dataloader_seed=dataloader_seed,
    )

    loader_generator = (
        torch.Generator().manual_seed(dataloader_seed)
        if dataloader_seed is not None
        else None
    )
    worker_init_fn = make_worker_init_fn(data_seed, dataloader_seed)

    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=loader_generator,
        worker_init_fn=worker_init_fn,
    )
