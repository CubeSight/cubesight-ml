import multiprocessing as mp
import random

import numpy as np
import torch


def resolve_seeds(
    master_seed: int | None = None,
    model_seed: int | None = None,
    data_seed: int | None = None,
    dataloader_seed: int | None = None,
) -> tuple[int, int, int, int]:
    """
    Resolves the 3 distinct seeds (model weight initialization, data generation,
    and dataloader worker) from explicit CLI inputs or deterministically from a master seed.

    If no seeds are provided, a secure random master seed is generated.
    If master_seed is provided, any omitted individual seeds are deterministically derived
    using non-overlapping linear congruential offsets to avoid correlated random streams.
    """
    if (
        master_seed is None
        and model_seed is None
        and data_seed is None
        and dataloader_seed is None
    ):
        resolved_master = random.SystemRandom().randint(0, 2**31 - 1)
    else:
        resolved_master = master_seed if master_seed is not None else 42

    resolved_model = model_seed if model_seed is not None else resolved_master
    resolved_data = (
        data_seed
        if data_seed is not None
        else (resolved_master * 10007 + 1) % (2**31)
    )
    resolved_dataloader = (
        dataloader_seed
        if dataloader_seed is not None
        else (resolved_master * 10007 + 2) % (2**31)
    )

    return resolved_master, resolved_model, resolved_data, resolved_dataloader


def seed_model_and_frameworks(model_seed: int) -> None:
    """
    Seeds Python random, NumPy, and PyTorch (CPU and CUDA) for model weight initialization
    and model-level stochasticity (e.g. dropout).
    """
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class WorkerInitFn:
    """
    Picklable worker_init_fn callable for PyTorch DataLoader.
    Fully compatible across Linux, Windows, and macOS under all
    multiprocessing start methods ('fork', 'forkserver', and 'spawn').
    """

    def __init__(
        self, data_seed: int | None, dataloader_seed: int | None
    ) -> None:
        self.data_seed = data_seed
        self.dataloader_seed = dataloader_seed

    def __call__(self, worker_id: int) -> None:
        if self.data_seed is not None:
            w_data_seed = (self.data_seed + worker_id * 10007) % (2**31)
            np.random.seed(w_data_seed)
            random.seed(w_data_seed)

        if self.dataloader_seed is not None:
            w_loader_seed = (self.dataloader_seed + worker_id * 10007) % (2**31)
            torch.manual_seed(w_loader_seed)


def make_worker_init_fn(
    data_seed: int | None, dataloader_seed: int | None
) -> WorkerInitFn:
    """
    Creates a picklable WorkerInitFn for PyTorch DataLoader ensuring each worker process
    receives distinct, independent seeds for both data generation (NumPy/Python)
    and DataLoader sampling (PyTorch).
    """
    return WorkerInitFn(data_seed, dataloader_seed)


def init_search_worker(seed: int) -> None:
    """
    Initializer for worker processes in parallel search (e.g. BeamSearchMP).
    Deterministically seeds Python, NumPy, and PyTorch for each worker based on process rank.
    """

    proc = mp.current_process()
    worker_id = proc._identity[0] if getattr(proc, "_identity", None) else 0
    worker_seed = (seed + worker_id * 10007) % (2**31)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
