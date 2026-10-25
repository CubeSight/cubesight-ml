import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from cubesight_ml.cube.moves import (
    BASE_MOVE_COSTS_STM,
    INDEX_TO_MOVE,
    NUM_MOVES,
    parse_move_penalties,
)
from cubesight_ml.models import build_model
from cubesight_ml.search.beam_search.utils import BeamSearchRubiksCubeHandler


def _load_model_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device,
) -> nn.Module:
    """Loads a model and attached penalties from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = build_model(cfg).to(device)
    sd = ckpt.get("model_state_dict", ckpt.get("target_model_state_dict", ckpt))
    if isinstance(sd, dict):
        cleaned_sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        model.load_state_dict(cleaned_sd)
    model.eval()

    penalties = ckpt.get("move_penalties")
    model.move_penalties = penalties
    model.checkpoint_path = Path(checkpoint_path)
    if hasattr(model, "config") and isinstance(model.config, dict):
        model.config["move_penalties"] = penalties
    return model


def _extract_model_penalties(model_or_path: Any) -> np.ndarray:
    """Extracts trained ergonomic move penalties from model or model file."""
    zero_arr = np.zeros(NUM_MOVES, dtype=np.float32)
    if model_or_path is None:
        return zero_arr

    # 1. Direct attribute on model or wrapper
    raw_penalties = getattr(model_or_path, "move_penalties", None)
    if raw_penalties is None:
        backbone = getattr(model_or_path, "backbone", None)
        raw_penalties = getattr(backbone, "move_penalties", None)

    # 2. Inside model.config dict
    if raw_penalties is None:
        cfg = getattr(model_or_path, "config", None)
        if isinstance(cfg, dict):
            raw_penalties = cfg.get("move_penalties")

    if raw_penalties is not None:
        penalties_arr = parse_move_penalties(raw_penalties)
        if np.any(penalties_arr > 0.0):
            return penalties_arr

    # 3. Checkpoint path (either passed directly or attached to model)
    ckpt_path = model_or_path if isinstance(model_or_path, str | Path) else None
    if ckpt_path is None:
        ckpt_path = getattr(
            model_or_path,
            "checkpoint_path",
            getattr(model_or_path, "checkpoint", None),
        )

    if not ckpt_path:
        return zero_arr

    p = Path(ckpt_path)
    model_dir = p.parent if p.is_file() else p
    penalties_arr = None

    # Load from model file (.pt)
    if p.is_file():
        try:
            ckpt_data = torch.load(p, map_location="cpu", weights_only=False)
            if ckpt_data.get("move_penalties"):
                raw = ckpt_data["move_penalties"]
                penalties_arr = parse_move_penalties(raw)
        except Exception:  # noqa: BLE001
            pass

    # Fallback to model directory: move_penalties.json
    if (
        penalties_arr is None or not np.any(penalties_arr > 0.0)
    ) and model_dir.is_dir():
        pen_file = model_dir / "move_penalties.json"
        if pen_file.is_file():
            try:
                with pen_file.open(encoding="utf-8") as f:
                    data = json.load(f)
                penalties_arr = parse_move_penalties(data)
            except Exception:  # noqa: BLE001
                pass

    # Fallback to model directory: config.json
    if (
        penalties_arr is None or not np.any(penalties_arr > 0.0)
    ) and model_dir.is_dir():
        cfg_file = model_dir / "config.json"
        if cfg_file.is_file():
            try:
                with cfg_file.open(encoding="utf-8") as f:
                    cfg = json.load(f)
                if cfg.get("move_penalties"):
                    raw_cfg = cfg["move_penalties"]
                    penalties_arr = parse_move_penalties(raw_cfg)
            except Exception:  # noqa: BLE001
                pass

    if penalties_arr is not None and np.any(penalties_arr > 0.0):
        return penalties_arr

    return zero_arr


class BeamSearch:
    """Batched GPU/CPU Beam Search solver supporting STM, rotations, and
    configurable move execution penalties.
    """

    def __init__(
        self,
        model: nn.Module | str | Path,
        puzzle_handler: BeamSearchRubiksCubeHandler,
        batch_size: int = 512,
        move_penalties: dict[str, float]
        | np.ndarray
        | torch.Tensor
        | str
        | Path
        | None = None,
        use_stm: bool = True,
        verbose: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        if isinstance(model, str | Path):
            self.model = _load_model_checkpoint(model, device=self.device)
        else:
            self.model = model.to(self.device)
        self.model.eval()

        self.batch_size = batch_size
        self.use_stm = use_stm
        self.verbose = verbose
        self.puzzle_handler = puzzle_handler
        self.state_size = puzzle_handler.get_state_size()
        self.moves = puzzle_handler.get_moves().to(self.device)
        self.n_moves = self.moves.size(0)

        # Default to ergonomic penalties inside model file if not specified
        if move_penalties is None:
            penalties_np = _extract_model_penalties(self.model)
        else:
            penalties_np = parse_move_penalties(move_penalties)

        self.move_penalties = penalties_np
        if use_stm:
            base_costs_np = BASE_MOVE_COSTS_STM.copy()
        else:
            base_costs_np = np.ones(NUM_MOVES, dtype=np.float32)

        total_costs_np = base_costs_np + penalties_np
        self.move_costs = torch.tensor(
            total_costs_np, dtype=torch.float32, device=self.device
        )

        # Fixed seed for hash vector generation
        gen = torch.Generator().manual_seed(42)
        self.hash_vec = torch.randint(
            0,
            int(1e15),
            (self.state_size,),
            dtype=torch.int64,
            generator=gen,
        ).to(self.device)

    def _hash_states(self, states: torch.Tensor) -> torch.Tensor:
        size = min(states.size(1), self.hash_vec.size(0))
        return torch.sum(
            states[:, :size].to(torch.int64) * self.hash_vec[:size], dim=1
        )

    def _get_unique_hashed_states_idx(
        self, hashed_states: torch.Tensor
    ) -> torch.Tensor:
        valid_idx = torch.arange(
            hashed_states.size(0), device=hashed_states.device
        )
        hashed_sorted, sort_idx = torch.sort(hashed_states, stable=True)

        unique_mask = torch.empty(
            hashed_sorted.size(0), dtype=torch.bool, device=hashed_states.device
        )
        unique_mask[0] = True
        unique_mask[1:] = hashed_sorted[1:] != hashed_sorted[:-1]

        return valid_idx[sort_idx[unique_mask]]

    def _get_model_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        inputs = self.puzzle_handler.state_to_model_input(batch)
        if self.device.type == "cuda":
            with (
                torch.autocast(device_type="cuda", dtype=torch.float16),
                torch.no_grad(),
            ):
                return self.model(*inputs)["value"]

        with torch.no_grad():
            return self.model(*inputs)["value"]

    def _perform_step(
        self,
        states: torch.Tensor,
        g_costs: torch.Tensor,
        beam_width: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
    ]:
        prev_state_idx = torch.arange(
            states.size(0), device=self.device
        ).repeat_interleave(self.n_moves)
        moves = self.moves.repeat(states.size(0))
        step_costs = self.move_costs[moves]
        candidate_g = g_costs[prev_state_idx] + step_costs

        # 1. Generate & hash neighbors in chunks
        neighbors_hashed = torch.empty(
            moves.size(0), dtype=torch.int64, device=self.device
        )
        for i in range(0, states.size(0), self.batch_size):
            batch = states[i : i + self.batch_size]
            current_b_size = batch.size(0)
            neighbors = self.puzzle_handler.get_neighbors(batch)
            neighbors_hashed[
                i * self.n_moves : (i + current_b_size) * self.n_moves
            ] = self._hash_states(neighbors)

        unique_idx_1 = self._get_unique_hashed_states_idx(neighbors_hashed)
        unique_g = candidate_g[unique_idx_1]

        # 2. Evaluate model on unique child states
        values = torch.empty(
            unique_idx_1.size(0), dtype=torch.float32, device=self.device
        )
        forward_passes = 0
        items_evaluated = unique_idx_1.size(0)

        for i in range(0, unique_idx_1.size(0), self.batch_size):
            forward_passes += 1
            chunk_slice = unique_idx_1[i : i + self.batch_size]
            parent_batch = states[prev_state_idx[chunk_slice]]
            move_batch = moves[chunk_slice]
            child_batch = self.puzzle_handler.apply_moves(
                parent_batch, move_batch
            )

            actual_size = child_batch.size(0)
            if actual_size < self.batch_size:
                pad_size = self.batch_size - actual_size
                pad_tensor = child_batch[0:1].expand(pad_size, -1)
                padded_batch = torch.cat([child_batch, pad_tensor], dim=0)
                preds = self._get_model_predictions(padded_batch).view(-1)
                values[i : i + actual_size] = preds[:actual_size].float()
            else:
                values[i : i + self.batch_size] = (
                    self._get_model_predictions(child_batch).view(-1).float()
                )

        # 3. Select top beam_width candidates ranked by f = g + h
        f_scores = unique_g + values
        best_idx_2 = torch.argsort(f_scores, stable=True)[:beam_width]

        next_states = torch.empty(
            best_idx_2.size(0),
            states.size(1),
            dtype=states.dtype,
            device=self.device,
        )
        for i in range(0, best_idx_2.size(0), self.batch_size):
            batch_best_idx = best_idx_2[i : i + self.batch_size]
            next_states[i : i + self.batch_size] = (
                self.puzzle_handler.apply_moves(
                    states[prev_state_idx[unique_idx_1[batch_best_idx]]],
                    moves[unique_idx_1[batch_best_idx]],
                )
            )

        return (
            next_states,
            unique_g[best_idx_2],
            values[best_idx_2],
            moves[unique_idx_1[best_idx_2]],
            prev_state_idx[unique_idx_1[best_idx_2]],
            forward_passes,
            items_evaluated,
        )

    def find_solution(
        self,
        state: np.ndarray | torch.Tensor | tuple[np.ndarray, ...],
        beam_width: int,
        num_steps: int,
    ) -> tuple[list[int] | None, int, dict[str, Any]]:
        """Executes beam search from the given start state with move penalties.
        Returns:
            (solution_moves, steps_taken, search_stats)
            solution_moves is None if search failed within num_steps.
        """
        states = self.puzzle_handler.state_from_env(state)
        if states.ndim == 1:
            states = states.unsqueeze(0)

        stats = {
            "batched_forward_passes": 0,
            "states_evaluated": 0,
            "states_expanded": 0,
            "generated_states": 0,
            "stm_length": 0,
            "total_length": 0,
            "total_cost": 0.0,
            "move_names": [],
        }

        # Check if already solved (e.g. skip stage)
        if bool(self.puzzle_handler.is_solved(states).any().item()):
            return [], 0, stats

        g_costs = torch.zeros(
            states.size(0), dtype=torch.float32, device=self.device
        )

        tree_move = -torch.ones(
            (num_steps, beam_width), dtype=torch.int16, device=self.device
        )
        tree_idx = -torch.ones(
            (num_steps, beam_width), dtype=torch.int32, device=self.device
        )
        tree_g = torch.zeros(
            (num_steps, beam_width), dtype=torch.float32, device=self.device
        )

        steps_range = (
            tqdm(range(num_steps), desc="Beam Search")
            if self.verbose
            else range(num_steps)
        )

        solved = False
        step = 0
        for step in steps_range:
            num_expanded = states.size(0)
            stats["states_expanded"] += num_expanded
            stats["generated_states"] += num_expanded * self.n_moves

            states, g_costs, pred, moves, idx, fp, ie = self._perform_step(
                states, g_costs, beam_width
            )
            stats["batched_forward_passes"] += fp
            stats["states_evaluated"] += ie

            if self.verbose and hasattr(steps_range, "set_description"):
                desc_str = (
                    f"y_min={pred.min().item():.2f}, "
                    f"g_min={g_costs.min().item():.2f}"
                )
                steps_range.set_description(desc_str)

            tree_move[step, : states.size(0)] = moves.to(torch.int16)
            tree_idx[step, : states.size(0)] = idx.to(torch.int32)
            tree_g[step, : states.size(0)] = g_costs

            is_solved_mask = self.puzzle_handler.is_solved(states)
            if bool(is_solved_mask.any().item()):
                solved = True
                break

        if not solved:
            return None, step + 1, stats

        solved_indices = torch.where(self.puzzle_handler.is_solved(states))[0]
        best_solved_idx = int(
            solved_indices[torch.argmin(g_costs[solved_indices])].item()
        )

        solution = []
        idx = best_solved_idx
        for i in range(step, -1, -1):
            solution.append(int(tree_move[i, idx].item()))
            idx = int(tree_idx[i, idx].item())

        solution_moves = list(reversed(solution))
        total_cost = float(g_costs[best_solved_idx].item())
        stm_length = sum(1 for m in solution_moves if m < 45)
        total_length = len(solution_moves)
        move_names = [INDEX_TO_MOVE.get(m, str(m)) for m in solution_moves]

        stats["stm_length"] = stm_length
        stats["total_length"] = total_length
        stats["total_cost"] = total_cost
        stats["move_names"] = move_names

        return solution_moves, step + 1, stats
