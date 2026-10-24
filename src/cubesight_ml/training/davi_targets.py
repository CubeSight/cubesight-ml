import torch
import torch.nn as nn

from cubesight_ml.cube.torch_ops import TorchCubeOps
from cubesight_ml.stages.goal_checker import StageGoalChecker


@torch.no_grad()
def compute_davi_targets(
    target_model: nn.Module,
    parents: torch.Tensor,
    cube_ops: TorchCubeOps,
    goal_checker: StageGoalChecker,
    chunk_size: int = 10000,
    clamp_min: float | None = 0.0,
    move_costs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes DeepCubeA DAVI Bellman targets for a batch of parent states:

    y(s) = 0 if s is goal, else min_a [ c(a) + J_target(s') ]
    where J_target(s') = 0 if s' is goal.

    Args:
        target_model: ResMLP value model wrapper.
        parents: Tensor (B, 40) or (B, 46) of cube states on device.
        cube_ops: TorchCubeOps instance on the same device.
        goal_checker: StageGoalChecker module on the same device.
        chunk_size: Maximum batch size for child evaluation chunks.
        clamp_min: Minimum value to clamp targets to (default 0.0).
        move_costs: Tensor of shape (A,) or (NUM_MOVES,) representing action
            execution costs c(a) = base_cost + penalty. If None, defaults to
            unit cost 1.0 for all actions.

    Returns:
        Tensor (B,) containing Bellman cost-to-go targets.
    """
    # 1. Expand children on device
    children = cube_ops.expand(parents)  # (B, A, state_dim)
    b, a, _ = children.shape

    # 2. Identify goal children
    is_goal_children = goal_checker.is_goal(children)  # (B, A) bool

    # 3. Evaluate target model on children in chunks
    flat_children = children.reshape(b * a, -1)
    val_chunks = []
    for chunk in flat_children.split(chunk_size):
        chunk_val = target_model(chunk)["value"].squeeze(-1).float()
        val_chunks.append(chunk_val)

    child_values = torch.cat(val_chunks, dim=0).view(b, a)

    # 4. Target Masking: if child s' is in goal set, J_target(s') = 0.0
    child_values = child_values.masked_fill(is_goal_children, 0.0)

    # 5. Bellman target: min_a [ c(a) + J_target(s') ]
    if move_costs is None:
        action_costs = torch.ones(
            a, dtype=torch.float32, device=child_values.device
        )
    else:
        mc = move_costs.squeeze()
        max_m = int(cube_ops.moves.max().item()) + 1
        if mc.ndim == 1 and mc.size(0) == a:
            action_costs = mc.to(
                device=child_values.device, dtype=torch.float32
            )
        elif mc.ndim == 1 and mc.size(0) >= max_m:
            action_costs = mc.to(
                device=child_values.device, dtype=torch.float32
            )[cube_ops.moves]
        else:
            raise ValueError(
                f"Expected move_costs of shape ({a},) or >= {max_m}, "
                f"got {move_costs.shape}"
            )

    candidate_costs = child_values + action_costs.unsqueeze(0)
    targets = candidate_costs.min(dim=-1).values

    if clamp_min is not None:
        targets = targets.clamp_min(clamp_min)

    # 6. Target Masking: if parent s is already in goal set, y(s) = 0.0
    is_goal_parents = goal_checker.is_goal(parents)
    return targets.masked_fill(is_goal_parents, 0.0)
