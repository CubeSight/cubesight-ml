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
) -> torch.Tensor:
    """
    Computes DeepCubeA DAVI Bellman targets for a batch of parent states:
    y(s) = 0 if s is a goal state, else 1 + min_a J_target(s') where J_target(s')=0 if s' is goal.

    Args:
        target_model: ResMLP value model wrapper.
        parents: Tensor (B, 40) of cube states on device.
        cube_ops: TorchCubeOps instance on the same device.
        goal_checker: StageGoalChecker module on the same device.
        chunk_size: Maximum batch size for child evaluation chunks.
        clamp_min: Minimum value to clamp targets to (default 0.0).

    Returns:
        Tensor (B,) containing Bellman cost-to-go targets.
    """
    # 1. Expand children on device
    children = cube_ops.expand(parents)  # (B, A, 40)
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

    # 4. DeepCubeA Target Masking: if child s' is in goal set, J_target(s') = 0.0
    child_values = child_values.masked_fill(is_goal_children, 0.0)

    # 5. Bellman target: 1 + min_a J(s')
    best_child_values = child_values.min(dim=-1).values
    targets = 1.0 + best_child_values

    if clamp_min is not None:
        targets = targets.clamp_min(clamp_min)

    # 6. DeepCubeA Target Masking: if parent s is already in goal set, y(s) = 0.0
    is_goal_parents = goal_checker.is_goal(parents)
    return targets.masked_fill(is_goal_parents, 0.0)
