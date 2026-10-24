from typing import Any, Protocol


class TargetSyncPolicy(Protocol):
    """Protocol for deciding when to synchronize the active model weights to the target model."""

    def should_sync(self, step: int, loss: float, interval_loss: float) -> bool:
        """Determines whether to trigger a target network synchronization at this step."""
        ...

    def on_sync(self, step: int) -> None:
        """Called immediately after a synchronization is executed."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Returns internal state dictionary for checkpointing."""
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads internal state from dictionary."""
        ...


class DeepCubeASyncPolicy:
    """
    DeepCubeA synchronization policy:
    Sync target network every C steps if the average loss over the interval is below threshold epsilon.
    """

    def __init__(
        self,
        sync_interval: int = 100,
        loss_threshold: float = 0.05,
    ) -> None:
        self.sync_interval = sync_interval
        self.loss_threshold = loss_threshold
        self.last_sync_step = 0
        self.num_syncs = 0

    def should_sync(self, step: int, loss: float, interval_loss: float) -> bool:
        if step <= 0 or step % self.sync_interval != 0:
            return False
        return interval_loss < self.loss_threshold

    def on_sync(self, step: int) -> None:
        self.last_sync_step = step
        self.num_syncs += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "deepcubea",
            "sync_interval": self.sync_interval,
            "loss_threshold": self.loss_threshold,
            "last_sync_step": self.last_sync_step,
            "num_syncs": self.num_syncs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.sync_interval = state_dict.get("sync_interval", self.sync_interval)
        self.loss_threshold = state_dict.get(
            "loss_threshold", self.loss_threshold
        )
        self.last_sync_step = state_dict.get("last_sync_step", 0)
        self.num_syncs = state_dict.get("num_syncs", 0)


class PlateauSyncPolicy:
    """
    Plateau synchronization policy from combinatorial-transformer:
    Detects plateauing loss over consecutive intervals and syncs when progress stalls.
    """

    def __init__(
        self,
        loss_interval: int = 100,
        min_steps_between_syncs: int = 500,
        patience: int = 3,
        safety_ceiling: float = 2.0,
        improvement_threshold: float = 0.01,
    ) -> None:
        self.loss_interval = loss_interval
        self.min_steps_between_syncs = min_steps_between_syncs
        self.patience = patience
        self.safety_ceiling = safety_ceiling
        self.improvement_threshold = improvement_threshold

        self.best_interval_loss = float("inf")
        self.plateau_strikes = 0
        self.steps_since_last_sync = 0
        self.num_syncs = 0

    def should_sync(self, step: int, loss: float, interval_loss: float) -> bool:
        if step <= 0 or step % self.loss_interval != 0:
            return False

        self.steps_since_last_sync += self.loss_interval

        if self.best_interval_loss == float("inf"):
            self.best_interval_loss = interval_loss
            self.plateau_strikes = 0
        else:
            delta = self.best_interval_loss - interval_loss
            if delta > (self.best_interval_loss * self.improvement_threshold):
                self.best_interval_loss = interval_loss
                self.plateau_strikes = 0
            else:
                self.plateau_strikes += 1

        return bool(
            (
                self.steps_since_last_sync >= self.min_steps_between_syncs
                and self.plateau_strikes >= self.patience
            )
            and interval_loss < self.safety_ceiling
        )

    def on_sync(self, step: int) -> None:
        self.best_interval_loss = float("inf")
        self.plateau_strikes = 0
        self.steps_since_last_sync = 0
        self.num_syncs += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "plateau",
            "loss_interval": self.loss_interval,
            "min_steps_between_syncs": self.min_steps_between_syncs,
            "patience": self.patience,
            "safety_ceiling": self.safety_ceiling,
            "improvement_threshold": self.improvement_threshold,
            "best_interval_loss": self.best_interval_loss,
            "plateau_strikes": self.plateau_strikes,
            "steps_since_last_sync": self.steps_since_last_sync,
            "num_syncs": self.num_syncs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.loss_interval = state_dict.get("loss_interval", self.loss_interval)
        self.min_steps_between_syncs = state_dict.get(
            "min_steps_between_syncs", self.min_steps_between_syncs
        )
        self.patience = state_dict.get("patience", self.patience)
        self.safety_ceiling = state_dict.get(
            "safety_ceiling", self.safety_ceiling
        )
        self.improvement_threshold = state_dict.get(
            "improvement_threshold", self.improvement_threshold
        )
        self.best_interval_loss = state_dict.get(
            "best_interval_loss", float("inf")
        )
        self.plateau_strikes = state_dict.get("plateau_strikes", 0)
        self.steps_since_last_sync = state_dict.get("steps_since_last_sync", 0)
        self.num_syncs = state_dict.get("num_syncs", 0)
