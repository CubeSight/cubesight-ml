from cubesight_ml.training.target_sync import (
    DeepCubeASyncPolicy,
    PlateauSyncPolicy,
)


def test_deepcubea_sync_policy():
    policy = DeepCubeASyncPolicy(sync_interval=100, loss_threshold=0.05)

    # Step not divisible by sync_interval
    assert not policy.should_sync(step=50, loss=0.01, interval_loss=0.01)

    # Step divisible, but loss too high
    assert not policy.should_sync(step=100, loss=0.02, interval_loss=0.08)

    # Step divisible and loss low enough
    assert policy.should_sync(step=100, loss=0.04, interval_loss=0.04)

    policy.on_sync(step=100)
    assert policy.num_syncs == 1
    assert policy.last_sync_step == 100

    # Serialization
    state = policy.state_dict()
    new_policy = DeepCubeASyncPolicy()
    new_policy.load_state_dict(state)
    assert new_policy.num_syncs == 1
    assert new_policy.last_sync_step == 100


def test_plateau_sync_policy():
    policy = PlateauSyncPolicy(
        loss_interval=100,
        min_steps_between_syncs=300,
        patience=2,
        safety_ceiling=1.0,
        improvement_threshold=0.01,
    )

    # Step 100: sets best loss to 0.5
    assert not policy.should_sync(step=100, loss=0.5, interval_loss=0.5)

    # Step 200: loss 0.499 (less than 1% improvement -> strike 1)
    assert not policy.should_sync(step=200, loss=0.499, interval_loss=0.499)

    # Step 300: loss 0.498 (strike 2 >= patience, steps 300 >= min_steps) -> SYNC!
    assert policy.should_sync(step=300, loss=0.498, interval_loss=0.498)

    policy.on_sync(step=300)
    assert policy.num_syncs == 1
    assert policy.plateau_strikes == 0
    assert policy.steps_since_last_sync == 0

    # Test safety ceiling
    policy2 = PlateauSyncPolicy(
        loss_interval=100,
        min_steps_between_syncs=100,
        patience=1,
        safety_ceiling=0.5,
    )
    # Loss 2.0 is above safety ceiling 0.5
    policy2.best_interval_loss = 2.0
    policy2.steps_since_last_sync = 100
    policy2.plateau_strikes = 1
    assert not policy2.should_sync(step=100, loss=2.0, interval_loss=2.0)
