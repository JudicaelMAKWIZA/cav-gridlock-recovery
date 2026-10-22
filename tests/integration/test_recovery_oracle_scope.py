"""Contrôle natif limité du garde expérimental omniscient, sans revendication générale."""

import pytest
from cav_recovery.recovery.scenarios import crossing
from cav_recovery.recovery.environment import RecoveryEnv, nominal_action
from cav_recovery.recovery.feasibility import GlobalBodyGuard


@pytest.mark.integration
def test_native_crossing_global_body_envelopes_preserve_contacts_and_budget(tmp_path):
    env = RecoveryEnv(crossing(2, horizon_s=80), tmp_path / "ideal-guard")
    try:
        env.reset()
        env.guard = GlobalBodyGuard(env)
        while not env.done:
            env.step(
                {
                    ego: nominal_action(view, env.contract)
                    for ego, view in env.views.items()
                }
            )
        assert not env.errors
        assert all(value <= 8 for value in env.reverse_used.values())
    finally:
        env.close()
