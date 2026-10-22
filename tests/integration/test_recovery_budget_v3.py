"""Budget vérifié à 50 Hz, y compris la dérive pendant le suivi de l'arrêt."""

import pytest
from cav_recovery.recovery.scenarios import crossing
from cav_recovery.recovery.environment import RecoveryEnv
from cav_recovery.recovery.contracts import Action
from cav_recovery.recovery.evaluation import rows


@pytest.mark.integration
def test_reverse_budget_stays_bounded_through_native_stop_and_long_hold(tmp_path):
    spec = crossing(2, horizon_s=180)
    spec["agents"] = spec["agents"][:1]
    spec["active_limit"] = 1
    folder = tmp_path / "budget"
    env = RecoveryEnv(spec, folder)
    stopped = False
    try:
        env.reset()
        while not env.done:
            receipt = env.guard.check(
                env.views["agent0"], Action(-0.6), require_agreement=False
            )
            stopped |= receipt["reason"] in {
                "REVERSE_BUDGET",
                "REVERSE_STOPPING_BUDGET",
            }
            env.step({"agent0": Action(0 if stopped else -0.6)})
        assert stopped
        assert 4 < env.reverse_used["agent0"] <= 8
        assert not env.errors
        assert abs(env.views["agent0"].own["speed_mps"]) <= 0.01
    finally:
        env.close()
    frames = list(rows(folder / "physics.jsonl.gz"))
    assert len(frames) == 9001
    assert all(frame["reverse_m"]["agent0"] <= 8 for frame in frames)
