"""Arrêt puis reprise effectifs dans Bullet, sans setter de mouvement."""

import importlib.util
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("metadrive") is None, reason="MetaDrive natif requis"
)


@pytest.mark.integration
def test_native_repeated_stop_resume_and_stopping_envelope(tmp_path):
    from cav_recovery.recovery.scenarios import crossing
    from cav_recovery.recovery.environment import RecoveryEnv
    from cav_recovery.recovery.contracts import Action

    spec = crossing(2, horizon_s=40.0)
    spec["agents"] = spec["agents"][:1]
    spec["active_limit"] = 1
    env = RecoveryEnv(spec, tmp_path / "actuation")
    try:
        env.reset()
        assert type(env.native.agents["agent0"]).__name__ == "RecoveryVehicle"
        for cycle in range(3):
            initial = env.views["agent0"].own["progress_m"]
            for _ in range(20):
                env.step({"agent0": Action(0.8)})
            moving = env.views["agent0"].own.copy()
            assert moving["progress_m"] > initial + 1.0
            for _ in range(10):
                env.step({"agent0": Action(0.0)})
            stopped = env.views["agent0"].own
            assert abs(stopped["speed_mps"]) < 0.01
            assert stopped["progress_m"] - moving["progress_m"] < 1.0
            assert not env.errors
    finally:
        env.close()
