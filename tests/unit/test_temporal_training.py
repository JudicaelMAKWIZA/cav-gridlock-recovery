"""Causalité, restauration réelle et invariance d'un entraînement interrompu."""
# ruff: noqa: E402

import pytest

np = pytest.importorskip("numpy", reason="install the laboratory extra")
torch = pytest.importorskip("torch", reason="install the laboratory extra")
pytest.importorskip("orjson", reason="install the laboratory extra")

from cav_recovery.laboratory.data import atomic_npz
from cav_recovery.laboratory.storage import atomic_json
from cav_recovery.laboratory.model import TemporalClassifier
from cav_recovery.laboratory.checkpoints import CheckpointStore
from cav_recovery.laboratory.training import train


def small_cache(root):
    root.mkdir()
    rng = np.random.default_rng(17)
    for part in ["train", "validation", "test_ood"]:
        x = rng.normal(size=(100, 3)).astype(np.float32)
        np.save(root / f"{part}-x.npy", x)
        ends = np.arange(10, 90)
        atomic_npz(
            root / f"{part}-windows.npz",
            start=np.maximum(0, ends - 10),
            end=ends,
            y=np.arange(len(ends)) % 2,
            available=np.ones(len(ends), bool),
            time_s=ends * 0.2,
            episode=np.zeros(len(ends), np.int32),
        )
    atomic_json(
        root / "normalization.json",
        dict(mean=[0.0] * 3, std=[1.0] * 3, fitted_partition="train"),
    )
    atomic_json(root / "complete.json", dict(fixture=True))
    return root


def config():
    return dict(
        seed=17,
        threads=1,
        hidden=3,
        dilations=[1, 2],
        kernel_size=3,
        architecture="tcn",
        batch_size=16,
        epochs=2,
        train_stride=1,
        validation_selection_stride=1,
        checkpoint_steps=2,
        patience=3,
        learning_rate=0.001,
    )


def test_future_suffix_never_changes_past_logits():
    torch.manual_seed(4)
    model = TemporalClassifier(3, hidden=4, dilations=[1, 2, 4])
    first = torch.randn(2, 3, 30)
    second = first.clone()
    second[:, :, 15:] = 1000.0
    mask = torch.ones(2, 1, 30)
    torch.testing.assert_close(
        model.sequence_logits(first, mask)[:, :15],
        model.sequence_logits(second, mask)[:, :15],
        rtol=0,
        atol=0,
    )


def test_stop_resume_matches_uninterrupted_weights_optimizer_and_history(tmp_path):
    cache = small_cache(tmp_path / "cache")
    cfg = config()
    uninterrupted = train(cache, tmp_path / "straight", cfg)
    interrupted = train(cache, tmp_path / "resumed", cfg, stop_after_steps=3)
    assert interrupted["interrupted"]
    resumed = train(cache, tmp_path / "resumed", cfg)
    assert uninterrupted["history"] == resumed["history"]

    def latest(folder, result):
        return CheckpointStore(folder / "checkpoints", result["compatibility"]).load()[
            0
        ]

    a = latest(tmp_path / "straight", uninterrupted)
    b = latest(tmp_path / "resumed", resumed)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    for index, weights in a["optimizer"]["state"].items():
        for key, value in weights.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(
                    value, b["optimizer"]["state"][index][key], rtol=0, atol=0
                )
    assert a["global_step"] == b["global_step"] == 10
    assert list((tmp_path / "resumed").glob("resume-*.json"))


def test_corrupted_latest_falls_back_and_configuration_mismatch_refuses(tmp_path):
    store = CheckpointStore(tmp_path, dict(data="original"))
    store.save(
        dict(
            global_step=1,
            extra_state=dict(policy="future_rl_state", episode_boundary=True),
        )
    )
    newest = store.save(dict(global_step=2))
    (tmp_path / newest["file"]).write_bytes(b"power loss")
    recovered, receipt = store.load()
    assert recovered["global_step"] == 1 and receipt["rejected"]
    assert recovered["extra_state"]["episode_boundary"]
    with pytest.raises(ValueError, match="incompatibles"):
        CheckpointStore(tmp_path, dict(data="different")).load()
