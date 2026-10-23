"""Contrôles sans moteur natif ni épisode expérimental."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cav_recovery.recovery.cooperative import CooperativePolicy, option_bank
from cav_recovery.recovery.graph_marl_core import (
    discounted_returns, load_config, ppo_update, scene_plan, status_of_episode,
    weight_hash,
)
from cav_recovery.laboratory.checkpoints import CheckpointStore
from cav_recovery.laboratory.storage import file_hash
from cav_recovery.recovery.graph_marl_policy import P4GraphPolicy
from cav_recovery.recovery.graph_marl_training import RewardTracker, _recoverable_folder
from cav_recovery.laboratory.storage import atomic_json
from cav_recovery.recovery.graph_policy import GraphActorCritic
from cav_recovery.recovery.verified_passage_policy import GuardEnvelopePassagePolicy


CONFIG = Path(__file__).parents[2] / "configs/recovery/graph-marl-p4-v1.json"


def test_plan_has_disjoint_development_parents_and_no_reserved_test():
    plan = scene_plan(load_config(CONFIG))
    training = {row["parent_id"] for row in plan if row["split"] == "train"}
    validation = {row["parent_id"] for row in plan if row["split"] == "validation"}
    assert training.isdisjoint(validation)
    assert {row["split"] for row in plan} == {"preflight", "train", "validation"}
    assert max(row["vehicles"] for row in plan) == 8


def test_p4_protocol_is_inherited_and_network_can_change_actual_option(monkeypatch):
    assert issubclass(P4GraphPolicy, GuardEnvelopePassagePolicy)
    bank = option_bank()
    monkeypatch.setattr(CooperativePolicy, "candidates",
                        lambda self, view, guard: [bank[1], bank[2]])
    monkeypatch.setattr("cav_recovery.recovery.graph_marl_policy.local_graph",
                        lambda view: (np.zeros((1, 24), np.float32),
                                      np.eye(1, dtype=np.float32)))

    class FixedNetwork(torch.nn.Module):
        def __init__(self, chosen):
            super().__init__()
            self.chosen = chosen

        def forward(self, nodes, adjacency):
            values = torch.full((len(bank),), -10.0)
            values[self.chosen] = 10.0
            return values, torch.tensor(0.0)

    decisions = []
    policy = object.__new__(P4GraphPolicy)
    policy.record = decisions.append
    policy.sample = False
    policy.model = FixedNetwork(1)
    view = SimpleNamespace(ego="V0", at_s=11.0)
    assert policy.candidates(view, object()) == [bank[1]]
    policy.model = FixedNetwork(2)
    assert policy.candidates(view, object()) == [bank[2]]
    assert [row["action"] for row in decisions] == [1, 2]
    assert all(row["mask"][0] for row in decisions)
    assert all(not row["mask"][-1] for row in decisions)


def test_shared_graph_accepts_variable_vehicle_resource_counts():
    model = GraphActorCritic()
    for count in (1, 4, 12):
        nodes = torch.zeros((count, 24))
        adjacency = torch.eye(count)
        logits, value = model(nodes, adjacency)
        assert logits.shape == (len(option_bank()),)
        assert value.ndim == 0


def test_returns_use_physics_time_and_keep_single_decision_gradient():
    rows = [dict(at_s=0.0), dict(at_s=0.2), dict(at_s=0.2)]
    assert discounted_returns(rows, [1.0, 2.0], 0.5).tolist() == [2.0, 2.0, 2.0]


def test_ppo_changes_weights_with_one_valid_physical_choice():
    config = load_config(CONFIG)
    torch.manual_seed(7)
    model = GraphActorCritic(config["hidden"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    nodes = [[0.0] * 24]
    adjacency = [[1.0]]
    with torch.no_grad():
        logits, value = model(torch.tensor(nodes), torch.tensor(adjacency))
        mask = [True] * len(option_bank())
        mask[-1] = False
        distribution = torch.distributions.Categorical(
            logits=logits.masked_fill(~torch.tensor(mask), -1e9))
        logprob = float(distribution.log_prob(torch.tensor(1)))
    decision = dict(at_s=0.0, value=float(value), features=nodes,
                    adjacency=adjacency, mask=mask, action=1,
                    old_logprob=logprob)
    before = weight_hash(model)
    result = ppo_update(model, optimizer, [decision], [1.0], config)
    assert result["updates"] == config["ppo_epochs"]
    assert result["decisions"] == 1
    assert before != weight_hash(model)


def test_reverse_budget_failure_is_not_a_success():
    status = status_of_episode(dict(errors=[], reverse_m={"V0": 8.01},
                                    arrivals={"V0": 20.0}, introduced=1,
                                    pending=0))
    assert not status["valid"]
    assert status["problems"] == ["reverse_budget_exceeded"]


def test_checkpoint_integrity_and_resume_contract(tmp_path):
    model = GraphActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    contract = {"config_sha256": "prepared", "p4": "fixed"}
    store = CheckpointStore(tmp_path / "checkpoints", contract)
    receipt = store.save(dict(global_step=3, model=model.state_dict(),
                              optimizer=optimizer.state_dict(), history=[{"index": 0}],
                              weight_hash=weight_hash(model)))
    state, loaded = store.load()
    assert state["weight_hash"] == weight_hash(model)
    assert receipt["sha256"] == file_hash(tmp_path / "checkpoints" / receipt["file"])
    assert loaded["receipt"] == receipt
    with pytest.raises(ValueError, match="incompatibles|incompatible"):
        CheckpointStore(tmp_path / "checkpoints", {"config_sha256": "changed"}).load()


def test_passage_reward_is_counted_once_per_actual_done_attempt():
    weights = load_config(CONFIG)["reward"]
    message = dict(kind="DONE", sender="V0", attempt="V0:10",
                   payload={"reason": "beneficiaries_passed"})
    policy = SimpleNamespace(outbox=[message])
    tracker = RewardTracker(weights, {"V0": policy})
    view = SimpleNamespace(own={"progress_m": 0.0})
    instance = SimpleNamespace(at_s=0.2, views={"V0": view},
                               metadata={"vehicles": {"V0": {"length_m": 4.5}}},
                               reverse_used={"V0": 0.0}, arrivals={}, errors=[])
    tracker(instance)
    instance.at_s = 0.4
    view.own["progress_m"] = 0.9
    tracker(instance)
    assert sum(row["passage"] for row in tracker.components) == weights["confirmed_passage"]
    assert sum(row["progress"] for row in tracker.components) == pytest.approx(0.2)


def test_incomplete_validation_trace_is_preserved_on_resume(tmp_path):
    base = tmp_path / "scene-00"
    base.mkdir()
    (base / "partial.jsonl").write_text("incomplet", encoding="utf-8")
    replacement, complete = _recoverable_folder(base)
    assert not complete
    assert replacement != base
    assert (base / "partial.jsonl").read_text(encoding="utf-8") == "incomplet"
    replacement.mkdir()
    (replacement / "summary.json").write_text("{}", encoding="utf-8")
    atomic_json(replacement / "graph-marl-manifest.json", dict(files={
        "summary.json": file_hash(replacement / "summary.json")}))
    found, complete = _recoverable_folder(base)
    assert complete and found == replacement
