"""Contrôles temporels et on-policy de Graph-PPO, sans MetaDrive."""

import pytest
import torch
from types import SimpleNamespace
import json

from cav_recovery.recovery.graph_ppo import option_transitions, ppo_update_batch
from cav_recovery.recovery.graph_policy import GraphActorCritic
from cav_recovery.recovery.cooperative import option_bank
from cav_recovery.recovery.graph_marl_core import scenario
from cav_recovery.recovery.graph_marl_training import RewardTracker
from cav_recovery.recovery.graph_ppo_campaign import readiness_gate
from cav_recovery.recovery import graph_ppo_campaign
from cav_recovery.recovery import graph_marl_training
from cav_recovery.recovery.verified_passage_policy import GuardEnvelopePassagePolicy
from cav_recovery.recovery.graph_marl_notebook import render_figures
from cav_recovery.recovery.graph_marl_notebook import close_training
from cav_recovery.recovery.graph_ppo_validation import select_checkpoint
from cav_recovery.laboratory.storage import atomic_json, file_hash
from cav_recovery.laboratory.checkpoints import CheckpointStore


def decision(model, ego, at_s, action=1, free=True):
    nodes = torch.zeros((1, 24))
    adjacency = torch.eye(1)
    mask = [True, free] + [False] * (len(option_bank()) - 2)
    chosen = action if free else 0
    with torch.no_grad():
        logits, value = model(nodes, adjacency)
        distribution = torch.distributions.Categorical(
            logits=logits.masked_fill(~torch.tensor(mask), -1e9))
        old_logprob = float(distribution.log_prob(torch.tensor(chosen)))
    return dict(ego=ego, at_s=at_s, action=chosen, mask=mask,
                admissible_count=sum(mask),
                features=nodes.tolist(), adjacency=adjacency.tolist(),
                value=float(value), old_logprob=old_logprob)


def test_option_duration_and_simultaneous_team_reward():
    model = GraphActorCritic()
    rows = [decision(model, "V0", 0.0), decision(model, "V1", 0.0),
            decision(model, "V0", 0.4)]
    transitions = option_transitions(rows, [1, 2, 3], gamma=0.5,
                                     gae_lambda=0, terminal=True)
    first_v0 = next(row for row in transitions if row["ego"] == "V0"
                    and row["at_s"] == 0.0)
    first_v1 = next(row for row in transitions if row["ego"] == "V1")
    assert first_v0["option_steps"] == 2
    assert first_v0["option_reward"] == 2.0
    assert first_v1["option_steps"] == 3
    assert first_v1["option_reward"] == 2.75


def test_gae_lambda_is_discounted_over_physical_duration():
    model = GraphActorCritic()
    rows = [decision(model, "V0", 0.0), decision(model, "V0", 0.4)]
    for row in rows:
        row["value"] = 0.0
    transitions = option_transitions(rows, [0, 0, 1], gamma=0.5,
                                     gae_lambda=0.5, terminal=True)
    assert transitions[0]["advantage"] == pytest.approx(0.0625)
    assert transitions[1]["advantage"] == pytest.approx(1.0)


def test_censoring_drops_unobserved_tail_and_forced_choice():
    model = GraphActorCritic()
    rows = [decision(model, "V0", 0.0),
            decision(model, "V0", 0.2, free=False),
            decision(model, "V0", 0.4)]
    transitions = option_transitions(rows, [1, 2, 3], gamma=0.9,
                                     terminal=False)
    assert len(transitions) == 1
    assert transitions[0]["at_s"] == 0.0
    assert transitions[0]["option_steps"] == 1


def test_censored_option_uses_final_local_value_and_native_arrival_stops_reward():
    model = GraphActorCritic()
    rows = [decision(model, "active", 0.2), decision(model, "arrived", 0.0)]
    transitions = option_transitions(
        rows, [1, 2, 3, 4], gamma=0.5, gae_lambda=0,
        terminal=False, termination_times={"arrived": 0.4},
        bootstrap_values={"active": 2.0},
    )
    active = next(row for row in transitions if row["ego"] == "active")
    arrived = next(row for row in transitions if row["ego"] == "arrived")
    assert active["censored"] and not active["terminal"]
    assert active["option_steps"] == 3
    assert active["option_reward"] == 2 + 0.5 * 3 + 0.25 * 4
    assert active["target_return"] == pytest.approx(
        active["option_reward"] + 0.5 ** 3 * 2.0)
    assert arrived["terminal"] and not arrived["censored"]
    assert arrived["option_steps"] == 2
    assert arrived["option_reward"] == 1 + 0.5 * 2


def test_invalid_grids_and_nonfinite_rewards_fail():
    model = GraphActorCritic()
    with pytest.raises(ValueError, match="grille"):
        option_transitions([decision(model, "V0", 0.13)], [1],
                           gamma=0.9, terminal=True)
    with pytest.raises(ValueError, match="non finie"):
        option_transitions([], [float("nan")], gamma=0.9)


def test_ppo_minibatches_change_weights_and_reject_stale_batch():
    torch.manual_seed(4)
    model = GraphActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rows = [decision(model, "V0", 0.0), decision(model, "V1", 0.0)]
    transitions = option_transitions(rows, [1, 2], gamma=0.9,
                                     terminal=True)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    settings = dict(epochs=2, minibatch_size=1, clip_ratio=0.2,
                    value_coefficient=0.5, entropy_coefficient=0.01,
                    gradient_norm=0.5)
    result = ppo_update_batch(model, optimizer, transitions, **settings)
    assert result["updates"] == 4
    assert result["decisions"] == 2
    assert any(not torch.equal(old, parameter) for old, parameter in
               zip(before, model.parameters()))
    assert all(torch.isfinite(torch.tensor(list(row.values()))).all()
               for row in result["losses"])
    with pytest.raises(ValueError, match="poids courants"):
        ppo_update_batch(model, optimizer, transitions, **settings)


def test_frozen_twelve_transition_batch_has_four_real_optimizer_steps():
    torch.manual_seed(12)
    model = GraphActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    decisions = [decision(model, f"V{index}", 0.0)
                 for index in range(12)]
    transitions = option_transitions(decisions, [0.5, 1.0],
                                     gamma=0.99, terminal=True)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    result = ppo_update_batch(
        model, optimizer, transitions, epochs=2, minibatch_size=6,
        clip_ratio=0.2, value_coefficient=0.5,
        entropy_coefficient=0.01, gradient_norm=0.5)
    assert result["decisions"] == 12
    assert result["updates"] == 4
    assert len(result["losses"]) == 4
    assert all(torch.isfinite(torch.tensor(list(loss.values()))).all()
               for loss in result["losses"])
    assert any(not torch.equal(previous, parameter)
               for previous, parameter in zip(before, model.parameters()))


def test_catalogue_constructs_requested_population_without_prefix_cut():
    row = dict(name="collective-3", seed=231004, vehicles=8, horizon_s=240.0)
    spec = scenario(row, "train")
    assert len(spec["agents"]) == 8
    assert spec["active_limit"] == 8
    assert spec["construction"] == "direct_agent_generation"
    assert len({(a["origin"], a["destination"], a["lane"])
                for a in spec["agents"]}) > 1


def test_archived_source_requires_exact_hash(tmp_path):
    path = tmp_path / "spec.json"
    spec = scenario(dict(name="collective-3", seed=231004,
                         vehicles=8, horizon_s=240.0), "train")
    path.write_text(json.dumps(spec), encoding="utf-8")
    row = dict(name="archive", seed=spec["seed"], vehicles=8,
               horizon_s=240.0, source_spec_path=str(path),
               source_spec_sha256=file_hash(path), source_id=spec["id"])
    assert len(scenario(row, "train")["agents"]) == 8
    row["source_spec_id"], row["source_id"] = row["source_id"], "control-alias"
    assert len(scenario(row, "train")["agents"]) == 8
    row["source_spec_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Empreinte"):
        scenario(row, "train")


def test_team_reward_excludes_unaffected_progress_and_arrivals():
    weights = dict(progress_per_vehicle_length=1.0, confirmed_passage=1.0,
                   arrival=2.0, collective_recovery=5.0,
                   per_step_cost=0.0, reverse_per_m=0.0,
                   idle_after_10s_per_step=0.0, invalid_event=-10.0)
    tracker = RewardTracker(weights, {}, affected={"V0"})
    views = {ego: SimpleNamespace(own={"progress_m": 0.0})
             for ego in ("V0", "V1")}
    instance = SimpleNamespace(at_s=0.2, views=views,
                               metadata={"vehicles": {ego: {"length_m": 4.0}
                                                     for ego in views}},
                               reverse_used={}, arrivals={}, errors=[])
    tracker(instance)
    views["V1"].own["progress_m"] = 20.0
    instance.arrivals = {"V1": 0.4}
    instance.at_s = 0.4
    tracker(instance)
    assert tracker.rewards == [0.0, 0.0]


def test_readiness_gate_requires_positive_matching_evidence(tmp_path):
    proof = tmp_path / "proof.json"
    atomic_json(proof, {"qualified": 0})
    atomic_json(tmp_path / "TRAINING_READINESS.json",
                dict(verdict="TRAIN_NOT_READY", fingerprints={"proof.json": file_hash(proof)}))
    with pytest.raises(RuntimeError, match="TRAIN_NOT_READY"):
        readiness_gate(tmp_path)
    atomic_json(tmp_path / "TRAINING_READINESS.json",
                dict(verdict="TRAIN_READY", fingerprints={"proof.json": file_hash(proof)}))
    readiness_gate(tmp_path)
    atomic_json(proof, {"qualified": 1})
    with pytest.raises(RuntimeError, match="modifiée"):
        readiness_gate(tmp_path)


def test_readiness_gate_detects_external_qualification_change(tmp_path):
    external = tmp_path / "qualification.json"
    atomic_json(external, {"labels": ["POSITIVE_CONFIRMED"]})
    atomic_json(tmp_path / "TRAINING_READINESS.json", dict(
        verdict="TRAIN_READY", fingerprints={},
        external_fingerprints={str(external): file_hash(external)},
    ))
    readiness_gate(tmp_path)
    atomic_json(external, {"labels": []})
    with pytest.raises(RuntimeError, match="Source scientifique modifiée"):
        readiness_gate(tmp_path)


def test_periodic_validation_preserves_weights_and_random_state(tmp_path, monkeypatch):
    import random

    source = tmp_path / "qualification"
    source.mkdir()
    reference = source / "nominal" / "validation" / "positive"
    reference.mkdir(parents=True)
    atomic_json(reference / "physical-annotation.json", {"label": "POSITIVE_CONFIRMED"})
    atomic_json(source / "qualification.json", dict(scenes=[dict(
        split="validation", source_id="positive",
        qualified_label="POSITIVE_CONFIRMED",
        annotation_sha256=file_hash(reference / "physical-annotation.json"),
    )]))
    row = dict(source_id="positive", name="positive", seed=1,
               vehicles=4, horizon_s=210.0,
               qualified_label="POSITIVE_CONFIRMED")
    config = dict(qualification_root=str(source),
                  periodic_validation_ids=["positive"],
                  scenes={"validation": [row]})
    model = GraphActorCritic()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    calls = []

    def fake_episode(spec, folder, configuration, network, *, sample):
        calls.append(folder)
        random.random()
        folder.mkdir(parents=True)
        atomic_json(folder / "graph-marl-manifest.json", {"files": {}})
        summary = dict(errors=[], arrivals={}, introduced=4, pending=0,
                       reverse_m={}, physical_label="POSITIVE_CONFIRMED")
        atomic_json(folder / "summary.json", summary)
        return summary, [], None

    monkeypatch.setattr(graph_ppo_campaign, "scenario", lambda candidate, split: {})
    monkeypatch.setattr(graph_ppo_campaign, "_episode", fake_episode)
    monkeypatch.setattr(graph_ppo_campaign, "compare", lambda *args: {"success": False})
    random.seed(7)
    expected = random.random()
    random.seed(7)
    result = graph_ppo_campaign._periodic_validation(config, tmp_path / "campaign",
                                                      model, 10)
    again = graph_ppo_campaign._periodic_validation(config, tmp_path / "campaign",
                                                     model, 20)
    assert result[0]["source_id"] == "positive"
    assert again == result
    assert len(calls) == 1
    assert random.random() == expected
    assert all(torch.equal(old, parameter) for old, parameter in zip(before, model.parameters()))


def test_training_resumes_at_boundary_with_pending_on_policy_options(tmp_path,
                                                                      monkeypatch):
    root = tmp_path / "campaign"
    (root / "reports").mkdir(parents=True)
    atomic_json(root / "reports/preflight.json", {"proceed": True})
    rows = [dict(name=f"scene-{i}", seed=i, vehicles=4,
                 qualified_label="POSITIVE_CONFIRMED") for i in range(2)]
    config = dict(scenes={"train": rows}, maximum_wall_minutes=360,
                  ppo_batch_size=2, validation_every_episodes=10,
                  gamma_per_step=0.9, gae_lambda=0.95,
                  ppo_epochs=2, minibatch_size=1, clip_ratio=0.2,
                  value_coefficient=0.5, entropy_coefficient=0.01,
                  gradient_norm=0.5)
    monkeypatch.setattr(graph_ppo_campaign, "_prepare", lambda *args: (config, root))
    monkeypatch.setattr(graph_ppo_campaign, "readiness_gate", lambda target: {})
    monkeypatch.setattr(graph_ppo_campaign, "_memory_ready", lambda *args: True)
    monkeypatch.setattr(graph_ppo_campaign, "_qualified_reference",
                        lambda *args: ([], None, None))
    monkeypatch.setattr(graph_ppo_campaign, "scenario", lambda *args: {})
    monkeypatch.setattr(graph_ppo_campaign, "_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_ppo_campaign, "_physical_metrics",
                        lambda *args: {})

    def initialize(*args):
        torch.manual_seed(5)
        model = GraphActorCritic()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        store = CheckpointStore(root / "checkpoints", {"contract": "test"})
        saved, receipt = store.load()
        if saved:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
        return config, model, optimizer, store, saved, receipt

    def episode(spec, folder, settings, model, **kwargs):
        folder.mkdir(parents=True)
        if folder.name == "episode-000":
            (root / "stop-requested").touch()
        return (dict(errors=[], reverse_m={}, arrivals={"V0": 0.2},
                     introduced=1, pending=0, wall_s=1.0),
                [decision(model, "V0", 0.0)],
                SimpleNamespace(rewards=[1.0], final_views={}))

    monkeypatch.setattr(graph_ppo_campaign, "initialize", initialize)
    monkeypatch.setattr(graph_ppo_campaign, "_episode", episode)
    assert graph_ppo_campaign.train(root / "config.json", root) == 86
    saved, _ = CheckpointStore(root / "checkpoints", {"contract": "test"}).load()
    assert len(saved["history"]) == len(saved["pending_transitions"]) == 1
    (root / "stop-requested").unlink()
    assert graph_ppo_campaign.train(root / "config.json", root) == 0
    resumed, _ = CheckpointStore(root / "checkpoints", {"contract": "test"}).load()
    assert len(resumed["history"]) == 2
    assert resumed["pending_transitions"] == []
    assert resumed["history"][-1]["learning"]["updates"] == 4
    assert resumed["weight_hash"] != saved["weight_hash"]
    assert resumed["consumed_wall_s"] >= saved["consumed_wall_s"]
    assert [row["global_update_step"] for row in
            resumed["history"][-1]["learning"]["losses"]] == [1, 2, 3, 4]
    assert resumed["history"][-1]["learning"]["ppo_batches"] == 1


def test_checkpoint_selection_uses_periodic_evidence_not_recency(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoints", {"contract": "test"})
    receipts = []
    for index, fingerprint in enumerate(("initial", "good", "late", "late")):
        receipts.append(store.save(dict(
            global_step=index, model={}, optimizer={}, weight_hash=fingerprint,
            history=[] if index == 0 else [dict(learning={"updates": 1})],
            pending_transitions=[],
        )))
    for episode, fingerprint, gain in ((12, "good", True), (24, "late", False)):
        folder = tmp_path / "validation" / "periodic" / f"after-{episode:03d}"
        folder.mkdir(parents=True)
        atomic_json(folder / "summary.json", dict(
            episode_index=episode, weight_hash=fingerprint,
            outcomes=[dict(validity={"valid": True, "arrived": 4},
                           paired={"improvement_not_spontaneous": gain})],
        ))
    selected, record = select_checkpoint(tmp_path, store)
    assert selected == receipts[1]
    assert record["distinct_trained_weights"] == 2
    assert record["rule"] == "periodic_gains_then_arrivals_then_earliest"


def test_voluntary_closure_keeps_checkpoint_and_pending_count(tmp_path, monkeypatch):
    import cav_recovery.recovery.graph_marl_notebook as notebook

    folder = tmp_path / "checkpoints"
    folder.mkdir()
    checkpoint = folder / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint conserve")
    receipt = dict(file=checkpoint.name, sha256=file_hash(checkpoint))
    atomic_json(folder / "latest.json", receipt)
    atomic_json(tmp_path / "reports" / "experiment_status.json", dict(
        state="stopped_at_boundary", completed_episodes=3,
        pending_decisions=7))
    atomic_json(tmp_path / "training" / "history.json", [
        dict(learning={"updates": 1})])
    (tmp_path / "stop-requested").touch()
    monkeypatch.setattr(notebook, "process_status", lambda root: dict(
        alive=False, experiment=json.loads((tmp_path / "reports" /
                                             "experiment_status.json").read_text())))
    result = close_training(tmp_path)
    assert result["pending_decisions"] == 7
    assert not (tmp_path / "stop-requested").exists()
    assert json.loads((tmp_path / "reports" / "experiment_status.json").read_text())["state"] == "closed_at_boundary"
    assert checkpoint.read_bytes() == b"checkpoint conserve"


def test_collective_reward_requires_non_spontaneous_native_intervention(
        tmp_path, monkeypatch):
    import gzip

    nominal = tmp_path / "nominal"
    actual = tmp_path / "actual"
    for folder in (nominal, actual):
        folder.mkdir()
        for name in ("physical-annotation.json", "metadata.json", "contract.json"):
            atomic_json(folder / name, {})
    atomic_json(nominal / "summary.json", {"arrivals": {}})
    with gzip.open(actual / "commands.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(guard={"accepted": True},
                                     applied={"reason": "cooperative_release"})) + "\n")
    monkeypatch.setattr(graph_marl_training, "physical_rows", lambda path: iter(()))
    outcomes = iter((dict(success=True, status="DURABLE_RECOVERY",
                          t_conf_s=0.2, timeline=[]),
                     dict(success=True, status="DURABLE_RECOVERY",
                          t_conf_s=0.2, timeline=[])))
    monkeypatch.setattr(graph_marl_training, "physical_outcome",
                        lambda *args: next(outcomes))
    tracker = RewardTracker({"collective_recovery": 10.0}, {})
    tracker.components = [dict(collective_recovery=0.0)]
    tracker.rewards = [0.0]
    result = tracker.finalize_physical(actual, {"errors": [], "arrivals": {},
                                                "at_s": 0.2},
                                       ["V0", "V1"], 10.0, nominal)
    assert result["nominal_success"]
    assert not result["bonus_credited"]
    assert tracker.rewards == [0.0]

    outcomes = iter((dict(success=True, status="DURABLE_RECOVERY",
                          t_conf_s=0.4, timeline=[]),
                     dict(success=False, status="NOT_RECOVERED", timeline=[])))
    monkeypatch.setattr(graph_marl_training, "physical_outcome",
                        lambda *args: next(outcomes))
    tracker = RewardTracker({"collective_recovery": 10.0}, {})
    tracker.components = [dict(collective_recovery=0.0) for _ in range(4)]
    tracker.rewards = [0.0] * 4
    result = tracker.finalize_physical(actual, {"errors": [], "arrivals": {},
                                                "at_s": 0.8},
                                       ["V0", "V1"], 10.0, nominal)
    assert result["bonus_credited"]
    assert tracker.rewards == [0.0, 10.0, 0.0, 0.0]


def test_oversized_guard_corridor_abstains_without_invalid_v2v_message():
    from shapely.geometry import box

    policy = object.__new__(GuardEnvelopePassagePolicy)
    policy.ego = "V0"
    policy._request_guard = SimpleNamespace(
        action_envelope=lambda view, action: (None, None, box(0, 0, 4, 2)))
    policy.contract = SimpleNamespace(margin_m=0.5, step_s=0.2,
                                      neighbor_acceleration_bound_mps2=3.0)
    policy.request_domain_rejections = 0
    policy.request_domain_examples = []
    policy._fresh_state = lambda view, target: dict(
        sent_s=10.0, payload={"speed_mps": 4.0})
    policy.emit = lambda *args, **kwargs: pytest.fail("Message invalide émis")
    view = SimpleNamespace(at_s=10.0)
    assert policy.request_clearance(view, box(0, 0, 4, 2),
                                    target="V1", purpose="NOMINAL_PASSAGE",
                                    action=object()) is False
    assert policy.request_domain_rejections == 1
    assert policy.request_domain_examples[0]["requested"]["received_area_m2"] > 200
    policy._fresh_state = lambda view, target: dict(
        sent_s=10.0, payload={"speed_mps": 0.0})
    policy._request_guard.check = lambda *args, **kwargs: {"reason": "LOCAL_ENVELOPE"}
    policy.emit = lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("Espace demandé hors domaine."))
    view.own = {"progress_m": 0.0}
    view.route = SimpleNamespace(agent={"movement_id": "m0"})
    assert policy.request_clearance(view, box(0, 0, 4, 2),
                                    target="V1", purpose="NOMINAL_PASSAGE",
                                    action=object()) is False
    assert policy.request_domain_rejections == 2
    assert "post_serialization_rejection" in policy.request_domain_examples[1]


def test_exported_figures_show_real_ppo_losses_and_physical_status(tmp_path):
    episode = tmp_path / "training/episodes/episode-000"
    episode.mkdir(parents=True)
    atomic_json(episode / "reward-summary.json", dict(
        by_component={"progress": 1.0, "collective_recovery": 0.0}))
    history = [dict(index=0, name="m02-71", seed=1071, reward=1.0,
                    qualified_label="POSITIVE_CONFIRMED",
                    folder="training/episodes/episode-000",
                    summary={"wall_s": 3.0, "cpu_s": 3.0},
                    validity={"arrived": 1, "valid": True,
                              "reverse_max_m": 0.0},
                    learning={"decisions": 2, "updates": 1,
                              "losses": [dict(policy=0.2, value=0.4,
                                              total=0.3, entropy=0.1)]},
                    physical=dict(recovery_success=False,
                                  recovery_status="NOT_RECOVERED",
                                  guard_refusals=2))]
    atomic_json(tmp_path / "training/history.json", history)
    figures = render_figures(tmp_path)
    assert {"ppo_losses", "physical_recovery", "training_table"} <= figures.keys()
    assert all((tmp_path / "figures" / name).exists()
               for name in ("ppo_losses.png", "physical_recovery.png"))
    assert "guard_refusals" in (tmp_path / "training/metrics.csv").read_text()


def test_quality_runner_flags_native_fault_even_when_pytest_returns_zero(
        tmp_path, monkeypatch):
    from scripts import check_graph_ppo_quality

    fake = SimpleNamespace(returncode=0, stdout="49 passed\n",
                           stderr="Windows fatal exception: access violation\n")
    monkeypatch.setattr(check_graph_ppo_quality.subprocess, "run",
                        lambda *args, **kwargs: fake)
    result = check_graph_ppo_quality.command("native", ["pytest"], tmp_path)
    assert result["exit_code"] == 0
    assert result["native_fault"]
    assert file_hash(tmp_path / "native.log") == result["log_sha256"]
