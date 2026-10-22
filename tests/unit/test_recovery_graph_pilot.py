"""Frontière locale et invariance de l'encodeur, sans apprentissage massif."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from cav_recovery.recovery.graph_policy import GraphActorCritic, local_graph, FEATURES  # noqa: E402
from cav_recovery.recovery.cooperative import option_bank  # noqa: E402


def test_graph_encoder_is_invariant_to_neighbor_permutation():
    torch.manual_seed(5)
    model = GraphActorCritic()
    nodes = torch.randn(4, FEATURES)
    adjacency = torch.tensor(
        [[0.25] * 4, [0.5, 0.5, 0, 0], [0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5]]
    )
    order = torch.tensor([0, 3, 1, 2])
    logits, value = model(nodes, adjacency)
    changed, other = model(nodes[order], adjacency[order][:, order])
    assert torch.allclose(logits, changed, atol=1e-6)
    assert torch.allclose(value, other, atol=1e-6)
    assert len(logits) == len(option_bank())


def test_projection_rejects_future_knowledge_and_ignores_privileged_attributes():
    from tests.unit.test_recovery_prerequisites import view, state
    from cav_recovery.recovery.protocol import Endpoint
    from shapely.geometry import LineString
    from types import SimpleNamespace

    v = view()
    line = LineString([[0.0, 0.0], [50.0, 0.0]])
    v.route = SimpleNamespace(line=line, position=lambda s: (s, 0.0))
    v.oracle_group = ["hidden"]
    v.future_success = True
    first = local_graph(v)
    v.messages = [state(Endpoint("future"), [3.0, 0.0], now=1.0)]
    second = local_graph(v)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_pilot_refuses_to_train_without_local_physical_proof(tmp_path):
    from cav_recovery.recovery.pilot import train

    with pytest.raises(ValueError, match="preuve locale"):
        train(tmp_path / "absent-config.json", tmp_path / "run")


def test_single_decision_preserves_a_nonzero_policy_learning_signal():
    from cav_recovery.recovery.pilot import update

    torch.manual_seed(7)
    model = GraphActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    nodes = torch.ones(1, FEATURES)
    adjacency = torch.ones(1, 1)
    logits, value = model(nodes, adjacency)
    distribution = torch.distributions.Categorical(logits=logits)
    decision = dict(
        at_s=0.0,
        features=nodes.tolist(),
        adjacency=adjacency.tolist(),
        mask=[True] * len(option_bank()),
        action=0,
        old_logprob=float(distribution.log_prob(torch.tensor(0)).detach()),
        value=float(value.detach()),
    )
    before = model.actor.weight.detach().clone()
    config = dict(
        gamma_per_physics_decision=0.995,
        ppo_epochs=1,
        clip_ratio=0.2,
        value_coefficient=0.0,
        entropy_coefficient=0.0,
        gradient_norm=0.5,
    )
    metrics = update(model, optimizer, [decision], [2.0], config)
    assert metrics["policy_advantage_abs_mean"] > 0.01
    assert not torch.equal(before, model.actor.weight)
