"""Contrôles du transport applicatif sans accès aux états globaux des CAV."""

import copy

import pytest

from cav_recovery.simulation.v2v import V2VExchange


def reading(x, *, distance=0, speed=0):
    return {"position": (x, 0.0), "distance": distance, "speed": speed,
            "lane": "road_0", "route": ("road", "exit"), "route_index": 0}


def test_tous_les_presents_participent_mais_pas_les_non_inseres():
    exchange = V2VExchange(range_m=20)
    events = exchange.observe(0.5, {"a": reading(0), "b": reading(10), "loin": reading(100)})
    assert {row["vehicle_id"] for row in events if row["event"] == "joined"} == {"a", "b", "loin"}
    assert exchange.snapshot()["active_cav"] == 3
    assert exchange.snapshot()["participants"]["a"]["received"] == {}
    exchange.observe(1.0, {"a": reading(0, distance=1), "b": reading(10, distance=1),
                           "loin": reading(100, distance=1)})
    state = exchange.snapshot()
    assert state["counts"]["emitted"] == 3
    assert state["counts"]["delivered"] == 2
    assert state["counts"]["out_of_range_pairs"] == 4
    assert set(state["participants"]["a"]["received"]) == {"b"}
    assert set(state["participants"]["b"]["received"]) == {"a"}
    assert state["participants"]["loin"]["received"] == {}
    exchange.observe(1.5, {"a": reading(0, distance=2), "b": reading(10, distance=2),
                           "loin": reading(100, distance=2)})
    assert exchange.snapshot()["recent_emissions"]
    assert "route" not in state["participants"]["a"]["received"]["b"]["message"]
    assert state["participants"]["a"]["own"]["progress_delta_m"] == 1
    assert state["participants"]["a"]["own"]["intended_next_edge"] == "exit"


def test_voisins_changent_et_information_perime_sans_devenir_connue():
    exchange = V2VExchange(range_m=20, interval_s=1, lifetime_s=1)
    exchange.observe(0.5, {"a": reading(0), "b": reading(30)})
    exchange.observe(1.0, {"a": reading(0), "b": reading(10)})
    received = copy.deepcopy(exchange.local["a"]["received"]["b"])
    assert received["message"]["observed_at_s"] == 1.0
    assert received["received_at_s"] == 1.0
    exchange.observe(1.5, {"a": reading(0), "b": reading(50)})
    assert exchange.neighbors["a"] == []
    assert exchange.local["a"]["received"]["b"] == received
    exchange.observe(2.0, {"a": reading(0), "b": reading(50)})
    assert "b" in exchange.local["a"]["received"]
    events = exchange.observe(2.5, {"a": reading(0), "b": reading(50)})
    assert "b" not in exchange.local["a"]["received"]
    assert exchange.local["a"]["expired"]["b"]["valid_until_s"] == 2.0
    assert any(row["event"] == "expired" and row["vehicle_id"] == "a" for row in events)
    exchange.observe(3.0, {"a": reading(0), "b": reading(5)})
    assert "b" in exchange.local["a"]["received"]
    assert "b" not in exchange.local["a"]["expired"]


def test_insertions_arrivees_et_doublons():
    exchange = V2VExchange(range_m=20)
    exchange.observe(0.5, {"a": reading(0)})
    exchange.observe(1.0, {"a": reading(0), "b": reading(2)})
    message = exchange.local["a"]["received"]["b"]["message"]
    assert exchange._deliver("a", message, 1.0) is False
    assert exchange.counts["duplicate_rejected"] == 1
    events = exchange.observe(1.5, {"b": reading(2)})
    assert {row["vehicle_id"] for row in events if row["event"] == "left"} == {"a"}
    assert "a" not in exchange.local
    assert exchange.snapshot()["active_cav"] == 1
    exchange.observe(2.0, {"b": reading(2)})
    assert "delivered_to" not in exchange.knowledge("b")["last_sent"]


def test_connaissance_locale_et_diagnostic_transport_restent_separes():
    exchange = V2VExchange(range_m=20)
    exchange.observe(0.5, {"a": reading(0), "b": reading(200)})
    state = exchange.snapshot()
    assert state["participants"]["a"]["received"] == {}
    assert "b" not in state["participants"]["a"]
    assert state["transport_distances_m"]["a|b"] == 200
    assert "transport_distances_m" not in exchange.knowledge("a")
    assert exchange.knowledge("missing") is None
    exchange.knowledge("a")["own"]["lane_id"] = "invented"
    assert exchange.knowledge("a")["own"]["lane_id"] == "road_0"
    state["participants"]["a"]["own"]["lane_id"] = "invented"
    assert exchange.local["a"]["own"]["lane_id"] == "road_0"


def test_rejeu_identique_et_temps_non_causal_refuse():
    def trace():
        exchange = V2VExchange(range_m=20)
        exchange.observe(0.5, {"b": reading(5), "a": reading(0)})
        exchange.observe(1.0, {"a": reading(0), "b": reading(5)})
        return exchange.snapshot()

    assert trace() == trace()
    exchange = V2VExchange()
    with pytest.raises(ValueError, match="temps V2V"):
        exchange.observe(1.0, {})
    exchange.observe(0.5, {})
    with pytest.raises(ValueError, match="temps V2V"):
        exchange.observe(0.5, {})


def test_message_du_futur_et_mesure_non_finie_refuses():
    exchange = V2VExchange(range_m=20)
    bad = reading(0)
    bad["speed"] = float("nan")
    with pytest.raises(ValueError, match="V2V invalide"):
        exchange.observe(0.5, {"a": bad})
    exchange = V2VExchange(range_m=20)
    exchange.observe(0.5, {"a": reading(0), "b": reading(5)})
    exchange.observe(1.0, {"a": reading(0), "b": reading(5)})
    message = copy.deepcopy(exchange.local["a"]["received"]["b"]["message"])
    message["sequence_number"] += 1
    message["observed_at_s"] = 2.0
    with pytest.raises(ValueError, match="hors de sa période"):
        exchange._deliver("a", message, 1.0)
    exchange = V2VExchange()
    exchange.observe(0.5, {"a": reading(0, distance=5)})
    with pytest.raises(ValueError, match="ne peut pas reculer"):
        exchange.observe(1.0, {"a": reading(0, distance=4)})


@pytest.mark.parametrize("arguments", [
    {"range_m": 0}, {"interval_s": 0.75}, {"lifetime_s": float("nan")},
    {"step_s": -0.5}, {"range_m": True},
])
def test_parametres_invalides_refuses(arguments):
    with pytest.raises(ValueError):
        V2VExchange(**arguments)
