"""Valeurs partagées et contrôles d'identité du scénario C3."""

import json
from pathlib import Path

import pytest

from cav_recovery import c3_reference as reference
from cav_recovery.empirical import traffic_inputs
from cav_recovery.simulation import traffic_demand, traffic_run


def test_shared_c3_values():
    assert reference.C3_NODE_ID == 250691665
    assert reference.C3_SECTOR_ID == "C3"
    assert reference.ENTRY_GATES == ("W23183369_IN", "W284241336_IN")
    assert reference.EXIT_GATES == ("W23183369_OUT", "W284241336_OUT")
    assert reference.CONTRACT_SCHEMA == "CGR-E03-1"
    assert reference.SOURCE_CATEGORIES == ("Car", "Taxi")
    assert reference.CAV_POPULATION_ID == "passenger_CAV"
    assert reference.LOAD_LEVELS == ("LOW", "MID", "HIGH")
    assert reference.OBSERVATION_INTERVAL_S == (0.0, 802.8)


def test_general_window_validation_does_not_require_c3_period():
    windows = [(12.0, 15.0), (10.0, 12.0)]
    assert traffic_inputs._validate_window_structure(windows) == sorted(windows)
    with pytest.raises(traffic_inputs.ContractInputError, match="14 fenêtres"):
        traffic_inputs._validate_windows(windows)


def test_profile_structure_can_be_checked_before_c3_identity():
    configuration = {"sector_seed": {"sector": {"center_osm_node_id": 123}}}
    manifest = {
        "execution": {"status": "succeeded"}, "configuration": configuration,
        "inputs": {name: "synthetic" for name in (
            "cgr_e01_manifest_sha256", "geometry_sha256",
            "locally_computed_cgr_e01_export_sha256", "runtime_config_sha256", "sector_seed_sha256")},
        "software": {},
    }
    summary = {"execution": {"status": "succeeded"}}
    traffic_inputs._validate_profile_structure(manifest, configuration, summary)
    with pytest.raises(traffic_inputs.ContractInputError):
        traffic_inputs._validate_c3_identity(manifest, configuration, summary, {})


@pytest.mark.parametrize("windows", [[(0, 0)], [(0, float("inf"))], [(0, 2), (1, 3)]])
def test_general_window_validation_refuses_invalid_structure(windows):
    with pytest.raises(traffic_inputs.ContractInputError):
        traffic_inputs._validate_window_structure(windows)


def test_coherent_non_c3_counts_remain_refused():
    path = Path(__file__).parent / "fixtures/traffic/demand.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert traffic_demand.demand_plans(contract)["LOW"]["entries"] == {
        "W23183369_IN": 2, "W284241336_IN": 3}
    with pytest.raises(traffic_demand.TrafficInputError, match="SHA"):
        traffic_demand.read_contract(path)


def test_c3_demand_checks_gate_keys_not_dictionary_order():
    contract = {
        "schema_version": "CGR-E03-1", "status": "complete",
        "empirical_context": {
            "sector": {"id": "C3", "osm_node_id": 250691665,
                       "entry_gates": ["W23183369_IN", "W284241336_IN"],
                       "exit_gates": ["W23183369_OUT", "W284241336_OUT"]},
            "coverage": {"status": "known", "intervals": [[0.0, 802.8]]},
        },
        "passenger_cav_contract": {"source_categories": ["Car", "Taxi"],
                                   "population_id": "passenger_CAV",
                                   "regimes": [{"regime_id": "HIGH", "classifiable_visits": 154,
                                                "censored_exit": 2}]},
    }
    plans = {}
    for name, duration, first, second, movements in [
        ("LOW", 240, 23, 72, (18, 5, 3, 69)),
        ("MID", 300, 22, 144, (16, 6, 3, 141)),
        ("HIGH", 240, 15, 141, (7, 8, 8, 133)),
    ]:
        plans[name] = {
            "injection_s": duration,
            "entries": {"W284241336_IN": second, "W23183369_IN": first},
            "allocation": {
                "W284241336_IN": {"W284241336_OUT": movements[3], "W23183369_OUT": movements[2]},
                "W23183369_IN": {"W284241336_OUT": movements[1], "W23183369_OUT": movements[0]},
            },
        }
    traffic_demand._validate_c3_contract(contract, plans)
    plans["LOW"]["allocation"]["W23183369_IN"] = {"W23183369_OUT": 5, "W284241336_OUT": 18}
    with pytest.raises(traffic_demand.TrafficInputError, match="Effectifs"):
        traffic_demand._validate_c3_contract(contract, plans)


def test_code_digest_includes_c3_reference(monkeypatch):
    original = Path.read_bytes
    before = traffic_run.code_provenance()["sha256"]

    def read_bytes(path):
        contents = original(path)
        return contents + b"\n" if path.name == "c3_reference.py" else contents

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert traffic_run.code_provenance()["sha256"] != before
