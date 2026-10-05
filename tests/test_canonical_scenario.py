"""Non-régression des attentes C3 et de leurs frontières de validation."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from cav_recovery.canonical_scenario import CANONICAL_SCENARIO
from cav_recovery.empirical import scenario_contract, traffic_inputs
from cav_recovery.simulation import road_network, traffic_demand, traffic_run, traffic_scenario


def test_canonical_values_preserve_empirical_and_simulation_contracts():
    config = CANONICAL_SCENARIO
    assert config.sector.node_id == 250691665
    assert (config.sector.latitude, config.sector.longitude) == (37.9809145, 23.7308597)
    assert config.sector.observation_interval_s == (0.0, 802.8)
    assert config.sector.window_s == 60
    assert config.sector.categories == ("Car", "Taxi", "Motorcycle", "Bus", "Medium Vehicle", "Heavy Vehicle")
    assert config.contract.schema_version == "CGR-E03-1"
    assert config.contract.source_categories == ("Car", "Taxi")
    assert config.contract.load_levels == ("LOW", "MID", "HIGH")
    assert config.contract.group_sizes == (4, 5, 4)
    assert [(row.name, row.injection_s, row.entry_counts, row.mission_counts,
             row.classifiable_visits, row.censored_exit) for row in config.contract.loads] == [
        ("LOW", 240, (23, 72), (18, 5, 3, 69), 95, 0),
        ("MID", 300, (22, 144), (16, 6, 3, 141), 166, 0),
        ("HIGH", 240, (15, 141), (7, 8, 8, 133), 154, 2),
    ]
    assert config.network.center_node == str(config.sector.node_id)
    assert config.network.controller_id == config.network.center_node
    assert config.network.phases == ((39, "GGrrr"), (6, "yyrrr"), (39, "rrGGG"), (6, "rryyy"))
    assert sum(duration for duration, _ in config.network.phases) == config.network.cycle_s == 90
    assert (config.network.tls_type, config.network.program_id, config.network.offset_s) == ("static", "0", 0)
    assert (config.simulation.version, config.simulation.step_s, config.simulation.seed) == ("1.27.1", 0.5, 0)
    assert config.simulation.time_to_teleport_s == config.simulation.max_depart_delay_s == -1
    assert config.simulation.drain_horizon_s == 600
    assert dict(config.simulation.vehicle_type) == {
        "id": "passenger_CAV", "vClass": "passenger", "carFollowModel": "Krauss", "length": "5.0",
        "minGap": "2.5", "accel": "2.6", "decel": "4.5", "tau": "1.0", "sigma": "0",
        "speedFactor": "1.0", "guiShape": "passenger/sedan",
    }
    assert (config.empirical.source.size_bytes, config.empirical.source.sha256) == (
        199512534, "17970bd3f8e167df3ef54792e8fb7f874f89a9a0aceb14571321e9a48fbeea0d")
    assert (config.contract.identity.size_bytes, config.contract.identity.sha256) == (
        13055, "409564667c1ea1466bc41b70e4bdff3ef4922dbb648308fc526dabeb5c7f980a")
    assert (config.network.source.size_bytes, config.network.source.sha256) == (
        76486, "c5c2105c28807e8bcb2743ca53079bacef23ccca0bdf18d7314d218f51b1f2bf")
    assert len(config.empirical.profiles) == 6


@pytest.mark.parametrize("record,field", [
    (CANONICAL_SCENARIO, "sector"), (CANONICAL_SCENARIO.sector, "node_id"),
    (CANONICAL_SCENARIO.empirical.source, "sha256"), (CANONICAL_SCENARIO.contract, "group_sizes"),
    (CANONICAL_SCENARIO.contract.loads[0], "entry_counts"), (CANONICAL_SCENARIO.network, "phases"),
    (CANONICAL_SCENARIO.simulation, "step_s"),
])
def test_canonical_records_are_frozen(record, field):
    with pytest.raises(FrozenInstanceError):
        setattr(record, field, "changed")


@pytest.mark.parametrize("mapping", [CANONICAL_SCENARIO.network.routes,
                                    CANONICAL_SCENARIO.network.gate_edges,
                                    CANONICAL_SCENARIO.simulation.vehicle_type])
def test_canonical_collections_cannot_be_modified(mapping):
    key = next(iter(mapping))
    with pytest.raises(TypeError):
        mapping[key] = "changed"
    assert isinstance(CANONICAL_SCENARIO.network.routes[next(iter(CANONICAL_SCENARIO.network.routes))], tuple)


def test_compatibility_names_share_immutable_values():
    assert traffic_inputs.ENTRY_GATES is traffic_demand.ENTRY_GATES is CANONICAL_SCENARIO.sector.entry_gates
    assert traffic_inputs.EXIT_GATES is traffic_demand.EXIT_GATES is CANONICAL_SCENARIO.sector.exit_gates
    assert road_network.ROUTES is traffic_scenario.ROUTES is CANONICAL_SCENARIO.network.routes
    assert traffic_demand.VEHICLE_TYPE is CANONICAL_SCENARIO.simulation.vehicle_type
    assert scenario_contract.GROUP_SIZES is CANONICAL_SCENARIO.contract.group_sizes
    assert dict(traffic_inputs.EXPECTED_INPUT_SHA256) == {
        item.filename: item.sha256 for item in CANONICAL_SCENARIO.empirical.profiles}
    with pytest.raises(TypeError):
        traffic_inputs.EXPECTED_INPUT_SHA256["manifest.json"] = "changed"


@pytest.mark.parametrize("module,wrapper,core,args", [
    (traffic_inputs, "load_traffic_inputs", "_load_traffic_inputs", ("profiles", "coverage.json")),
    (traffic_demand, "read_contract", "_read_contract", ("contract.json",)),
    (traffic_demand, "demand_plans", "_demand_plans", ({},)),
    (road_network, "convert_network", "_convert_network", (Path("road.osm"), Path("scenario"))),
    (road_network, "inspect_network", "_inspect_network", (Path("network.net.xml"), Path("road.osm"))),
    (traffic_scenario, "validate_prepared_missions", "_validate_prepared_missions", (Path("scenario"), {})),
    (traffic_run, "verify_loaded_scenario", "_verify_loaded_scenario", (object(), [])),
])
def test_historical_wrappers_inject_canonical_scenario(monkeypatch, module, wrapper, core, args):
    operation = Mock(return_value={})
    monkeypatch.setattr(module, core, operation)
    getattr(module, wrapper)(*args)
    operation.assert_called_once()
    assert operation.call_args.args[-1] is CANONICAL_SCENARIO


def test_general_window_validation_does_not_require_canonical_period():
    windows = [(12.0, 15.0), (10.0, 12.0)]
    assert traffic_inputs._validate_window_structure(windows) == sorted(windows)
    with pytest.raises(traffic_inputs.ContractInputError, match="14 fenêtres"):
        traffic_inputs._validate_windows(windows, CANONICAL_SCENARIO.sector)


@pytest.mark.parametrize("windows", [[(0, 0)], [(0, float("inf"))], [(0, 2), (1, 3)]])
def test_general_window_validation_refuses_invalid_structure(windows):
    with pytest.raises(traffic_inputs.ContractInputError):
        traffic_inputs._validate_window_structure(windows)


def test_coherent_noncanonical_counts_remain_refused():
    path = Path(__file__).parent / "fixtures/traffic/demand.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert traffic_demand.demand_plans(contract)["LOW"]["entries"] == {
        "W23183369_IN": 2, "W284241336_IN": 3}
    with pytest.raises(traffic_demand.TrafficInputError, match="SHA"):
        traffic_demand.read_contract(path)


def test_code_digest_includes_canonical_configuration(monkeypatch):
    original = Path.read_bytes
    before = traffic_run.code_provenance()["sha256"]

    def read_bytes(path):
        contents = original(path)
        return contents + b"\n" if path.name == "canonical_scenario.py" else contents

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert traffic_run.code_provenance()["sha256"] != before
