"""Valeurs figées de l'expérience C3, communes aux profils et à la simulation.

Ces attentes protègent une expérience déjà vérifiée ; elles ne constituent pas
une interface de configuration pour d'autres secteurs.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class FileIdentity:
    filename: str
    sha256: str
    size_bytes: int | None = None

    def source_record(self) -> dict:
        return {"filename": self.filename, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class EmpiricalProvenance:
    source: FileIdentity
    profiles: tuple[FileIdentity, ...]
    profile_schema: str
    coverage_schema: str
    extraction_method: str


@dataclass(frozen=True)
class Sector:
    name: str
    seed_id: str
    node_id: int
    latitude: float
    longitude: float
    entry_gates: tuple[str, ...]
    exit_gates: tuple[str, ...]
    observation_interval_s: tuple[float, float]
    window_s: int
    interval_convention: str
    categories: tuple[str, ...]

    @property
    def gates(self) -> tuple[str, ...]:
        return self.entry_gates + self.exit_gates


@dataclass(frozen=True)
class LoadExpectation:
    name: str
    injection_s: int
    entry_counts: tuple[int, ...]
    mission_counts: tuple[int, ...]
    classifiable_visits: int
    censored_exit: int


@dataclass(frozen=True)
class ContractRules:
    schema_version: str
    identity: FileIdentity
    source_categories: tuple[str, ...]
    load_levels: tuple[str, ...]
    group_sizes: tuple[int, ...]
    loads: tuple[LoadExpectation, ...]


@dataclass(frozen=True)
class Network:
    source: FileIdentity
    center_node: str
    gate_edges: Mapping[str, str]
    routes: Mapping[str, tuple[str, ...]]
    connector_entry_gate: str
    entry_connector: str
    controller_id: str
    program_id: str
    tls_type: str
    offset_s: int
    cycle_s: int
    phases: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class SimulationSettings:
    version: str
    step_s: float
    seed: int
    vehicle_type: Mapping[str, str]
    time_to_teleport_s: int
    max_depart_delay_s: int
    drain_horizon_s: int


@dataclass(frozen=True)
class CanonicalScenario:
    empirical: EmpiricalProvenance
    sector: Sector
    contract: ContractRules
    network: Network
    simulation: SimulationSettings


CANONICAL_SCENARIO = CanonicalScenario(
    empirical=EmpiricalProvenance(
        source=FileIdentity("20181024_d3_0830_0900.csv", "17970bd3f8e167df3ef54792e8fb7f874f89a9a0aceb14571321e9a48fbeea0d", 199512534),
        profiles=(
            FileIdentity("manifest.json", "d6b659330d708c2d766088750a7e27ef44a25b435531a206d04d799e774da1af"),
            FileIdentity("sector_config.json", "c62bd8543bfbe655e78ae067bf3fe8ae0f6bee14cc19ccda2deb50e8ca6300d2"),
            FileIdentity("quality_summary.json", "22039ca750de97e74f0c6412f59f34514232d18fb7491c6e63686648986021b8"),
            FileIdentity("flow_profile.csv", "b33be95fe52a1968a01f9b3e83e1423a8c883883a9e91287d610ee2edf56e2a8"),
            FileIdentity("movement_profile.csv", "2fc9915f7eb4433873d8e7de37e413b6dfb962abf8769c57b27d50c45a442dda"),
            FileIdentity("coverage.json", "c856992dc1733bd0aa131787edf8609d44a9928343b1919c1bec00285d02038d"),
        ),
        profile_schema="CGR-E02-1",
        coverage_schema="CGR-E02-coverage-evidence-2",
        extraction_method="oriented_finite_virtual_gates",
    ),
    sector=Sector(
        name="C3", seed_id="PNEUMA_D3_NODE_250691665", node_id=250691665,
        latitude=37.9809145, longitude=23.7308597,
        entry_gates=("W23183369_IN", "W284241336_IN"),
        exit_gates=("W23183369_OUT", "W284241336_OUT"),
        observation_interval_s=(0.0, 802.8), window_s=60, interval_convention="[a,b)",
        categories=("Car", "Taxi", "Motorcycle", "Bus", "Medium Vehicle", "Heavy Vehicle"),
    ),
    contract=ContractRules(
        schema_version="CGR-E03-1",
        identity=FileIdentity("empirical_contract.json", "409564667c1ea1466bc41b70e4bdff3ef4922dbb648308fc526dabeb5c7f980a", 13055),
        source_categories=("Car", "Taxi"), load_levels=("LOW", "MID", "HIGH"), group_sizes=(4, 5, 4),
        loads=(
            LoadExpectation("LOW", 240, (23, 72), (18, 5, 3, 69), 95, 0),
            LoadExpectation("MID", 300, (22, 144), (16, 6, 3, 141), 166, 0),
            LoadExpectation("HIGH", 240, (15, 141), (7, 8, 8, 133), 154, 2),
        ),
    ),
    network=Network(
        source=FileIdentity("cgr_e02_roads_2018.osm", "c5c2105c28807e8bcb2743ca53079bacef23ccca0bdf18d7314d218f51b1f2bf", 76486),
        center_node="250691665",
        gate_edges=MappingProxyType({
            "W23183369_IN": "23183369#1", "W284241336_IN": "284241336#1",
            "W23183369_OUT": "23183369#5", "W284241336_OUT": "284241336#3",
        }),
        routes=MappingProxyType({
            "W23183369_IN__W23183369_OUT": ("23183369#1", "23183369#2", "23183369#3", "23183369#4", "23183369#5"),
            "W23183369_IN__W284241336_OUT": ("23183369#1", "23183369#2", "23183369#3", "284241336#3"),
            "W284241336_IN__W23183369_OUT": ("284241336#1", "284241336#2", "23183369#4", "23183369#5"),
            "W284241336_IN__W284241336_OUT": ("284241336#1", "284241336#2", "284241336#3"),
        }),
        connector_entry_gate="W23183369_IN", entry_connector=":2725672310_0",
        controller_id="250691665", program_id="0", tls_type="static", offset_s=0, cycle_s=90,
        phases=((39, "GGrrr"), (6, "yyrrr"), (39, "rrGGG"), (6, "rryyy")),
    ),
    simulation=SimulationSettings(
        version="1.27.1", step_s=0.5, seed=0,
        vehicle_type=MappingProxyType({
            "id": "passenger_CAV", "vClass": "passenger", "carFollowModel": "Krauss",
            "length": "5.0", "minGap": "2.5", "accel": "2.6", "decel": "4.5",
            "tau": "1.0", "sigma": "0", "speedFactor": "1.0", "guiShape": "passenger/sedan",
        }),
        time_to_teleport_s=-1, max_depart_delay_s=-1, drain_horizon_s=600,
    ),
)
