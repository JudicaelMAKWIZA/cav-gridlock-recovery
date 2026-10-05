"""Préparation des fichiers nécessaires aux trois simulations nominales."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

from ..canonical_scenario import CANONICAL_SCENARIO, CanonicalScenario, SimulationSettings

from .road_network import ROUTES, GATE_EDGES, check_environment, convert_network, inspect_network, build_scenery, write_xml
from .traffic_demand import (TrafficInputError, STEP_S, read_contract,
                             allocate_counts, demand_plans, build_missions, mission_records, file_hash, CONTRACT_IDENTITY)


def new_output_directory(path: str | Path) -> Path:
    directory = Path(path).resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise TrafficInputError("Le dossier de sortie doit être absent ou vide.")
    return directory


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_view(path: Path, center: dict) -> None:
    """Le zoom et les formes automobiles n'agissent pas sur la dynamique."""
    root = ET.Element("viewsettings")
    scheme = ET.SubElement(root, "scheme", name="real world")
    ET.SubElement(scheme, "background", backgroundColor="238,240,235", showGrid="0")
    ET.SubElement(scheme, "edges", laneShowBorders="1", showLinkRules="1", showLinkDecals="1")
    # La valeur 2 est le mode natif « simple shapes », pas le mode triangle.
    ET.SubElement(scheme, "vehicles", vehicleQuality="2", vehicleMode="0",
                  vehicle_exaggeration="1.5", vehicle_minSize="1", showBlinker="1")
    ET.SubElement(root, "viewport", x=center["x"], y=center["y"], zoom="650")
    write_xml(path, root)


def write_traffic_files(directory: Path, missions: list, routes: dict, scenery: bool) -> None:
    _write_traffic_files(directory, missions, routes, scenery, CANONICAL_SCENARIO.simulation)


def _write_traffic_files(directory: Path, missions: list, routes: dict, scenery: bool, settings: SimulationSettings) -> None:
    root = ET.Element("routes")
    ET.SubElement(root, "vType", dict(settings.vehicle_type))
    for route_id, edges in sorted(routes.items()):
        ET.SubElement(root, "route", id=route_id, edges=" ".join(edges))
    for mission in missions:
        ET.SubElement(root, "vehicle", id=mission.vehicle_id, type=settings.vehicle_type["id"],
                      route=mission.route_id, depart=str(mission.scheduled_s))
    write_xml(directory / "traffic.rou.xml", root)
    config = ET.Element("configuration")
    inputs = ET.SubElement(config, "input")
    ET.SubElement(inputs, "net-file", value="../network.net.xml")
    ET.SubElement(inputs, "route-files", value="traffic.rou.xml")
    if scenery:
        ET.SubElement(inputs, "additional-files", value="../scenery.add.xml")
    time = ET.SubElement(config, "time")
    ET.SubElement(time, "step-length", value=str(settings.step_s))
    processing = ET.SubElement(config, "processing")
    ET.SubElement(processing, "time-to-teleport", value=str(settings.time_to_teleport_s))
    ET.SubElement(processing, "max-depart-delay", value=str(settings.max_depart_delay_s))
    random = ET.SubElement(config, "random_number")
    ET.SubElement(random, "seed", value=str(settings.seed))
    gui = ET.SubElement(config, "gui_only")
    ET.SubElement(gui, "gui-settings-file", value="../view.xml")
    write_xml(directory / "simulation.sumocfg", config)


def prepare_traffic(osm_path: str | Path, contract_path: str | Path, output_dir: str | Path, *,
                    failure_diagnostics_dir: str | Path | None = None) -> dict:
    """Prépare un réseau contrôlé et trois demandes, sans lancer de véhicules.

    Les sources restent inchangées. Le dossier entier n'est publié qu'après
    conversion, vérification et écriture de toutes les configurations.
    Sur demande explicite, les fichiers d'une préparation interrompue sont
    copiés séparément avant le nettoyage temporaire ; ils ne sont pas un scénario.
    """
    destination = new_output_directory(output_dir)
    diagnostics = None
    if failure_diagnostics_dir is not None:
        diagnostics = new_output_directory(failure_diagnostics_dir)
        if diagnostics.is_relative_to(destination) or destination.is_relative_to(diagnostics):
            raise TrafficInputError("Les dossiers de scénario et de diagnostic doivent être séparés.")
    contract = read_contract(contract_path)
    versions = check_environment("sumo", "netconvert")
    plans = demand_plans(contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".traffic-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        try:
            conversion = convert_network(Path(osm_path), stage)
            inspection = inspect_network(stage / "network.net.xml", Path(osm_path))
            # Le format écrit conserve des listes, même si les routes internes sont immuables.
            inspection = {**inspection, "routes": {name: list(edges) for name, edges in inspection["routes"].items()}}
            scenery = build_scenery(Path(osm_path), stage)
            write_view(stage / "view.xml", inspection["center"])
            regimes = {}
            for name, plan in plans.items():
                directory = stage / name
                directory.mkdir()
                missions = build_missions(name, plan, inspection["routes"])
                write_traffic_files(directory, missions, inspection["routes"], (stage / "scenery.add.xml").exists())
                regimes[name] = {"plan": plan, "missions": mission_records(missions)}
            manifest = {"schema_version": "traffic-scenario-1", "status": "prepared", "versions": versions,
                        "contract": {"filename": Path(contract_path).name, "sha256": CONTRACT_IDENTITY[1]},
                        "conversion": conversion, "network": inspection, "scenery": scenery,
                        "vehicle_type": dict(CANONICAL_SCENARIO.simulation.vehicle_type), "step_s": STEP_S,
                        "seed": CANONICAL_SCENARIO.simulation.seed, "regimes": regimes,
                        "limits": contract["limitations"],
                        "files_sha256": {p.relative_to(stage).as_posix(): file_hash(p)
                                         for p in sorted(stage.rglob("*")) if p.is_file() and p.suffix != ".log"}}
            write_json(stage / "scenario.json", manifest)
            existed = destination.exists()
            if existed:
                destination.rmdir()
            try:
                os.replace(stage, destination)
            except Exception:
                if existed and not destination.exists():
                    destination.mkdir()
                raise
        except BaseException as error:
            # Sauvegarder aussi sur interruption, puis relancer la même erreur.
            if diagnostics is not None:
                try:
                    if stage.exists() and any(stage.iterdir()):
                        new_output_directory(diagnostics)
                        shutil.copytree(stage, diagnostics, dirs_exist_ok=True)
                except BaseException as copy_error:
                    error.add_note(f"Sauvegarde des diagnostics impossible dans {diagnostics} : {copy_error}")
            raise
    return manifest


def validate_prepared_missions(directory: Path, manifest: dict) -> None:
    """Vérifie aussi la cohérence du manifeste avec les missions et les fichiers XML."""
    _validate_prepared_missions(directory, manifest, CANONICAL_SCENARIO)


def _validate_prepared_missions(directory: Path, manifest: dict, config: CanonicalScenario) -> None:
    settings = config.simulation
    routes_expected = {name: list(edges) for name, edges in config.network.routes.items()}
    if (manifest["vehicle_type"] != settings.vehicle_type or manifest["step_s"] != settings.step_s or manifest["seed"] != settings.seed
            or manifest["network"]["routes"] != routes_expected
            or manifest["network"]["gate_mapping"] != {**config.network.gate_edges, "entry_connector": config.network.entry_connector}
            or set(manifest["regimes"]) != set(config.contract.load_levels)):
        raise TrafficInputError("Paramètres ou mapping du scénario incompatibles.")
    for name, record in manifest["regimes"].items():
        plan = record["plan"]
        observed = plan["observed_movements"]
        for gate in config.sector.entry_gates:
            if (set(observed[gate]) != set(config.sector.exit_gates)
                    or sum(observed[gate].values()) != plan["observed_denominators"][gate]
                    or allocate_counts(plan["entries"][gate], observed[gate]) != plan["allocation"][gate]):
                raise TrafficInputError("Allocation ou distribution observée incohérente.")
        if sum(plan["entries"].values()) != sum(plan["observed_denominators"].values()) + plan["censored_exit"]:
            raise TrafficInputError("Les comptes préparés ne conservent pas les censures.")
        expected = build_missions(name, plan, routes_expected)
        if record["missions"] != mission_records(expected):
            raise TrafficInputError("Les missions ne correspondent pas à la demande préparée.")
        root = ET.parse(directory / name / "traffic.rou.xml").getroot()
        vehicle_types = root.findall("vType")
        routes = {r.get("id"): r.get("edges").split() for r in root.findall("route")}
        vehicles = [v.attrib for v in root.findall("vehicle")]
        expected_vehicles = [{"id": m.vehicle_id, "type": settings.vehicle_type["id"], "route": m.route_id,
                              "depart": str(m.scheduled_s)} for m in expected]
        if (len(vehicle_types) != 1 or vehicle_types[0].attrib != settings.vehicle_type or routes != routes_expected
                or len(root.findall("route")) != len(routes_expected) or vehicles != expected_vehicles):
            raise TrafficInputError("Les routes XML ne correspondent pas aux missions.")
        simulation_config = ET.parse(directory / name / "simulation.sumocfg").getroot()
        checks = {"input/net-file": "../network.net.xml", "input/route-files": "traffic.rou.xml",
                  "time/step-length": str(settings.step_s), "processing/time-to-teleport": str(settings.time_to_teleport_s),
                  "processing/max-depart-delay": str(settings.max_depart_delay_s), "random_number/seed": str(settings.seed),
                  "gui_only/gui-settings-file": "../view.xml"}
        for path, value in checks.items():
            row = simulation_config.find(path)
            if row is None or row.get("value") != value:
                raise TrafficInputError(f"Paramètre de simulation incompatible : {path}")


def read_scenario(directory: str | Path) -> dict:
    """Refuse des fichiers préparés qui ont changé depuis la construction."""
    directory = Path(directory).resolve()
    try:
        manifest = json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
        if manifest["schema_version"] != "traffic-scenario-1" or manifest["status"] != "prepared":
            raise TrafficInputError("Format de scénario incompatible.")
        required = {"network.net.xml", "view.xml"}
        required.update(f"{name}/{file}" for name in CANONICAL_SCENARIO.contract.load_levels
                        for file in ("traffic.rou.xml", "simulation.sumocfg"))
        if not required.issubset(manifest["files_sha256"]):
            raise TrafficInputError("Fichiers préparés obligatoires absents du manifeste.")
        for name, digest in manifest["files_sha256"].items():
            path = directory / name
            if path.resolve().is_relative_to(directory) is False or not path.is_file() or file_hash(path) != digest:
                raise TrafficInputError(f"Fichier préparé absent ou modifié : {name}")
        validate_prepared_missions(directory, manifest)
        return manifest
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ET.ParseError) as error:
        raise TrafficInputError(f"Scénario invalide : {error}") from error
