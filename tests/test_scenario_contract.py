import csv
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cav_recovery.empirical import contract_files
from cav_recovery.empirical import scenario_contract
from cav_recovery.empirical import traffic_inputs
from cav_recovery.empirical.scenario_contract import (
    ContractInputError,
    _aggregate_movements,
    build_empirical_contract,
)
from cav_recovery.empirical.traffic_inputs import COVERAGE_SCHEMA_VERSION, PROFILE_SCHEMA_VERSION


FIXTURES = Path(__file__).parent / "fixtures" / "scenario_contract"
ROOT = Path(__file__).parents[1]
CATEGORIES = traffic_inputs.ALL_CATEGORIES
ENTRIES = traffic_inputs.ENTRY_GATES
EXITS = traffic_inputs.EXIT_GATES
GATES = traffic_inputs.ALL_GATES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes(directory, coverage):
    names = ["manifest.json", "sector_config.json", "quality_summary.json", "flow_profile.csv", "movement_profile.csv"]
    return {**{name: digest(directory / name) for name in names}, "coverage.json": digest(coverage)}


def patch_input_hashes(hashes):
    return patch.object(traffic_inputs, "EXPECTED_INPUT_SHA256", hashes)


def load_cli_module():
    spec = importlib.util.spec_from_file_location("build_empirical_contract_cli", ROOT / "scripts" / "build_empirical_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_identity_documents(root):
    source = {
        "filename": "20181024_d3_0830_0900.csv",
        "sha256": "17970bd3f8e167df3ef54792e8fb7f874f89a9a0aceb14571321e9a48fbeea0d",
        "size_bytes": 199512534,
    }
    windows = [(index * 60.0, min((index + 1) * 60.0, 802.8)) for index in range(14)]
    coverage_by_gate = {gate: [[0.0, 802.8]] for gate in GATES}
    branches = [{"id": gate, "role": "entry" if gate in ENTRIES else "exit"} for gate in GATES]
    config = {
        "schema": PROFILE_SCHEMA_VERSION.replace("-1", "-runtime-used-1"),
        "aggregation": {"origin_s": 0.0, "terminal_s": 802.8, "window_s": 60.0, "interval_convention": "[a,b)"},
        "coverage": coverage_by_gate,
        "sector_seed": {
            "sector": {"id": "PNEUMA_D3_NODE_250691665", "center_osm_node_id": 250691665},
            "branches": branches,
        },
    }
    profiles = root / "profiles"
    profiles.mkdir()
    (profiles / "sector_config.json").write_text(json.dumps(config), encoding="utf-8")
    manifest = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "execution": {"status": "succeeded"},
        "source": source,
        "configuration": config,
        "inputs": {
            "cgr_e01_manifest_sha256": "a" * 64,
            "geometry_sha256": "b" * 64,
            "locally_computed_cgr_e01_export_sha256": {"observations.csv.gz": "c" * 64},
            "runtime_config_sha256": "d" * 64,
            "sector_seed_sha256": "e" * 64,
        },
        "software": {"package_version": "synthetic", "cgr_e02_code_sha256": "f" * 64},
    }
    (profiles / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    coverage = root / "coverage.json"
    coverage.write_text(json.dumps({
        "schema": COVERAGE_SCHEMA_VERSION,
        "status": "known",
        "intervals": [[0.0, 802.8]],
        "source_time_range": source,
    }), encoding="utf-8")
    return profiles, coverage, windows


def prepare_inputs(root):
    profile = json.loads((FIXTURES / "profile.json").read_text(encoding="utf-8"))
    profiles, coverage, windows = write_identity_documents(root)

    entry_counts = Counter()
    for index, (start, _) in enumerate(windows):
        total = profile["passenger_in_total"][index]
        first = profile["passenger_in_w23183369"][index]
        entry_counts[(ENTRIES[0], "Car", start)] = first
        entry_counts[(ENTRIES[1], "Car", start)] = total - first
    for category, count in profile["other_category_entries_at_zero"].items():
        entry_counts[(ENTRIES[0], category, 0.0)] = count

    censored_exit = profile["censored_exit"]
    censored_key = (censored_exit["entry_gate"], censored_exit["category"], censored_exit["window_start_s"])
    movement_counts = Counter()
    denominators = {}
    for entry in ENTRIES:
        for category in CATEGORIES:
            for start, _ in windows:
                denominator = entry_counts[(entry, category, start)]
                if (entry, category, start) == censored_key:
                    denominator -= censored_exit["count"]
                denominators[(entry, category, start)] = denominator
                first_exit = denominator // 2
                movement_counts[(entry, EXITS[0], category, start)] = first_exit
                movement_counts[(entry, EXITS[1], category, start)] = denominator - first_exit

    censored_entry = profile["censored_entry"]
    exit_counts = Counter()
    for entry in ENTRIES:
        for exit_gate in EXITS:
            for category in CATEGORIES:
                for start, _ in windows:
                    exit_counts[(exit_gate, category, start)] += movement_counts[(entry, exit_gate, category, start)]
    exit_counts[(EXITS[0], censored_entry["category"], censored_entry["window_start_s"])] += censored_entry["count"]

    flow_rows = []
    for gate in GATES:
        for category in CATEGORIES:
            for start, end in windows:
                count = entry_counts[(gate, category, start)] if gate in ENTRIES else exit_counts[(gate, category, start)]
                duration = end - start
                flow_rows.append({
                    "gate_id": gate,
                    "direction": "entry" if gate in ENTRIES else "exit",
                    "category": category,
                    "window_start_s": start,
                    "window_end_s": end,
                    "raw_passages": count,
                    "passages_in_exposure": count,
                    "exposure_s": duration,
                    "flow_veh_per_h": 3600 * count / duration,
                    "coverage_status": "known",
                    "coverage_assumption": "intervals_explicitly_provided",
                })
    flow_fields = ["gate_id", "direction", "category", "window_start_s", "window_end_s", "raw_passages", "passages_in_exposure", "exposure_s", "flow_veh_per_h", "coverage_status", "coverage_assumption"]
    write_csv(profiles / "flow_profile.csv", flow_fields, flow_rows)

    movement_rows = []
    for entry in ENTRIES:
        for exit_gate in EXITS:
            for category in CATEGORIES:
                for start, end in windows:
                    count = movement_counts[(entry, exit_gate, category, start)]
                    denominator = denominators[(entry, category, start)]
                    movement_rows.append({
                        "record_type": "movement",
                        "entry_gate": entry,
                        "exit_gate": exit_gate,
                        "category": category,
                        "window_start_s": start,
                        "window_end_s": end,
                        "count": count,
                        "denominator_classifiable": denominator,
                        "proportion": "" if denominator == 0 else count / denominator,
                        "visit_status": "classifiable",
                    })
    movement_rows.extend([
        {
            "record_type": "unclassified_visit",
            "entry_gate": censored_exit["entry_gate"],
            "exit_gate": "",
            "category": censored_exit["category"],
            "window_start_s": censored_exit["window_start_s"],
            "window_end_s": censored_exit["window_start_s"] + 60.0,
            "count": censored_exit["count"],
            "denominator_classifiable": "",
            "proportion": "",
            "visit_status": "censored_exit",
        },
        {
            "record_type": "unclassified_visit",
            "entry_gate": "NO_OBSERVED_ENTRY",
            "exit_gate": "",
            "category": censored_entry["category"],
            "window_start_s": censored_entry["window_start_s"],
            "window_end_s": censored_entry["window_start_s"] + 60.0,
            "count": censored_entry["count"],
            "denominator_classifiable": "",
            "proportion": "",
            "visit_status": "censored_entry",
        },
    ])
    movement_fields = ["record_type", "entry_gate", "exit_gate", "category", "window_start_s", "window_end_s", "count", "denominator_classifiable", "proportion", "visit_status"]
    write_csv(profiles / "movement_profile.csv", movement_fields, movement_rows)

    by_gate = {gate: sum(row["passages_in_exposure"] for row in flow_rows if row["gate_id"] == gate) for gate in GATES}
    by_category = {category: sum(row["passages_in_exposure"] for row in flow_rows if row["category"] == category) for category in CATEGORIES}
    classifiable = sum(denominators.values())
    summary = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "execution": {"status": "succeeded"},
        "method": "oriented_finite_virtual_gates",
        "counts": {
            "crossings": sum(by_gate.values()),
            "visits": classifiable + censored_exit["count"] + censored_entry["count"],
            "visits_by_status": {
                "classifiable": classifiable,
                "censored_exit": censored_exit["count"],
                "censored_entry": censored_entry["count"],
            },
        },
        "crossings_by_gate": by_gate,
        "crossings_by_category": by_category,
    }
    (profiles / "quality_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return profiles, coverage


class ScenarioContractTests(unittest.TestCase):
    def run_builder(self, profiles, coverage, output):
        with patch_input_hashes(input_hashes(profiles, coverage)):
            return build_empirical_contract(profiles, coverage, output)

    def test_builds_four_deterministic_artifacts_and_applies_4_5_4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, coverage = prepare_inputs(root)
            first, second = root / "first", root / "second"
            self.run_builder(profiles, coverage, first)
            self.run_builder(profiles, coverage, second)
            self.assertEqual(sorted(path.name for path in first.iterdir()), sorted(contract_files.OUTPUTS))
            for name in contract_files.OUTPUTS:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
            with (first / "regime_profile.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 14)
            self.assertEqual(Counter(row["regime_id"] for row in rows), {"LOW": 4, "MID": 5, "HIGH": 4, "": 1})
            self.assertEqual(rows[0]["regime_id"], "LOW")
            self.assertEqual(rows[1]["regime_id"], "MID")
            self.assertEqual(rows[-1]["eligible_for_stratification"], "false")
            self.assertEqual(rows[-1]["regime_id"], "")

    def test_contract_uses_only_car_and_taxi_but_keeps_six_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, coverage = prepare_inputs(root)
            output = root / "out"
            self.run_builder(profiles, coverage, output)
            contract = json.loads((output / "empirical_contract.json").read_text(encoding="utf-8"))
            self.assertEqual(contract["passenger_cav_contract"]["source_categories"], ["Car", "Taxi"])
            self.assertIn("input_artifacts_sha256", contract["inputs"])
            self.assertEqual(contract["passenger_cav_contract"]["all_window_passenger_entries"], 84)
            composition = contract["empirical_context"]["composition_observed"]
            self.assertEqual(set(composition["categories"]), set(CATEGORIES))
            self.assertEqual(composition["categories"]["Motorcycle"]["observed_entries"], 1)
            self.assertEqual(composition["passenger_entries"], 84)

    def test_rates_shares_movements_and_censures_use_pooled_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, coverage = prepare_inputs(root)
            output = root / "out"
            self.run_builder(profiles, coverage, output)
            contract = json.loads((output / "empirical_contract.json").read_text(encoding="utf-8"))
            regimes = {row["regime_id"]: row for row in contract["passenger_cav_contract"]["regimes"]}
            for regime in regimes.values():
                self.assertAlmostEqual(sum(regime["passenger_entry_shares"].values()), 1.0)
                self.assertEqual(regime["combined_reference_flow_veh_per_h"], 3600 * regime["passenger_entry_total"] / regime["exposure_s"])
                self.assertEqual(regime["passenger_entry_total"], regime["classifiable_visits"] + regime["censored_exit"])
                for entry in ENTRIES:
                    self.assertEqual(sum(regime["passenger_movement_counts"][entry].values()), regime["passenger_movement_denominators"][entry])
            self.assertEqual(regimes["HIGH"]["censored_exit"], 1)
            self.assertEqual(regimes["LOW"]["censored_exit"], 0)

    def test_zero_movement_denominator_produces_null_probabilities(self):
        rows = {}
        for entry in ENTRIES:
            for exit_gate in EXITS:
                for category in scenario_contract.SOURCE_CATEGORIES:
                    denominator = 3 if entry == ENTRIES[1] and category == "Car" else 0
                    count = denominator if exit_gate == EXITS[0] else 0
                    rows[(entry, exit_gate, category, 0.0)] = {"count": count, "denominator": denominator}
        counts, denominators, probabilities = _aggregate_movements({0.0}, rows)
        self.assertEqual(denominators[ENTRIES[0]], 0)
        self.assertEqual(probabilities[ENTRIES[0]], {EXITS[0]: None, EXITS[1]: None})
        self.assertEqual(denominators[ENTRIES[1]], 3)
        self.assertEqual(sum(counts[ENTRIES[1]].values()), 3)

    def test_missing_required_file_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, coverage = prepare_inputs(root)
            expected = input_hashes(profiles, coverage)
            (profiles / "flow_profile.csv").unlink()
            with patch_input_hashes(expected):
                with self.assertRaisesRegex(ContractInputError, "obligatoire absent"):
                    build_empirical_contract(profiles, coverage, root / "out")

    def test_wrong_fingerprint_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, coverage = prepare_inputs(root)
            expected = input_hashes(profiles, coverage)
            (profiles / "flow_profile.csv").write_text("altéré\n", encoding="utf-8")
            with patch_input_hashes(expected):
                with self.assertRaisesRegex(ContractInputError, "Empreinte SHA-256"):
                    build_empirical_contract(profiles, coverage, root / "out")

    def test_missing_csv_column_is_refused_after_hash_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, coverage = prepare_inputs(root)
            path = profiles / "flow_profile.csv"
            rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
            fields = [field for field in rows[0] if field != "passages_in_exposure"]
            write_csv(path, fields, ({key: value for key, value in row.items() if key in fields} for row in rows))
            with patch_input_hashes(input_hashes(profiles, coverage)):
                with self.assertRaisesRegex(ContractInputError, "Colonnes absentes"):
                    build_empirical_contract(profiles, coverage, root / "out")

    def test_incompatible_identity_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, coverage = prepare_inputs(root)
            manifest_path = profiles / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source"]["filename"] = "autre.csv"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch_input_hashes(input_hashes(profiles, coverage)):
                with self.assertRaisesRegex(ContractInputError, "source pNEUMA"):
                    build_empirical_contract(profiles, coverage, root / "out")

    def test_nonempty_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, coverage = prepare_inputs(root)
            output = root / "out"
            output.mkdir()
            (output / "keep.txt").write_text("à conserver", encoding="utf-8")
            with self.assertRaisesRegex(ContractInputError, "n'est pas vide"):
                self.run_builder(profiles, coverage, output)
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "à conserver")

    def test_provenance_limits_and_epistemic_statuses_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, coverage = prepare_inputs(root)
            output = root / "out"
            self.run_builder(profiles, coverage, output)
            contract = json.loads((output / "empirical_contract.json").read_text(encoding="utf-8"))
            report = (output / "regime_report.md").read_text(encoding="utf-8")
            for limitation in scenario_contract.PROVENANCE_LIMITATIONS:
                self.assertIn(limitation, contract["limitations"])
                self.assertIn(limitation, report)
            self.assertIn("## Statut des informations", report)
            self.assertIn("OBSERVÉ", report)
            self.assertIn("DÉRIVÉ", report)
            self.assertIn("SUPPOSÉ", report)

    def test_outputs_use_censoring_term(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, coverage = prepare_inputs(root)
            output = root / "out"
            self.run_builder(profiles, coverage, output)
            contents = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir())
            forbidden_term = "censor" + "ship"
            self.assertNotIn(forbidden_term, contents.lower())
            contract = json.loads((output / "empirical_contract.json").read_text(encoding="utf-8"))
            statuses = contract["empirical_context"]["epistemic_status"]
            self.assertIn("observed_counts_and_censoring", statuses)

    def test_promotion_failure_leaves_existing_destination_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, coverage = prepare_inputs(root)
            output = root / "out"
            output.mkdir()
            with patch.object(contract_files.os, "replace", side_effect=OSError("promotion impossible")):
                with self.assertRaisesRegex(OSError, "promotion impossible"):
                    self.run_builder(profiles, coverage, output)
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(list(root.glob(".empirical-contract-*")), [])

    def test_empty_output_directory_accepts_atomic_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, coverage = prepare_inputs(root)
            output = root / "out"
            output.mkdir()
            self.run_builder(profiles, coverage, output)
            self.assertEqual(sorted(path.name for path in output.iterdir()), sorted(contract_files.OUTPUTS))

    def test_cli_refuses_incompatible_inputs_with_code_2(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([
                sys.executable,
                "scripts/build_empirical_contract.py",
                "--profile-dir", directory,
                "--coverage", str(Path(directory) / "missing.json"),
                "--output-dir", str(Path(directory) / "out"),
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("Contrat refusé", result.stderr)

    def test_cli_returns_code_0_on_success(self):
        module = load_cli_module()
        summary = {"status": "complete", "windows": {"total": 14, "complete": 13, "partial": 1}}
        arguments = ["build_empirical_contract.py", "--profile-dir", "profiles", "--coverage", "coverage.json", "--output-dir", "out"]
        with patch.object(module, "build_empirical_contract", return_value=summary), patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(), 0)

    def test_cli_returns_code_1_on_technical_error(self):
        module = load_cli_module()
        arguments = ["build_empirical_contract.py", "--profile-dir", "profiles", "--coverage", "coverage.json", "--output-dir", "out"]
        error_output = io.StringIO()
        with patch.object(module, "build_empirical_contract", side_effect=OSError("écriture impossible")), patch.object(sys, "argv", arguments), redirect_stderr(error_output):
            self.assertEqual(module.main(), 1)
        self.assertIn("Erreur technique", error_output.getvalue())
