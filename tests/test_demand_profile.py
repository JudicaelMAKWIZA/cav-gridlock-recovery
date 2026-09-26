import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cav_recovery.empirical.demand_profile import (
    CoverageInterval,
    ProfileInputError,
    Visit,
    build_flow_rows,
    build_movement_rows,
    load_runtime_configuration,
    profile_pneuma,
    reconstruct_visits,
)
from cav_recovery.empirical.sector import Crossing, load_sector


FIXTURES = Path(__file__).parent / "fixtures" / "sector_profile"
ROOT = Path(__file__).parents[1]
DEGREE_PER_METER = 1.0 / 111_195.0


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crossing(gate, role, time_s, continuity=0):
    return Crossing(2, "t", "Car", gate, role, continuity, int(time_s * 10), int(time_s * 10 + 1), time_s, time_s - 0.1, time_s + 0.1)


def prepare_inputs(root):
    source = root / "synthetic.csv"
    geometry = root / "synthetic.osm"
    source.write_text("source synthétique réservée au contrôle d'identité\n", encoding="utf-8")
    geometry.write_text("<osm version=\"0.6\"></osm>\n", encoding="utf-8")
    seed = json.loads((FIXTURES / "sector_seed.json").read_text(encoding="utf-8"))
    seed["source"]["pneuma_sha256"] = digest(source)
    seed["geometry"]["osm_sha256"] = digest(geometry)
    seed_path = root / "seed.json"
    seed_path.write_text(json.dumps(seed), encoding="utf-8")
    runtime = root / "runtime.json"
    runtime.write_text((FIXTURES / "runtime_config.json").read_text(encoding="utf-8"), encoding="utf-8")
    e01 = root / "e01"
    e01.mkdir()
    manifest = {"schema_version": "CGR-E01-1", "status": "complete", "source": {"filename": source.name, "sha256": digest(source), "size_bytes": source.stat().st_size}}
    (e01 / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary = {"schema_version": "CGR-E01-1", "status": "complete", "time_seconds": {"min": 0.0, "max": 8.0}}
    (e01 / "quality_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (e01 / "trajectories.csv").write_text("source_line,track_id,type\n2,t,Car\n", encoding="utf-8")
    (e01 / "issues.csv").write_text("source_line,code\n", encoding="utf-8")
    headers = ["source_line", "group_index", "track_id", "type", "lat", "lon", "time_s", "coordinate_in_range"]
    positions = [(0, 30), (0, 21), (0, 19), (0, 10), (0, 0), (0, -10), (0, -19), (0, -21), (0, -30)]
    with gzip.open(e01 / "observations.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for index, (x_m, y_m) in enumerate(positions):
            writer.writerow({"source_line": 2, "group_index": index, "track_id": "t", "type": "Car", "lat": y_m * DEGREE_PER_METER, "lon": x_m * DEGREE_PER_METER, "time_s": index * 0.5, "coordinate_in_range": "true"})
    return source, geometry, seed_path, runtime, e01


class DemandProfileTests(unittest.TestCase):
    def setUp(self):
        self.sector = load_sector(FIXTURES / "sector_seed.json")

    def test_true_exit_then_new_entry_creates_two_visits(self):
        events = [crossing("N", "entry", 1, 0), crossing("S", "exit", 2, 0), crossing("W", "entry", 4, 1), crossing("E", "exit", 5, 1)]
        visits = reconstruct_visits(events, {0: ("N", "S"), 1: ("W", "E")}, set())
        self.assertEqual([(visit.entry_gate, visit.exit_gate, visit.status) for visit in visits], [("N", "S", "classifiable"), ("W", "E", "classifiable")])

    def test_censorship_and_ambiguity_never_create_certain_movement(self):
        censored = reconstruct_visits([crossing("S", "exit", 2)], {0: ("S",)}, set())
        ambiguous = reconstruct_visits([crossing("N", "entry", 1), crossing("S", "exit", 2)], {0: ("N", "S")}, {0})
        self.assertEqual(censored[0].status, "censored_entry")
        self.assertEqual(ambiguous[0].status, "ambiguous")

    def test_coverage_unknown_differs_from_observed_zero(self):
        events = [crossing("N", "entry", 10)]
        coverage = {"N": (CoverageInterval(0, 60),), "W": None, "E": (CoverageInterval(0, 60),), "S": ()}
        rows = build_flow_rows(events, self.sector, ["Car"], coverage, [(0, 60)])
        by_gate = {row["gate_id"]: row for row in rows}
        self.assertEqual(by_gate["N"]["passages"], 1)
        self.assertEqual(by_gate["N"]["flow_veh_per_h"], 60.0)
        self.assertEqual(by_gate["E"]["passages"], 0)
        self.assertEqual(by_gate["E"]["coverage_status"], "known")
        self.assertEqual(by_gate["E"]["flow_veh_per_h"], 0.0)
        self.assertEqual(by_gate["W"]["coverage_status"], "unknown")
        self.assertIsNone(by_gate["W"]["flow_veh_per_h"])
        self.assertEqual(by_gate["S"]["coverage_status"], "zero_exposure")

    def test_event_on_window_boundary_belongs_to_next_window(self):
        events = [crossing("N", "entry", 60)]
        coverage = {"N": (CoverageInterval(0, 120),), "W": None, "E": None, "S": None}
        rows = build_flow_rows(events, self.sector, ["Car"], coverage, [(0, 60), (60, 120)])
        north = [row for row in rows if row["gate_id"] == "N"]
        self.assertEqual([row["passages"] for row in north], [0, 1])

    def test_movement_proportions_show_known_denominator(self):
        visits = [
            Visit("1", 2, "a", "Car", 0, "N", "S", 10, 20, "classifiable", ("N", "S"), "ok"),
            Visit("2", 3, "b", "Car", 0, "N", "E", 11, 21, "classifiable", ("N", "E"), "ok"),
            Visit("3", 4, "c", "Car", 0, "N", None, 12, None, "censored_exit", ("N",), "censurée"),
        ]
        rows = build_movement_rows(visits, self.sector, ["Car"], [(0, 60)])
        movements = {(row["entry_gate"], row["exit_gate"]): row for row in rows if row["record_type"] == "movement"}
        self.assertEqual(movements[("N", "S")]["denominator_classifiable"], 2)
        self.assertEqual(movements[("N", "S")]["proportion"], 0.5)
        self.assertEqual(movements[("N", "E")]["proportion"], 0.5)
        self.assertTrue(any(row["visit_status"] == "censored_exit" for row in rows))

    def test_incompatible_provenance_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, geometry, seed, runtime, e01 = prepare_inputs(root)
            source.write_text("source modifiée\n", encoding="utf-8")
            with self.assertRaises(ProfileInputError):
                profile_pneuma(source, e01, seed, geometry, runtime, root / "out")

    def test_runtime_configuration_requires_coverage_evidence(self):
        document = json.loads((FIXTURES / "runtime_config.json").read_text(encoding="utf-8"))
        del document["coverage_evidence"]["N"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Justification de couverture"):
                load_runtime_configuration(path, self.sector)

    def test_pipeline_is_reproducible_on_complete_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, geometry, seed, runtime, e01 = prepare_inputs(root)
            first, second = root / "first", root / "second"
            first_summary = profile_pneuma(source, e01, seed, geometry, runtime, first)
            second_summary = profile_pneuma(source, e01, seed, geometry, runtime, second)
            self.assertEqual(first_summary["status"], "complete")
            self.assertEqual(first_summary, second_summary)
            for name in ["crossings.csv", "partial_routes.csv", "flow_profile.csv", "movement_profile.csv", "quality_summary.json", "sector.geojson", "sector_config.json", "profile_report.md"]:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
            manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "CGR-E02-1")
            self.assertNotIn(str(root), json.dumps(manifest))
            with (first / "partial_routes.csv").open(encoding="utf-8", newline="") as handle:
                routes = list(csv.DictReader(handle))
            self.assertEqual([(row["entry_gate"], row["exit_gate"], row["status"]) for row in routes], [("N", "S", "classifiable")])
            with (first / "flow_profile.csv").open(encoding="utf-8", newline="") as handle:
                flow_rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["window_end_s"] == "8.0" for row in flow_rows))

    def test_public_cli_runs_without_sumo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, geometry, seed, runtime, e01 = prepare_inputs(root)
            output = root / "profile"
            result = subprocess.run([
                sys.executable, "scripts/profile_pneuma.py", "--source", str(source),
                "--cgr-e01-dir", str(e01), "--sector-seed", str(seed),
                "--geometry-source", str(geometry), "--runtime-config", str(runtime),
                "--output-dir", str(output),
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "manifest.json").is_file())
