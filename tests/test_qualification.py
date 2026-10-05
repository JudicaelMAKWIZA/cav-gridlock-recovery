import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cav_recovery.empirical.qualification import EXPORTS, qualify_pneuma


ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "pneuma"


class QualificationTests(unittest.TestCase):
    def test_known_file_exports_and_units(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            summary = qualify_pneuma(FIXTURES / "known.csv", output)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["counts"]["candidate_lines"], 2)
            self.assertEqual(summary["counts"]["observation_export_rows"], 4)
            self.assertEqual(summary["time_seconds"]["extent"], 4.0)
            self.assertEqual(set(path.name for path in output.iterdir()), set(EXPORTS))
            with gzip.open(output / "observations.csv.gz", "rt", encoding="utf-8", newline="") as handle:
                observations = list(csv.DictReader(handle))
            self.assertEqual(observations[0]["speed_mps"], "10.0")
            self.assertEqual([row["group_index"] for row in observations], ["0", "1", "0", "1"])

    def test_five_observations_have_known_coverage_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            summary = qualify_pneuma(FIXTURES / "five_observations.csv", output)
            self.assertEqual(summary["counts"]["observation_export_rows"], 5)
            self.assertEqual(summary["time_seconds"]["extent"], 6.0)
            self.assertEqual(summary["intervals_seconds"]["positive_bounds"]["min"], 1.0)
            report = (output / "qualification_report.md").read_text(encoding="utf-8")
            self.assertIn("Couverture temporelle observée", report)
            self.assertIn("Emprise géographique recevable", report)
            self.assertIn("Car : 1", report)
            self.assertIn("## Exclusions", report)
            self.assertIn("## Conclusion", report)

    def test_anomalies_preserved_and_reconciled(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            summary = qualify_pneuma(FIXTURES / "anomalies.csv", output)
            self.assertEqual(summary["status"], "complete_with_issues")
            self.assertEqual(summary["counts"]["candidate_lines"], 3)
            self.assertEqual(summary["counts"]["structure_excluded_lines"], 1)
            self.assertEqual(summary["counts"]["observation_export_rows"], 4)
            self.assertEqual(summary["intervals_seconds"]["zero"], 1)
            self.assertEqual(summary["intervals_seconds"]["negative"], 1)
            with (output / "issues.csv").open(encoding="utf-8", newline="") as handle:
                issues = list(csv.DictReader(handle))
            codes = {row["code"] for row in issues}
            self.assertTrue({"STRUCTURE_ERROR", "DUPLICATE_TRACK_ID", "INVALID_NUMBER", "NEGATIVE_SPEED", "DUPLICATE_TIME", "NON_MONOTONIC_TIME", "UNKNOWN_CATEGORY"} <= codes)
            with (output / "trajectories.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_unusable_and_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "header.csv"
            source.write_text("track_id; type; traveled_d; avg_speed; lat; lon; speed; lon_acc; lat_acc; time\n", encoding="utf-8")
            output = root / "out"
            summary = qualify_pneuma(source, output)
            self.assertEqual(summary["status"], "unusable")
            before = source.read_bytes()
            with self.assertRaises(ValueError):
                qualify_pneuma(source, output)
            self.assertEqual(source.read_bytes(), before)

    def test_empty_file_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "empty.csv"
            source.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "vide"):
                qualify_pneuma(source, Path(directory) / "out")

    def test_no_usable_group_keeps_invalid_values_and_identity_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            summary = qualify_pneuma(FIXTURES / "no_usable.csv", output)
            self.assertEqual(summary["status"], "unusable")
            self.assertEqual(summary["categories"], {"unknown": 1})
            self.assertEqual(summary["issues_by_code"]["MISSING_TRACK_ID"], 1)
            self.assertEqual(summary["issues_by_code"]["INVALID_NUMBER"], 1)
            self.assertEqual(summary["issues_by_code"]["UNKNOWN_CATEGORY"], 1)
            with gzip.open(output / "observations.csv.gz", "rt", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["lat"], "100.0")

    def test_two_executions_are_scientifically_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "one", root / "two"
            qualify_pneuma(FIXTURES / "five_observations.csv", first)
            qualify_pneuma(FIXTURES / "five_observations.csv", second)
            for name in ["trajectories.csv", "issues.csv", "quality_summary.json", "qualification_report.md"]:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            self.assertEqual(gzip.decompress((first / "observations.csv.gz").read_bytes()), gzip.decompress((second / "observations.csv.gz").read_bytes()))
            one_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
            two_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(one_manifest["source"], two_manifest["source"])
            self.assertEqual(one_manifest["software"]["code_state"], two_manifest["software"]["code_state"])
            self.assertEqual(one_manifest["source"]["filename"], "five_observations.csv")
            self.assertNotIn("path", one_manifest["source"])
            self.assertNotIn("input", one_manifest["parameters"])
            self.assertNotIn("output_dir", one_manifest["parameters"])

    def test_cli_is_independent_of_sumo(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            result = subprocess.run([sys.executable, "scripts/qualify_pneuma.py", "--input", str(FIXTURES / "known.csv"), "--output-dir", str(output)], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Qualification complete", result.stdout)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")

    def test_cli_returns_two_for_refused_or_unusable_input(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.csv"
            refused = subprocess.run([sys.executable, "scripts/qualify_pneuma.py", "--input", str(missing), "--output-dir", str(Path(directory) / "out")], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("Qualification refusée", refused.stderr)
            unusable = subprocess.run([sys.executable, "scripts/qualify_pneuma.py", "--input", str(FIXTURES / "no_usable.csv"), "--output-dir", str(Path(directory) / "unusable")], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(unusable.returncode, 2)

    def test_cli_returns_one_for_technical_output_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output_file = Path(directory) / "not-a-directory"
            output_file.write_text("occupied", encoding="utf-8")
            result = subprocess.run([sys.executable, "scripts/qualify_pneuma.py", "--input", str(FIXTURES / "known.csv"), "--output-dir", str(output_file)], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("Erreur technique de qualification", result.stderr)
