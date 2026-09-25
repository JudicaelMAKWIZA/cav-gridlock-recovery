"""Exports reproductibles de qualification pNEUMA, sans nettoyage ni inférence."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .pneuma import GROUP_FIELDS, Issue, parse_number, read_candidates, source_issue

SCHEMA_VERSION = "CGR-E01-1"
EXPORTS = ["manifest.json", "trajectories.csv", "observations.csv.gz", "issues.csv", "quality_summary.json", "qualification_report.md"]
PNEUMA_CATEGORIES = {"Car", "Taxi", "Bus", "Medium Vehicle", "Heavy Vehicle", "Motorcycle"}


def _csv_value(value: Any) -> str:
    return "" if value is None else str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _RunningStats:
    """Minimum, maximum and count without retaining individual values."""
    def __init__(self) -> None:
        self.count = 0
        self.excluded = 0
        self.minimum: float | None = None
        self.maximum: float | None = None

    def add(self, value: float) -> None:
        self.count += 1
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def exclude(self) -> None:
        self.excluded += 1

    def result(self, additional_exclusions: int = 0) -> dict[str, Any]:
        return {"count": self.count, "excluded": self.excluded + additional_exclusions, "min": self.minimum, "max": self.maximum,
                "reason": None if self.count else "aucune valeur finie"}


class _StreamingCsvRows:
    """CSV writer retaining only a row count."""
    def __init__(self, path: Path, headers: list[str]) -> None:
        self._handle = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=headers)
        self._writer.writeheader()
        self.count = 0

    def append(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self.count += 1

    def __len__(self) -> int:
        return self.count

    def close(self) -> None:
        self._handle.close()


class _StreamingIssues(_StreamingCsvRows):
    """List-compatible sink used by the parser, with incremental diagnostics counts."""
    def __init__(self, path: Path, headers: list[str]) -> None:
        super().__init__(path, headers)
        self.by_code: Counter[str] = Counter()

    def append(self, issue: Issue) -> None:  # type: ignore[override]
        super().append({"source_line": issue.line, "group_index": "" if issue.group is None else issue.group,
                        "field": issue.field, "code": issue.code, "severity": issue.severity,
                        "source_value": issue.value, "message": issue.message})
        self.by_code[issue.code] += 1


class _StreamingGzipRows:
    """CSV writer which retains only a row count, never all observations."""
    def __init__(self, path: Path, headers: list[str]) -> None:
        self._raw = path.open("wb")
        self._gzip = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
        self._text = __import__("io").TextIOWrapper(self._gzip, encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._text, fieldnames=headers)
        self._writer.writeheader()
        self.count = 0

    def append(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self.count += 1

    def __len__(self) -> int:
        return self.count

    def close(self) -> None:
        self._text.close()
        self._raw.close()


def _code_state() -> dict[str, str]:
    """Identify executable code without recording local source paths."""
    module_paths = [Path(__file__), Path(__file__).with_name("pneuma.py")]
    digest = hashlib.sha256()
    for module_path in module_paths:
        digest.update(module_path.name.encode("utf-8"))
        digest.update(module_path.read_bytes())
    try:
        package_version = importlib.metadata.version("cav-gridlock-recovery")
    except importlib.metadata.PackageNotFoundError:
        package_version = "not-installed"
    return {"package_version": package_version, "cgr_e01_code_sha256": digest.hexdigest()}


def _git_state() -> dict[str, str | None]:
    """Return repository state when Git is available, without exposing a local path."""
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, text=True, capture_output=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repository, text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree": "unavailable"}
    return {"commit": commit, "working_tree": "dirty" if dirty else "clean"}


def qualify_pneuma(input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Qualify one pNEUMA CSV, writing only a new/empty private output directory.

    The source is opened read-only and observations are emitted as each source row is read.
    """
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Entrée absente ou illisible : {source}")
    if destination == source or source in destination.parents:
        raise ValueError("Le dossier de sortie ne peut pas contenir le fichier source.")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Le dossier de sortie existe déjà et n'est pas vide : {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".cgr-e01-", dir=destination.parent))
    trajectories: _StreamingCsvRows | None = None
    observations: _StreamingGzipRows | None = None
    issues: _StreamingIssues | None = None
    try:
        trajectory_headers = ["source_line", "track_id", "type", "traveled_d_m", "avg_speed_kmh", "avg_speed_mps", "structural_status", "observation_count", "diagnostic_count"]
        observation_headers = ["source_line", "group_index", "track_id", "type", "lat", "lon", "speed_kmh", "speed_mps", "lon_acc_mps2", "lat_acc_mps2", "time_s", "all_numeric_finite", "coordinate_in_range"]
        issue_headers = ["source_line", "group_index", "field", "code", "severity", "source_value", "message"]
        trajectories = _StreamingCsvRows(stage / "trajectories.csv", trajectory_headers)
        observations = _StreamingGzipRows(stage / "observations.csv.gz", observation_headers)
        issues = _StreamingIssues(stage / "issues.csv", issue_headers)
        category_counts: Counter[str] = Counter()
        seen_ids: set[str] = set()
        duplicate_ids: set[str] = set()
        time_stats = _RunningStats()
        latitude_stats = _RunningStats()
        longitude_stats = _RunningStats()
        coordinate_exclusions = 0
        interval_counts = Counter()
        positive_interval_stats = _RunningStats()
        unavailable_intervals = 0
        valid_complete_groups = 0
        invalid_numeric_groups = 0
        minimally_usable = False
        candidates = decomposable = excluded_structure = 0
        expected_group_count = 0

        for candidate in read_candidates(source):
            candidates += 1
            fields = candidate.fields
            track_id = fields[0] if len(fields) > 0 else ""
            category = fields[1] if len(fields) > 1 else ""
            if not candidate.decomposable:
                excluded_structure += 1
                issues.append(source_issue(candidate.line, None, "row", "STRUCTURE_ERROR", "error", ";".join(fields), "Nombre de champs incompatible avec 4 + 6 × n."))
                trajectories.append({"source_line": candidate.line, "track_id": track_id, "type": category, "traveled_d_m": "", "avg_speed_kmh": "", "avg_speed_mps": "", "structural_status": "excluded_structure", "observation_count": 0, "diagnostic_count": 1})
                continue
            decomposable += 1
            expected_group_count += candidate.groups
            metadata_issues_before = len(issues)
            traveled = parse_number(fields[2], line=candidate.line, group=None, field="traveled_d_m", issues=issues)
            avg_speed = parse_number(fields[3], line=candidate.line, group=None, field="avg_speed_kmh", issues=issues)
            if track_id == "":
                issues.append(source_issue(candidate.line, None, "track_id", "MISSING_TRACK_ID", "warning", "", "Identifiant de trajectoire manquant."))
            elif track_id in seen_ids:
                duplicate_ids.add(track_id)
                issues.append(source_issue(candidate.line, None, "track_id", "DUPLICATE_TRACK_ID", "warning", track_id, "Identifiant répété ; les lignes restent distinctes."))
            else:
                seen_ids.add(track_id)
            if category == "":
                issues.append(source_issue(candidate.line, None, "type", "MISSING_CATEGORY", "warning", "", "Catégorie déclarée manquante."))
            else:
                category_counts[category] += 1
                if category not in PNEUMA_CATEGORIES:
                    issues.append(source_issue(candidate.line, None, "type", "UNKNOWN_CATEGORY", "warning", category, "Catégorie absente du contrat pNEUMA."))
            previous_time: float | None = None
            has_previous_group = False
            for index in range(candidate.groups):
                tokens = fields[4 + index * 6: 10 + index * 6]
                values = [parse_number(token, line=candidate.line, group=index, field=name, issues=issues) for token, name in zip(tokens, GROUP_FIELDS)]
                lat, lon, speed, lon_acc, lat_acc, time_s = values
                numeric_valid = all(value is not None for value in values)
                if numeric_valid:
                    valid_complete_groups += 1
                else:
                    invalid_numeric_groups += 1
                coordinate_valid = lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180
                if lat is not None and not -90 <= lat <= 90:
                    issues.append(source_issue(candidate.line, index, "lat", "OUT_OF_RANGE_COORDINATE", "warning", tokens[0], "Latitude hors de [-90, 90]."))
                if lon is not None and not -180 <= lon <= 180:
                    issues.append(source_issue(candidate.line, index, "lon", "OUT_OF_RANGE_COORDINATE", "warning", tokens[1], "Longitude hors de [-180, 180]."))
                if coordinate_valid:
                    latitude_stats.add(lat)  # type: ignore[arg-type]
                    longitude_stats.add(lon)  # type: ignore[arg-type]
                else:
                    coordinate_exclusions += 1
                    latitude_stats.exclude()
                    longitude_stats.exclude()
                if speed is not None and speed < 0:
                    issues.append(source_issue(candidate.line, index, "speed_kmh", "NEGATIVE_SPEED", "warning", tokens[2], "Vitesse négative conservée."))
                previous_time_missing = has_previous_group and previous_time is None
                if time_s is not None:
                    time_stats.add(time_s)
                    if time_s < 0:
                        issues.append(source_issue(candidate.line, index, "time_s", "NEGATIVE_TIME", "warning", tokens[5], "Temps négatif conservé."))
                    if has_previous_group and previous_time is not None:
                        interval = time_s - previous_time
                        if interval == 0:
                            interval_counts["zero"] += 1
                            issues.append(source_issue(candidate.line, index, "time_s", "DUPLICATE_TIME", "warning", tokens[5], "Timestamp identique au groupe précédent."))
                        elif interval < 0:
                            interval_counts["negative"] += 1
                            issues.append(source_issue(candidate.line, index, "time_s", "NON_MONOTONIC_TIME", "warning", tokens[5], "Timestamp décroissant dans l'ordre source."))
                        else:
                            interval_counts["positive"] += 1
                            positive_interval_stats.add(interval)
                    previous_time = time_s
                else:
                    time_stats.exclude()
                    previous_time = None
                if has_previous_group and (previous_time_missing or time_s is None):
                    unavailable_intervals += 1
                has_previous_group = True
                if numeric_valid and coordinate_valid and speed is not None and speed >= 0 and time_s is not None and time_s >= 0:
                    minimally_usable = True
                observations.append({
                    "source_line": candidate.line, "group_index": index, "track_id": track_id,
                    "type": category, "lat": _csv_value(lat), "lon": _csv_value(lon),
                    "speed_kmh": _csv_value(speed), "speed_mps": _csv_value(None if speed is None else speed / 3.6),
                    "lon_acc_mps2": _csv_value(lon_acc), "lat_acc_mps2": _csv_value(lat_acc), "time_s": _csv_value(time_s),
                    "all_numeric_finite": str(numeric_valid).lower(), "coordinate_in_range": str(coordinate_valid).lower(),
                })
            trajectories.append({"source_line": candidate.line, "track_id": track_id, "type": category, "traveled_d_m": _csv_value(traveled), "avg_speed_kmh": _csv_value(avg_speed), "avg_speed_mps": _csv_value(None if avg_speed is None else avg_speed / 3.6), "structural_status": "decomposable", "observation_count": candidate.groups, "diagnostic_count": len(issues) - metadata_issues_before})

        observations.close()
        trajectories.close()
        issues.close()
        usable = minimally_usable
        status = "complete" if not issues and usable else "complete_with_issues" if usable else "unusable"
        summary = {
            "schema_version": SCHEMA_VERSION, "status": status,
            "counts": {"candidate_lines": candidates, "decomposable_lines": decomposable, "structure_excluded_lines": excluded_structure, "trajectory_export_rows": len(trajectories), "expected_groups_decomposable_lines": expected_group_count, "observation_export_rows": len(observations), "all_numeric_finite_groups": valid_complete_groups, "numeric_invalid_groups": invalid_numeric_groups, "distinct_nonempty_track_ids": len(seen_ids), "duplicate_track_ids": sorted(duplicate_ids)},
            "categories": dict(sorted(category_counts.items())),
            "issues_by_code": dict(sorted(issues.by_code.items())),
            "time_seconds": {**time_stats.result(), "extent": None if time_stats.count == 0 else time_stats.maximum - time_stats.minimum},
            "acceptable_geographic_bounds": {"latitude": latitude_stats.result(), "longitude": longitude_stats.result(), "acceptable_coordinate_groups": latitude_stats.count, "coordinate_group_exclusions": coordinate_exclusions},
            "intervals_seconds": {"zero": interval_counts["zero"], "negative": interval_counts["negative"], "positive": interval_counts["positive"], "unavailable_missing_time": unavailable_intervals, "positive_bounds": positive_interval_stats.result(interval_counts["zero"] + interval_counts["negative"])},
            "exclusions": {"invalid_time_values": time_stats.excluded, "coordinate_groups_unacceptable": coordinate_exclusions, "groups_with_invalid_numeric_values": invalid_numeric_groups, "structure_excluded_lines": excluded_structure, "intervals_unavailable_missing_time": unavailable_intervals},
            "limitations": ["Les lignes structurellement invalides ont un nombre d'observations original indéterminé.", "Cette qualification ne constitue ni une analyse de flux, ni une reconstruction de routes ou de réseau."],
        }
        if candidates != decomposable + excluded_structure or len(trajectories) != candidates or len(observations) != summary["counts"]["expected_groups_decomposable_lines"]:
            raise RuntimeError("Bilan de qualification incohérent.")
        (stage / "quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        time_summary = summary["time_seconds"]
        geo_summary = summary["acceptable_geographic_bounds"]
        interval_summary = summary["intervals_seconds"]
        report = "# Qualification pNEUMA — CGR-E01\n\n"
        report += f"Statut : **{status}**. {candidates} lignes candidates, {len(observations)} groupes exportés.\n\n"
        report += "## Couverture temporelle observée\n\n"
        report += f"- Bornes : {time_summary['min']} s à {time_summary['max']} s ; étendue : {time_summary['extent']} s.\n"
        report += f"- Effectif fini : {time_summary['count']} ; exclusions numériques : {time_summary['excluded']}.\n"
        report += f"- Intervalles consécutifs : {interval_summary['zero']} nuls, {interval_summary['negative']} négatifs, {interval_summary['positive']} positifs ; bornes positives : {interval_summary['positive_bounds']['min']} à {interval_summary['positive_bounds']['max']} s.\n\n"
        report += "## Emprise géographique recevable\n\n"
        report += f"- Latitude : {geo_summary['latitude']['min']} à {geo_summary['latitude']['max']} ({geo_summary['latitude']['count']} valeurs).\n"
        report += f"- Longitude : {geo_summary['longitude']['min']} à {geo_summary['longitude']['max']} ({geo_summary['longitude']['count']} valeurs).\n"
        report += f"- Groupes exclus de l’emprise : {geo_summary['coordinate_group_exclusions']}.\n\n"
        report += "## Catégories et qualité\n\n"
        report += "- Catégories déclarées : " + (", ".join(f"{key} : {value}" for key, value in summary["categories"].items()) or "aucune") + ".\n"
        report += f"- Groupes entièrement numériques finis : {summary['counts']['all_numeric_finite_groups']} ; avec au moins une valeur numérique invalide : {summary['counts']['numeric_invalid_groups']}.\n"
        report += "- Anomalies par code : " + (", ".join(f"{key} : {value}" for key, value in summary["issues_by_code"].items()) or "aucune") + ".\n\n"
        exclusions = summary["exclusions"]
        report += "## Exclusions\n\n"
        report += f"- Valeurs temporelles invalides : {exclusions['invalid_time_values']} ; valeurs temporelles finies : {time_summary['count']}.\n"
        report += f"- Groupes aux coordonnées recevables : {geo_summary['acceptable_coordinate_groups']} ; hors emprise ou coordonnées invalides : {exclusions['coordinate_groups_unacceptable']}.\n"
        report += f"- Groupes avec au moins une valeur numérique invalide : {exclusions['groups_with_invalid_numeric_values']}.\n"
        report += f"- Lignes exclues pour erreur structurelle : {exclusions['structure_excluded_lines']}.\n"
        report += f"- Intervalles non calculés à cause d’un temps manquant : {exclusions['intervals_unavailable_missing_time']}.\n\n"
        report += "## Limites\n\n" + "\n".join(f"- {item}" for item in summary["limitations"]) + "\n"
        report += "\n## Conclusion\n\n"
        if status == "complete":
            report += "La qualification est terminée sans diagnostic enregistré ; ce constat décrit uniquement le fichier observé.\n"
        elif status == "complete_with_issues":
            report += "La qualification est terminée avec des diagnostics localisés ; les exports conservent les valeurs et anomalies observées.\n"
        else:
            report += "Aucun groupe minimalement utilisable n’a été observé ; le fichier est qualifié inutilisable selon le critère minimal défini.\n"
        (stage / "qualification_report.md").write_text(report, encoding="utf-8")
        manifest = {"schema_version": SCHEMA_VERSION, "status": status, "source": {"filename": source.name, "sha256": _sha256(source), "size_bytes": source.stat().st_size}, "references": ["EPFL pNEUMA downloads/FAQ (S1)", "Zenodo pNEUMA dataset, record 10491409 (S2)"], "parameters": {"format": "pNEUMA CSV ;, quatre métadonnées puis groupes de six", "exports": EXPORTS}, "units": {"speed_source": "km/h", "speed_export": "m/s", "coordinates": "degrees", "acceleration": "m/s²", "time": "s"}, "software": {"python": sys.version.split()[0], "reader": "cav_recovery.empirical.pneuma", "platform": platform.platform(), "code_state": _code_state(), "git": _git_state()}, "exports": EXPORTS}
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        destination.mkdir(exist_ok=True)
        # The manifest is published last, so a write/move failure cannot look complete.
        for name in [name for name in EXPORTS if name != "manifest.json"] + ["manifest.json"]:
            os.replace(stage / name, destination / name)
        return summary
    finally:
        # Idempotent close calls also cover failures while parsing an empty/bad file.
        for writer in (observations, trajectories, issues):
            if writer is not None:
                writer.close()
        shutil.rmtree(stage, ignore_errors=True)
