"""Lecteur progressif et conservateur du format pNEUMA (une ligne = trajectoire)."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


EXPECTED_HEADER = [
    "track_id", "type", "traveled_d", "avg_speed", "lat", "lon", "speed",
    "lon_acc", "lat_acc", "time",
]
GROUP_FIELDS = ("lat", "lon", "speed_kmh", "lon_acc", "lat_acc", "time_s")


@dataclass(frozen=True)
class Issue:
    line: int
    group: int | None
    field: str
    code: str
    severity: str
    value: str
    message: str


@dataclass
class Candidate:
    line: int
    fields: list[str]
    decomposable: bool
    groups: int


def read_candidates(source: Path) -> Iterator[Candidate]:
    """Yield non-empty data rows, retaining source line numbers and source order."""
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("Le fichier est vide : en-tête pNEUMA absent.") from error
        normalized_header = [item.strip() for item in header]
        if normalized_header != EXPECTED_HEADER:
            raise ValueError("En-tête pNEUMA inconnu ou incompatible.")
        for row in reader:
            line = reader.line_num
            if not row or not any(item.strip() for item in row):
                continue
            fields = [item.strip() for item in row]
            # Only remove an optional terminal separator when the remaining shape is exact.
            if fields[-1] == "" and len(fields) > 1 and len(fields) - 1 >= 10 and (len(fields) - 1 - 4) % 6 == 0:
                fields.pop()
            decomposable = len(fields) >= 10 and (len(fields) - 4) % 6 == 0
            yield Candidate(line, fields, decomposable, (len(fields) - 4) // 6 if decomposable else 0)


def parse_number(token: str, *, line: int, group: int | None, field: str, issues: list[Issue]) -> float | None:
    if token == "":
        issues.append(Issue(line, group, field, "INVALID_NUMBER", "error", token, "Valeur numérique manquante."))
        return None
    try:
        number = float(token)
    except ValueError:
        issues.append(Issue(line, group, field, "INVALID_NUMBER", "error", token, "Valeur numérique illisible."))
        return None
    if not math.isfinite(number):
        issues.append(Issue(line, group, field, "INVALID_NUMBER", "error", token, "Valeur numérique non finie."))
        return None
    return number


def source_issue(line: int, group: int | None, field: str, code: str, severity: str, value: str, message: str) -> Issue:
    return Issue(line, group, field, code, severity, value, message)
