"""Portable audit I/O must preserve an immutable historical corpus."""

import hashlib
import json
from pathlib import Path
from cav_recovery.dataset.contract import Contract
from cav_recovery.dataset.export import validate
from cav_recovery.dataset.pipeline import file_sha256


def test_hashing_does_not_read_a_whole_archive_into_memory(tmp_path, monkeypatch):
    payload = b"compressed reference\x00" * 100000
    archive = tmp_path / "trace.gz"
    archive.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    def forbidden(*args, **kwargs):
        raise AssertionError("Whole-file allocation forbidden")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert file_sha256(archive) == expected


def test_external_validation_report_preserves_existing_dataset_files(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "episodes").mkdir()
    (root / "campaign.json").write_text(
        json.dumps({"contract": Contract().record()}), encoding="utf-8"
    )
    original = root / "validation.json"
    original.write_bytes(b"historical audit must survive")
    before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    report = tmp_path / "windows-audit.json"
    assert validate(root, report_path=report)["valid"]
    assert report.is_file()
    assert before == {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
