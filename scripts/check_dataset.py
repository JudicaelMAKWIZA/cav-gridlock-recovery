"""Contrôle parallèle borné d’épisodes terminés immuables, indexé par leurs manifestes."""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import time
from cav_recovery.dataset.pipeline import records, write_json, code_hashes
from cav_recovery.dataset.export import validate


def check_one(task):
    root, folder, cache = map(Path, task)
    ident = folder.name
    manifest_before = hashlib.sha256(
        (folder / "manifest.json").read_bytes()
    ).hexdigest()
    workspace = root / ".qc-views" / str(os.getpid())
    if not workspace.resolve().is_relative_to(root.resolve() / ".qc-views"):
        raise ValueError("QC scratch directory resolves outside its dataset workspace")
    (workspace / "episodes").mkdir(parents=True, exist_ok=True)
    campaign = workspace / "campaign.json"
    if not campaign.exists():
        if os.name == "nt":
            campaign.write_bytes((root / "campaign.json").read_bytes())
        else:
            campaign.symlink_to(root.resolve() / "campaign.json")
    link = workspace / "episodes/current"
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        if os.name != "nt":
            raise ValueError("QC view is not the expected disposable symlink")
        if (
            link.is_junction()
            or link.resolve() != workspace.resolve() / "episodes/current"
        ):
            raise ValueError(
                "QC view is a reparse point, not an owned hard-link directory"
            )
        # Only remove hard links in this explicitly owned scratch directory.
        entries = list(link.iterdir())
        if any(
            not entry.is_file() or entry.is_symlink() or entry.stat().st_nlink < 2
            for entry in entries
        ):
            raise ValueError(
                "QC scratch contains an unexpected entry or a last file link"
            )
        for entry in entries:
            entry.unlink()
        link.rmdir()
    if os.name == "nt":
        link.mkdir()
        for entry in folder.iterdir():
            if entry.is_file():
                os.link(entry, link / entry.name)
    else:
        link.symlink_to(folder.resolve(), target_is_directory=True)
    begun = time.perf_counter()
    cpu = time.process_time()
    result = validate(workspace, deep=True)
    after = hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()
    if after != manifest_before:
        raise ValueError("Evidence changed while checking")
    result.update(
        id=ident,
        manifest_sha256=manifest_before,
        wall_s=time.perf_counter() - begun,
        cpu_s=time.process_time() - cpu,
        validator_sha256=hashlib.sha256(
            Path(
                __import__("cav_recovery.dataset.export", fromlist=["x"]).__file__
            ).read_bytes()
        ).hexdigest(),
        verification_code=code_hashes(),
        orchestration_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    write_json(cache / (ident + ".json"), result)
    return result


def check(root, workers=4, follow=False):
    root = Path(root).resolve()
    cache = root / "qc-reports"
    cache.mkdir(exist_ok=True)
    pending = {}
    known = set()
    results = {}
    begun = time.perf_counter()
    verification_code = code_hashes()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as pool:
        while True:
            snapshot = list(records(root))
            for folder, row in snapshot:
                ident = row["id"]
                if ident in known:
                    continue
                saved = cache / (ident + ".json")
                if saved.exists():
                    report = json.loads(saved.read_text())
                    if (
                        report.get("verification_code") == verification_code
                        and report["manifest_sha256"]
                        == hashlib.sha256(
                            (folder / "manifest.json").read_bytes()
                        ).hexdigest()
                    ):
                        results[ident] = report
                        known.add(ident)
                        continue
                if len(pending) >= workers * 2:
                    break
                future = pool.submit(check_one, (str(root), str(folder), str(cache)))
                pending[future] = ident
                known.add(ident)
            if pending:
                completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    ident = pending.pop(future)
                    report = future.result()
                    results[ident] = report
                    if len(results) % 20 == 0:
                        print(
                            json.dumps(
                                dict(
                                    checked=len(results),
                                    valid=all(r["valid"] for r in results.values()),
                                )
                            ),
                            flush=True,
                        )
            active = (root / "campaign.lock").exists()
            if not pending and all(row["id"] in results for _, row in snapshot):
                if not follow or not active:
                    # An episode may finish between the directory snapshot and
                    # the coordinator releasing its lock. Reconcile once more.
                    latest = {row["id"] for _, row in records(root)}
                    if latest <= set(results):
                        break
                    continue
                time.sleep(2)
    global_check = validate(root, deep=False)
    current = {row["id"] for _, row in records(root)}
    reports = {ident: r for ident, r in results.items() if ident in current}
    issues = list(global_check["issues"]) + [
        dict(id=ident, issues=report["issues"])
        for ident, report in reports.items()
        if not report["valid"]
    ]
    if set(reports) != current:
        issues.append(dict(kind="deep_qc_coverage"))
    report = dict(
        valid=not issues,
        deep=True,
        episodes=len(current),
        labels=global_check["labels"],
        issues=issues,
        method="per-episode independent deep checks plus global file/provenance/duplicate checks",
        cache_manifest_sha256={
            ident: r["manifest_sha256"] for ident, r in reports.items()
        },
        episode_qc_cpu_s=sum(r["cpu_s"] for r in reports.values()),
        wall_s=time.perf_counter() - begun,
        contract_sha256=global_check["contract_sha256"],
        workers=workers,
    )
    write_json(root / "validation-deep.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    result = check(args.dataset, args.workers, args.follow)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "cache_manifest_sha256"}, indent=2
        )
    )
    raise SystemExit(0 if result["valid"] else 1)
