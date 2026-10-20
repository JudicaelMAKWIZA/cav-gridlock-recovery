"""Production bornée et reprenable, avec validation atomique de chaque épisode."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import time

from .contract import Contract, digest
from .generation import candidate
from .qualify import POSITIVE, NEGATIVE, UNKNOWN, qualify
from ..platform_support import available_memory_kib, process_alive


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def file_sha256(path):
    """Calcule l’empreinte par blocs sans charger toute la trace en mémoire."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def code_hashes():
    package = Path(__file__).parents[1]
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((package / "metadrive").glob("*.py")) + [
        package / "communication/v2v.py",
        package / "cli.py",
    ]
    paths += [package / "platform_support.py"]
    return {
        p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def records(root):
    for path in sorted((Path(root) / "episodes").glob("*/annotation.json")):
        if (path.parent / "manifest.json").exists():
            yield path.parent, json.loads(path.read_text(encoding="utf-8"))


def summary(root):
    counts = Counter()
    families = {}
    scopes = Counter()
    parents = set()
    wall = 0.0
    substeps = 0
    cpu = 0.0
    peak_rss = 0
    for folder, row in records(root):
        counts[row["label"]] += 1
        parents.add(row["parent_id"])
        wall += row["wall_s"]
        substeps += row["integrity"]["substeps"]
        cpu += row.get("cpu_s", 0.0)
        peak_rss = max(peak_rss, row.get("peak_rss_kib", 0))
        families.setdefault(row["family"], Counter())[row["label"]] += 1
        if row["label"] == POSITIVE:
            scopes.update(set(row["scopes"]))
    total_bytes = sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())
    return dict(
        episodes=sum(counts.values()),
        labels=dict(counts),
        positive_scopes_overlapping=dict(scopes),
        families={k: dict(v) for k, v in families.items()},
        unique_parents=len(parents),
        episode_wall_s=wall,
        substeps=substeps,
        bytes=total_bytes,
        episode_cpu_s=cpu,
        maximum_worker_peak_rss_kib=peak_rss,
        incomplete=[
            p.name
            for p in (Path(root) / "episodes").glob("*")
            if not (p / "manifest.json").exists()
        ],
    )


def _execute_one(task):
    spec, root, parameters, expected = task
    from .native import simulate

    folder = Path(root) / "episodes" / spec["id"]
    if folder.exists():
        if (folder / "manifest.json").exists():
            return json.loads((folder / "annotation.json").read_text())
        # Les preuves interrompues sont conservées au lieu d’écraser un épisode incomplet.
        saved = Path(root) / "interrupted" / (spec["id"] + "-" + str(time.time_ns()))
        saved.parent.mkdir(exist_ok=True)
        folder.replace(saved)
    before = code_hashes()
    if before != expected:
        raise ValueError("Source changed after campaign freeze.")
    result = simulate(spec, folder, Contract(**parameters))
    if code_hashes() != expected:
        raise ValueError("Source changed during episode; evidence not committed.")
    result["storage_bytes"] = sum(
        p.stat().st_size for p in folder.iterdir() if p.is_file()
    )
    write_json(folder / "annotation.json", result)
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.iterdir()
        if p.is_file()
    }
    write_json(
        folder / "manifest.json",
        dict(
            id=spec["id"],
            spec_sha256=digest(spec),
            files=files,
            code=expected,
            contract_sha256=Contract(**parameters).sha256,
            complete=True,
        ),
    )
    return result


def execute_one(task):
    spec, root, _, _ = task
    directory = Path(root) / "worker-locks"
    directory.mkdir(exist_ok=True)
    lock = directory / (spec["id"] + ".lock")
    if lock.exists():
        if process_alive(int(lock.read_text())):
            raise ValueError(
                "Episode worker is still active; concurrent replay refused."
            )
        lock.unlink()
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        return _execute_one(task)
    finally:
        lock.unlink(missing_ok=True)


def generate(
    root,
    *,
    count=0,
    positive=0,
    negative=0,
    max_candidates=4000,
    workers=2,
    max_seconds=7200,
    max_gib=4.0,
    reserve_gib=8.0,
    phase="production",
    contract_file=None,
):
    if (
        workers not in (1, 2, 3, 4)
        or min(count, positive, negative) < 0
        or max_candidates < 1
        or max_seconds <= 0
        or max_gib <= 0
        or reserve_gib < 0
    ):
        raise ValueError("Invalid batch or resource limits; at most four workers.")
    if not all(math.isfinite(v) for v in (max_seconds, max_gib, reserve_gib)):
        raise ValueError("Resource limits must be finite.")
    if not any((count, positive, negative)):
        raise ValueError("Specify a count or class quotas.")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "episodes").mkdir(exist_ok=True)
    frozen = Contract()
    if contract_file:
        document = json.loads(Path(contract_file).read_text())
        frozen = Contract(**document["contract"])
        if (
            not document.get("frozen")
            or document.get("contract_sha256") != frozen.sha256
        ):
            raise ValueError("A valid frozen calibration contract is required.")
    elif phase == "production":
        raise ValueError("Production requires --contract from a completed calibration.")
    campaign = dict(
        schema="cgr-campaign-1",
        phase=phase,
        contract=frozen.record(),
        contract_sha256=frozen.sha256,
        code=code_hashes(),
        generator="single-dense-junction-v4",
        variants_per_parent=1 if phase == "calibration" else 4,
    )
    manifest = root / "campaign.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != campaign:
            raise ValueError(
                "Resume refused: campaign contract/source/generator differs."
            )
    else:
        write_json(manifest, campaign)
    lock = root / "campaign.lock"
    if lock.exists():
        owner = int(lock.read_text())
        if not process_alive(owner):
            if any(
                process_alive(int(path.read_text()))
                for path in (root / "worker-locks").glob("*.lock")
            ):
                raise ValueError(
                    "Interrupted campaign workers still active; wait for their completion before resuming."
                )
            lock.unlink()
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError(
            "Campaign is locked; verify the owning process before recovering its lock."
        )
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    begun = time.monotonic()
    done = {row["id"] for _, row in records(root)}
    stats = summary(root)
    counts = Counter(stats["labels"])
    index = 0
    submitted = 0
    status = "candidate_limit"
    pending = {}
    bytes_used = stats["bytes"]
    try:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            while True:
                quotas_met = (
                    counts[POSITIVE] >= positive and counts[NEGATIVE] >= negative
                )
                if (count and sum(counts.values()) >= count) or (
                    not count and quotas_met
                ):
                    status = "quotas_reached"
                    break
                if time.monotonic() - begun > max_seconds:
                    status = "time_limit"
                    break
                if shutil.disk_usage(root).free < reserve_gib * 1024**3:
                    status = "disk_reserve"
                    break
                available = available_memory_kib()
                if available is not None and available < 512 * 1024:
                    status = "memory_reserve"
                    break
                if bytes_used > max_gib * 1024**3:
                    status = "storage_limit"
                    break
                while (
                    len(pending) < workers
                    and index < max_candidates
                    and (not count or sum(counts.values()) + len(pending) < count)
                ):
                    spec = candidate(
                        index, phase=phase, variants=campaign["variants_per_parent"]
                    )
                    index += 1
                    if spec["id"] in done:
                        continue
                    future = pool.submit(
                        execute_one,
                        (spec, str(root), frozen.record(), campaign["code"]),
                    )
                    pending[future] = spec["id"]
                    submitted += 1
                if not pending:
                    break
                completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    episode_id = pending.pop(future)
                    row = future.result()
                    done.add(episode_id)
                    counts[row["label"]] += 1
                    bytes_used += row.get("storage_bytes", 0)
                    if sum(counts.values()) % 20 == 0:
                        print(
                            json.dumps(
                                dict(
                                    episodes=sum(counts.values()),
                                    labels=dict(counts),
                                    elapsed_s=round(time.monotonic() - begun, 1),
                                )
                            ),
                            flush=True,
                        )
                        write_json(
                            root / "progress.json",
                            dict(
                                labels=dict(counts), elapsed_s=time.monotonic() - begun
                            ),
                        )
            # Les épisodes déjà engagés terminent et restent disponibles après l’arrêt de ressources.
            for future in pending:
                future.result()
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        lock.unlink(missing_ok=True)
    stats = summary(root)
    stats.update(
        stop_reason=status,
        batch_wall_s=time.monotonic() - begun,
        quotas=dict(positive=positive, negative=negative, count=count),
        contract_sha256=frozen.sha256,
        resource_limits=dict(
            workers=workers,
            max_seconds=max_seconds,
            max_gib=max_gib,
            reserve_gib=reserve_gib,
            minimum_available_ram_mib=512,
        ),
    )
    write_json(root / "summary.json", stats)
    return stats


def calibrate(root):
    root = Path(root)
    base = Contract()
    results = []
    # Les épisodes de calibration ont une identité distincte et sont exclus des partitions finales.
    rows = list(records(root))
    if not rows or any(row["phase"] != "calibration" for _, row in rows):
        raise ValueError(
            "Only calibration episodes may freeze the operational contract."
        )
    from .generation import FAMILIES, DENSITIES

    if set(row["family"] for _, row in rows) != set(FAMILIES):
        raise ValueError("Calibration must cover every candidate family.")
    specs = [json.loads((folder / "spec.json").read_text()) for folder, _ in rows]
    if set(s["density_level"] for s in specs) != set(DENSITIES):
        raise ValueError("Calibration must cover all requested density levels.")
    if any(row["integrity"]["errors"] for _, row in rows):
        raise ValueError(
            "Calibration contains physical/technical failures; investigate before freeze."
        )
    if not {POSITIVE, NEGATIVE} <= {row["label"] for _, row in rows}:
        raise ValueError(
            "Development must contain independently qualified positive and negative circulation."
        )
    for persistence in (3.0, 5.0, 8.0):
        for followup in (15.0, 30.0, 35.0):
            contract = replace(
                base, persistence_s=persistence, nominal_followup_s=followup
            )
            counts = Counter()
            by_family = {}
            for folder, row in rows:
                metadata = json.loads((folder / "metadata.json").read_text())
                result = qualify(
                    list(read_rows(folder / "reference.jsonl.gz")),
                    metadata,
                    row["integrity"],
                    contract,
                )
                counts[result["label"]] += 1
                by_family.setdefault(row["family"], Counter())[result["label"]] += 1
            results.append(
                dict(
                    persistence_s=persistence,
                    nominal_followup_s=followup,
                    labels=dict(counts),
                    families=by_family,
                )
            )
    frozen = dict(
        frozen=True,
        contract=base.record(),
        contract_sha256=base.sha256,
        calibration_ids=[r["id"] for _, r in rows],
        sensitivity=results,
        selection="5s effect windows and conservative 30s movement/resource persistence retained after development sensitivity; successive occupants share the movement clock; no test data used",
        class_coverage=dict(Counter(r["label"] for _, r in rows)),
        scientific_acceptance=False,
        limitations="operational stalled-movement subset; no universal thresholds; no exhaustive alternate-action search",
    )
    write_json(root / "frozen_contract.json", frozen)
    return frozen


def requalify(source, output, *, contract_file=None):
    """Conserve les preuves brutes et crée une révision d’annotation avec sa provenance."""
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Requalification output must be absent or empty.")
    old = json.loads((source / "campaign.json").read_text())
    parameters = Contract()
    if contract_file:
        frozen = json.loads(Path(contract_file).read_text())
        parameters = Contract(**frozen["contract"])
        if (
            not frozen.get("frozen")
            or frozen.get("contract_sha256") != parameters.sha256
        ):
            raise ValueError("Invalid frozen contract.")
    elif old["phase"] != "calibration":
        raise ValueError("Production requalification requires a frozen contract.")
    from .export import validate

    if not validate(source)["valid"]:
        raise ValueError("Source evidence integrity must pass before requalification.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir(exist_ok=True)
    code = code_hashes()
    campaign = dict(
        schema="cgr-campaign-1",
        phase=old["phase"],
        contract=parameters.record(),
        contract_sha256=parameters.sha256,
        code=code,
        generator=old["generator"],
        variants_per_parent=old["variants_per_parent"],
    )
    write_json(output / "campaign.json", campaign)
    for folder, row in records(source):
        target = output / "episodes" / row["id"]
        target.mkdir()
        for path in folder.iterdir():
            if path.name not in ("annotation.json", "manifest.json"):
                shutil.copy2(path, target / path.name)
        metadata = json.loads((folder / "metadata.json").read_text())
        result = {
            **row,
            **qualify(
                list(read_rows(folder / "reference.jsonl.gz")),
                metadata,
                row["integrity"],
                parameters,
            ),
        }
        result["annotation_supersedes"] = {
            "contract_sha256": old["contract_sha256"],
            "source_episode": row["id"],
        }
        write_json(target / "annotation.json", result)
        previous = json.loads((folder / "manifest.json").read_text())
        files = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in target.iterdir()
            if p.is_file()
        }
        write_json(
            target / "manifest.json",
            dict(
                id=row["id"],
                spec_sha256=previous["spec_sha256"],
                files=files,
                code=code,
                measurement_code=previous.get("measurement_code", previous["code"]),
                qualification_code=code,
                contract_sha256=parameters.sha256,
                complete=True,
            ),
        )
    result = summary(output)
    write_json(output / "summary.json", result)
    return result
