"""Lancement et reprise d'expériences privées, sans toucher à la campagne M02."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .catalog import catalog, specification


def source_identity():
    package = Path(__file__).parents[1]
    return {
        path.relative_to(package).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }


def run_one(task):
    name, seed, horizon, output, gui, expected = task
    return run_spec(specification(name, seed, horizon_s=horizon), output, gui, expected)


def run_spec(spec, output, gui=False, expected=None):
    """Exécute une configuration de développement immuable et conserve ses sources."""
    expected = source_identity() if expected is None else expected
    if (
        spec.get("phase") != "integrated_development"
        or spec.get("development_only") is not True
    ):
        raise ValueError("only new integrated development configurations are accepted")
    if spec.get("intersection_count") != 1 or not 1 <= spec["active_limit"] <= 30:
        raise ValueError("one intersection and at most 30 active CAV are supported")
    if not 1 <= len(spec["agents"]) <= 300:
        raise ValueError("bounded CAV admissions are required")
    from ..dataset.contract import digest

    configured = dict(spec)
    identity = configured.pop("id", None)
    if identity != digest(configured)[:24]:
        raise ValueError("configuration id must match its complete content")
    name = spec["experiment_name"]
    from ..dataset.contract import Contract
    from ..dataset.native import simulate
    from ..dataset.pipeline import file_sha256
    from .session import DetectionSession

    output = Path(output)
    complete = output / "complete.json"
    if complete.exists():
        old = json.loads(complete.read_text("utf-8"))
        if old["sources"] != expected or old["spec_id"] != identity:
            raise ValueError("experiment source identity changed; choose a new output")
        if any(
            file_sha256(output / relative) != fingerprint
            for relative, fingerprint in old.get("files", {}).items()
        ):
            raise ValueError("completed experimental evidence changed")
        return old
    if output.exists():
        raise ValueError(
            "partial experiment preserved; choose a new output for its explicit retry"
        )
    protected = Path(r"D:\cav-recovery-data\metadrive-dense-20261018").resolve()
    if output.resolve().is_relative_to(protected):
        raise ValueError("frozen M02 corpus cannot be an experimental output")
    if source_identity() != expected:
        raise ValueError("sources changed before experiment")
    existing_parent = output.resolve().parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < 8 * 1024**3:
        raise ValueError("experimental output requires an 8 GiB disk reserve")
    package = Path(__file__).parents[1]
    snapshot = {relative: (package / relative).read_bytes() for relative in expected}
    if any(
        hashlib.sha256(payload).hexdigest() != expected[name]
        for name, payload in snapshot.items()
    ):
        raise ValueError("source capture does not match experiment identity")
    session = DetectionSession(output, gui=gui)
    annotation = simulate(spec, output, Contract(), observer=session)
    for relative, payload in snapshot.items():
        target = output / "source-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    if source_identity() != expected:
        raise ValueError("sources changed during experiment")
    result = dict(
        name=name,
        seed=spec["seed"],
        family=spec["family"],
        spec_id=spec["id"],
        sources=expected,
        files={
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in output.rglob("*")
            if path.is_file()
        },
        runtime=dict(python=sys.version, platform=sys.platform),
        annotation=annotation,
        complete=True,
    )
    complete.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cgr demo", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    run = commands.add_parser("run")
    run.add_argument("scenario", choices=[row["name"] for row in catalog()])
    run.add_argument("--seed", type=int)
    run.add_argument("--horizon-s", type=float)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--gui", action="store_true")
    configured_run = commands.add_parser("run-config")
    configured_run.add_argument("configuration", type=Path)
    configured_run.add_argument("--output-dir", type=Path, required=True)
    configured_run.add_argument("--gui", action="store_true")
    replay_command = commands.add_parser("replay")
    replay_command.add_argument("recorded_experiment", type=Path)
    replay_command.add_argument("--output-dir", type=Path, required=True)
    campaign = commands.add_parser("evaluate")
    campaign.add_argument("--output-dir", type=Path, required=True)
    campaign.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(catalog(), indent=2))
        return 0
    sources = source_identity()
    if args.command == "replay":
        from .replay import replay

        print(json.dumps(replay(args.recorded_experiment, args.output_dir), indent=2))
        return 0
    if args.command == "run-config":
        from .configuration import load_specification

        result = run_spec(
            load_specification(args.configuration),
            args.output_dir,
            args.gui,
            sources,
        )
        print(
            json.dumps(
                {key: result[key] for key in ("name", "seed", "family", "spec_id")}
            )
        )
        return int(bool(result["annotation"]["integrity"]["errors"]))
    if args.command == "run":
        result = run_one(
            (
                args.scenario,
                args.seed,
                args.horizon_s,
                str(args.output_dir),
                args.gui,
                sources,
            )
        )
        print(
            json.dumps(
                {key: result[key] for key in ("name", "seed", "family", "spec_id")}
            )
        )
        return int(bool(result["annotation"]["integrity"]["errors"]))
    root = args.output_dir.resolve()
    if root.is_relative_to(
        Path(r"D:\cav-recovery-data\metadrive-dense-20261018").resolve()
    ):
        raise ValueError("frozen M02 corpus cannot be an experimental output")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "experiment.lock"
    if lock.exists():
        from ..platform_support import process_alive

        previous_pid = int(lock.read_text("utf-8"))
        if process_alive(previous_pid):
            raise ValueError("another experiment producer is active")
        lock.rename(root / f"stale-experiment-{previous_pid}-{time.time_ns()}.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    started = time.perf_counter()
    try:
        tasks = [
            (row["name"], row["seed"], None, str(root / row["name"]), False, sources)
            for row in catalog()
        ]
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(run_one, tasks):
                results.append(result)
                progress = dict(
                    completed=len(results),
                    planned=len(tasks),
                    elapsed_s=time.perf_counter() - started,
                    episodes=[
                        {key: row[key] for key in ("name", "seed", "family", "spec_id")}
                        for row in results
                    ],
                )
                (root / "progress.json").write_text(
                    json.dumps(progress, indent=2), encoding="utf-8"
                )
                print(json.dumps(progress["episodes"][-1]), flush=True)
        from .evaluation import evaluate_campaign

        evaluate_campaign(root)
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
