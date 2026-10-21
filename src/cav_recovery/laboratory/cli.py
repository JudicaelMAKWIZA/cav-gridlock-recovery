"""Commandes reproductibles du laboratoire, chemins toujours configurables."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cgr lab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Cache dérivé reprenable, hors du corpus M02."
    )
    prepare.add_argument("dataset", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--workers", type=int, default=2)
    fit = commands.add_parser(
        "train", help="Entraîner ou reprendre exactement un candidat CPU."
    )
    fit.add_argument("cache", type=Path)
    fit.add_argument("output", type=Path)
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--trial", required=True)
    fit.add_argument("--feature-count", type=int)
    fit.add_argument("--simulate-power-cut-after", type=int)
    score = commands.add_parser(
        "evaluate", help="Calibrer sur validation puis évaluer le modèle sélectionné."
    )
    score.add_argument("cache", type=Path)
    score.add_argument("output", type=Path)
    score.add_argument("--validation-only", action="store_true")
    score.add_argument("--feature-count", type=int)
    campaign = commands.add_parser(
        "run", help="Protocole complet, sélection validation et comparaisons appariées."
    )
    campaign.add_argument("output", type=Path)
    campaign.add_argument("--config", type=Path, required=True)
    campaign.add_argument("--m02-cache", type=Path, required=True)
    campaign.add_argument("--integrated-cache", type=Path, required=True)
    evidence = commands.add_parser(
        "evidence", help="Exporter les figures et tableaux réels, sans réentraînement."
    )
    evidence.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .data import prepare_m02

        if (args.output / "complete.json").exists():
            from .storage import verify_cache_files, file_hash

            result = verify_cache_files(args.output)
            for relative, expected in result["source_files"].items():
                if file_hash(args.dataset / relative) != expected:
                    raise ValueError(
                        "Les sources immuables ne correspondent plus au cache."
                    )
        else:
            result = prepare_m02(args.dataset, args.output, args.workers)
        print(
            json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2)
        )
    elif args.command == "train":
        from .storage import safe_destination, protect_frozen_dataset

        safe_destination(args.cache, args.output)
        protect_frozen_dataset(args.output)
        from .training import train

        config = json.loads(args.config.read_text("utf-8"))
        trial = next(t for t in config.pop("trials") if t["name"] == args.trial)
        config.update(trial)
        print(
            json.dumps(
                train(
                    args.cache,
                    args.output,
                    config,
                    feature_count=args.feature_count,
                    stop_after_steps=args.simulate_power_cut_after,
                    abrupt_exit=args.simulate_power_cut_after is not None,
                ),
                indent=2,
            )
        )
    elif args.command == "evaluate":
        from .training import evaluate

        result = evaluate(
            args.cache,
            args.output,
            partitions=("validation",)
            if args.validation_only
            else ("validation", "test_ood"),
            feature_count=args.feature_count,
        )
        print(json.dumps({k: v["micro"] for k, v in result.items()}, indent=2))
    elif args.command == "run":
        from .protocol import run

        print(
            json.dumps(
                run(
                    args.output,
                    args.config,
                    m02_cache=args.m02_cache,
                    integrated_cache=args.integrated_cache,
                ),
                indent=2,
            )
        )
    elif args.command == "evidence":
        from .workbench import Laboratory
        from .evidence import export_evidence

        lab = Laboratory.from_environment()
        if args.root is not None:
            import os

            os.environ["CGR_LAB_ROOT"] = str(args.root)
            lab = Laboratory.from_environment()
        print(json.dumps(export_evidence(lab), indent=2))
    return 0
