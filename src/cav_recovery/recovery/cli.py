"""Essais physiques, comparaison appariée et pilote soumis à preuve locale."""

import argparse
from dataclasses import replace
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cgr recovery", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="Essai natif privé, nominal ou baseline locale."
    )
    run.add_argument("specification", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--method",
        choices=["nominal", "deterministic", "cooperative", "global_search"],
        default="nominal",
    )
    run.add_argument("--reverse-budget-m", type=float, default=8.0)
    run.add_argument("--stall-s", type=float, default=35.0)
    run.add_argument(
        "--measurement-contract",
        choices=["raw", "REC-METRO-1"],
        default=None,
        help="Composition diagnostique passive ; aucun changement d'activation.",
    )
    comparison = commands.add_parser(
        "compare", help="Mesure hors ligne de deux essais appariés."
    )
    comparison.add_argument("nominal", type=Path)
    comparison.add_argument("intervention", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    pilot = commands.add_parser(
        "pilot", help="Pilote Graph-PPO court, soumis à une preuve locale vérifiée."
    )
    pilot.add_argument("--config", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--proof", type=Path, required=True)
    pilot.add_argument("--stop-after", type=int)
    args = parser.parse_args(argv)
    if args.command == "pilot":
        from .pilot import train

        return train(
            args.config, args.output, proof_path=args.proof, stop_after=args.stop_after
        )
    if args.command == "compare":
        from .assessment import paired_assessment as compare

        result = compare(args.nominal, args.intervention, args.output)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in {"nominal", "intervention"}
                },
                indent=2,
            )
        )
        return 0
    from .contracts import RecoveryContract
    from .environment import run as execute

    contract = replace(
        RecoveryContract(), reverse_budget_m=args.reverse_budget_m, stall_s=args.stall_s
    )
    spec = json.loads(args.specification.read_text("utf-8"))
    result = execute(
        spec,
        args.output,
        method=args.method,
        contract=contract,
        measurement_contract=args.measurement_contract,
    )
    print(json.dumps(result, indent=2))
    return 1 if result["errors"] else 0
