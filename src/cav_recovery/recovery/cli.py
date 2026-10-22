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
    audit = commands.add_parser(
        "audit-run", help="Une condition causale, sans entraînement."
    )
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--case", required=True)
    audit.add_argument("--condition", default="A")
    audit.add_argument(
        "--mode",
        choices=[
            "current",
            "plan",
            "ideal",
            "sequence",
            "motion-global",
            "motion-local",
        ],
        default="current",
        help="Variante expérimentale explicite ; les modes globaux sont privilégiés.",
    )
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--temporal-model", type=Path)
    audit.add_argument("--inference-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "audit-run":
        from .causal import case_specification, run_condition

        config = json.loads(args.config.read_text("utf-8"))
        spec = case_specification(config, args.case)
        if args.mode != "current" and (
            args.temporal_model is not None
            or args.condition != "A"
            or args.inference_threads != 1
        ):
            parser.error(
                "Les variantes dédiées définissent leur propre condition et leur déclencheur."
            )
        if args.mode == "current":
            result = run_condition(
                spec,
                args.output,
                config["conditions"][args.condition],
                temporal_model=args.temporal_model,
                inference_threads=args.inference_threads,
            )
        elif args.mode == "plan":
            from .plan_audit import run_planned_trigger

            result = run_planned_trigger(spec, args.output)
        elif args.mode == "ideal":
            from .oracle_audit import run_ideal_guard

            result = run_ideal_guard(spec, args.output)
        elif args.mode == "sequence":
            from .sequence_audit import run_sequence

            result = run_sequence(spec, args.output)
        else:
            from .motion_audit import run_motion_condition

            result = run_motion_condition(
                spec, args.output, local=args.mode == "motion-local"
            )
        print(json.dumps(result, indent=2))
        return 1 if result["errors"] else 0
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
