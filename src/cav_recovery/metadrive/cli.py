"""Lanceur MetaDrive ; les dépendances natives ne sont chargées qu'au lancement."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import os
from .scenes import PHYSICAL_SCENES, PHYSICAL_POLICIES


SCENARIOS = ("crossing", "convergence", "crossing_three", *PHYSICAL_SCENES)


def main(argv=None, *, runner=None):
    parser = argparse.ArgumentParser(
        prog="cgr metadrive",
        description="Scènes préparées multi-CAV, vue 2D uniquement.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scenarios", help="Décrire les scènes et leurs limites.")
    run = commands.add_parser("run", help="Exécuter une scène native MetaDrive.")
    run.add_argument("scenario", choices=SCENARIOS)
    run.add_argument(
        "--seed", type=int, default=1, help="Seed de la carte native, de 0 à 9."
    )
    view = run.add_mutually_exclusive_group()
    view.add_argument(
        "--gui", action="store_true", help="Afficher la fenêtre top-down 2D."
    )
    view.add_argument(
        "--headless",
        dest="gui",
        action="store_false",
        help="Ne pas ouvrir de fenêtre (défaut).",
    )
    run.set_defaults(gui=False)
    run.add_argument(
        "--gui-delay-ms",
        type=int,
        default=30,
        help="Délai par décision affichée, sans changer la physique.",
    )
    run.add_argument(
        "--horizon-s", type=float, default=90, help="Horizon simulé, multiple de 0,1 s."
    )
    run.add_argument(
        "--policy",
        choices=("courtesy_wait", "scheduled_passage", *PHYSICAL_POLICIES),
        default=None,
        help="M01 : courtesy_wait/scheduled_passage. Expériences physiques : road_following/global_clearance (état global).",
    )
    run.add_argument("--output-dir", type=Path, help="Dossier absent ou vide.")
    run.add_argument(
        "--no-sensors",
        dest="sensors",
        action="store_false",
        help="Témoin sans rayons de capteurs.",
    )
    run.add_argument(
        "--no-v2v",
        dest="v2v",
        action="store_false",
        help="Témoin sans transport applicatif.",
    )
    run.add_argument(
        "--no-crdg",
        dest="crdg",
        action="store_false",
        help="Témoin sans graphe d'observation.",
    )
    args = parser.parse_args(argv)
    if args.command == "scenarios":
        print(
            "crossing : deux CAV sur mouvements perpendiculaires ; scène F02 préparée."
        )
        print(
            "convergence : deux CAV convergent vers la même sortie ; scène F02 préparée."
        )
        print(
            "crossing_three : traversée et suiveur supplémentaire, trois missions natives."
        )
        for name, scene in PHYSICAL_SCENES.items():
            print(
                f"{name} : {scene['geometry']} ; {scene['formation']}. {scene['hypothesis']}"
            )
        print(
            "Aucun cas n'est présenté comme un gridlock spontané ou une détection D01 portée."
        )
        return 0
    output = args.output_dir or Path(
        os.environ.get("CGR_OUTPUT_ROOT", "outputs")
    ) / "metadrive" / args.scenario / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    try:
        if runner is None:
            from .run import run_metadrive

            runner = run_metadrive
        result = runner(
            output,
            case=args.scenario,
            seed=args.seed,
            horizon_s=args.horizon_s,
            gui=args.gui,
            gui_delay_ms=args.gui_delay_ms,
            policy=args.policy
            or (
                "road_following"
                if args.scenario in PHYSICAL_SCENES
                else "scheduled_passage"
            ),
            sensors=args.sensors,
            v2v=args.v2v,
            crdg=args.crdg,
        )
    except ImportError as error:
        print(
            f"MetaDrive indisponible : installer les dépendances MetaDrive épinglées. Détail : {error}",
            file=sys.stderr,
        )
        return 2
    except ValueError as error:
        print(f"Lancement refusé : {error}", file=sys.stderr)
        return 2
    print(f"MetaDrive / {args.scenario} / seed {args.seed} : {result['status']}")
    print(
        f"Programmés : {result['scheduled']} ; partis : {result['departed']} ; "
        f"arrivés : {len(result['arrived'])} ; actifs : {len(result['active'])} ; fin : {result['simulation_time_s']} s"
    )
    print(f"Bilan : {output}")
    if result.get("errors"):
        print(result["errors"], file=sys.stderr)
    return {"completed": 0, "horizon_reached": 3, "user_closed": 4, "failed": 1}[
        result["status"]
    ]
