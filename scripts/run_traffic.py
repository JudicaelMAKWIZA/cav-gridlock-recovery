"""Lance le trafic simulé de Kintambo depuis le dépôt."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cav_recovery.simulation.traffic_run import run_traffic
from cav_recovery.simulation.intersection_scenarios import scenario_names


def main() -> int:
    parser = argparse.ArgumentParser(description="Observer Kintambo ou un croisement contrôlé, sans récupération.")
    parser.add_argument("--scenario", choices=("kintambo", *scenario_names()), default="kintambo")
    parser.add_argument("--demand", choices=["LOW", "MEDIUM", "HIGH", "STRESS"], default="LOW")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gui", action="store_true", help="Utiliser sumo-gui avec démarrage automatique.")
    parser.add_argument("--street-names", action="store_true", help="Afficher les noms OSM dans la vue graphique.")
    parser.add_argument("--crdg", action="store_true", help="Observer les dépendances entre véhicules et espace aval.")
    parser.add_argument("--blockage-evidence", action="store_true",
                        help="Suivre la progression et les attentes observées ; active aussi le C-RDG, sans diagnostic de gridlock.")
    parser.add_argument("--gui-delay-ms", type=int, default=100, help="Délai visuel, sans effet sur le pas simulé.")
    parser.add_argument("--config", help="Configuration JSON de réseau et de demande.")
    parser.add_argument("--rate", type=float, help="Intensité en véhicules/heure/entrée.")
    parser.add_argument("--duration-s", type=float, help="Durée d'injection en secondes.")
    parser.add_argument("--drain-horizon-s", type=float, help="Attente maximale après injection.")
    parser.add_argument("--output-dir", help="Dossier absent ou vide ; sinon un nouveau dossier est créé.")
    args = parser.parse_args()
    output = args.output_dir or str(Path("outputs/simulation") / args.scenario /
                                   datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    try:
        result = run_traffic(output, demand=args.demand, seed=args.seed, gui=args.gui,
                                street_names=args.street_names,
                                gui_delay_ms=args.gui_delay_ms, config_path=args.config,
                                duration_s=args.duration_s, rate=args.rate,
                                drain_horizon_s=args.drain_horizon_s, crdg=args.crdg,
                                blockage_evidence=args.blockage_evidence, scenario=args.scenario)
    except ValueError as error:
        print(f"Entrée refusée : {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Expérience interrompue : {error}. Diagnostic : {output}", file=sys.stderr)
        return 1
    label = f"Kintambo / {args.demand}" if args.scenario == "kintambo" else args.scenario
    print(f"{label} / seed {args.seed} : {result['status']}")
    print(result["counts"])
    print(f"Bilan conservé : {output}")
    if result.get("reason"):
        print(result["reason"], file=sys.stderr)
    return {"completed": 0, "horizon_reached": 3, "failed": 1}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
