"""Lance une demande préparée et vérifie les arrivées sans assistance."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cav_recovery.simulation.traffic_demand import TrafficInputError
from cav_recovery.simulation.traffic_run import run_traffic


def main() -> int:
    parser = argparse.ArgumentParser(description="Simuler une demande avec SUMO ou sumo-gui.")
    parser.add_argument("--scenario-dir", required=True, help="Dossier de scénario préparé.")
    parser.add_argument("--regime", required=True, choices=("LOW", "MID", "HIGH"))
    parser.add_argument("--output-dir", required=True, help="Nouveau dossier du bilan.")
    parser.add_argument("--gui", action="store_true", help="Utiliser sumo-gui avec démarrage automatique.")
    parser.add_argument("--gui-delay-ms", type=int, default=100, help="Délai d'affichage en millisecondes, sans changer le pas simulé.")
    parser.add_argument("--drain-horizon-s", type=float, default=600, help="Attente maximale après injection, en secondes simulées.")
    args = parser.parse_args()
    try:
        result = run_traffic(args.scenario_dir, args.regime, args.output_dir, gui=args.gui,
                             gui_delay_ms=args.gui_delay_ms, drain_horizon_s=args.drain_horizon_s)
    except TrafficInputError as error:
        print(f"Entrée refusée : {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Exécution impossible : {error}", file=sys.stderr)
        return 1
    counts = result["counts"]
    print(f"{args.regime} : {result['status']}")
    for name in ("scheduled", "departed", "arrived", "active", "future", "delayed_not_inserted",
                 "teleport_starts", "teleport_ends", "max_insertion_delay_s", "mean_insertion_delay_s",
                 "simulation_time_s"):
        print(f"{name} : {counts[name]}")
    print(f"Connexion fermée : {result['connection_closed']} ; processus arrêté : {result['process_stopped']}")
    if result["reason"]:
        print(result["reason"], file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
