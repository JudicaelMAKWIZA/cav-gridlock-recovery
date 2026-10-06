"""Lance le trafic simulé de Kintambo depuis le dépôt."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cav_recovery.simulation.traffic_run import run_traffic


def main() -> int:
    parser = argparse.ArgumentParser(description="Observer une demande Poisson sur la topologie de Kintambo.")
    parser.add_argument("--demand", choices=["LOW", "MEDIUM", "HIGH", "STRESS"], default="LOW")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gui", action="store_true", help="Utiliser sumo-gui avec démarrage automatique.")
    parser.add_argument("--gui-delay-ms", type=int, default=100, help="Délai visuel, sans effet sur le pas simulé.")
    parser.add_argument("--config", help="Configuration JSON de réseau et de demande.")
    parser.add_argument("--rate", type=float, help="Intensité en véhicules/heure/entrée.")
    parser.add_argument("--duration-s", type=float, help="Durée d'injection en secondes.")
    parser.add_argument("--drain-horizon-s", type=float, help="Attente maximale après injection.")
    parser.add_argument("--output-dir", help="Dossier absent ou vide ; sinon un nouveau dossier est créé.")
    args = parser.parse_args()
    output = args.output_dir or str(Path("outputs/simulation/kintambo") /
                                   datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    try:
        result = run_traffic(output, demand=args.demand, seed=args.seed, gui=args.gui,
                                gui_delay_ms=args.gui_delay_ms, config_path=args.config,
                                duration_s=args.duration_s, rate=args.rate,
                                drain_horizon_s=args.drain_horizon_s)
    except ValueError as error:
        print(f"Entrée refusée : {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Expérience interrompue : {error}. Diagnostic : {output}", file=sys.stderr)
        return 1
    print(f"Kintambo / {args.demand} / seed {args.seed} : {result['status']}")
    print(result["counts"])
    print(f"Bilan conservé : {output}")
    if result.get("reason"):
        print(result["reason"], file=sys.stderr)
    return {"completed": 0, "horizon_reached": 3, "failed": 1}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
