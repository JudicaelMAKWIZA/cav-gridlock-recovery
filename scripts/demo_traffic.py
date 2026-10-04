"""Prépare puis lance une démonstration graphique du trafic validé."""

import argparse
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cav_recovery.simulation.traffic_demand import TrafficInputError
from cav_recovery.simulation.traffic_run import run_traffic
from cav_recovery.simulation.traffic_scenario import new_output_directory, prepare_traffic


def main() -> int:
    """Ne nettoie la démonstration qu'après succès et fermeture normale confirmés."""
    parser = argparse.ArgumentParser(description="Préparer et observer le trafic dans sumo-gui.")
    parser.add_argument("--osm", required=True, help="Source OSM historique validée.")
    parser.add_argument("--contract", required=True, help="Contrat de trafic validé.")
    parser.add_argument("--regime", required=True, choices=("LOW", "MID", "HIGH"))
    parser.add_argument("--gui-delay-ms", type=int, default=100, help="Délai d'affichage en millisecondes.")
    parser.add_argument("--drain-horizon-s", type=float, default=600, help="Attente maximale après injection.")
    parser.add_argument("--keep-artifacts", help="Conserver les fichiers dans ce dossier absent ou vide.")
    args = parser.parse_args()
    workspace = None
    try:
        if args.keep_artifacts is not None:
            workspace = new_output_directory(args.keep_artifacts)
            workspace.mkdir(parents=True, exist_ok=True)
        else:
            # Pas de TemporaryDirectory : une erreur doit laisser les fichiers accessibles.
            workspace = Path(tempfile.mkdtemp(prefix="traffic-demo-")).resolve()
        scenario = workspace / "scenario"
        prepare_traffic(args.osm, args.contract, scenario)
        result = run_traffic(scenario, args.regime, workspace / "result", gui=True,
                             gui_delay_ms=args.gui_delay_ms, drain_horizon_s=args.drain_horizon_s)
        closed_normally = (
            result.get("connection_closed") is True
            and result.get("process_stopped") is True
            and result.get("process_returncode") == 0
            and result.get("forced_process_stop") is False
            and result.get("cleanup_errors") == []
        )
        if result.get("status") != "passed" or not closed_normally:
            details = [result.get("reason") or "Exécution ou fermeture normale non confirmée."]
            details.extend(result.get("cleanup_errors") or [])
            raise RuntimeError("; ".join(details))
        if args.keep_artifacts is None:
            # Seul l'espace créé par cette invocation peut être supprimé automatiquement.
            shutil.rmtree(workspace)
            print("Démonstration terminée. Les fichiers temporaires ont été nettoyés.")
        else:
            print(f"Démonstration terminée. Fichiers conservés : {workspace}")
        return 0
    except TrafficInputError as error:
        print(f"Entrée refusée : {error}", file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        print("Démonstration interrompue.", file=sys.stderr)
        code = 130
    except Exception as error:
        print(f"Démonstration impossible : {error}", file=sys.stderr)
        code = 1
    if workspace is not None and workspace.exists():
        print(f"Dossier conservé pour diagnostic : {workspace}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
