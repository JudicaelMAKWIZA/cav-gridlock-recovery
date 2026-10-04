"""Prépare les fichiers de réseau et les demandes de trafic validées."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cav_recovery.simulation.traffic_demand import TrafficInputError
from cav_recovery.simulation.traffic_scenario import prepare_traffic


def main() -> int:
    parser = argparse.ArgumentParser(description="Préparer le réseau et trois niveaux de trafic.")
    parser.add_argument("--osm", required=True, help="Source OSM historique validée.")
    parser.add_argument("--contract", required=True, help="Contrat empirique validé.")
    parser.add_argument("--output-dir", required=True, help="Dossier absent ou vide.")
    args = parser.parse_args()
    try:
        result = prepare_traffic(args.osm, args.contract, args.output_dir)
    except TrafficInputError as error:
        print(f"Entrée refusée : {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Préparation impossible : {error}", file=sys.stderr)
        return 1
    print("Réseau vérifié ; demandes LOW, MID et HIGH prêtes.")
    for name, record in result["regimes"].items():
        print(f"{name} : {len(record['missions'])} véhicules, injection {record['plan']['injection_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
