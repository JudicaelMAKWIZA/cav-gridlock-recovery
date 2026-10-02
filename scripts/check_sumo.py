"""Commande de vérification du trajet synthétique SUMO/TraCI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Le contrôle doit aussi fonctionner depuis un checkout sans installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cav_recovery.simulation.sumo_smoke import SmokeInputError, run_sumo_smoke


def main() -> int:
    """Affiche le bilan et signale les échecs au shell."""
    parser = argparse.ArgumentParser(description="Vérifier un trajet synthétique avec SUMO et TraCI.")
    parser.add_argument("--sumo-binary", default="sumo", help="Binaire SUMO, disponible dans PATH par défaut")
    parser.add_argument("--horizon", type=float, default=60.0, help="Horizon maximal en secondes simulées")
    args = parser.parse_args()
    try:
        result = run_sumo_smoke(ROOT / "tests/fixtures/sumo_smoke",
                                sumo_binary=args.sumo_binary, horizon_s=args.horizon)
    except SmokeInputError as error:
        print(f"Contrôle refusé : {error}", file=sys.stderr)
        return 2
    print("Contrôle SUMO/TraCI : " + ("réussi" if result["status"] == "passed" else "échec"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
