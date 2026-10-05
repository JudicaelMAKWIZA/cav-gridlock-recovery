"""Commande de qualification d'un fichier pNEUMA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# La commande fonctionne aussi depuis un checkout sans installation préalable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cav_recovery.empirical.qualification import qualify_pneuma


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualifier un fichier pNEUMA sans le modifier.")
    parser.add_argument("--input", required=True, help="CSV pNEUMA source (lecture seule)")
    parser.add_argument("--output-dir", required=True, help="Dossier absent ou vide pour les exports")
    args = parser.parse_args()
    try:
        summary = qualify_pneuma(args.input, args.output_dir)
    except ValueError as error:
        print(f"Qualification refusée : {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Erreur technique de qualification : {error}", file=sys.stderr)
        return 1
    print(f"Qualification {summary['status']} : {summary['counts']['observation_export_rows']} observations exportées.")
    return 0 if summary["status"] in {"complete", "complete_with_issues"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
