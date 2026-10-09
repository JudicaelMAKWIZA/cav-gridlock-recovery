"""Construit une page C-RDG autonome, à ouvrir dans un navigateur local."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cav_recovery.crdg_view import write_view


def main():
    parser = argparse.ArgumentParser(description="Visualiser les graphes C-RDG exportés, sans calcul de diagnostic.")
    parser.add_argument("source", type=Path, help="Fichier crdg.jsonl issu d'une simulation.")
    parser.add_argument("--output", type=Path, help="Nouveau fichier HTML ; défaut : crdg_view.html près de la source.")
    parser.add_argument("--time", type=float, help="Ne garder que cet instant exactement exporté, utile pour un gros fichier.")
    args = parser.parse_args()
    try:
        result = write_view(args.source, args.output or args.source.with_name("crdg_view.html"), time_s=args.time)
    except (ValueError, OSError) as error:
        print(f"Visualisation refusée : {error}", file=sys.stderr)
        return 2
    print(f"Ouvrir localement : {result['output']}")
    print(f"Instants exportés inclus : {result['snapshots']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
