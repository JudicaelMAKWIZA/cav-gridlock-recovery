"""CLI CGR-E01 : qualification descriptive d'un fichier pNEUMA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow the documented command to work from a source checkout, without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cav_recovery.empirical.qualification import qualify_pneuma


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualifier un fichier pNEUMA sans le modifier.")
    parser.add_argument("--input", required=True, help="CSV pNEUMA source (lecture seule)")
    parser.add_argument("--output-dir", required=True, help="Dossier privé, nouveau ou vide, des exports")
    args = parser.parse_args()
    try:
        summary = qualify_pneuma(args.input, args.output_dir)
    except ValueError as error:
        print(f"CGR-E01 refusé : {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"CGR-E01 erreur technique : {error}", file=sys.stderr)
        return 1
    print(f"CGR-E01 {summary['status']} : {summary['counts']['observation_export_rows']} observations exportées.")
    return 0 if summary["status"] in {"complete", "complete_with_issues"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
