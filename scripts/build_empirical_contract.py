"""Interface de construction du contrat empirique CGR-E03."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# La commande doit fonctionner depuis un checkout sans installation préalable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cav_recovery.empirical.scenario_contract import ContractInputError, build_empirical_contract


def main() -> int:
    """Valide les chemins, exécute CGR-E03 et traduit les erreurs en codes stables."""
    parser = argparse.ArgumentParser(description="Construire le contrat empirique CGR-E03 depuis les agrégats CGR-E02.")
    parser.add_argument("--cgr-e02-dir", required=True, help="Dossier privé de l'exécution nominale CGR-E02")
    parser.add_argument("--coverage", required=True, help="Preuve privée de couverture CGR-E02")
    parser.add_argument("--output-dir", required=True, help="Nouveau dossier privé de résultats CGR-E03")
    arguments = parser.parse_args()
    try:
        summary = build_empirical_contract(arguments.cgr_e02_dir, arguments.coverage, arguments.output_dir)
    except ContractInputError as error:
        print(f"CGR-E03 refusé : {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"CGR-E03 erreur technique : {error}", file=sys.stderr)
        return 1
    windows = summary["windows"]
    print(
        f"CGR-E03 {summary['status']} : {windows['total']} fenêtres, "
        f"dont {windows['complete']} complètes et {windows['partial']} partielle."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
