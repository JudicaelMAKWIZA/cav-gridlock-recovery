"""Interface de construction du contrat empirique de trafic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# La commande doit fonctionner depuis un checkout sans installation préalable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cav_recovery.empirical.scenario_contract import ContractInputError, build_empirical_contract


def main() -> int:
    """Construit le contrat et traduit les erreurs en codes de sortie stables."""

    parser = argparse.ArgumentParser(
        description="Construire un contrat empirique depuis des profils de trafic validés."
    )
    parser.add_argument("--profile-dir", required=True, help="Dossier contenant les profils de trafic validés")
    parser.add_argument("--coverage", required=True, help="Fichier décrivant la couverture temporelle")
    parser.add_argument("--output-dir", required=True, help="Nouveau dossier de résultats")
    arguments = parser.parse_args()
    try:
        summary = build_empirical_contract(arguments.profile_dir, arguments.coverage, arguments.output_dir)
    except ContractInputError as error:
        print(f"Contrat refusé : {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Erreur technique : {error}", file=sys.stderr)
        return 1
    windows = summary["windows"]
    print(
        f"Contrat {summary['status']} : {windows['total']} fenêtres, "
        f"dont {windows['complete']} complètes et {windows['partial']} partielle."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
