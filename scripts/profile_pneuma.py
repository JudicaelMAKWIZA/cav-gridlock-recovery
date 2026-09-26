"""Interface de lancement du profil CGR-E02."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# La commande doit fonctionner depuis un checkout sans installation préalable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cav_recovery.empirical.demand_profile import ProfileInputError, profile_pneuma
from cav_recovery.empirical.sector import SectorConfigurationError


def main() -> int:
    """Valide les arguments, exécute CGR-E02 et traduit les erreurs en codes stables."""
    parser = argparse.ArgumentParser(description="Profiler un secteur pNEUMA depuis les exports validés de CGR-E01.")
    parser.add_argument("--source", required=True, help="CSV pNEUMA canonique, utilisé uniquement pour le contrôle d'identité")
    parser.add_argument("--cgr-e01-dir", required=True, help="Dossier des exports validés CGR-E01")
    parser.add_argument("--sector-seed", required=True, help="Configuration géométrique figée du secteur")
    parser.add_argument("--geometry-source", required=True, help="Fichier géométrique dont l'empreinte figure dans le seed")
    parser.add_argument("--runtime-config", required=True, help="Seuils et couverture explicitement gelés pour l'exécution")
    parser.add_argument("--output-dir", required=True, help="Nouveau dossier privé de résultats CGR-E02")
    arguments = parser.parse_args()
    try:
        summary = profile_pneuma(arguments.source, arguments.cgr_e01_dir, arguments.sector_seed, arguments.geometry_source, arguments.runtime_config, arguments.output_dir)
    except (ProfileInputError, SectorConfigurationError) as error:
        print(f"CGR-E02 refusé : {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"CGR-E02 erreur technique : {error}", file=sys.stderr)
        return 1
    print(
        f"CGR-E02 exécution {summary['execution']['status']} ; "
        f"admissibilité {summary['empirical_admissibility']['status']} ; "
        f"validation scientifique {summary['scientific_validation']['status']} : "
        f"{summary['counts']['crossings']} franchissements, {summary['counts']['visits']} visites."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
