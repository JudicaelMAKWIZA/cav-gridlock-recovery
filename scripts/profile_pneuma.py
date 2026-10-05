"""Commande de construction des profils de trafic d'un secteur."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# La commande doit fonctionner depuis un checkout sans installation préalable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cav_recovery.empirical.demand_profile import ProfileInputError, profile_pneuma
from cav_recovery.empirical.sector import SectorConfigurationError


def main() -> int:
    """Construit les profils et traduit les erreurs en codes de sortie stables."""
    parser = argparse.ArgumentParser(description="Profiler un secteur pNEUMA depuis ses observations qualifiées.")
    parser.add_argument("--source", required=True, help="CSV pNEUMA de référence, utilisé pour le contrôle d'identité")
    parser.add_argument("--cgr-e01-dir", required=True, help="Dossier des exports de qualification")
    parser.add_argument("--sector-seed", required=True, help="Configuration géométrique du secteur")
    parser.add_argument("--geometry-source", required=True, help="Fichier géométrique identifié dans la configuration du secteur")
    parser.add_argument("--runtime-config", required=True, help="Seuils et couverture déclarés pour l'exécution")
    parser.add_argument("--output-dir", required=True, help="Dossier absent ou vide pour les profils")
    arguments = parser.parse_args()
    try:
        summary = profile_pneuma(
            arguments.source, arguments.cgr_e01_dir, arguments.sector_seed,
            arguments.geometry_source, arguments.runtime_config, arguments.output_dir,
        )
    except (ProfileInputError, SectorConfigurationError) as error:
        print(f"Profil refusé : {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Erreur technique de profilage : {error}", file=sys.stderr)
        return 1
    print(
        f"Profil : exécution {summary['execution']['status']} ; "
        f"admissibilité {summary['empirical_admissibility']['status']} ; "
        f"validation scientifique {summary['scientific_validation']['status']} : "
        f"{summary['counts']['crossings']} franchissements, {summary['counts']['visits']} visites."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
