"""Commandes actives MetaDrive et préparation des données scientifiques."""

import argparse
import sys


def main(argv=None, *, runner=None):
    """Charge uniquement le composant demandé, sans moteur au démarrage."""
    values = list(sys.argv[1:] if argv is None else argv)
    if values[:1] == ["demo"]:
        from .experiments.cli import main as run

        return run(values[1:])
    if values[:1] == ["metadrive"]:
        from .metadrive.cli import main as run

        return run(values[1:], runner=runner)
    if values[:1] == ["dataset"]:
        from .dataset.cli import main as run

        return run(values[1:])
    if values[:1] == ["d01"]:
        from .d01.native import main as run

        return run(values[1:])
    parser = argparse.ArgumentParser(
        prog="cgr", description="Simulation MetaDrive et diagnostic CAV."
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("metadrive", help="Scénarios et simulation physique.")
    commands.add_parser("dataset", help="Génération, qualification et exports M02.")
    commands.add_parser("d01", help="Environnement d’essai individuel D01.")
    commands.add_parser("demo", help="Détection intégrée, C-RDG et démonstrations.")
    parser.parse_args(values)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
