"""Commande commune ; les exécuteurs MetaDrive et SUMO restent indépendants."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import shutil
import sys

from .simulation.kintambo_scenarios import case_names, read_case


def run_traffic(*args, **kwargs):
    # MetaDrive peut se lancer sans charger TraCI ni l'interface SUMO.
    from .simulation.traffic_run import run_traffic as execute
    return execute(*args, **kwargs)


def add_run_options(parser, *, gui_default=False, delay=100, v2v_default=False):
    parser.add_argument("--demand", choices=("LOW", "MEDIUM", "HIGH", "STRESS"), default="LOW")
    parser.add_argument("--seed", type=int, default=1)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--gui", action="store_true", help="Ouvrir SUMO animé et INFO C-RDG.")
    display.add_argument("--no-gui", dest="gui", action="store_false", help="Exécuter sans fenêtre.")
    parser.set_defaults(gui=gui_default)
    parser.add_argument("--street-names", action="store_true", help="Afficher les noms OSM dans SUMO.")
    parser.add_argument("--crdg", action="store_true", help="Enregistrer les dépendances observées, sans diagnostic.")
    parser.add_argument("--crdg-live", action="store_true", help="Ancien nom ; --gui ouvre déjà INFO C-RDG.")
    parser.add_argument("--blockage-evidence", action="store_true", help="Enregistrer les épisodes physiques, sans diagnostic.")
    detection = parser.add_mutually_exclusive_group()
    detection.add_argument("--detect-obstructions", dest="detect_obstructions", action="store_true",
                           help="Caractériser les obstructions et leurs incertitudes, sans confirmer un gridlock.")
    detection.add_argument("--no-detect-obstructions", dest="detect_obstructions", action="store_false",
                           help="Désactiver le détecteur dans un raccourci graphique.")
    parser.set_defaults(detect_obstructions=None)
    communication = parser.add_mutually_exclusive_group()
    communication.add_argument("--v2v", dest="v2v", action="store_true",
                               help="Échanger les états propres des CAV à portée, sans agir sur le trafic.")
    communication.add_argument("--no-v2v", dest="v2v", action="store_false",
                               help="Désactiver les échanges pour un essai témoin.")
    parser.set_defaults(v2v=v2v_default)
    parser.add_argument("--v2v-range-m", type=float, default=120.0, help="Portée directe en mètres (défaut : 120).")
    parser.add_argument("--v2v-interval-s", type=float, default=1.0, help="Cadence d'émission en secondes (défaut : 1).")
    parser.add_argument("--v2v-lifetime-s", type=float, default=2.0, help="Validité d'un message en secondes (défaut : 2).")
    parser.add_argument("--gui-delay-ms", type=int, default=delay, help="Délai graphique en millisecondes, sans changer le pas SUMO.")
    parser.add_argument("--close-on-end", action="store_true", help="Fermer à la fin plutôt que garder la vue inspectable.")
    parser.add_argument("--output-mode", choices=("full", "interactive"), default="full", help="full garde les traces fines ; interactive garde le bilan et les preuves consultées.")
    parser.add_argument("--config", help="Configuration JSON du réseau et de la demande Poisson.")
    parser.add_argument("--rate", type=float, help="Véhicules/heure/entrée, pour Poisson uniquement.")
    parser.add_argument("--duration-s", type=float, help="Durée d'injection Poisson en secondes.")
    parser.add_argument("--drain-horizon-s", type=float, help="Horizon après injection, sans changer les missions.")
    parser.add_argument("--output-dir", help="Dossier absent ou vide ; un dossier daté est créé par défaut.")
    parser.add_argument("--crdg-scene-at", type=float, action="append", default=[], help="Garder un état SUMO natif et ses preuves à cet instant ; répétable.")
    parser.add_argument("--crdg-focus", help="ID SUMO à sélectionner au départ dans INFO C-RDG.")
    parser.add_argument("--crdg-depth", type=int, choices=(1, 2, 3), default=2, help=argparse.SUPPRESS)


def install_launcher(project, *, shell_path=False, home=None):
    """Installe un petit relais utilisateur ; le pipeline reste celui de main."""
    if os.name != "posix":
        raise ValueError("Ce lanceur est prévu pour Ubuntu/WSL.")
    project = Path(project).resolve()
    binary = project / ".venv/bin"
    python = binary / "python"
    if not (project / "pyproject.toml").is_file() or not (project / "src/cav_recovery/cli.py").is_file() or not python.is_file():
        raise ValueError("Projet ou .venv absent ; installer le package dans .venv avant le lanceur.")
    task_home = Path(home) if home is not None else Path.home()
    target = task_home / ".local/bin/cgr"
    marker = "# Lanceur utilisateur CAV Gridlock Recovery"
    if target.is_symlink() or (target.exists() and (not target.is_file() or marker.encode() not in target.read_bytes())):
        raise ValueError(f"Commande existante non remplacée : {target}")
    existing = shutil.which("cgr")
    if existing and Path(existing).absolute() not in (target.absolute(), (binary / "cgr").absolute()):
        raise ValueError(f"Une autre commande cgr existe déjà : {existing}")
    target.parent.mkdir(parents=True, exist_ok=True)
    code = ("#!/bin/sh\n" + marker + "\n"
            f"if [ ! -x {shlex.quote(str(python))} ]; then\n"
            "  echo 'cgr : environnement .venv indisponible ; réinstaller le lanceur depuis le projet.' >&2\n  exit 1\nfi\n"
            f"if [ ! -d {shlex.quote(str(project))} ]; then\n"
            "  echo 'cgr : dépôt déplacé ; réinstaller le lanceur au nouvel emplacement.' >&2\n  exit 1\nfi\n"
            f"export PATH={shlex.quote(str(binary))}:\"$PATH\"\n"
            f"cd {shlex.quote(str(project))} || exit 1\n"
            f"exec {shlex.quote(str(python))} -m cav_recovery.cli \"$@\"\n")
    target.write_text(code, encoding="utf-8")
    target.chmod(0o755)
    print(f"Lanceur installé : {target}\nRésultats par défaut dans : {project / 'outputs/simulation'}")
    if shell_path:
        # Ce changement de démarrage n'est effectué que sur demande explicite.
        block = b'\n# cgr : commandes utilisateur\nexport PATH="$HOME/.local/bin:$PATH"\n'
        profile = next((task_home / name for name in (".bash_profile", ".bash_login", ".profile")
                        if (task_home / name).exists()), task_home / ".profile")
        for startup in (task_home / ".bashrc", profile):
            content = startup.read_bytes() if startup.exists() else b""
            if block.strip() not in content:
                startup.write_bytes(content + block)
            print(f"PATH utilisateur ajouté explicitement dans {startup}.")
        print("Ouvrir un nouveau terminal Ubuntu.")
    else:
        print("Aucun fichier de démarrage modifié. Ajouter ~/.local/bin au PATH, ou réinstaller avec --shell-path.")
    return 0


def prompt_integer(label, default):
    while True:
        value = input(f"{label} [{default}] : ").strip()
        try:
            number = int(value) if value else default
            if number >= 0:
                return number
        except ValueError:
            pass
        print("Saisir un entier positif ou nul.")


def interactive_menu(*, runner=None):
    """Le menu utilise les mêmes scénarios et le même run que les commandes directes."""
    try:
        while True:
            print("\nCAV GRIDLOCK RECOVERY\n1. Lancer SUMO et observer le C-RDG\n2. Consulter les scénarios\n3. Aide\n0. Quitter")
            print("Diagnostic confirmé, récupération, entraînement et évaluation : non implémentés.")
            choice = input("Votre choix : ").strip()
            if choice == "0":
                print("Menu fermé.")
                return 0
            if choice in ("2", "3"):
                if choice == "2":
                    print("\nSCÉNARIOS DISPONIBLES\n")
                    main(["scenarios"], runner=runner)
                else:
                    print("\nAIDE CAV GRIDLOCK RECOVERY\n"
                          "1 : choisir un scénario et lancer la simulation ; en GUI, le détecteur est proposé.\n"
                          "2 : lire les scénarios disponibles et leurs objectifs.\n"
                          "0 : quitter le menu principal.\n\n"
                          "cgr crossing : ouvrir SUMO et INFO, seed 1, délai 300 ms.\n"
                          "cgr crossing --seed 2 --gui-delay-ms 100 : modifier les valeurs.\n"
                          "cgr run crossing --help : consulter les options avancées.\n"
                          "cgr --help : consulter les commandes disponibles.")
                input("\nAppuyez sur Entrée pour revenir au menu principal… ")
                continue
            if choice != "1":
                print("Choix inconnu.")
                continue
            names = ("kintambo", *case_names())
            print("\nSCÉNARIOS DISPONIBLES")
            for index, name in enumerate(names, 1):
                print(f"{index}. {name}")
            print("0. Retour")
            selection = input("Scénario : ").strip()
            if selection == "0":
                continue
            if not selection.isdigit() or not 1 <= int(selection) <= len(names):
                print("Scénario inconnu.")
                continue
            name = names[int(selection) - 1]
            seed = prompt_integer("Seed", 1)
            display = input("Ouvrir SUMO et INFO C-RDG ? [O/n] : ").strip().lower()
            if display not in ("", "o", "oui", "n", "non"):
                print("Répondre oui ou non ; lancement annulé.")
                continue
            gui = display in ("", "o", "oui")
            if gui:
                detection = input("Activer le détecteur ? [O/n] : ").strip().lower()
                if detection not in ("", "o", "oui", "n", "non"):
                    print("Répondre oui ou non ; lancement annulé.")
                    continue
                communication = input("Activer les échanges V2V ? [O/n] : ").strip().lower()
                if communication not in ("", "o", "oui", "n", "non"):
                    print("Répondre oui ou non ; lancement annulé.")
                    continue
            delay = prompt_integer("Délai graphique en ms", 300) if gui else 100
            names_on = input("Afficher les noms des rues ? [o/N] : ").strip().lower() if gui else "n"
            if names_on not in ("", "o", "oui", "n", "non"):
                print("Répondre oui ou non ; lancement annulé.")
                continue
            args = ["run", name, "--seed", str(seed), "--gui-delay-ms", str(delay)]
            if gui:
                args.append("--gui")
                if detection in ("", "o", "oui"):
                    args.append("--detect-obstructions")
                if communication in ("", "o", "oui"):
                    args.append("--v2v")
            if names_on in ("o", "oui"):
                args.append("--street-names")
            if name == "kintambo":
                level = input("Demande LOW/MEDIUM/HIGH/STRESS [LOW] : ").strip().upper() or "LOW"
                if level not in ("LOW", "MEDIUM", "HIGH", "STRESS"):
                    print("Niveau inconnu ; lancement annulé.")
                    continue
                args.extend(["--demand", level])
            print(f"Lancement : {' '.join(['cgr', *args])} ; traces complètes conservées.")
            code = main(args, runner=runner)
            if code == 4:
                print("Démo fermée à votre demande. Retour au menu.")
            elif code in (0, 3):
                print("Démo fermée ; simulation terminée. Retour au menu.")
            else:
                print(f"Exécution interrompue (code {code}). Retour au menu.")
    except (EOFError, KeyboardInterrupt):
        print("\nMenu fermé.")
        return 0


def main(argv=None, *, legacy=False, runner=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if not legacy and values[:1] == ["dataset"]:
        from .dataset.cli import main as dataset_main
        return dataset_main(values[1:])
    if not legacy and values[:1] == ["metadrive"]:
        from .metadrive.cli import main as metadrive_main
        return metadrive_main(values[1:], runner=runner)
    parser = argparse.ArgumentParser(prog="cgr", description="Lancer MetaDrive 2D ou le banc historique SUMO/Kintambo.")
    if legacy:
        add_run_options(parser)
        parser.add_argument("--scenario", default="kintambo", help="Compatibilité : seul Kintambo est conservé.")
        parser.add_argument("--kintambo-case", choices=case_names())
        parser.add_argument("--crdg-gui", action="store_true", help=argparse.SUPPRESS)
    else:
        commands = parser.add_subparsers(dest="command", required=True)
        commands.add_parser("metadrive", help="Lancer le socle MetaDrive 2D, indépendant de SUMO.")
        commands.add_parser("dataset", help="Générer et qualifier les épisodes physiques MetaDrive.")
        commands.add_parser("scenarios", help="Lister les scénarios disponibles et leurs hypothèses.")
        run = commands.add_parser("run", help="Préparer le réseau, lancer SUMO et garder les résultats.")
        run.add_argument("case", choices=("kintambo", *case_names()))
        add_run_options(run)
        install = commands.add_parser("install", help="Installer le lanceur utilisateur depuis le dépôt.")
        install.add_argument("--project", type=Path, default=Path.cwd(), help="Dépôt contenant .venv ; réinstaller après un déplacement.")
        install.add_argument("--shell-path", action="store_true", help="Ajouter explicitement ~/.local/bin au PATH de .bashrc et du profil de connexion.")
        for name in ("kintambo", *case_names()):
            shortcut = commands.add_parser(name, help=f"Ouvrir {name} dans SUMO et INFO C-RDG.")
            shortcut.set_defaults(case=name)
            add_run_options(shortcut, gui_default=True, delay=300, v2v_default=True)
        if not values:
            if sys.stdin.isatty():
                return interactive_menu(runner=runner)
            parser.print_help()
            print("Le menu demande un terminal interactif ; utiliser cgr run ou un raccourci explicite.")
            return 0
    args = parser.parse_args(values)
    if not legacy and args.command == "install":
        try:
            return install_launcher(args.project, shell_path=args.shell_path)
        except (OSError, ValueError) as error:
            print(f"Installation refusée : {error}", file=sys.stderr)
            return 2
    if not legacy and args.command == "scenarios":
        print("kintambo : trafic Poisson canonique ; LOW / MEDIUM / HIGH / STRESS")
        for name in case_names():
            row = read_case(name)
            variant = " [variante adverse déclarée]" if row.get("network_variant") else ""
            print(f"{name}{variant} : {row['aim']}")
        return 0
    if legacy:
        if args.scenario != "kintambo":
            parser.error("Les anciens carrefours indépendants sont retirés. Consulter cgr scenarios.")
        if args.crdg_gui:
            parser.error("Le visualiseur figé est remplacé par INFO C-RDG : utiliser --gui. Les états natifs restent enregistrables avec --crdg-scene-at.")
        case = args.kintambo_case
    else:
        case = None if args.case == "kintambo" else args.case
    detect_obstructions = (args.detect_obstructions if args.detect_obstructions is not None
                           else bool(args.gui and not legacy and args.command != "run"))
    output = args.output_dir or str(Path("outputs/simulation") / (case or "kintambo") /
                                   datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    try:
        result = (runner or run_traffic)(output, demand=args.demand, seed=args.seed, gui=args.gui,
                 street_names=args.street_names, gui_delay_ms=args.gui_delay_ms, config_path=args.config,
                 duration_s=args.duration_s, rate=args.rate, drain_horizon_s=args.drain_horizon_s,
                 crdg=args.crdg or not legacy, blockage_evidence=args.blockage_evidence,
                 detect_obstructions=detect_obstructions,
                 v2v=args.v2v, v2v_range_m=args.v2v_range_m,
                 v2v_interval_s=args.v2v_interval_s, v2v_lifetime_s=args.v2v_lifetime_s,
                 scenario="kintambo", kintambo_case=case, crdg_scene_times=tuple(args.crdg_scene_at),
                 crdg_focus=args.crdg_focus, crdg_depth=args.crdg_depth,
                 crdg_live=args.gui or args.crdg_live, close_on_end=args.close_on_end, output_mode=args.output_mode)
    except ValueError as error:
        print(f"Entrée refusée : {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Exécution interrompue : {error}. Diagnostic : {output}", file=sys.stderr)
        return 1
    print(f"Kintambo / {case or args.demand} / seed {args.seed} : {result['status']}")
    print(result["counts"])
    print(f"Bilan conservé : {output}")
    if result.get("reason"):
        print(result["reason"], file=sys.stderr)
    if result.get("info_crdg", {}).get("errors"):
        print(f"Affichage en erreur : {result['info_crdg']['errors']}", file=sys.stderr)
        return 1
    return {"completed": 0, "horizon_reached": 3, "failed": 1, "user_closed": 4}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
