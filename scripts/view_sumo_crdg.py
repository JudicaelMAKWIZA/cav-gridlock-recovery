"""Ouvre un état natif figé de SUMO avec ses dépendances observées."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cav_recovery.simulation.crdg_gui import render_scene
from cav_recovery.simulation.road_network import require_binary


def main():
    parser = argparse.ArgumentParser(description="Examiner une scène C-RDG figée sur le réseau SUMO réel.")
    parser.add_argument("scene", type=Path, help="Dossier crdg_scenes/<instant> produit par run_traffic.")
    parser.add_argument("--focus", help="Véhicule présent à mettre en évidence ; défaut : choix enregistré.")
    parser.add_argument("--depth", type=int, choices=(1, 2, 3), help="Profondeur du voisinage.")
    parser.add_argument("--list-vehicles", action="store_true", help="Lister les IDs sans ouvrir de fenêtre.")
    args = parser.parse_args()
    try:
        if args.list_vehicles:
            data = json.loads((args.scene / "scene.json").read_text(encoding="utf-8"))
            print("\n".join(sorted(data["readings"])))
            return 0
        shown = render_scene(args.scene, focus=args.focus, depth=args.depth)
        print(f"État figé {shown['time_s']:g} s / {shown['focus']}. Fermer la fenêtre pour terminer.", flush=True)
        subprocess.run([require_binary("sumo-gui"), "-c", str(args.scene.resolve() / "scene.sumocfg"), "--start", "false"], check=True)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Scène refusée : {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
