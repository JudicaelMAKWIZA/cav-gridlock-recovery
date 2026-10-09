"""Lance le trafic simulé de Kintambo depuis le dépôt."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cav_recovery.simulation.traffic_run import run_traffic
from cav_recovery.simulation.intersection_scenarios import scenario_names
from cav_recovery.simulation.kintambo_scenarios import case_names


def main() -> int:
    parser = argparse.ArgumentParser(description="Observer Kintambo ou un croisement contrôlé, sans récupération.")
    parser.add_argument("--scenario", choices=("kintambo", *scenario_names()), default="kintambo")
    parser.add_argument("--kintambo-case", choices=case_names(), help="Variante ciblée de demande, réseau canonique inchangé.")
    parser.add_argument("--demand", choices=["LOW", "MEDIUM", "HIGH", "STRESS"], default="LOW")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gui", action="store_true", help="Utiliser sumo-gui avec démarrage automatique.")
    parser.add_argument("--street-names", action="store_true", help="Afficher les noms OSM dans la vue graphique.")
    parser.add_argument("--crdg", action="store_true", help="Observer les dépendances entre véhicules et espace aval.")
    parser.add_argument("--crdg-gui", action="store_true", help="Ouvrir les scènes figées enregistrées dans SUMO-GUI, après le run.")
    parser.add_argument("--crdg-scene-at", type=float, action="append", default=[], help="Enregistrer l'état natif et le C-RDG à cet instant exact ; option répétable.")
    parser.add_argument("--crdg-focus", help="ID de véhicule à mettre en évidence dans les scènes.")
    parser.add_argument("--crdg-depth", type=int, choices=(1, 2, 3), default=2, help="Profondeur du voisinage affiché.")
    parser.add_argument("--blockage-evidence", action="store_true",
                        help="Suivre la progression et les attentes observées ; active aussi le C-RDG, sans diagnostic de gridlock.")
    parser.add_argument("--gui-delay-ms", type=int, default=100, help="Délai visuel, sans effet sur le pas simulé.")
    parser.add_argument("--config", help="Configuration JSON de réseau et de demande.")
    parser.add_argument("--rate", type=float, help="Intensité en véhicules/heure/entrée.")
    parser.add_argument("--duration-s", type=float, help="Durée d'injection en secondes.")
    parser.add_argument("--drain-horizon-s", type=float, help="Attente maximale après injection.")
    parser.add_argument("--output-dir", help="Dossier absent ou vide ; sinon un nouveau dossier est créé.")
    args = parser.parse_args()
    if args.crdg_gui and (not args.crdg_scene_at or not args.crdg_focus):
        parser.error("--crdg-gui demande --crdg-scene-at et --crdg-focus.")
    output = args.output_dir or str(Path("outputs/simulation") / args.scenario /
                                   datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    try:
        result = run_traffic(output, demand=args.demand, seed=args.seed, gui=args.gui,
                                street_names=args.street_names,
                                gui_delay_ms=args.gui_delay_ms, config_path=args.config,
                                duration_s=args.duration_s, rate=args.rate,
                                drain_horizon_s=args.drain_horizon_s, crdg=args.crdg,
                                blockage_evidence=args.blockage_evidence, scenario=args.scenario,
                                kintambo_case=args.kintambo_case, crdg_scene_times=tuple(args.crdg_scene_at),
                                crdg_focus=args.crdg_focus, crdg_depth=args.crdg_depth)
    except ValueError as error:
        print(f"Entrée refusée : {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Expérience interrompue : {error}. Diagnostic : {output}", file=sys.stderr)
        return 1
    label = f"Kintambo / {args.demand}" if args.scenario == "kintambo" else args.scenario
    if args.kintambo_case:
        label = f"Kintambo / {args.kintambo_case}"
    print(f"{label} / seed {args.seed} : {result['status']}")
    print(result["counts"])
    print(f"Bilan conservé : {output}")
    if result.get("reason"):
        print(result["reason"], file=sys.stderr)
    scenes = result.get("crdg_scenes", {})
    if scenes.get("unobserved_times_s"):
        print(f"Instants non observés : {scenes['unobserved_times_s']}", file=sys.stderr)
    if scenes.get("focus_absent_times_s"):
        print(f"Véhicule choisi absent à {scenes['focus_absent_times_s']} ; choisir un autre ID avec view_sumo_crdg.py.", file=sys.stderr)
    if args.crdg_gui and result["status"] != "failed":
        from cav_recovery.simulation.road_network import require_binary
        import subprocess
        for time in result["crdg_scenes"]["recorded_times_s"]:
            if time in scenes["focus_absent_times_s"]:
                continue
            subprocess.run([require_binary("sumo-gui"), "-c", str(Path(output) / "crdg_scenes" / f"{time:g}" / "scene.sumocfg"),
                            "--start", "false"], check=True)
    return {"completed": 0, "horizon_reached": 3, "failed": 1}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
