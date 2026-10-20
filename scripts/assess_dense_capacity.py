"""Estime les ressources depuis les traces v4 existantes, sans lancer de simulation."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from cav_recovery.dataset.pipeline import records, write_json, code_hashes


def scale_cpu(baseline, actors, horizon_s=110.0):
    ratio = actors / 30
    return dict(
        lower_s=baseline * ratio * horizon_s / 110,
        upper_s=baseline * ratio**2 * horizon_s / 110,
        assumption="linear to quadratic workload scaling from 30; scenario envelope, not a measured confidence interval",
    )


def approach_slots(length, arms, lanes, *, vehicle_length=4.515, gap=2.0, spawn=4.0):
    return (
        arms
        * lanes
        * max(
            0,
            1
            + math.floor(
                (length - vehicle_length / 2 - spawn) / (vehicle_length + gap)
            ),
        )
    )


def assess(root):
    root = Path(root)
    traffic = json.loads((root / "traffic-distributions.json").read_text())
    cpu = traffic["costs"]["30"]["normalized_cpu_110s_median"]
    nbytes = traffic["costs"]["30"]["mean_bytes"]
    current = []
    for folder, row in records(root):
        spec = json.loads((folder / "spec.json").read_text())
        t = spec["topology"]
        current.append(
            dict(
                family=spec["family"],
                arms=t["arms"],
                lanes=t["lane_num"],
                approach_length_m=t["approach_length_m"],
                approximate_wholly_on_approach_storage=approach_slots(
                    t["approach_length_m"], t["arms"], t["lane_num"]
                ),
            )
        )
    observed = traffic["groups"]["ALL|ALL"]["episode_peaks"]
    horizons = {
        str(n): {str(h): scale_cpu(cpu, n, h) for h in (110, 220, 330)}
        for n in (80, 100)
    }
    runtime = Path("outputs/dense-intersection/runtime-verification-20261019.json")
    snapshot = json.loads(runtime.read_text()) if runtime.exists() else None
    hardware_file = root.parent / "hardware.json"
    hardware = (
        json.loads(hardware_file.read_text(encoding="utf-8-sig"))
        if hardware_file.exists()
        else {}
    )
    if os.name == "nt":
        import psutil

        hardware.update(
            runtime_platform="Windows",
            guest_logical_cpus=os.cpu_count(),
            native_ram_mib=psutil.virtual_memory().total / 1024**2,
            swap_mib=psutil.swap_memory().total / 1024**2,
        )
    else:
        memory = {
            line.split(":")[0]: int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
        }
        hardware.update(
            guest_logical_cpus=os.cpu_count(),
            wsl_ram_mib=memory["MemTotal"] / 1024,
            swap_mib=memory["SwapTotal"] / 1024,
        )
    source = Path("src/cav_recovery")
    evidence = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            source / "dataset/native.py",
            source / "dataset/generation.py",
            source / "communication/v2v.py",
            Path("scripts/check_dataset.py"),
        )
    }
    result = dict(
        no_new_campaign=True,
        no_generator_or_contract_changes=True,
        assessed_at_unix_s=time.time(),
        measured_maximum_active=observed["active"]["maximum"],
        measured_maximum_central=observed["central"]["maximum"],
        measured_maximum_affected=observed["affected_confirmed"]["maximum"],
        measured_maximum_affected_central=observed["affected_central"]["maximum"],
        measured_maximum_affected_approaches=observed["affected_approach"]["maximum"],
        integrity_failure_cases=traffic.get("integrity_failures", []),
        normalized_30_cav_cpu_s_per_110_simulated_s=cpu,
        cpu_scenarios=horizons,
        cpu_scope="native simulate() measurement: setup/physics/control/oracle/trace streams/initial qualification; excludes independent deep QC, later exports and gallery",
        baseline_median_mean_active=traffic["costs"]["30"]["mean_active_median"],
        scaling_assumption="comparable fill/drain profiles scaled by active cap; not 100 actors present from t=0 or an assertion that all reach the cap",
        known_hardware=hardware,
        current_approach_storage=dict(
            minimum=min(v["approximate_wholly_on_approach_storage"] for v in current),
            maximum=max(v["approximate_wholly_on_approach_storage"] for v in current),
            assumptions="stationary 4.515m bodies, 2m gap, no body straddling central boundary, all incoming lanes used; upper layout estimate, not validated reachable demand",
        ),
        proposed_approach_storage=dict(
            four_arms_three_lanes_60m=approach_slots(60, 4, 3),
            four_arms_three_lanes_100m=approach_slots(100, 4, 3),
        ),
        memory=dict(
            measured_worker_highwater_mib_by_cap={
                k: dict(
                    minimum=v["minimum_worker_highwater_mib"],
                    maximum=v["maximum_worker_highwater_mib"],
                )
                for k, v in traffic["costs"].items()
            },
            interpretation="getrusage ru_maxrss is a cumulative process high-water mark across many episodes, not isolated per-episode memory",
            runtime_snapshot=runtime.name if snapshot else None,
            unmeasured_planning_budget_gib_per_100_cav_worker=[1.5, 2.5],
            budget_status="engineering reservation to test later; not a measured estimate or guarantee",
            suggested_initial_workers=1,
            second_worker="only after measuring stable USS/RSS and coordinator/annotation memory under 8GiB",
            required="bound native caches and worker lifetime; bound in-simulation oracle frames and stream QC manifests/references rather than retaining the full campaign",
        ),
        modifications=[
            "Future generator version only: add 80/100 caps and suitable total missions; current hardcoded cap list ends at 30.",
            "Reach the active count through demand and trip residence time, not a cap alone. Current 0.5-1.3s dense headways and 110s horizon cannot guarantee 100 simultaneously moving actors.",
            "Use longer physically valid approaches (indicatively 60-100m), balanced actual lane use and occupancy-checked per-approach admission queues instead of one global FIFO.",
            "Allow roughly 220-330s horizons for 120-150 staggered missions plus sufficient observation; calibrate in future development, never on the v4 independent test.",
            "Replace all-pair Python scans by conservative spatial broad phases, preserving all actual collisions, required-space witnesses, 30m sensing and safety margins.",
            "Share immutable route/footprint templates; bound growing native engine caches and recycle workers between completed episodes.",
            "Native simulate currently retains every 5Hz oracle frame until qualify(). Before longer 100-CAV runs, use bounded temporal qualification state with complete witnesses preserved on disk; prove equivalence to the frozen physical criteria.",
            "Stream annotation/QC indices; avoid duplicating identical required-pose lists for each blocker. Keep local observations separate from global oracle.",
            "Hash large trace/export files incrementally instead of Path.read_bytes(); future 100-CAV partition archives could exceed WSL RAM. Consider bounded shards with indexed manifests while preserving causal ownership and parent partitions.",
            "Revalidate steering, braking, birth clearance, local V2V/sensor load and hard dense negatives. Never weaken safety to obtain 20-50 constrained vehicles.",
            "Use per-episode lazy gallery payloads and zoom while preserving metre-to-pixel proportions; a single embedded 16-episode payload grows strongly at 100 CAV and longer horizons.",
        ],
        feasibility=dict(
            active_80_100="technically plausible with a revised and requalified future configuration, not demonstrated by v4",
            affected_20_50_total="plausible through several physically constrained queues on approaches plus central interactions; not guaranteed by active count or density",
            affected_20_50_central="not demonstrated; 50 in the shared centre needs a separate reachable-geometry and safety validation, not artificial vehicle scaling",
            distinction="approach storage is not central occupancy; marginal peaks from different times cannot be added",
        ),
        compressed_trace_scenarios_mib_for_100_220s=[
            nbytes * (100 / 30) * 2 / 1024**2,
            nbytes * (100 / 30) ** 2 * 2 / 1024**2,
        ],
        source_evidence_sha256=evidence,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_code_unchanged=code_hashes()
        == json.loads((root / "campaign.json").read_text())["code"],
    )
    write_json(root / "capacity-assessment.json", result)
    lines = [
        "# Extension 80–100 CAV — étude sans nouvelle campagne",
        "",
        "Aucun simulateur lancé, aucun générateur/contrat v4 modifié. Les valeurs extrapolées sont des scénarios techniques, pas des performances mesurées à 80–100 CAV.",
        "",
        f"Base mesurée : médiane normalisée de **{cpu:.1f} s CPU** pour 110 s simulées au plafond 30 CAV.",
        "",
        "La mesure couvre la simulation native et sa qualification initiale ; les contrôles approfondis, les exports et la galerie ajoutent un coût séparé. Les profils de remplissage/vidange sont supposés comparables ; ce ne sont pas des mesures de 100 acteurs présents dès le départ.",
        "",
        "| Plafond actif visé, profils de remplissage comparables | Horizon simulé | CPU par épisode, enveloppe linéaire–quadratique |",
        "|---:|---:|---:|",
    ]
    for n in (80, 100):
        for h in (110, 220, 330):
            d = horizons[str(n)][str(h)]
            lines.append(
                f"| {n} | {h} s | {d['lower_s'] / 60:.1f}–{d['upper_s'] / 60:.1f} min CPU |"
            )
    lines += [
        "",
        "La CPU par épisode ne se divise pas en augmentant le nombre de workers ; le parallélisme améliore le débit de plusieurs épisodes sous réserve de RAM. Les scans de voisins/paires, les empreintes balayées, le V2V, le LiDAR et les annotations expliquent l’enveloppe de coût.",
        "",
        f"Stockage statique théorique sur les approches actuelles : {result['current_approach_storage']['minimum']}–{result['current_approach_storage']['maximum']} voitures, selon la géométrie, avec un interstice de 2 m. Cette borne suppose toutes les voies entrantes utilisées et ne démontre ni leur admission ni un blocage.",
        f"Quatre approches, trois voies entrantes et 60–100 m donneraient environ {approach_slots(60, 4, 3)}–{approach_slots(100, 4, 3)} places entièrement sur les approches. Le carrefour reste unique.",
        "",
        "20–50 CAV contraints au total pourraient se répartir entre plusieurs files et le centre. Cela ne signifie pas 20–50 voitures au centre. Les maximums v4 observés sont conservés dans le JSON ; atteindre 50 centraux n’est pas démontré.",
        f"La v4 contient {len(traffic.get('integrity_failures', []))} épisode(s) d’intégrité invalide, conservés dans les inconnus. L’augmentation de population ne constitue pas une garantie de sécurité ; le contrôle nominal devra être requalifié sans désactiver ses protections.",
        "",
        f"RAM : les pics ru_maxrss v4 sont cumulatifs pour des processus ayant exécuté des centaines d’épisodes. Une réserve de planification de 1,5–2,5 Gio par worker à 100 CAV est à vérifier, pas une mesure ni une borne supérieure. Le simulateur conserve actuellement tous les cadres oracle à 5 Hz jusqu’à la qualification : leur croissance avec la durée/population doit aussi être bornée dans une version future. Sous {hardware.get('native_ram_mib', hardware.get('wsl_ram_mib', 0)):.0f} Mio du système mesuré, commencer ultérieurement avec un worker ; envisager deux après mesure USS/RSS stable et contrôle de la RAM du QC. Quatre workers à 100 CAV ne sont pas justifiés par les données actuelles.",
        "",
        "Modifications estimées :",
        "",
    ]
    lines += ["- " + m for m in result["modifications"]]
    (root / "capacity-assessment.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    r = assess(args.dataset)
    print(
        json.dumps(
            dict(
                no_new_campaign=True,
                cpu_scenarios=r["cpu_scenarios"],
                feasibility=r["feasibility"],
            ),
            indent=2,
        )
    )
    raise SystemExit(0 if r["frozen_code_unchanged"] else 1)
