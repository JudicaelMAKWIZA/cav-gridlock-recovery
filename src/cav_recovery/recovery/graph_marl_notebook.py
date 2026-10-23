"""Commandes légères du notebook ; aucune simulation lors de l'import."""

import json
import gzip
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from ..laboratory.storage import atomic_json, file_hash, json_file
from .graph_marl_core import load_config, scene_plan


def environment_check(project, root):
    """Diagnostic statique ; l'import du moteur ne crée aucun épisode."""
    import importlib.metadata as metadata
    import torch
    import ipykernel
    import matplotlib
    import metadrive

    project, root = Path(project).resolve(), Path(root).resolve()
    if not (project / "src/cav_recovery/recovery/graph_marl_training.py").is_file():
        raise FileNotFoundError("Le worktree Graph-MARL attendu est introuvable.")
    if not (root / "experiment-config.json").is_file():
        raise FileNotFoundError("La configuration privée est absente.")
    if Path(sys.executable).resolve() != Path(
        "D:/cav-recovery-venvs/metadrive/Scripts/python.exe").resolve():
        raise RuntimeError("Sélectionner le noyau Python MetaDrive situé sur D:.")
    disk = shutil.disk_usage(root.anchor)
    if disk.free < 5 * 1024**3:
        raise RuntimeError("Moins de 5 Gio libres sur le volume de sortie.")
    check = subprocess.run([sys.executable, "-m", "pip", "check"],
                           capture_output=True, text=True, timeout=30)
    if check.returncode:
        raise RuntimeError("Dépendances incohérentes : " + check.stdout + check.stderr)
    return dict(
        python=sys.version.split()[0], interpreter=sys.executable,
        torch=torch.__version__, metadrive=metadata.version("metadrive-simulator"),
        ipykernel=ipykernel.__version__, matplotlib=matplotlib.__version__,
        cpu_logical=psutil.cpu_count(), memory_available_gib=round(
            psutil.virtual_memory().available / 1024**3, 2),
        output_free_gib=round(disk.free / 1024**3, 2),
        project=str(project), output=str(root),
        metadrive_imported=bool(metadrive), pip_check="OK",
    )


def _live(record):
    try:
        process = psutil.Process(record["pid"])
        return abs(process.create_time() - record["created"]) < 0.1
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def process_status(root):
    root = Path(root)
    process_file = root / "reports" / "process.json"
    process = json_file(process_file) if process_file.exists() else None
    experiment_file = root / "reports" / "experiment_status.json"
    experiment = json_file(experiment_file) if experiment_file.exists() else None
    writer_file = root / "writer" / "writer.lock"
    worker = json_file(writer_file) if writer_file.exists() else None
    worker_alive = _live(worker) if worker else False
    terminal = bool(process and experiment
                    and experiment.get("updated_at", 0) >= process["started_at"]
                    and experiment.get("state") in {
                        "training_complete", "memory_pressure_at_boundary",
                        "wall_budget_at_boundary", "stopped_at_boundary",
                        "invalid_episode_stop", "interrupted_at_episode",
                    })
    starting = bool(process and not terminal and not worker_alive
                    and time.time() - process["started_at"] < 120)
    alive = worker_alive or starting
    interrupted = bool(process and experiment and not alive
                       and experiment.get("updated_at", 0) >= process["started_at"]
                       and experiment.get("state") == "running")
    return dict(process=process, worker=worker,
                worker_pid=worker["pid"] if worker_alive else None,
                alive=alive, interrupted=interrupted,
                effective_state=("interrupted_with_error" if interrupted else
                                 experiment.get("state") if experiment else None),
                experiment=experiment,
                preflight=json_file(root / "reports" / "preflight.json")
                if (root / "reports" / "preflight.json").exists() else None,
                validation=json_file(root / "reports" / "validation_status.json")
                if (root / "reports" / "validation_status.json").exists() else None)


def launch(task, project, root):
    """Démarre au plus un producteur externe ; le noyau reste interactif."""
    if task not in {"train", "validate"}:
        raise ValueError("Tâche inconnue.")
    project, root = Path(project).resolve(), Path(root).resolve()
    status = process_status(root)
    if status["alive"]:
        pid = status["worker_pid"] or status["process"]["pid"]
        raise RuntimeError(f"Une tâche est déjà active : PID {pid}.")
    config_path = root / "experiment-config.json"
    config = load_config(config_path)
    if config["version"] == "REC-GRAPH-PPO-P4-2":
        from .graph_ppo_campaign import readiness_gate
        readiness_gate(root)
    if task == "train":
        gate = status["preflight"]
        if not gate or not gate.get("proceed"):
            raise RuntimeError("Exécuter d'abord la cellule 08 et résoudre ses alertes.")
        if status["experiment"] and status["experiment"].get("state") == "training_complete":
            raise RuntimeError("L'entraînement est déjà complet ; aucune relance nécessaire.")
    elif not (root / "checkpoints" / "latest.json").exists():
        raise RuntimeError("Aucun checkpoint entraîné pour la validation.")
    elif (not status["experiment"] or status["experiment"].get("state") not in
          ({"training_complete", "wall_budget_at_boundary"}
           if config["version"] == "REC-GRAPH-PPO-P4-2" else {"training_complete"})):
        raise RuntimeError("Terminer l'entraînement avant la validation.")
    elif (root / "reports" / "validation_summary.json").exists():
        raise RuntimeError("La validation est déjà complète ; consulter ses résultats.")
    (root / "stop-requested").unlink(missing_ok=True)
    folder = root / "training" / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / f"{task}-{time.time_ns()}.log"
    command = [sys.executable, "-m", "cav_recovery.recovery.graph_marl_training",
               task, "--config", str(config_path), "--output", str(root)]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(project / "src"),
               PYTHONPYCACHEPREFIX="D:/cav-recovery-cache/graph-marl-pycache",
               PIP_CACHE_DIR="D:/cav-recovery-cache/pip",
               TEMP="D:/cav-recovery-temp", TMP="D:/cav-recovery-temp",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=project, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=flags)
    record = dict(task=task, pid=process.pid,
                  created=psutil.Process(process.pid).create_time(),
                  launcher_pid=process.pid,
                  started_at=time.time(), log=str(log),
                  config_sha256=file_hash(config_path))
    atomic_json(root / "reports" / "process.json", record)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = process_status(root)
        if state["worker_pid"] is not None:
            record["pid"] = state["worker_pid"]
            record["created"] = state["worker"]["created"]
            atomic_json(root / "reports" / "process.json", record)
            return record
        if state["experiment"] and state["experiment"].get("updated_at", 0) >= record["started_at"]:
            name = state["experiment"].get("state")
            if name == "memory_pressure_at_boundary":
                raise RuntimeError(
                    "Entraînement arrêté avant l'épisode : mémoire libre "
                    f"{state['experiment'].get('available_memory_gib', 'inconnue')} Gio ; "
                    f"seuil {state['experiment'].get('required_memory_gib', 'inconnu')} Gio. "
                    "Aucun producteur ne reste actif."
                )
            if name in {"training_complete", "stopped_at_boundary",
                        "wall_budget_at_boundary", "invalid_episode_stop"}:
                return record
        time.sleep(0.1)
    return record


def request_stop(root):
    status = process_status(root)
    if not status["alive"]:
        return "Aucun processus actif."
    (Path(root) / "stop-requested").write_text(
        f"Demande volontaire à {time.time()}; arrêt à la frontière d'épisode.\n",
        encoding="utf-8")
    return "Arrêt demandé. L'épisode en cours se termine avant le checkpoint."


def recent_history(root):
    path = Path(root) / "training" / "history.json"
    return json_file(path) if path.exists() else []


def render_figures(root):
    """Figures issues exclusivement des mesures écrites après l'expérience."""
    import matplotlib.pyplot as plt

    root = Path(root)
    history = recent_history(root)
    if not history:
        raise RuntimeError("Aucun épisode terminé : aucune figure à fabriquer.")
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    xs = [row["index"] + 1 for row in history]
    reward = [row["reward"] for row in history]
    arrival = [row["validity"]["arrived"] for row in history]
    valid = [row["validity"]["valid"] for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    axes[0].plot(xs, reward, "o-")
    axes[0].set(ylabel="Récompense", title="Apprentissage : épisodes observés")
    axes[1].bar(xs, arrival, color=["#287a61" if row else "#b74949" for row in valid])
    axes[1].set(xlabel="Épisode", ylabel="Arrivées physiques")
    target = folder / "learning_curve.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    outputs = {"learning_curve": str(target)}
    table = root / "training" / "metrics.csv"
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "episode", "scene", "seed", "reward", "arrivals", "valid",
            "decisions", "ppo_updates", "wall_s", "cpu_s", "reverse_max_m",
            "reverse_total_m", "guard_refusals", "domain_rejections",
            "recovery_status", "recovery_success", "confirmed_passages"])
        writer.writeheader()
        for row in history:
            writer.writerow(dict(
                episode=row["index"] + 1, scene=row["name"], seed=row["seed"],
                reward=row["reward"], arrivals=row["validity"]["arrived"],
                valid=row["validity"]["valid"],
                decisions=row["learning"]["decisions"],
                ppo_updates=row["learning"]["updates"],
                wall_s=row["summary"]["wall_s"],
                cpu_s=row["summary"]["cpu_s"],
                reverse_max_m=row["validity"]["reverse_max_m"],
                reverse_total_m=row.get("physical", {}).get("reverse_total_m"),
                guard_refusals=row.get("physical", {}).get("guard_refusals"),
                domain_rejections=row.get("physical", {}).get("domain_rejections"),
                recovery_status=row.get("physical", {}).get("recovery_status"),
                recovery_success=row.get("physical", {}).get("recovery_success"),
                confirmed_passages=row.get("physical", {}).get("confirmed_passages")))
    outputs["training_table"] = str(table)
    components = []
    for row in history:
        receipt = root / row["folder"] / "reward-summary.json"
        components.append(json_file(receipt)["by_component"] if receipt.exists() else {})
    keys = ["progress", "passage", "arrival", "collective_recovery",
            "time", "reverse", "idle", "invalid"]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for key in keys:
        values = [row.get(key, 0.0) for row in components]
        ax.plot(xs, values, marker="o", label=key)
    ax.set(xlabel="Épisode", ylabel="Contribution réelle",
           title="Composantes de la récompense observée")
    ax.legend(ncol=4)
    target = folder / "reward_components.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    outputs["reward_components"] = str(target)
    updates = [(row["index"] + 1, loss) for row in history
               for loss in row["learning"].get("losses", [])]
    if updates:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for key in ("policy", "value", "total", "entropy"):
            ax.plot([episode for episode, _ in updates],
                    [loss[key] for _, loss in updates], ".", label=key)
        ax.set(xlabel="Épisode", ylabel="Valeur mesurée",
               title="Pertes et entropie à chaque mise à jour PPO")
        ax.legend()
        target = folder / "ppo_losses.png"
        fig.savefig(target, dpi=220)
        plt.close(fig)
        outputs["ppo_losses"] = str(target)
    if any("physical" in row for row in history):
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.step(xs, [int(row.get("physical", {}).get("recovery_success", False))
                     for row in history], where="mid", label="reprise durable")
        ax.step(xs, [int(row["qualified_label"] == "POSITIVE_CONFIRMED")
                     for row in history], where="mid", alpha=0.5,
                label="obstruction nominale qualifiée")
        ax.set(xlabel="Épisode", ylabel="0 ou 1",
               title="Obstruction et récupération physique observées", ylim=(-0.1, 1.1))
        ax.legend()
        target = folder / "physical_recovery.png"
        fig.savefig(target, dpi=220)
        plt.close(fig)
        outputs["physical_recovery"] = str(target)
    validation = root / "reports" / "validation_summary.json"
    if validation.exists():
        comparisons = json_file(validation)["comparisons"]
        table = root / "validation" / "comparisons.csv"
        table.parent.mkdir(parents=True, exist_ok=True)
        with table.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "scene", "seed", "method", "incident_qualified",
                "intervention_success", "nominal_success", "gain_non_spontaneous"])
            writer.writeheader()
            for row in comparisons:
                paired = row["paired"]
                writer.writerow(dict(
                    scene=row["scene"]["name"], seed=row["scene"]["seed"],
                    method=row["method"],
                    incident_qualified=bool(paired["intervention"]["affected"]),
                    intervention_success=paired["intervention"]["success"],
                    nominal_success=paired["nominal"]["success"],
                    gain_non_spontaneous=paired["improvement_not_spontaneous"]))
        outputs["comparison_table"] = str(table)
        neural_name = ("graph_ppo" if any(row["method"] == "graph_ppo"
                                       for row in comparisons) else "graph_marl")
        methods = ["b3", "p4", neural_name]
        gains = [sum(row["paired"]["improvement_not_spontaneous"]
                     for row in comparisons if row["method"] == method)
                 for method in methods]
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.bar(methods, gains)
        ax.set(ylabel="Scènes avec gain REC-EVAL-1",
               title="Validation de développement appariée")
        target = folder / "recovery_success.png"
        fig.savefig(target, dpi=220)
        plt.close(fig)
        outputs["recovery_success"] = str(target)
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        scene_names = sorted({row["scene"]["name"] for row in comparisons})
        for index, method in enumerate(methods):
            values = [next((row["arrivals_affected"]
                            for row in comparisons
                            if row["method"] == method
                            and row["scene"]["name"] == scene), 0)
                      for scene in scene_names]
            ax.bar([j + (index - 1) * 0.25 for j in range(len(scene_names))],
                   values, width=0.25, label=method)
        ax.set_xticks(range(len(scene_names)), scene_names, rotation=20)
        ax.set(ylabel="Arrivées de la cohorte affectée", title="Comparaison appariée")
        ax.legend()
        target = folder / "baseline_comparison.png"
        fig.savefig(target, dpi=220)
        plt.close(fig)
        outputs["baseline_comparison"] = str(target)
    return outputs


def write_review(root, runs, target):
    """Ajoute décisions et messages horodatés au lecteur physique existant."""
    from .inspection import ConstructiveStudy

    root, target = Path(root), Path(target)
    ConstructiveStudy(root).write_player(runs, target)
    events = {}
    for run in runs:
        folder = root / run
        records = []
        packets = folder / "messages.jsonl.gz"
        if packets.exists():
            with gzip.open(packets, "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    message = row.get("message", {})
                    if row.get("event") == "SENT" and message.get("kind") != "STATE":
                        records.append(dict(at_s=row["at_s"],
                                            description=(message["sender"] + " → "
                                                         + str(message.get("target"))
                                                         + " : " + message["kind"]
                                                         + " / " + message["attempt"])))
        decisions = folder / "neural-decisions.jsonl"
        if decisions.exists():
            for line in decisions.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                records.append(dict(at_s=row["at_s"],
                                    description=(row["ego"] + " : option neuronale "
                                                 + str(row["action"]) + " "
                                                 + str(row["chosen"]))))
        commands = folder / "commands.jsonl.gz"
        if commands.exists():
            with gzip.open(commands, "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if not row["guard"]["accepted"]:
                        records.append(dict(at_s=row["at_s"],
                                            description=(row["ego"] + " : refus garde "
                                                         + row["guard"]["reason"])))
        events[run] = sorted(records, key=lambda row: row["at_s"])
    html = target.read_text(encoding="utf-8")
    payload = json.dumps(events, ensure_ascii=False).replace("<", "\\u003c")
    html = html.replace("function draw(){", (
        "const graphEvents=" + payload + ";\n"
        "function eventInfo(name,at){let a=graphEvents[name]||[];"
        "return a.filter(x=>Math.abs(x.at_s-at)<0.11).slice(0,12)"
        ".map(x=>'\\n'+x.at_s.toFixed(1)+' s : '+x.description).join('');}\n"
        "function draw(){"))
    html = html.replace("' | Échelle : segment de 10 m';}",
                        "' | Échelle : segment de 10 m'+eventInfo(r.name,f.at_s);}")
    target.write_text(html, encoding="utf-8")
    return target


def final_manifest(root):
    """Inventaire de l'existant, sans annoncer de réussite scientifique."""
    root = Path(root)
    files = {}
    for name in ("experiment-config.json", "experiment-identity.json",
                 "environment.json", "source-manifest.json", "README_EXPERIENCE.md"):
        path = root / name
        if path.exists():
            files[name] = file_hash(path)
    for folder_name in ("checkpoints", "training", "validation", "figures", "reports"):
        folder = root / folder_name
        if folder.exists():
            for path in folder.rglob("*"):
                if path.is_file() and path.name != "writer.lock":
                    files[path.relative_to(root).as_posix()] = file_hash(path)
    result = dict(files=files, created_at=time.time(),
                  completed_episodes=len(recent_history(root)),
                  validation_present=(root / "reports/validation_summary.json").exists(),
                  scientific_success_claimed=False)
    atomic_json(root / "final" / "artifact_manifest.json", result)
    return result
