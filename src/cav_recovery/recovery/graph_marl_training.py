"""Campagne Graph-MARL à lancer par l'étudiant, jamais à l'import."""

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import psutil
import torch

from ..laboratory.checkpoints import CheckpointStore, restore_random_state
from ..laboratory.storage import atomic_json, file_hash, json_file, single_writer
from .environment import run
from .evaluation import compare
from .evaluation import physical_outcome, rows as physical_rows
from .graph_marl_core import (
    compatibility, load_config, make_model, ppo_update, scenario,
    status_of_episode, weight_hash,
)
from .graph_marl_policy import P4GraphPolicy
from .passage_policy import PassagePolicy
from .verified_passage_policy import GuardEnvelopePassagePolicy


class RewardTracker:
    """Récompense hors acteur calculée après le pas natif."""

    def __init__(self, weights, policies, affected=None):
        self.weights, self.policies = weights, policies
        self.affected = None if affected is None else frozenset(affected)
        self.rewards, self.components = [], []
        self.progress, self.reverse, self.arrivals = {}, {}, set()
        self.passage_attempts = set()
        self.last_progress_at = {}
        self.error_count = 0
        self.physical_finalized = False
        self.final_views = {}

    def __call__(self, instance):
        at = instance.at_s
        self.final_views = dict(instance.views)
        part = dict(progress=0.0, passage=0.0, arrival=0.0,
                    collective_recovery=0.0, time=-self.weights["per_step_cost"],
                    reverse=0.0, idle=0.0, invalid=0.0)
        for ego, view in instance.views.items():
            if self.affected is not None and ego not in self.affected:
                continue
            progress = view.own["progress_m"]
            previous = self.progress.setdefault(ego, progress)
            length = instance.metadata["vehicles"][ego]["length_m"]
            delta = progress - previous
            part["progress"] += self.weights["progress_per_vehicle_length"] * delta / length
            self.progress[ego] = progress
            if delta > 0.02:
                self.last_progress_at[ego] = at
            elif at - self.last_progress_at.setdefault(ego, at) >= 10.0:
                part["idle"] -= self.weights["idle_after_10s_per_step"]
        for ego, used in instance.reverse_used.items():
            earlier = self.reverse.setdefault(ego, used)
            part["reverse"] -= self.weights["reverse_per_m"] * max(0.0, used - earlier)
            self.reverse[ego] = used
        new_arrivals = set(instance.arrivals) - self.arrivals
        if self.affected is not None:
            new_arrivals &= self.affected
        part["arrival"] += self.weights["arrival"] * len(new_arrivals)
        self.arrivals.update(new_arrivals)
        for policy in self.policies.values():
            for message in policy.outbox:
                if (message["kind"] == "DONE"
                    and message["payload"].get("reason") == "beneficiaries_passed"):
                    key = (message["sender"], message["attempt"])
                    if key not in self.passage_attempts:
                        self.passage_attempts.add(key)
                        part["passage"] += self.weights["confirmed_passage"]
        if len(instance.errors) > self.error_count:
            part["invalid"] += self.weights["invalid_event"] * (
                len(instance.errors) - self.error_count)
            self.error_count = len(instance.errors)
        self.components.append(part)
        total = sum(part.values())
        if not math.isfinite(total):
            raise ValueError("Récompense non finie.")
        self.rewards.append(total)

    def finalize_physical(self, folder, summary, affected, reference_start_s,
                          nominal_folder=None):
        """Ne prime un succès collectif que s'il n'est pas spontané."""
        if self.physical_finalized:
            raise RuntimeError("Récompense terminale déjà évaluée.")
        self.physical_finalized = True
        if not self.components or len(affected) < 2 or summary["errors"]:
            return None
        outcome = physical_outcome(
            list(physical_rows(folder / "reference.jsonl.gz")),
            json_file(folder / "physical-annotation.json"),
            json_file(folder / "metadata.json"), summary["arrivals"],
            set(affected), reference_start_s,
            json_file(folder / "contract.json"),
        )
        nominal_success = None
        applied_releases = None
        if nominal_folder is not None:
            nominal_folder = Path(nominal_folder)
            nominal_summary = json_file(nominal_folder / "summary.json")
            nominal = physical_outcome(
                list(physical_rows(nominal_folder / "reference.jsonl.gz")),
                json_file(nominal_folder / "physical-annotation.json"),
                json_file(nominal_folder / "metadata.json"),
                nominal_summary["arrivals"], set(affected), reference_start_s,
                json_file(nominal_folder / "contract.json"),
            )
            nominal_success = nominal["success"]
            with gzip.open(folder / "commands.jsonl.gz", "rt",
                           encoding="utf-8") as stream:
                applied_releases = sum(
                    row["guard"]["accepted"]
                    and row["applied"]["reason"] in {
                        "cooperative_release", "graph_policy_option"}
                    for row in (json.loads(line) for line in stream)
                )
        bonus_credited = (outcome["success"] and not nominal_success
                          and (nominal_folder is None or applied_releases > 0))
        bonus_at_s = None
        if bonus_credited:
            bonus = self.weights["collective_recovery"]
            bonus_at_s = outcome.get("t_conf_s")
            if bonus_at_s is None:
                bonus_at_s = summary["at_s"]
            step_s = summary["at_s"] / len(self.rewards)
            index = min(len(self.rewards) - 1,
                        max(0, round(bonus_at_s / step_s) - 1))
            self.components[index]["collective_recovery"] += bonus
            self.rewards[index] += bonus
        return dict({key: value for key, value in outcome.items()
                     if key != "timeline"},
                    nominal_success=nominal_success,
                    applied_cooperative_commands=applied_releases,
                    bonus_credited=bonus_credited,
                    bonus_at_s=bonus_at_s,
                    attribution_to_graph="not_established_without_P4_counterfactual")


def _json_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _command_evidence(folder, decisions):
    """Conserve proposition, commande gardée et motif réel au pas du choix."""
    commands = {}
    with gzip.open(folder / "commands.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            commands[(row["ego"], round(row["at_s"], 6))] = row
    result = []
    for decision in decisions:
        command = commands.get((decision["ego"], round(decision["at_s"], 6)))
        result.append(dict(
            ego=decision["ego"], at_s=decision["at_s"],
            neural_action=decision["action"], selected=decision["chosen"],
            proposed=None if command is None else command["proposed"],
            applied=None if command is None else command["applied"],
            guard=None if command is None else command["guard"],
            note="Le choix peut initier une négociation V2V avant sa commande physique."
        ))
    _json_lines(folder / "neural-versus-command.jsonl", result)


def _recoverable_folder(base):
    """Réutilise un résultat complet ou conserve un fragment puis repart à neuf."""
    candidates = [base] + sorted(base.parent.glob(base.name + "-retry-*"))
    complete = next((folder for folder in reversed(candidates)
                     if (folder / "summary.json").is_file()
                     and (folder / "graph-marl-manifest.json").is_file()), None)
    if complete is not None:
        manifest = json_file(complete / "graph-marl-manifest.json")
        for name, expected in manifest["files"].items():
            if file_hash(complete / name) != expected:
                raise ValueError(f"Trace de validation modifiée : {complete / name}")
        return complete, True
    if any(folder.exists() for folder in candidates):
        return base.with_name(base.name + f"-retry-{time.time_ns()}"), False
    return base, False


def _episode(spec, folder, config, model=None, *, method="graph_policy", updates=0,
             sample=True, record_rewards=False, affected=None,
             reference_start_s=None, nominal_folder=None, detailed_trace=False):
    if folder.exists():
        raise FileExistsError(f"Épisode déjà présent : {folder}")
    decisions, policies = [], {}
    tracker = (RewardTracker(config["reward"], policies, affected=affected)
               if record_rewards else None)

    def factory(ego, endpoint, contract):
        cls = P4GraphPolicy if method == "graph_policy" else (
            PassagePolicy if method == "b3" else GuardEnvelopePassagePolicy)
        policy = (cls(ego, endpoint, contract, model, decisions.append, sample=sample)
                  if cls is P4GraphPolicy else cls(ego, endpoint, contract))
        policies[ego] = policy
        return policy

    actual_method = ("nominal" if method == "nominal" else
                     "graph_policy" if method == "graph_policy" else "causal_audit")
    summary = run(
        spec, folder, method=actual_method,
        measurement_contract=config["measurement_contract"],
        policy_factory=None if method == "nominal" else factory,
        instrumentation_hook=tracker,
        record_diagnostic_inputs=(detailed_trace or
                                  not config.get("light_training_trace", False)),
        policy_version=dict(
            updates=updates,
            source=("Graph-PPO-P4" if config["version"].startswith("REC-GRAPH-PPO-P4-")
                    else "Graph-MARL-P4"))
        if method == "graph_policy" else None,
    )
    if method == "graph_policy":
        _json_lines(folder / "neural-decisions.jsonl", decisions)
        _command_evidence(folder, decisions)
    if tracker is not None:
        outcome = None
        if affected and reference_start_s is not None:
            outcome = tracker.finalize_physical(
                folder, summary, affected, reference_start_s,
                nominal_folder=nominal_folder)
        _json_lines(folder / "reward-components.jsonl", tracker.components)
        atomic_json(folder / "reward-summary.json", dict(
            total=sum(tracker.rewards),
            by_component={key: sum(row[key] for row in tracker.components)
                          for key in tracker.components[0]} if tracker.components else {},
            confirmed_passage_attempts=len(tracker.passage_attempts),
            physical_recovery=outcome,
        ))
    rejected = {ego: policy.request_domain_rejections
                for ego, policy in policies.items()
                if getattr(policy, "request_domain_rejections", 0)}
    atomic_json(folder / "policy-domain-summary.json", dict(
        rejected_clearance_requests=sum(rejected.values()),
        by_vehicle=rejected,
        examples={ego: policy.request_domain_examples
                  for ego, policy in policies.items()
                  if getattr(policy, "request_domain_rejections", 0)},
        reason="requested_space_outside_REC_V2V_domain",
    ))
    atomic_json(folder / "graph-marl-manifest.json", dict(files={
        path.name: file_hash(path) for path in folder.iterdir() if path.is_file()
    }))
    return summary, decisions, tracker


def _prepare(config_path, root):
    config_path, root = Path(config_path).resolve(), Path(root).resolve()
    config = load_config(config_path)
    if root.drive.upper() != "D:":
        raise ValueError("Les résultats doivent être enregistrés sur D:.")
    if file_hash(root / "experiment-config.json") != file_hash(config_path):
        raise ValueError("La configuration de l'expérience a changé.")
    manifest = json_file(root / "source-manifest.json")
    project = Path(__file__).resolve().parents[3]
    for name, expected in manifest["files"].items():
        if name.endswith(".py") and file_hash(project / name) != expected:
            raise ValueError(f"Source modifiée après préparation : {name}")
    return config, root


def _memory_ready(config, vehicles):
    threshold = (config["minimum_available_gib_eight_cav"] if vehicles >= 8
                 else config["minimum_available_gib_before_episode"])
    return psutil.virtual_memory().available / 1024**3 >= threshold


def initialize(config_path, root):
    """Prépare uniquement des poids aléatoires et une identité ; aucun épisode."""
    config, root = _prepare(config_path, root)
    torch.set_num_threads(config["cpu_threads"])
    random.seed(config["training_seed"])
    np.random.seed(config["training_seed"])
    torch.manual_seed(config["training_seed"])
    model = make_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    store = CheckpointStore(root / "checkpoints", compatibility(config_path, config))
    saved, receipt = store.load()
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if weight_hash(model) != saved["weight_hash"]:
            raise ValueError("Empreinte des poids repris incohérente.")
    return config, model, optimizer, store, saved, receipt


def _preflight(config_path, root):
    """Un seul épisode de contrôle : appel exclusivement dans la cellule 08."""
    config, root = _prepare(config_path, root)
    report = root / "reports" / "preflight.json"
    if report.exists():
        return json_file(report)
    _, model, _, _, saved, _ = initialize(config_path, root)
    if saved is not None:
        raise ValueError("Le contrôle préalable doit précéder l'entraînement.")
    row = config["scenes"]["preflight"][0]
    if not _memory_ready(config, row["vehicles"]):
        raise RuntimeError("Mémoire disponible insuffisante pour le précontrôle.")
    folder = root / "preflight" / "episode-000"
    complete = next((candidate for candidate in sorted(folder.parent.glob("episode-000*"))
                     if all((candidate / name).exists() for name in
                            ("summary.json", "reward-summary.json",
                             "neural-decisions.jsonl", "reward-components.jsonl"))),
                    None)
    if complete is not None:
        folder = complete
        summary = json_file(folder / "summary.json")
        decisions = [json.loads(line) for line in
                     (folder / "neural-decisions.jsonl").read_text(encoding="utf-8").splitlines()]
        reward_sum = json_file(folder / "reward-summary.json")["total"]
        reward_steps = (folder / "reward-components.jsonl").read_text(
            encoding="utf-8").splitlines()
    else:
        if folder.exists():
            folder = folder.with_name(folder.name + f"-retry-{time.time_ns()}")
        affected, reference_start, nominal_folder = None, None, None
        if config["version"].startswith("REC-GRAPH-PPO-P4-"):
            from .graph_ppo_campaign import _qualified_reference
            affected, reference_start, nominal_folder = _qualified_reference(config, row)
        summary, decisions, tracker = _episode(
            scenario(row, "preflight"), folder, config, model,
            record_rewards=True, affected=affected,
            reference_start_s=reference_start,
            nominal_folder=nominal_folder,
            detailed_trace=True,
        )
        reward_sum = sum(tracker.rewards)
        reward_steps = tracker.rewards
    validity = status_of_episode(summary)
    available_choices = sum(row["admissible_count"] > 1 for row in decisions)
    with gzip.open(folder / "commands.jsonl.gz", "rt", encoding="utf-8") as stream:
        applied_options = sum(
            json.loads(line)["applied"]["reason"] == "cooperative_release"
            for line in stream
        )
    validity.update(neural_decisions=len(decisions),
                    neural_non_hold=sum(row["action"] != 0 for row in decisions),
                    choices_with_physical_alternative=available_choices,
                    applied_cooperative_commands=applied_options,
                    reward_sum=reward_sum, folder=str(folder.relative_to(root)),
                    proceed=validity["valid"] and bool(decisions)
                    and bool(available_choices) and bool(reward_steps))
    atomic_json(report, validity)
    return validity


def preflight(config_path, root):
    with single_writer(Path(root) / "writer"):
        return _preflight(config_path, root)


def _progress(root, history, state, **extra):
    completed = len(history)
    durations = [row["summary"]["wall_s"] for row in history]
    atomic_json(root / "reports" / "experiment_status.json", dict(
        state=state, pid=os.getpid(), updated_at=time.time(),
        available_memory_gib=round(psutil.virtual_memory().available / 1024**3, 3),
        completed_episodes=completed, episode_wall_mean_s=(sum(durations) / completed
                                                           if completed else None),
        estimated_remaining_s=(None if not completed else
                               (extra.get("total", completed) - completed)
                               * sum(durations) / completed),
        **extra,
    ))


def train(config_path, root):
    """Reprend uniquement à une frontière d'épisode complet vérifiée."""
    config, root = _prepare(config_path, root)
    if config["version"].startswith("REC-GRAPH-PPO-P4-"):
        from .graph_ppo_campaign import train as train_graph_ppo
        return train_graph_ppo(config_path, root)
    gate = json_file(root / "reports" / "preflight.json")
    if not gate["proceed"]:
        raise ValueError("Le précontrôle physique ou neuronal a échoué.")
    with single_writer(root / "writer"):
        _, model, optimizer, store, saved, receipt = initialize(config_path, root)
        history = [] if saved is None else saved["history"]
        step = 0 if saved is None else saved["global_step"]
        if history and not history[-1]["validity"]["valid"]:
            raise ValueError("Dernier épisode invalide : contrôle scientifique requis.")
        if saved is not None:
            restore_random_state(saved["random_state"])
        started = time.monotonic()
        total = len(config["scenes"]["train"])
        _progress(root, history, "running", total=total, checkpoint=receipt)
        for index in range(len(history), total):
            if (root / "stop-requested").exists():
                _progress(root, history, "stopped_at_boundary", total=total)
                return 86
            if history and time.monotonic() - started >= 60 * config["maximum_wall_minutes"]:
                _progress(root, history, "wall_budget_at_boundary", total=total)
                return 86
            row = config["scenes"]["train"][index]
            if not _memory_ready(config, row["vehicles"]):
                threshold = (config["minimum_available_gib_eight_cav"]
                             if row["vehicles"] >= 8 else
                             config["minimum_available_gib_before_episode"])
                _progress(root, history, "memory_pressure_at_boundary",
                          total=total, required_memory_gib=threshold,
                          next_vehicles=row["vehicles"])
                return 86
            folder = root / "training" / "episodes" / f"episode-{index:03d}"
            if folder.exists():
                ignored = root / "reports" / "uncheckpointed_fragments.json"
                old = json_file(ignored) if ignored.exists() else []
                old.append(dict(index=index, folder=str(folder.relative_to(root)),
                                reason="episode_without_checkpoint_not_counted"))
                atomic_json(ignored, old)
                folder = folder.with_name(folder.name + f"-retry-{time.time_ns()}")
            before = weight_hash(model)
            _progress(root, history, "running", total=total, current_episode=index)
            summary, decisions, tracker = _episode(
                scenario(row, "train"), folder, config, model,
                updates=sum(item["learning"]["updates"] for item in history),
                record_rewards=True,
            )
            validity = status_of_episode(summary)
            metrics = (ppo_update(model, optimizer, decisions, tracker.rewards, config)
                       if validity["valid"] else
                       dict(updates=0, decisions=len(decisions), reason="episode_invalide"))
            step += len(tracker.rewards)
            entry = dict(index=index, name=row["name"], seed=row["seed"],
                         folder=str(folder.relative_to(root)), summary=summary,
                         validity=validity, learning=metrics,
                         reward=sum(tracker.rewards),
                         before_weight_hash=before,
                         after_weight_hash=weight_hash(model))
            history.append(entry)
            receipt = store.save(dict(
                global_step=step, model=model.state_dict(),
                optimizer=optimizer.state_dict(), history=history,
                weight_hash=weight_hash(model)))
            atomic_json(root / "training" / "history.json", history)
            _progress(root, history, "running", total=total, checkpoint=receipt)
            if not validity["valid"]:
                _progress(root, history, "invalid_episode_stop", total=total,
                          checkpoint=receipt, problems=validity["problems"])
                return 87
        _progress(root, history, "training_complete", total=total,
                  checkpoint=receipt)
    return 0


def _validate(config_path, root):
    """Évalue sans gradient les checkpoints terminaux sur des parents distincts."""
    config, root = _prepare(config_path, root)
    if config["version"] == "REC-GRAPH-PPO-P4-3":
        from .graph_ppo_validation import validate as validate_graph_ppo
        return validate_graph_ppo(config_path, root)
    training_status = json_file(root / "reports" / "experiment_status.json")
    allowed_states = {"training_complete"}
    if config["version"].startswith("REC-GRAPH-PPO-P4-"):
        allowed_states.add("wall_budget_at_boundary")
    if training_status.get("state") not in allowed_states:
        raise ValueError("L'entraînement prévu n'est pas terminé.")
    _, model, _, store, saved, receipt = initialize(config_path, root)
    if saved is None or not saved["history"]:
        raise ValueError("Aucun checkpoint entraîné à valider.")
    if not any(item["learning"]["updates"] for item in saved["history"]):
        raise ValueError("Aucune mise à jour neuronale dans les checkpoints.")
    stop = root / "stop-requested"
    if stop.exists():
        raise ValueError("Une demande d'arrêt est présente ; reprendre explicitement.")
    # Chaque historique est une génération immuable ; la sélection emploie
    # uniquement les scènes de validation, sans résultat de test réservé.
    receipts = []
    for path in sorted((root / "checkpoints").glob("checkpoint-*.json")):
        record = json_file(path)
        receipts.append(record)
    candidates = receipts[-min(2, len(receipts)):]
    nominal_folders = {}
    for index, scene_row in enumerate(config["scenes"]["validation"]):
        if stop.exists():
            atomic_json(root / "reports" / "validation_status.json",
                        dict(state="stopped_at_scene_boundary"))
            return []
        if not _memory_ready(config, scene_row["vehicles"]):
            raise RuntimeError("Mémoire insuffisante avant la référence nominale.")
        atomic_json(root / "reports" / "validation_status.json",
                    dict(state="running", phase="nominal_reference", scene=index,
                         checkpoint=None))
        if config["version"].startswith("REC-GRAPH-PPO-P4-"):
            qualified = next((item for item in json_file(
                Path(config["qualification_root"]) / "qualification.json")["scenes"]
                if item["split"] == "validation"
                and item["source_id"] == scene_row["source_id"]), None)
            folder = (Path(config["qualification_root"]) / "nominal" /
                      "validation" / scene_row["source_id"])
            if (qualified is None or not qualified["status"]["valid"]
                    or file_hash(folder / "physical-annotation.json")
                    != qualified["annotation_sha256"]):
                raise ValueError("Référence nominale de validation invalide.")
        else:
            base = root / "validation" / "baselines" / f"scene-{index:02d}" / "nominal"
            folder, completed = _recoverable_folder(base)
            if not completed:
                _episode(scenario(scene_row, "validation"), folder, config,
                         method="nominal")
        nominal_folders[index] = folder
    scores = []
    for candidate in candidates:
        path = root / "checkpoints" / candidate["file"]
        if file_hash(path) != candidate["sha256"]:
            raise ValueError("Empreinte du checkpoint de validation incorrecte.")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["compatibility"] != store.compatibility:
            raise ValueError("Checkpoint incompatible avec la configuration.")
        if not any(entry["learning"]["updates"] for entry in state["history"]):
            continue
        model.load_state_dict(state["model"])
        model.eval()
        rows = []
        for index, scene_row in enumerate(config["scenes"]["validation"]):
            if stop.exists():
                atomic_json(root / "reports" / "validation_status.json",
                            dict(state="stopped_at_scene_boundary", checkpoint=candidate))
                return scores
            if not _memory_ready(config, scene_row["vehicles"]):
                raise RuntimeError("Mémoire insuffisante avant l'épisode de validation.")
            atomic_json(root / "reports" / "validation_status.json",
                        dict(state="running", phase="candidate", scene=index,
                             checkpoint=candidate))
            spec = scenario(scene_row, "validation")
            base = root / "validation" / "episodes" / candidate["file"].replace(".pt", "") / f"scene-{index:02d}"
            folder, completed = _recoverable_folder(base)
            if completed:
                summary = json_file(folder / "summary.json")
            else:
                summary, _, _ = _episode(spec, folder, config, model,
                                         updates=sum(e["learning"]["updates"] for e in state["history"]),
                                         sample=False)
            paired_path = (root / "validation" / "comparisons" /
                           f"{path.stem}-scene-{index:02d}.json")
            paired = compare(nominal_folders[index], folder, paired_path)
            rows.append(dict(scene=scene_row, folder=str(folder.relative_to(root)),
                             summary=summary,
                             validity=status_of_episode(summary), paired=paired))
        score_gains = sum(row["paired"]["improvement_not_spontaneous"]
                          for row in rows if row["validity"]["valid"])
        score_arrivals = sum(row["validity"]["arrived"]
                             for row in rows if row["validity"]["valid"])
        scores.append(dict(checkpoint=candidate, score_recovery_gains=score_gains,
                           score_arrivals=score_arrivals, scenes=rows))
        atomic_json(root / "validation" / "summaries" / (path.stem + ".json"), scores[-1])
    selected = max(scores, key=lambda row: (row["score_recovery_gains"],
                                            row["score_arrivals"],
                                            -sum(bool(not x["validity"]["valid"])
                                                 for x in row["scenes"])))
    atomic_json(root / "checkpoints" / "selected.json", selected["checkpoint"])
    selected_path = root / "checkpoints" / selected["checkpoint"]["file"]
    selected_state = torch.load(selected_path, map_location="cpu", weights_only=False)
    model.load_state_dict(selected_state["model"])
    comparisons = []
    for index, scene_row in enumerate(config["scenes"]["validation"]):
        if stop.exists():
            atomic_json(root / "reports" / "validation_status.json",
                        dict(state="stopped_at_scene_boundary"))
            return scores
        if not _memory_ready(config, scene_row["vehicles"]):
            raise RuntimeError("Mémoire insuffisante avant la scène comparative.")
        atomic_json(root / "reports" / "validation_status.json",
                    dict(state="running", phase="baseline_comparison", scene=index,
                         checkpoint=selected["checkpoint"]))
        spec = scenario(scene_row, "validation")
        folders = {}
        for method in ("b3", "p4"):
            if stop.exists():
                atomic_json(root / "reports" / "validation_status.json",
                            dict(state="stopped_at_scene_boundary"))
                return scores
            base = root / "validation" / "baselines" / f"scene-{index:02d}" / method
            folder, completed = _recoverable_folder(base)
            if not completed:
                _episode(spec, folder, config, method=method)
            folders[method] = folder
        folders["nominal"] = nominal_folders[index]
        graph_folder = root / selected["scenes"][index]["folder"]
        neural_name = ("graph_ppo" if config["version"].startswith("REC-GRAPH-PPO-P4-")
                       else "graph_marl")
        for method, folder in [("b3", folders["b3"]),
                               ("p4", folders["p4"]),
                               (neural_name, graph_folder)]:
            comparison = compare(
                folders["nominal"], folder,
                root / "validation" / "comparisons" /
                f"scene-{index:02d}-{method}.json",
            )
            comparisons.append(dict(
                scene=scene_row, method=method, paired=comparison,
                arrivals_total=len(json_file(folder / "summary.json")["arrivals"]),
                arrivals_affected=sum(
                    ego in json_file(folder / "summary.json")["arrivals"]
                    for ego in comparison["intervention"]["affected"]),
            ))
    atomic_json(root / "reports" / "validation_summary.json", dict(
        selected=selected["checkpoint"], scores=scores,
        comparisons=comparisons,
        claim="Développement seulement ; aucun test réservé ni généralisation."))
    atomic_json(root / "reports" / "validation_status.json",
                dict(state="complete", selected=selected["checkpoint"]))
    return scores


def validate(config_path, root, *, full=False):
    with single_writer(Path(root) / "writer"):
        if full:
            config, root = _prepare(config_path, root)
            if config["version"] != "REC-GRAPH-PPO-P4-3":
                raise ValueError("Extension complète réservée à Graph-PPO P4-3.")
            from .graph_ppo_validation import validate as validate_graph_ppo
            return validate_graph_ppo(config_path, root, full=True)
        return _validate(config_path, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["preflight", "train", "validate",
                                         "validate_full"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.task == "preflight":
        print(preflight(args.config, args.output), flush=True)
    elif args.task == "train":
        print("Code de sortie :", train(args.config, args.output), flush=True)
    else:
        print(validate(args.config, args.output,
                       full=args.task == "validate_full"), flush=True)


if __name__ == "__main__":
    main()
