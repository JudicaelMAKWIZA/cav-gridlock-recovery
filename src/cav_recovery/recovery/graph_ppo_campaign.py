"""Collecte Graph-PPO par lots d'options, reprise à la frontière d'épisode."""

import gzip
import json
from pathlib import Path
import time
import torch

from ..laboratory.checkpoints import random_state, restore_random_state
from ..laboratory.storage import atomic_json, file_hash, json_file, single_writer
from .graph_ppo import option_transitions, ppo_update_batch
from .graph_marl_core import scenario, status_of_episode, weight_hash
from .graph_policy import local_graph
from .evaluation import compare
from .graph_marl_training import (
    _episode, _memory_ready, _prepare, _progress, _recoverable_folder, initialize,
)


def readiness_gate(root):
    """Refuse tout lancement tant que le rapport exact n'est pas positif."""
    root = Path(root)
    report_path = root / "TRAINING_READINESS.json"
    if not report_path.is_file():
        raise RuntimeError("Rapport TRAINING_READINESS.json absent.")
    report = json_file(report_path)
    if report.get("verdict") != "TRAIN_READY":
        raise RuntimeError("TRAIN_NOT_READY : entraînement long interdit.")
    for relative, expected in report["fingerprints"].items():
        target = root / relative
        if file_hash(target) != expected:
            raise RuntimeError(f"Qualification modifiée : {relative}")
    for absolute, expected in report.get("external_fingerprints", {}).items():
        if file_hash(Path(absolute)) != expected:
            raise RuntimeError(f"Source scientifique modifiée : {absolute}")
    return report


def _qualified_reference(config, row):
    qualification_root = Path(config["qualification_root"])
    folder = qualification_root / "nominal" / "train" / row["source_id"]
    record = json_file(qualification_root / "qualification.json")
    entry = next((item for item in record["scenes"]
                  if item["source_id"] == row["source_id"]
                  and item["split"] == "train"), None)
    if entry is None or not entry["status"]["valid"]:
        raise RuntimeError("Scène non qualifiée ou invalide.")
    if entry["qualified_label"] != row["qualified_label"]:
        raise RuntimeError("Label de qualification modifié.")
    if sorted(row["qualified_affected"]) != entry["affected"]:
        raise RuntimeError("Cohorte qualifiée modifiée.")
    if file_hash(folder / "physical-annotation.json") != entry["annotation_sha256"]:
        raise RuntimeError("Annotation nominale modifiée.")
    if file_hash(folder / "spec.json") != entry["spec_sha256"]:
        raise RuntimeError("Spécification nominale modifiée.")
    starts = [interval["start_s"] for interval in entry["intervals"]]
    return entry["affected"], min(starts) if starts else None, folder


def _learning_config(config):
    return dict(epochs=config["ppo_epochs"],
                minibatch_size=config["minibatch_size"],
                clip_ratio=config["clip_ratio"],
                value_coefficient=config["value_coefficient"],
                entropy_coefficient=config["entropy_coefficient"],
                gradient_norm=config["gradient_norm"])


def _bootstrap_values(model, final_views, arrivals):
    """Évalue uniquement le dernier état observable des CAV encore actifs."""
    values = {}
    with torch.no_grad():
        for ego, view in final_views.items():
            if ego in arrivals:
                continue
            nodes, adjacency = local_graph(view)
            _, value = model(torch.from_numpy(nodes), torch.from_numpy(adjacency))
            values[ego] = float(value)
    return values


def _episode_folder(root, index):
    """Ne compte jamais un fragment sans checkpoint lors d'une reprise."""
    folder = root / "training" / "episodes" / f"episode-{index:03d}"
    if not folder.exists():
        return folder
    fragments = root / "reports" / "uncheckpointed_fragments.json"
    previous = json_file(fragments) if fragments.exists() else []
    previous.append(dict(index=index, folder=str(folder.relative_to(root)),
                         reason="episode_without_checkpoint_not_counted"))
    atomic_json(fragments, previous)
    return folder.with_name(folder.name + f"-retry-{time.time_ns()}")


def _learn_from_episode(model, optimizer, config, folder, summary, decisions,
                        tracker, validity, pending):
    """Ajoute les options complètes au lot puis met à jour seulement au seuil."""
    if not validity["valid"]:
        return dict(updates=0, decisions=0, collected_transitions=0,
                    trained_transitions=0, ppo_batches=0,
                    reason="episode_invalide"), pending
    bootstrap = _bootstrap_values(model, tracker.final_views,
                                  summary["arrivals"])
    new = option_transitions(
        decisions, tracker.rewards, gamma=config["gamma_per_step"],
        gae_lambda=config["gae_lambda"], terminal=validity["all_arrived"],
        termination_times=summary["arrivals"], bootstrap_values=bootstrap,
    )
    atomic_json(folder / "ppo-collection.json", dict(
        bootstrap_values=bootstrap, transitions=new,
        rewards_count=len(tracker.rewards), neural_calls=len(decisions),
        free_choices=sum(row["admissible_count"] > 1 for row in decisions),
    ))
    pending.extend(new)
    if len(pending) < config["ppo_batch_size"]:
        return (dict(updates=0, decisions=len(new),
                     collected_transitions=len(new), trained_transitions=0,
                     ppo_batches=0,
                     reason="accumulation_on_policy", pending=len(pending)),
                pending)
    metrics = ppo_update_batch(model, optimizer, pending,
                               **_learning_config(config))
    metrics.update(collected_transitions=len(new),
                   trained_transitions=len(pending), ppo_batches=1,
                   pending=0)
    return metrics, []


def _physical_metrics(folder, summary, tracker):
    """Rend les refus et la récupération consultables pendant la campagne."""
    with gzip.open(folder / "commands.jsonl.gz", "rt", encoding="utf-8") as stream:
        guard_refusals = sum(not json.loads(line)["guard"]["accepted"]
                             for line in stream)
    recovery = json_file(folder / "reward-summary.json")["physical_recovery"]
    return dict(
        guard_refusals=guard_refusals,
        reverse_total_m=sum(summary["reverse_m"].values()),
        confirmed_passages=len(tracker.passage_attempts),
        recovery_status=None if recovery is None else recovery["status"],
        recovery_success=False if recovery is None else recovery["success"],
        recovery_time_s=None if recovery is None else recovery.get("t_conf_s"),
        affected_count=len(tracker.affected or ()),
        affected_arrivals=sum(ego in summary["arrivals"]
                              for ego in (tracker.affected or ())),
        relapse=False if recovery is None else recovery.get("relapse", False),
        domain_rejections=json_file(folder / "policy-domain-summary.json")[
            "rejected_clearance_requests"],
    )


def _periodic_validation(config, root, model, episode_index):
    """Mesure un sous-ensemble fixé, sans gradient ni mise à jour des poids.

    Les références nominales proviennent de la qualification indépendante et
    leur empreinte est revérifiée. La simulation peut consommer des générateurs
    aléatoires globaux ; leur état est restauré avant l'épisode d'entraînement
    suivant afin qu'une reprise au checkpoint conserve la même trajectoire.
    """
    source = Path(config["qualification_root"])
    identifiers = config["periodic_validation_ids"]
    rows = {row["source_id"]: row for row in config["scenes"]["validation"]}
    if len(identifiers) != len(set(identifiers)) or any(key not in rows for key in identifiers):
        raise ValueError("Sous-ensemble de validation périodique invalide.")
    before = weight_hash(model)
    started = time.monotonic()
    target_summary = (root / "validation" / "periodic" /
                      f"after-{episode_index:03d}" / "summary.json")
    for earlier in sorted((root / "validation" / "periodic").glob(
            "after-*/summary.json")):
        previous = json_file(earlier)
        if previous["weight_hash"] == before:
            atomic_json(target_summary, dict(
                episode_index=episode_index, weight_hash=before,
                outcomes=previous["outcomes"], elapsed_wall_s=0.0,
                reused_from=str(earlier.relative_to(root)),
            ))
            return previous["outcomes"]
    state = random_state()
    previous_mode = model.training
    model.eval()
    outcomes = []
    try:
        for identifier in identifiers:
            row = rows[identifier]
            reference = source / "nominal" / "validation" / identifier
            entry = next((item for item in json_file(source / "qualification.json")["scenes"]
                          if item["source_id"] == identifier
                          and item["split"] == "validation"), None)
            if (entry is None or entry["qualified_label"] != row["qualified_label"]
                    or file_hash(reference / "physical-annotation.json")
                    != entry["annotation_sha256"]):
                raise ValueError("Référence de validation modifiée.")
            base = root / "validation" / "periodic" / f"after-{episode_index:03d}" / identifier
            folder, complete = _recoverable_folder(base)
            if complete:
                policy_record = folder / "periodic-policy.json"
                if not policy_record.exists() or json_file(policy_record)["weight_hash"] != before:
                    folder = base.with_name(base.name + f"-retry-{time.time_ns()}")
                    complete = False
            if complete:
                summary = json_file(folder / "summary.json")
            else:
                summary, _, _ = _episode(scenario(row, "validation"), folder,
                                         config, model, sample=False)
                atomic_json(folder / "periodic-policy.json", dict(weight_hash=before))
            paired = compare(reference, folder,
                             base.parent / f"{identifier}-paired.json")
            outcomes.append(dict(source_id=identifier, summary=summary,
                                 validity=status_of_episode(summary), paired=paired))
    finally:
        model.train(previous_mode)
        restore_random_state(state)
    if weight_hash(model) != before:
        raise RuntimeError("La validation a modifié les poids du réseau.")
    atomic_json(target_summary,
                dict(episode_index=episode_index, weight_hash=before,
                     outcomes=outcomes, elapsed_wall_s=time.monotonic() - started))
    return outcomes


def train(config_path, root):
    """Conserve les choix et leurs anciens logits jusqu'au seuil de lot."""
    config, root = _prepare(config_path, root)
    readiness_gate(root)
    gate = json_file(root / "reports" / "preflight.json")
    if not gate["proceed"]:
        raise RuntimeError("Précontrôle neuronal non concluant.")
    with single_writer(root / "writer"):
        _, model, optimizer, store, saved, receipt = initialize(config_path, root)
        history = [] if saved is None else saved["history"]
        pending = [] if saved is None else saved.get("pending_transitions", [])
        step = 0 if saved is None else saved["global_step"]
        if saved is None:
            receipt = store.save(dict(
                global_step=0, model=model.state_dict(),
                optimizer=optimizer.state_dict(), history=[],
                pending_transitions=[], weight_hash=weight_hash(model),
                consumed_wall_s=0.0, episodes_completed=0,
                optimizer_steps=0, ppo_batches=0,
                collected_transitions=0, trained_transitions=0,
            ))
        if history and not history[-1]["validity"]["valid"]:
            raise RuntimeError("Dernier épisode invalide : reprise interdite.")
        if saved is not None:
            restore_random_state(saved["random_state"])
        started = time.monotonic()
        previously_consumed = 0.0 if saved is None else saved.get("consumed_wall_s", 0.0)
        scenes = config["scenes"]["train"]
        _progress(root, history, "running", total=len(scenes), checkpoint=receipt)
        for index in range(len(history), len(scenes)):
            if (root / "stop-requested").exists():
                _progress(root, history, "stopped_at_boundary", total=len(scenes),
                          checkpoint=receipt, pending_decisions=len(pending),
                          unused_on_policy_transitions=len(pending))
                return 86
            consumed = previously_consumed + time.monotonic() - started
            if history and consumed >= 60 * config["maximum_wall_minutes"]:
                _progress(root, history, "wall_budget_at_boundary", total=len(scenes),
                          checkpoint=receipt, pending_decisions=len(pending),
                          unused_on_policy_transitions=len(pending))
                return 86
            row = scenes[index]
            if not _memory_ready(config, row["vehicles"]):
                _progress(root, history, "memory_pressure_at_boundary",
                          total=len(scenes), next_vehicles=row["vehicles"],
                          checkpoint=receipt, pending_decisions=len(pending),
                          unused_on_policy_transitions=len(pending))
                return 86
            affected, reference_start, nominal_folder = _qualified_reference(config, row)
            folder = _episode_folder(root, index)
            before = weight_hash(model)
            summary, decisions, tracker = _episode(
                scenario(row, "train"), folder, config, model,
                updates=sum(item["learning"]["updates"] for item in history),
                record_rewards=True, affected=affected,
                reference_start_s=reference_start,
                nominal_folder=nominal_folder,
                detailed_trace=(not config.get("light_training_trace", False)
                                or index == 0),
            )
            validity = status_of_episode(summary)
            metrics, pending = _learn_from_episode(
                model, optimizer, config, folder, summary, decisions,
                tracker, validity, pending)
            completed_updates = sum(item["learning"]["updates"] for item in history)
            for offset, loss in enumerate(metrics.get("losses", []), start=1):
                loss["global_update_step"] = completed_updates + offset
                loss["episode"] = index + 1
            step += len(tracker.rewards)
            history.append(dict(
                index=index, name=row["name"], seed=row["seed"],
                qualified_label=row["qualified_label"],
                qualified_affected=affected,
                folder=str(folder.relative_to(root)), summary=summary,
                validity=validity, learning=metrics,
                neural_decisions=len(decisions),
                useful_choices=sum(item["admissible_count"] > 1 for item in decisions),
                reward=sum(tracker.rewards),
                physical=_physical_metrics(folder, summary, tracker),
                before_weight_hash=before, after_weight_hash=weight_hash(model),
            ))
            if (config.get("validation_every_episodes")
                    and (index + 1) % config["validation_every_episodes"] == 0
                    and any(item["learning"]["updates"] for item in history)
                    and previously_consumed + time.monotonic() - started
                    < 60 * config["maximum_wall_minutes"]):
                _periodic_validation(config, root, model, index + 1)
            receipt = store.save(dict(
                global_step=step, model=model.state_dict(),
                optimizer=optimizer.state_dict(), history=history,
                pending_transitions=pending, weight_hash=weight_hash(model),
                consumed_wall_s=previously_consumed + time.monotonic() - started,
                episodes_completed=len(history),
                optimizer_steps=sum(item["learning"]["updates"] for item in history),
                ppo_batches=sum(item["learning"].get("ppo_batches", 0)
                                for item in history),
                collected_transitions=sum(item["learning"].get(
                    "collected_transitions", 0) for item in history),
                trained_transitions=sum(item["learning"].get(
                    "trained_transitions", 0) for item in history)))
            atomic_json(root / "training" / "history.json", history)
            _progress(root, history, "running", total=len(scenes), checkpoint=receipt,
                      pending_decisions=len(pending),
                      collected_transitions=sum(item["learning"].get(
                          "collected_transitions", 0) for item in history),
                      trained_transitions=sum(item["learning"].get(
                          "trained_transitions", 0) for item in history),
                      optimizer_steps=sum(item["learning"]["updates"]
                                          for item in history))
            if not validity["valid"]:
                _progress(root, history, "invalid_episode_stop",
                          total=len(scenes), checkpoint=receipt,
                          problems=validity["problems"])
                return 87
        _progress(root, history, "training_complete", total=len(scenes),
                  checkpoint=receipt, pending_decisions=len(pending),
                  unused_on_policy_transitions=len(pending))
    return 0
