"""Sélection préenregistrée et validation de développement de Graph-PPO.

Les deux scènes périodiques choisissent les poids. Les autres scènes du lot
prioritaire ne participent jamais à cette sélection.
"""

from pathlib import Path

import torch

from ..laboratory.storage import atomic_json, file_hash, json_file
from .evaluation import compare
from .graph_marl_core import scenario, status_of_episode, weight_hash
from .graph_marl_training import _episode, _memory_ready, _recoverable_folder, initialize


def _receipts_by_weight(root, store):
    """Associe une seule génération vérifiée à chaque ensemble de poids."""
    receipts = {}
    for path in sorted((root / "checkpoints").glob("checkpoint-*.json")):
        receipt = json_file(path)
        target = root / "checkpoints" / receipt["file"]
        if file_hash(target) != receipt["sha256"]:
            raise ValueError(f"Checkpoint altéré : {target.name}")
        state = torch.load(target, map_location="cpu", weights_only=False)
        if state["compatibility"] != store.compatibility:
            raise ValueError("Checkpoint incompatible.")
        if not any(row["learning"]["updates"] for row in state["history"]):
            continue
        receipts.setdefault(state["weight_hash"], (receipt, state))
    return receipts


def select_checkpoint(root, store, *, expected=None):
    """Choisit sur validation périodique, puis fige la décision et sa preuve."""
    receipts = _receipts_by_weight(root, store)
    if not receipts:
        raise ValueError("Aucune mise à jour PPO vérifiée.")
    observations = []
    evaluated_weights = set()
    for path in sorted((root / "validation" / "periodic").glob("after-*/summary.json")):
        result = json_file(path)
        fingerprint = result["weight_hash"]
        if fingerprint not in receipts:
            continue
        outcomes = result["outcomes"]
        if (fingerprint in evaluated_weights or not outcomes
                or not all(row["validity"]["valid"] for row in outcomes)):
            continue
        evaluated_weights.add(fingerprint)
        gains = sum(bool(row["paired"]["improvement_not_spontaneous"])
                    for row in outcomes)
        arrivals = sum(row["validity"]["arrived"] for row in outcomes)
        observations.append(dict(
            weight_hash=fingerprint, receipt=receipts[fingerprint][0],
            gains=gains, arrivals=arrivals, episode=result["episode_index"],
            source=str(path.relative_to(root)),
        ))
    if observations:
        # À score égal, l'évaluation la plus ancienne est retenue.
        chosen = max(observations, key=lambda row: (
            row["gains"], row["arrivals"], -row["episode"]))
        rule = "periodic_gains_then_arrivals_then_earliest"
    else:
        # Un budget écourté peut ne contenir aucune validation périodique.
        # Cette règle de secours est déclarée avant l'entraînement.
        chosen_receipt, chosen_state = max(
            receipts.values(), key=lambda item: len(item[1]["history"]))
        chosen = dict(weight_hash=chosen_state["weight_hash"],
                      receipt=chosen_receipt, episode=len(chosen_state["history"]),
                      gains=None, arrivals=None, source="latest_trained_fallback")
        rule = "latest_trained_when_no_periodic_measurement"
    record = dict(rule=rule, chosen=chosen, observations=observations,
                  distinct_trained_weights=len(receipts))
    if expected is not None and expected != chosen["receipt"]:
        raise ValueError("La sélection figée a changé avant l'extension.")
    atomic_json(root / "checkpoints" / "selection.json", record)
    atomic_json(root / "checkpoints" / "selected.json", chosen["receipt"])
    return chosen["receipt"], record


def _nominal_reference(config, row):
    source = Path(config["qualification_root"])
    reference = source / "nominal" / "validation" / row["source_id"]
    entry = next((item for item in json_file(source / "qualification.json")["scenes"]
                  if item["split"] == "validation"
                  and item["source_id"] == row["source_id"]), None)
    if (entry is None or not entry["status"]["valid"]
            or entry["qualified_label"] != row["qualified_label"]
            or file_hash(reference / "physical-annotation.json")
            != entry["annotation_sha256"]):
        raise ValueError("Référence nominale de validation modifiée.")
    return reference


def _evaluate_method(config, root, row, index, method, model, reference, updates):
    base = root / "validation" / "episodes" / row["source_id"] / method
    folder, complete = _recoverable_folder(base)
    if not complete:
        summary, _, _ = _episode(
            scenario(row, "validation"), folder, config, model,
            method="graph_policy" if method == "graph_ppo" else method,
            sample=False, updates=updates)
    else:
        summary = json_file(folder / "summary.json")
    paired = compare(reference, folder, root / "validation" / "comparisons" /
                     f"{row['source_id']}-{method}.json")
    return dict(source_id=row["source_id"], scene=row, scene_index=index,
                method=method,
                folder=str(folder.relative_to(root)), summary=summary,
                validity=status_of_episode(summary), paired=paired,
                arrivals_affected=sum(ego in summary["arrivals"]
                                      for ego in paired["intervention"]["affected"]))


def validate(config_path, root, *, full=False):
    """Évalue le lot prioritaire figé ou étend ensuite aux dix scènes."""
    root = Path(root)
    config, model, _, store, _, _ = initialize(config_path, root)
    status = json_file(root / "reports" / "experiment_status.json")
    if status["state"] not in {"training_complete", "wall_budget_at_boundary",
                               "closed_at_boundary"}:
        raise ValueError("Campagne non clôturée pour évaluation.")
    if (root / "stop-requested").exists():
        raise ValueError("Demande d'arrêt non résolue.")
    prior_selection = (json_file(root / "checkpoints" / "selected.json")
                       if full and (root / "checkpoints" / "selected.json").exists()
                       else None)
    receipt, selection = select_checkpoint(root, store, expected=prior_selection)
    state = torch.load(root / "checkpoints" / receipt["file"],
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    if weight_hash(model) != selection["chosen"]["weight_hash"]:
        raise ValueError("Poids sélectionnés incohérents.")
    rows = config["scenes"]["validation"]
    chosen_ids = ([row["source_id"] for row in rows] if full else
                  config["priority_validation_ids"])
    previous = root / "reports" / "validation_priority_summary.json"
    if full and not previous.exists():
        raise ValueError("Exécuter d'abord la validation prioritaire.")
    target = root / "reports" / ("validation_full_summary.json" if full else
                                 "validation_priority_summary.json")
    if target.exists():
        return json_file(target)
    outcomes = []
    updates = sum(item["learning"]["updates"] for item in state["history"])
    for index, row in enumerate(rows):
        if row["source_id"] not in chosen_ids:
            continue
        if (root / "stop-requested").exists():
            atomic_json(root / "reports" / "validation_status.json",
                        dict(state="stopped_at_scene_boundary", scene=index))
            return outcomes
        if not _memory_ready(config, row["vehicles"]):
            raise RuntimeError("Mémoire insuffisante pour la validation.")
        reference = _nominal_reference(config, row)
        for method in ("graph_ppo", "b3", "p4"):
            if (root / "stop-requested").exists():
                atomic_json(root / "reports" / "validation_status.json",
                            dict(state="stopped_at_method_boundary", scene=index,
                                 next_method=method))
                return outcomes
            atomic_json(root / "reports" / "validation_status.json",
                        dict(state="running", full=full, scene=index,
                             method=method, checkpoint=receipt))
            outcomes.append(_evaluate_method(config, root, row, index, method,
                                             model, reference, updates))
        atomic_json(root / "reports" / "validation_progress.json",
                    dict(full=full, completed=outcomes))
    result = dict(scope="development_only", full=full, scene_ids=chosen_ids,
                  selected=receipt, selection=selection, comparisons=outcomes,
                  claim="Aucune validation sur le test réservé.")
    atomic_json(target, result)
    if not full:
        atomic_json(root / "reports" / "validation_summary.json", result)
    atomic_json(root / "reports" / "validation_status.json",
                dict(state="complete", full=full, selected=receipt))
    return result
