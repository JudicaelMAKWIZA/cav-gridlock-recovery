"""Contrats et calculs réutilisables du pilote Graph-MARL, sans simulation."""

import copy
import hashlib
from pathlib import Path

import torch

from ..experiments.catalog import specification
from ..dataset.contract import digest
from ..laboratory.storage import file_hash, json_file
from .cooperative import option_bank
from .contracts import RecoveryContract
from .graph_policy import GraphActorCritic


def load_config(path):
    config = json_file(path)
    versions = {
        "REC-GRAPH-MARL-P4-1": "shared_local_graph_PPO_P4_physical_option_only",
        "REC-GRAPH-PPO-P4-2": "shared_local_graph_option_PPO_P4",
    }
    if config["version"] not in versions:
        raise ValueError("Contrat Graph-MARL inconnu.")
    if config["algorithm"] != versions[config["version"]]:
        raise ValueError("Algorithme inattendu.")
    if config["maximum_wall_minutes"] <= 0 or config["cpu_threads"] < 1:
        raise ValueError("Budget de ressources invalide.")
    if (config["minimum_available_gib_before_episode"] < 2
            or config["minimum_available_gib_eight_cav"]
            < config["minimum_available_gib_before_episode"]):
        raise ValueError("Réserve mémoire insuffisante pour MetaDrive.")
    splits = config["scenes"]
    parents = {}
    for split in ("preflight", "train", "validation"):
        if not splits[split]:
            raise ValueError(f"Partition vide : {split}")
        for row in splits[split]:
            if row["vehicles"] not in range(2, 9) or row["horizon_s"] <= 0:
                raise ValueError("Effectif ou horizon hors du pilote.")
            if split != "preflight":
                parent = row.get("source_parent", row["name"])
                previous = parents.setdefault(parent, split)
                if previous != split:
                    raise ValueError("Même parent dans train et validation.")
    if config["version"] == "REC-GRAPH-PPO-P4-2":
        if config["ppo_batch_size"] < 32 or config["minibatch_size"] < 1:
            raise ValueError("Lot PPO de préparation insuffisant.")
        if not 0 <= config["gae_lambda"] <= 1:
            raise ValueError("Lambda GAE invalide.")
        if config["validation_every_episodes"] < 1:
            raise ValueError("Fréquence de validation invalide.")
        validation_ids = {row["source_id"] for row in splits["validation"]}
        periodic = config["periodic_validation_ids"]
        if (not periodic or len(periodic) != len(set(periodic))
                or not set(periodic) <= validation_ids):
            raise ValueError("Scènes de validation périodique non figées.")
    return config


def scenario(row, split):
    """Gèle la description avant toute interaction, sans lui attribuer de label."""
    if "source_spec_path" in row:
        path = Path(row["source_spec_path"])
        if file_hash(path) != row["source_spec_sha256"]:
            raise ValueError("Empreinte de la scène source incohérente.")
        spec = copy.deepcopy(json_file(path))
        if (row["seed"] != spec["seed"]
                or row["vehicles"] != spec["active_limit"]
                or row.get("source_spec_id", row["source_id"]) != spec["id"]):
            raise ValueError("Identité ou capacité M02 incohérente.")
        spec["horizon_s"] = row["horizon_s"]
    else:
        spec = specification(row["name"], seed=row["seed"],
                             horizon_s=row["horizon_s"],
                             vehicle_count=row["vehicles"])
        if len(spec["agents"]) != row["vehicles"]:
            raise ValueError("Le générateur n'a pas construit l'effectif demandé.")
    spec["phase"] = f"graph_marl_{split}_development_only"
    spec["experiment_name"] = f"graph-marl-{split}-{row['name']}-{row['seed']}"
    spec["id"] = digest(spec)[:24]
    return spec


def scene_plan(config):
    plan = []
    for split, rows in config["scenes"].items():
        for row in rows:
            spec = scenario(row, split)
            plan.append(dict(split=split, name=row["name"], seed=row["seed"],
                             vehicles=row["vehicles"], horizon_s=row["horizon_s"],
                             parent_id=spec["parent_id"], topology=spec["topology"]))
    return plan


def weight_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def compatibility(config_path, config):
    sources = Path(__file__).parent
    names = ["graph_marl_core.py", "graph_marl_policy.py", "graph_marl_training.py",
             "graph_ppo.py", "graph_ppo_campaign.py",
             "graph_policy.py", "cooperative.py", "verified_passage_policy.py",
             "passage_policy.py", "safety.py", "environment.py", "contracts.py"]
    return dict(config_sha256=file_hash(config_path),
                sources={name: file_hash(sources / name) for name in names},
                catalogue_sha256=file_hash(sources.parent / "experiments" / "catalog.py"),
                contract=RecoveryContract().record(),
                action_count=len(option_bank()),
                engine=config["engine_commit"], p4_commit=config["p4_commit"])


def make_model(config):
    torch.set_num_threads(config["cpu_threads"])
    return GraphActorCritic(config["hidden"])


def discounted_returns(decisions, rewards, gamma, step_s=0.2):
    """Retour Monte-Carlo du même pas physique, y compris choix simultanés.

    La fin de l'horizon est une troncature expérimentale ; le retour fini
    optimise cet horizon et ne certifie aucune récupération après celui-ci.
    """
    future = 0.0
    returns = [0.0] * (len(rewards) + 1)
    for index in range(len(rewards) - 1, -1, -1):
        future = float(rewards[index]) + gamma * future
        returns[index] = future
    return torch.tensor([
        returns[min(max(round(row["at_s"] / step_s), 0), len(rewards))]
        for row in decisions
    ], dtype=torch.float32)


def ppo_update(model, optimizer, decisions, rewards, config):
    """Mise à jour PPO sur les décisions réellement prises pendant l'épisode."""
    if not decisions:
        return dict(updates=0, decisions=0, reason="aucune_option_neuronale")
    returns = discounted_returns(decisions, rewards, config["gamma_per_step"])
    values_old = torch.tensor([row["value"] for row in decisions])
    advantages = returns - values_old
    if len(advantages) > 1 and advantages.std(unbiased=False) > 1e-6:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False)
    metrics = []
    for _ in range(config["ppo_epochs"]):
        policy_terms, value_terms, entropies = [], [], []
        for index, row in enumerate(decisions):
            logits, value = model(
                torch.tensor(row["features"], dtype=torch.float32),
                torch.tensor(row["adjacency"], dtype=torch.float32),
            )
            mask = torch.tensor(row["mask"], dtype=torch.bool)
            if not mask.any() or not mask[row["action"]]:
                raise ValueError("Décision hors du masque d'actions.")
            distribution = torch.distributions.Categorical(
                logits=logits.masked_fill(~mask, -1e9)
            )
            logprob = distribution.log_prob(torch.tensor(row["action"]))
            ratio = torch.exp(logprob - row["old_logprob"])
            clipped = ratio.clamp(1 - config["clip_ratio"], 1 + config["clip_ratio"])
            policy_terms.append(-torch.minimum(
                ratio * advantages[index], clipped * advantages[index]))
            value_terms.append((value - returns[index]) ** 2)
            entropies.append(distribution.entropy())
        policy_loss = torch.stack(policy_terms).mean()
        value_loss = torch.stack(value_terms).mean()
        entropy = torch.stack(entropies).mean()
        loss = (policy_loss + config["value_coefficient"] * value_loss
                - config["entropy_coefficient"] * entropy)
        if not torch.isfinite(loss):
            raise ValueError("Perte PPO non finie.")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_norm"])
        optimizer.step()
        metrics.append(dict(total=float(loss.detach()),
                            policy=float(policy_loss.detach()),
                            value=float(value_loss.detach()),
                            entropy=float(entropy.detach())))
    return dict(updates=len(metrics), decisions=len(decisions), losses=metrics,
                return_min=float(returns.min()), return_max=float(returns.max()),
                advantage_abs_mean=float(advantages.abs().mean()))


def status_of_episode(summary, contract=None):
    contract = contract or RecoveryContract()
    reverse = summary.get("reverse_m", {})
    problems = list(summary.get("errors", []))
    if any(distance > contract.reverse_budget_m + 1e-7 for distance in reverse.values()):
        problems.append("reverse_budget_exceeded")
    return dict(valid=not problems, problems=problems,
                arrived=len(summary.get("arrivals", {})),
                reverse_max_m=max(reverse.values(), default=0.0),
                physical_label=summary.get("physical_label"),
                all_arrived=len(summary.get("arrivals", {})) == summary.get("introduced", -1)
                and summary.get("pending", 1) == 0)
