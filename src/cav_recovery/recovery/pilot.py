"""Petit pilote Graph-PPO, sauvegardé uniquement entre épisodes complets."""

from pathlib import Path
import hashlib
import random
import time
import numpy as np
import torch

from ..experiments.catalog import specification
from ..laboratory.checkpoints import CheckpointStore, restore_random_state
from ..laboratory.storage import (
    atomic_json,
    file_hash,
    json_file,
    protect_frozen_dataset,
    single_writer,
)
from .graph_policy import GraphActorCritic, LearnedPolicy
from .environment import run
from .cooperative import option_bank
from .contracts import RecoveryContract


def weight_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def update(model, optimizer, decisions, rewards, config):
    """PPO sur les choix d'options réels ; retours causaux après ces choix."""
    if not decisions:
        return dict(updates=0, decisions=0)
    discounted = []
    future = 0.0
    for reward in reversed(rewards):
        future = reward + config["gamma_per_physics_decision"] * future
        discounted.append(future)
    discounted = list(reversed(discounted)) + [0.0]
    returns = torch.tensor(
        [discounted[min(round(d["at_s"] / 0.2), len(rewards))] for d in decisions]
    )
    old_values = torch.tensor([d["value"] for d in decisions])
    advantages = returns - old_values
    if len(advantages) > 1 and advantages.std(unbiased=False) > 1e-6:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False)
    losses = []
    for _ in range(config["ppo_epochs"]):
        policy = []
        values = []
        entropies = []
        for i, decision in enumerate(decisions):
            logits, value = model(
                torch.tensor(decision["features"]), torch.tensor(decision["adjacency"])
            )
            mask = torch.tensor(decision["mask"])
            distribution = torch.distributions.Categorical(
                logits=logits.masked_fill(~mask, -1e9)
            )
            logprob = distribution.log_prob(torch.tensor(decision["action"]))
            ratio = torch.exp(logprob - decision["old_logprob"])
            clipped = torch.clamp(
                ratio, 1 - config["clip_ratio"], 1 + config["clip_ratio"]
            )
            policy.append(
                -torch.minimum(ratio * advantages[i], clipped * advantages[i])
            )
            values.append((value - returns[i]) ** 2)
            entropies.append(distribution.entropy())
        loss = (
            torch.stack(policy).mean()
            + config["value_coefficient"] * torch.stack(values).mean()
            - config["entropy_coefficient"] * torch.stack(entropies).mean()
        )
        if not torch.isfinite(loss):
            raise ValueError("Perte PPO non finie.")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_norm"])
        optimizer.step()
        losses.append(float(loss.detach()))
    return dict(
        updates=config["ppo_epochs"],
        decisions=len(decisions),
        losses=losses,
        policy_advantage_abs_mean=float(advantages.abs().mean()),
        return_min=float(returns.min()),
        return_max=float(returns.max()),
        reward_sum=float(sum(rewards)),
    )


def train(config_path, output, *, proof_path=None, stop_after=None):
    """Ne restaure pas un monde Bullet partiellement collecté.

    La preuve locale exigée est vérifiée avant l'entrée par le script de
    campagne ; la configuration en garde l'identité. Les réservations de test
    ne sont ni chargées ni simulées par ce pilote de développement.
    """
    if proof_path is None:
        raise ValueError(
            "Une preuve locale appariée de récupération durable est requise."
        )
    proof_path = Path(proof_path)
    proof = json_file(proof_path)
    if not proof["intervention"]["success"] or proof["nominal"]["success"]:
        raise ValueError(
            "La preuve ne démontre pas un gain local durable face au nominal."
        )
    verified_folder = proof_path.parent / "cooperative"
    verified = json_file(verified_folder / "summary.json")
    if verified.get("global_audit_in_actor") or verified["method"] != "cooperative":
        raise ValueError("Une preuve privilégiée globale ne suffit pas à ce pilote.")
    for name, sha in json_file(verified_folder / "manifest.json")["files"].items():
        if file_hash(verified_folder / name) != sha:
            raise ValueError("Preuve locale modifiée.")
    config = json_file(config_path)
    output = Path(output)
    protect_frozen_dataset(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    random.seed(config["training_seed"])
    np.random.seed(config["training_seed"])
    torch.manual_seed(config["training_seed"])
    model = GraphActorCritic(config["hidden"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    dependencies = {
        "pilot.py",
        "graph_policy.py",
        "cooperative.py",
        "contracts.py",
        "safety.py",
        "actuation.py",
        "environment.py",
        "native_diagnostics.py",
        "diagnostics.py",
        "protocol.py",
    }
    sources = {
        name: file_hash(Path(__file__).parent / name) for name in sorted(dependencies)
    }
    compatibility = dict(
        config=config,
        sources=sources,
        contract=RecoveryContract().record(),
        proof_sha256=file_hash(proof_path),
        action_count=len(option_bank()),
        engine="fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1",
    )
    store = CheckpointStore(output / "checkpoints", compatibility)
    with single_writer(output / "coordinator"):
        saved, receipt = store.load()
        history = []
        global_step = 0
        if saved is not None:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            restore_random_state(saved["random_state"])
            history = saved["history"]
            global_step = saved["global_step"]
            atomic_json(
                output / f"resume-{len(history)}.json",
                dict(
                    checkpoint=receipt,
                    loaded_weight_hash=weight_hash(model),
                    expected_weight_hash=saved["weight_hash"],
                    optimizer_states=len(optimizer.state),
                    boundary="complete_episode_no_Bullet_mid_episode_restore",
                ),
            )
            if weight_hash(model) != saved["weight_hash"]:
                raise ValueError("Poids repris différents.")
        atomic_json(output / "config.json", config)
        for episode in range(len(history), config["episodes_total"]):
            spec = specification(
                config["development_scenario"],
                seed=config["scenario_seed"],
                horizon_s=config["native_horizon_s"],
            )
            spec["agents"] = spec["agents"][: config["vehicles"]]
            spec["active_limit"] = config["vehicles"]
            spec["phase"] = "graph_pilot_known_development_not_final_evaluation"
            decisions = []
            rewards = []
            progress = {}
            previous = 0.0
            arrived = 0

            def observe_reward(instance):
                nonlocal previous, arrived
                for ego, view in instance.views.items():
                    progress[ego] = view.own["progress_m"]
                potential = sum(progress.values()) / 4.515
                reward = (
                    potential
                    - previous
                    + config["reward"]["verified_native_arrival"]
                    * (len(instance.arrivals) - arrived)
                    - config["reward"]["per_step_cost"]
                )
                if instance.errors:
                    reward += config["reward"]["invalid_episode"]
                previous = potential
                arrived = len(instance.arrivals)
                rewards.append(reward)

            def factory(ego, endpoint, contract):
                return LearnedPolicy(ego, endpoint, contract, model, decisions.append)

            before = weight_hash(model)
            folder = output / f"episodes/episode-{episode:03d}"
            if folder.exists():
                # Une coupure dans Bullet conserve son fragment ; il ne peut
                # être repris comme une transition terminale ou un succès.
                folder = (
                    output / f"episodes/episode-{episode:03d}-restart-{time.time_ns()}"
                )
            summary = run(
                spec,
                folder,
                method="graph_policy",
                measurement_contract=config["measurement_contract"],
                policy_factory=factory,
                instrumentation_hook=observe_reward,
                policy_version=dict(updates=len(history)),
            )
            atomic_json(folder / "neural-decisions.json", decisions)
            atomic_json(folder / "learning-rewards.json", rewards)
            atomic_json(
                folder / "learning-manifest.json",
                dict(
                    files={
                        name: file_hash(folder / name)
                        for name in [
                            "neural-decisions.json",
                            "learning-rewards.json",
                            "manifest.json",
                        ]
                    },
                    proof_sha256=file_hash(proof_path),
                ),
            )
            metrics = update(model, optimizer, decisions, rewards, config)
            global_step += len(rewards)
            entry = dict(
                episode=episode,
                run=str(folder.relative_to(output)),
                summary=summary,
                learning=metrics,
                before_weight_hash=before,
                after_weight_hash=weight_hash(model),
            )
            history.append(entry)
            store.save(
                dict(
                    global_step=global_step,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    history=history,
                    weight_hash=weight_hash(model),
                ),
                best=True,
            )
            atomic_json(output / "history.json", history)
            if stop_after is not None and len(history) >= stop_after:
                atomic_json(
                    output / "controlled-stop.json",
                    dict(
                        after_complete_episodes=len(history),
                        exit_code=86,
                        weight_hash=weight_hash(model),
                        Bullet_closed=True,
                    ),
                )
                return 86
        atomic_json(
            output / "complete.json",
            dict(
                episodes=len(history),
                updates=sum(h["learning"]["updates"] for h in history),
                final_weight_hash=weight_hash(model),
                learned_performance_claimed=False,
                test_evaluated=False,
            ),
        )
    return 0
