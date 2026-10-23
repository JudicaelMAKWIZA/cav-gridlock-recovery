"""Vérifie une petite mise à jour PPO sur des décisions physiques conservées."""

import argparse
import json
from pathlib import Path
import time

import torch  # Ordre d'import Windows.

from cav_recovery.laboratory.checkpoints import CheckpointStore
from cav_recovery.laboratory.storage import atomic_json, file_hash, json_file
from cav_recovery.recovery.graph_ppo import option_transitions, ppo_update_batch
from cav_recovery.recovery.graph_marl_core import make_model, weight_hash


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json_file(Path(__file__).parents[1] / "configs/recovery/graph-marl-p4-v1.json")
    decisions_path = args.probe / "neural-decisions.jsonl"
    rewards_path = args.probe / "reward-components.jsonl"
    decisions = lines(decisions_path)
    rewards = [sum(row.values()) for row in lines(rewards_path)]
    summary = json_file(args.probe / "summary.json")
    collection_path = args.probe / "ppo-collection.json"
    collection = json_file(collection_path) if collection_path.exists() else {}
    torch.manual_seed(231023)
    model = make_model(config)
    before = weight_hash(model)
    transitions = option_transitions(
        decisions, rewards, gamma=config["gamma_per_step"],
        gae_lambda=0.995,
        terminal=len(summary["arrivals"]) == summary["introduced"]
        and summary["pending"] == 0,
        termination_times=summary["arrivals"],
        bootstrap_values=collection.get("bootstrap_values", {}),
    )
    if collection and len(collection["transitions"]) != len(transitions):
        raise ValueError("Collecte PPO et recalcul non concordants.")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    started = time.perf_counter()
    metrics = ppo_update_batch(
        model, optimizer, transitions, epochs=2, minibatch_size=2,
        clip_ratio=config["clip_ratio"],
        value_coefficient=config["value_coefficient"],
        entropy_coefficient=config["entropy_coefficient"],
        gradient_norm=config["gradient_norm"],
    )
    duration = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    compatibility = dict(decisions_sha256=file_hash(decisions_path),
                         rewards_sha256=file_hash(rewards_path))
    store = CheckpointStore(args.output / "mini-checkpoints", compatibility)
    receipt = store.save(dict(global_step=len(rewards),
                              model=model.state_dict(),
                              optimizer=optimizer.state_dict(),
                              history=[metrics], weight_hash=weight_hash(model)))
    restored, _ = store.load()
    if restored["weight_hash"] != weight_hash(model):
        raise ValueError("Reprise du mini-checkpoint incohérente.")
    report = dict(physical_decisions=len(decisions),
                  useful_transitions=len(transitions),
                  censored=not (len(summary["arrivals"]) == summary["introduced"]),
                  before_weight_hash=before, after_weight_hash=weight_hash(model),
                  weights_changed=before != weight_hash(model),
                  update=metrics, update_wall_s=duration,
                  optimizer_states=len(optimizer.state), checkpoint=receipt,
                  checkpoint_loaded=True, source=compatibility)
    atomic_json(args.output / "mini-ppo-report.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
