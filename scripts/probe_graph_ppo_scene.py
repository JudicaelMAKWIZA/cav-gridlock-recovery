"""Sonde courte d'influence neuronale sur une scène requalifiée."""

import argparse
from pathlib import Path
import time

import torch  # Avant MetaDrive sous Windows.

from cav_recovery.laboratory.storage import atomic_json, json_file
from cav_recovery.recovery.graph_marl_core import make_model, scenario, status_of_episode
from cav_recovery.recovery.graph_marl_training import _episode
from cav_recovery.recovery.graph_ppo_campaign import _bootstrap_values
from cav_recovery.recovery.graph_ppo import option_transitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("graph_policy", "p4"),
                        default="graph_policy")
    args = parser.parse_args()
    shortlist = json_file(args.shortlist)["splits"]
    pair = next(((split, row) for split, rows in shortlist.items() for row in rows
                 if row["source_id"] == args.source_id), None)
    if pair is None:
        raise ValueError("Scène hors de la présélection.")
    split, row = pair
    qualified = next((entry for entry in json_file(args.qualification)["scenes"]
                      if entry["source_id"] == args.source_id), None)
    if qualified is None or qualified["qualified_label"] != "POSITIVE_CONFIRMED":
        raise ValueError("La scène doit être positive requalifiée.")
    config = json_file(Path(__file__).parents[1] / "configs/recovery/graph-marl-p4-v1.json")
    config["reward"].update(progress_per_vehicle_length=0.2,
                            confirmed_passage=1.0, arrival=2.0,
                            collective_recovery=10.0,
                            per_step_cost=0.002, reverse_per_m=0.2,
                            idle_after_10s_per_step=0.001,
                            invalid_event=-100.0)
    torch.manual_seed(231023)
    model = make_model(config) if args.method == "graph_policy" else None
    start = min(interval["start_s"] for interval in qualified["intervals"])
    started = time.perf_counter()
    summary, decisions, tracker = _episode(
        scenario(row, split), args.output, config, model, method=args.method,
        record_rewards=args.method == "graph_policy", affected=qualified["affected"],
        reference_start_s=start,
        nominal_folder=args.qualification.parent / "nominal" / split / row["source_id"],
    )
    if tracker is not None:
        bootstrap = _bootstrap_values(model, tracker.final_views, summary["arrivals"])
        transitions = option_transitions(
            decisions, tracker.rewards, gamma=config["gamma_per_step"],
            gae_lambda=0.995,
            terminal=len(summary["arrivals"]) == summary["introduced"]
            and summary["pending"] == 0,
            termination_times=summary["arrivals"],
            bootstrap_values=bootstrap,
        )
        atomic_json(args.output / "ppo-collection.json", dict(
            bootstrap_values=bootstrap, transitions=transitions,
            rewards_count=len(tracker.rewards),
        ))
    else:
        transitions = []
    result = dict(source_id=row["source_id"], summary=summary,
                  validity=status_of_episode(summary),
                  decisions=len(decisions),
                  useful_choices=sum(item["admissible_count"] > 1 for item in decisions),
                  non_hold=sum(item["action"] != 0 for item in decisions),
                  useful_transitions=len(transitions),
                  reward_total=sum(tracker.rewards) if tracker else None,
                  wall_s=time.perf_counter() - started)
    atomic_json(args.output.parent / "probe-summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
