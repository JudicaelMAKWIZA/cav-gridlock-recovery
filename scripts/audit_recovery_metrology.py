"""Mesure l'arrêt natif et ses oscillations, sans changer K04 ou le moteur."""

import argparse
from pathlib import Path
import math

import numpy as np
from shapely.geometry import Point

from cav_recovery.experiments.catalog import specification
from cav_recovery.dataset.native import environment, Route
from cav_recovery.metadrive.control import apply_command, DriveCommand
from cav_recovery.metadrive.observation import signed_speed
from cav_recovery.metadrive.native import configuration_record
from cav_recovery.laboratory.storage import atomic_json, protect_frozen_dataset


def measure(output, seed, mode):
    output.mkdir(parents=True, exist_ok=False)
    spec = specification("nominal-1", seed=seed)
    env = environment(spec)
    records = []
    try:
        env.reset(seed=seed)
        atomic_json(output / "configuration.json", configuration_record(env, seed))
        vehicle = env.agents["agent0"]
        route = Route(env, spec["agents"][0])
        initial = np.array(vehicle.position).copy()
        for index in range(100):
            at = index * 0.2
            action = apply_command(vehicle, DriveCommand(0.0))
            if mode == "coast":
                vehicle.explicit_brake = False
                action = [0.0, 0.0]
            elif mode == "native_full_brake":
                vehicle.explicit_brake = False
                action = [0.0, -1.0]
            records.append(
                dict(
                    at_s=at,
                    speed_mps=signed_speed(vehicle),
                    position=list(map(float, vehicle.position)),
                    route_progress_m=float(route.line.project(Point(vehicle.position))),
                    displacement_m=float(np.linalg.norm(vehicle.position - initial)),
                    native_action=action,
                )
            )
            env.step({"agent0": action})
        tail = records[20:]
        speed = np.asarray([r["speed_mps"] for r in tail])
        differences = np.diff([r["route_progress_m"] for r in tail])
        report = dict(
            seed=seed,
            mode=mode,
            samples=len(tail),
            minimum_speed_mps=float(speed.min()),
            maximum_speed_mps=float(speed.max()),
            negative_speed_fraction=float(np.mean(speed < -0.001)),
            maximum_abs_speed_mps=float(abs(speed).max()),
            max_position_diameter_m=max(
                math.dist(a["position"], b["position"]) for a in tail for b in tail
            ),
            minimum_progress_delta_m=float(differences.min()),
            position_or_velocity_setters=False,
            scope="isolated_stop_after_4s_settling_not_dense_recovery",
        )
        atomic_json(output / "records.json", records)
        atomic_json(output / "summary.json", report)
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protect_frozen_dataset(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in [21001, 21002, 21003]:
        for mode in ["wrapper_hold", "coast", "native_full_brake"]:
            results.append(measure(args.output / f"{seed}-{mode}", seed, mode))
            atomic_json(args.output / "summary.json", results)
    print("Neuf essais d’arrêt natif enregistrés.")


if __name__ == "__main__":
    main()
