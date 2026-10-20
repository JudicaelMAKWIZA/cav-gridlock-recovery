"""Commandes globales de faisabilité déclarées, séparées des preuves physiques."""

from dataclasses import asdict
import math

import numpy as np
from shapely.ops import unary_union

from .control import (
    DriveCommand,
    apply_command,
    guard_blockers,
    path_poses,
    stopping_distance,
)
from .observation import body, rectangle, signed_speed

BAY_TARGET_Y_M = 2.7


class FeasibilityController:
    def __init__(self, case, policy, surface):
        self.case, self.policy, self.surface = case, policy, surface
        self.priority = None
        self.stage = "approach"

    @staticmethod
    def bay_tracking(vehicle, start_x, length, reverse, height=BAY_TARGET_Y_M):
        x, y = vehicle.position
        raw_u = (start_x - x) / length if reverse else (x - start_x) / length
        u = max(0.0, min(1.0, raw_u))
        target_y = height * (
            (3 * u * u - 2 * u * u * u) if reverse else (1 - 3 * u * u + 2 * u * u * u)
        )
        slope = -6 * height * u * (1 - u) / length
        second = (6 * height * (1 - 2 * u) / length**2) * (1 if reverse else -1)
        curvature = second / (1 + slope * slope) ** 1.5 if 0 <= raw_u <= 1 else 0.0
        feedforward = math.atan(2.7 * curvature) / math.radians(
            vehicle.config["max_steering"]
        )
        angle = math.atan(slope)
        error = math.atan2(
            math.sin(angle - vehicle.heading_theta),
            math.cos(angle - vehicle.heading_theta),
        )
        steering = (
            feedforward + (-1.8 if reverse else 1.8) * error - 0.4 * (y - target_y)
        )
        return float(np.clip(steering, -0.6, 0.6))

    def passage_command(self, key, env, time_s):
        vehicle = env.agents[key]
        if self.policy == "road_following" or time_s < 15:
            return 1.2, 0.0, "assigned_shared_strip_forward"
        if self.stage == "approach":
            self.stage = "retreat"
        west = env.agents.get("agent0")
        if (
            self.stage == "retreat"
            and west is not None
            and west.position[0] < 20.5
            and abs(signed_speed(west)) < 0.12
        ):
            self.stage = "bay"
        if self.stage == "retreat":
            if key == "agent0":
                return (
                    (-1.0 if vehicle.position[0] > 20.0 else 0.0),
                    0.0,
                    "global_reverse_to_bay_entry",
                )
            return 0.0, 0.0, "global_hold_for_retreat"
        if self.stage == "bay":
            if key == "agent0":
                x, y = vehicle.position
                steering = self.bay_tracking(vehicle, 20.0, 12.5, True)
                if x < 8.2:
                    if (
                        abs(signed_speed(vehicle)) < 0.12
                        and body(vehicle).bounds[1] > 1.7
                        and abs(vehicle.heading_theta) < 0.15
                    ):
                        self.stage = "pass"
                    return 0.0, 0.0, "global_brake_in_bay"
                return -0.6, steering, "global_reverse_steering_into_bay"
            return 0.0, 0.0, "global_hold_for_bay"
        if self.stage == "pass":
            if key == "agent0":
                if "agent1" not in env.agents:
                    self.stage = "rejoin"
                    self.rejoin_start = tuple(map(float, vehicle.position))
                return 0.0, 0.0, "global_wait_for_native_opposite_arrival"
            return 1.2, 0.0, "global_opposite_passage"
        if self.stage == "rejoin":
            x, y = vehicle.position
            steering = self.bay_tracking(
                vehicle, self.rejoin_start[0], 12.0, False, self.rejoin_start[1]
            )
            if x > 20 and abs(y) < 0.3 and abs(vehicle.heading_theta) < 0.15:
                self.stage = "finish"
            return 0.6, steering, "global_forward_rejoin"
        return (
            1.2,
            float(
                np.clip(
                    -2 * vehicle.heading_theta - 0.8 * vehicle.position[1], -0.5, 0.5
                )
            ),
            "global_finish_original_mission",
        )

    def local_guard(self, key, env, target, steering):
        """Vérifie un balayage cinématique court, complété par les contrôles physiques."""
        vehicle = env.agents[key]
        distance = max(0.35, stopping_distance(math.hypot(*vehicle.velocity)) + 0.2)
        sign = -1 if target < 0 else 1
        curvature = (
            math.tan(steering * math.radians(vehicle.config["max_steering"])) / 2.7
        )
        poses = []
        x, y = vehicle.position
        heading = vehicle.heading_theta
        for s in np.linspace(0, sign * distance, max(3, int(distance / 0.05) + 1)):
            angle = heading + curvature * s
            if abs(curvature) < 1e-8:
                px, py = x + s * math.cos(heading), y + s * math.sin(heading)
            else:
                px = x + (math.sin(angle) - math.sin(heading)) / curvature
                py = y - (math.cos(angle) - math.cos(heading)) / curvature
            poses.append(rectangle((px, py), angle, vehicle.LENGTH, vehicle.WIDTH))
        sweep = unary_union(poses)
        blockers = [
            other
            for other, v in env.agents.items()
            if other != key
            and sweep.distance(body(v))
            < 0.7 + stopping_distance(math.hypot(*v.velocity))
        ]
        return sorted(blockers), self.surface.covers(sweep.buffer(0.15))

    def commands(self, env, followers, time_s):
        ids = sorted(env.agents)
        if self.priority not in ids:
            self.priority = ids[0] if ids else None
        actions, desired, records = {}, {}, {}
        for key, vehicle in sorted(env.agents.items()):
            if self.case == "shared_passage":
                speed, steering, reason = self.passage_command(key, env, time_s)
                blockers, complete = self.local_guard(key, env, speed, steering)
            else:
                follower = followers[key]
                follower.move_to_next_road()
                target_lane = follower.routing_target_lane
                if self.case == "lane_merge":
                    # Cet essai historique de changement de voie utilise un voisinage global déclaré.
                    _, _, target_lane = follower.lane_change_policy(
                        list(env.agents.values())
                    )
                speed, reason = 1.5, "native_route_following_no_symmetric_courtesy"
                if self.policy == "global_clearance" and key != self.priority:
                    speed, reason = 0.0, "declared_global_serial_order_feasibility_test"
                # La mémoire du PID n’est pas accumulée pendant cet arrêt global déclaré.
                steering = (
                    float(np.clip(follower.steering_control(target_lane), -1, 1))
                    if speed
                    else 0.0
                )
                if self.case == "lane_merge":
                    speed = min(speed, 0.8)
                    blockers, complete = self.local_guard(key, env, speed, steering)
                else:
                    blockers, complete = guard_blockers(
                        key,
                        env,
                        max(0.5, stopping_distance(math.hypot(*vehicle.velocity))),
                    )
            accepted = speed if not blockers and complete else 0.0
            request, applied = (
                DriveCommand(speed, steering),
                DriveCommand(accepted, steering),
            )
            actions[key] = apply_command(vehicle, applied)
            poses, covered = path_poses(vehicle, 0.5)
            desired[key] = dict(
                requested_forward=speed > 0,
                path_poses=poses,
                path_complete=covered,
                native_mission=vehicle.config["destination"],
            )
            records[key] = dict(
                requested=asdict(request),
                accepted=asdict(applied),
                native_action=actions[key],
                explicit_brake=vehicle.explicit_brake,
                policy_reason=reason,
                guard_refused=speed != accepted,
                guard_blockers=blockers,
                guard_path_complete=complete,
                stage=self.stage,
                source="global_feasibility_controller_not_distributed_policy",
                guard_scope="sampled_execution_guard_not_independent_physical_impossibility",
            )
        return actions, desired, records, self.priority
