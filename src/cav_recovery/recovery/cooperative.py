"""Baseline REC-COOP-2 : dégagement choisi sur les seules demandes reçues."""

import math
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import unary_union
from ..metadrive.observation import rectangle
from ..metadrive.planning import requested_path
from .contracts import Action
from .safety import trajectory


def required_space(view):
    own = view.own
    poses = requested_path(
        view.route,
        own["position"],
        own["heading_rad"],
        max(0.0, own["speed_mps"]),
        own["wheelbase_m"],
        own["max_steering_rad"],
        1.2 * own["length_m"],
    )
    sweep = unary_union(
        [rectangle(p[:2], p[2], own["length_m"], own["width_m"]) for p in poses]
    )
    return sweep.difference(
        rectangle(own["position"], own["heading_rad"], own["length_m"], own["width_m"])
    )


def option_bank():
    """Même bibliothèque physique pour planification, baseline et apprentissage."""
    return (
        [Action(0.0, reason="hold")]
        + [
            Action(speed, steer, "cooperative_option")
            for speed in [0.4, 0.8, -0.6]
            for steer in [-0.6, -0.3, 0.0, 0.3, 0.6]
        ]
        + [Action(0.0, reason="refuse_commitment")]
    )


def endpoint_pose(own, action, distance=1.0):
    sign = 1 if action.speed_mps >= 0 else -1
    curvature = math.tan(action.steering * own["max_steering_rad"]) / own["wheelbase_m"]
    h = own["heading_rad"]
    x, y = own["position"]
    s = sign * distance
    angle = h + curvature * s
    if abs(curvature) < 1e-8:
        return [x + s * math.cos(h), y + s * math.sin(h), angle]
    return [
        x + (math.sin(angle) - math.sin(h)) / curvature,
        y - (math.cos(angle) - math.cos(h)) / curvature,
        angle,
    ]


class CooperativePolicy:
    """Chaque instance connaît seulement son endpoint, sa mission et ses réceptions."""

    def __init__(self, ego, endpoint, contract):
        self.ego, self.endpoint, self.contract = ego, endpoint, contract
        self.stopped_since = None
        self.attempt = None
        self.started = 0.0
        self.anchor = None
        self.action = None
        self.outbox = []
        self.yield_until = 0.0
        self.cooldown = 0.0
        self.partners = set()
        self.progress_history = []
        self.release_space = None
        self.activation_override = None
        self.request_distance_m = 2.5
        self.option_distance_m = 4.5
        self.option_duration_s = 25.0
        self.pose_predictor = endpoint_pose
        self.sweep_predictor = trajectory

    def emit(self, kind, view, payload, *, target=None, attempt=None):
        self.outbox.append(
            self.endpoint.send(
                kind,
                attempt or self.attempt or "request",
                view.own["position"],
                view.at_s,
                target=target,
                payload=payload,
            )
        )

    def candidates(self, view, guard):
        own = view.own
        requests = [
            wkt.loads(m["payload"]["space_wkt"])
            for m in view.messages
            if m["kind"] == "CLEAR_REQUEST" and m["target"] in (None, self.ego)
        ]
        present = rectangle(
            own["position"], own["heading_rad"], own["length_m"], own["width_m"]
        )
        requested = unary_union([r for r in requests if r.intersects(present)])
        options = []
        station = view.route.line.project(Point(own["position"]))
        for action in option_bank()[1:]:
            if action.speed_mps == 0:
                continue
            if not guard.check(view, action, require_agreement=False)["accepted"]:
                continue
            pose = self.pose_predictor(own, action, 1.0)
            end = rectangle(pose[:2], pose[2], own["length_m"], own["width_m"])
            release = (
                present.intersection(requested).area - end.intersection(requested).area
            )
            progress = view.route.line.project(Point(pose[:2])) - station
            if action.speed_mps < 0 and release <= 0.02:
                continue
            if not view.road.covers(self.sweep_predictor(own, action, 1.0)):
                continue
            route_error = view.route.line.distance(Point(pose[:2]))
            score = (
                2 * release
                + progress
                - 0.3 * route_error
                - 0.02 * abs(action.steering)
                + 0.05 * max(0.0, action.speed_mps)
            )
            options.append((score, action))
        return [a for _, a in sorted(options, key=lambda item: item[0], reverse=True)]

    def decide(self, view, nominal, guard):
        self.outbox = []
        now = view.at_s
        stationary = abs(view.own["speed_mps"]) < self.contract.stationary_speed_mps
        self.stopped_since = (
            (now if self.stopped_since is None else self.stopped_since)
            if stationary
            else None
        )
        self.progress_history.append((now, view.own["progress_m"]))
        while (
            len(self.progress_history) > 1
            and self.progress_history[1][0] <= now - self.contract.stall_s
        ):
            self.progress_history.pop(0)
        stalled = (
            now - self.progress_history[0][0] >= self.contract.stall_s - 1e-7
            and view.own["progress_m"] - self.progress_history[0][1] < 0.5
        )
        if self.activation_override is not None:
            stalled = self.activation_override
        if stalled and self.attempt is None and now >= self.yield_until:
            space = required_space(view)
            for message in view.messages:
                if message["kind"] != "STATE":
                    continue
                p = message["payload"]
                other = rectangle(
                    message["position"], p["heading_rad"], p["length_m"], p["width_m"]
                )
                if space.intersection(other).area > 0.01:
                    self.emit(
                        "CLEAR_REQUEST",
                        view,
                        dict(space_wkt=space.wkt, progress_m=view.own["progress_m"]),
                        target=message["sender"],
                    )
        intents = [
            m
            for m in view.messages
            if m["kind"] == "INTENT"
            and math.dist(m["position"], view.own["position"]) <= 10.0
        ]
        peers = {m["sender"] for m in intents}
        if self.attempt:
            peers.add(self.ego)
        winner = min(peers) if peers else None
        if winner is not None and winner != self.ego:
            invitation = next(m for m in intents if m["sender"] == winner)
            if self.attempt:
                self.emit(
                    "CANCEL", view, {"reason": "local_priority"}, attempt=self.attempt
                )
                self.attempt = None
            if stationary:
                self.emit(
                    "AGREE_HOLD",
                    view,
                    {"until_s": now + self.contract.message_lifetime_s},
                    target=winner,
                    attempt=invitation["attempt"],
                )
            return Action(0.0, reason="relevant_neighbor_hold")
        if now < view.own_commitment_until_s - 1e-7:
            return Action(0.0, reason="honour_hold")
        if now < self.yield_until:
            return Action(0.0, reason="resource_yield")
        if self.attempt:
            moved = math.dist(self.anchor, view.own["position"])
            space_clear = (
                self.release_space is not None
                and moved > 0.25
                and not rectangle(
                    view.own["position"],
                    view.own["heading_rad"],
                    view.own["length_m"],
                    view.own["width_m"],
                )
                .buffer(0.15)
                .intersects(self.release_space)
            )
            if (
                space_clear
                or moved >= self.option_distance_m
                or now - self.started >= self.option_duration_s
            ):
                self.emit(
                    "DONE",
                    view,
                    {
                        "reason": "announced_space_clear"
                        if space_clear
                        else "bounded_local_option_finished"
                    },
                )
                reverse = self.action.speed_mps < 0
                self.attempt = None
                self.cooldown = now + 2.0
                self.yield_until = now + (15.0 if reverse else 0.0)
                return Action(0.0, reason="option_end")
        nominal_allowed = guard.limit(view, nominal)[1]["accepted"]
        present = rectangle(
            view.own["position"],
            view.own["heading_rad"],
            view.own["length_m"],
            view.own["width_m"],
        )
        helping = any(
            m["kind"] == "CLEAR_REQUEST"
            and m["target"] == self.ego
            and wkt.loads(m["payload"]["space_wkt"]).intersects(present)
            for m in view.messages
        )
        if helping and self.attempt is None and now >= self.yield_until:
            requests = unary_union(
                [
                    wkt.loads(m["payload"]["space_wkt"])
                    for m in view.messages
                    if m["kind"] == "CLEAR_REQUEST" and m["target"] == self.ego
                ]
            )
            for steer in [0.0, -0.3, 0.3]:
                reverse = Action(-0.6, steer)
                pose = self.pose_predictor(view.own, reverse, self.request_distance_m)
                end = rectangle(
                    pose[:2], pose[2], view.own["length_m"], view.own["width_m"]
                )
                if (
                    present.intersection(requests).area
                    - end.intersection(requests).area
                    <= 0.05
                ):
                    continue
                sweep = self.sweep_predictor(view.own, reverse, self.request_distance_m)
                if not view.road.covers(sweep):
                    continue
                needed = sweep.difference(present)
                for message in view.messages:
                    if message["kind"] != "STATE":
                        continue
                    p = message["payload"]
                    other = rectangle(
                        message["position"],
                        p["heading_rad"],
                        p["length_m"],
                        p["width_m"],
                    )
                    if needed.intersection(other).area > 0.01:
                        self.emit(
                            "CLEAR_REQUEST",
                            view,
                            dict(
                                space_wkt=needed.wkt, progress_m=view.own["progress_m"]
                            ),
                            target=message["sender"],
                        )
                break
        if (
            self.attempt is None
            and (helping or stalled and not nominal_allowed)
            and now >= self.cooldown
        ):
            candidates = self.candidates(view, guard)
            if candidates:
                self.action = candidates[0]
                self.attempt = f"{self.ego}:{now:.1f}"
                self.started = now
                self.anchor = list(view.own["position"])
                self.partners = set()
                spaces = [
                    wkt.loads(m["payload"]["space_wkt"])
                    for m in view.messages
                    if m["kind"] == "CLEAR_REQUEST" and m["target"] == self.ego
                ]
                self.release_space = unary_union(spaces) if spaces else None
            else:
                self.cooldown = now + 1.0
        if self.attempt:
            a = self.action
            self.emit(
                "INTENT",
                view,
                dict(
                    speed_mps=a.speed_mps,
                    steering=a.steering,
                    distance_m=self.option_distance_m,
                ),
            )
            required = {
                m["sender"]
                for m in view.messages
                if m["kind"] == "STATE"
                and math.dist(m["position"], view.own["position"]) <= 10.0
            }
            self.partners.update(required)
            agreements = {
                m["sender"]
                for m in view.messages
                if m["kind"] == "AGREE_HOLD"
                and m["attempt"] == self.attempt
                and m["payload"]["until_s"] >= now
            }
            if self.partners <= agreements and now - self.started >= 0.4:
                return Action(
                    a.speed_mps, a.steering, "cooperative_release", self.attempt
                )
            return Action(0.0, reason="await_relevant_agreements")
        return nominal
