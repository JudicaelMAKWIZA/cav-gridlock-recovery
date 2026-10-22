"""Recherche globale de borne supérieure, interdite aux acteurs opérationnels."""

import math
from shapely.geometry import Point
from shapely.ops import unary_union
from ..metadrive.observation import rectangle
from .cooperative import option_bank, required_space, endpoint_pose
from .safety import trajectory
from .safety import LocalGuard
from .contracts import Action


def search(
    instance,
    relevant_requesters=None,
    *,
    pose_predictor=endpoint_pose,
    sweep_predictor=trajectory,
):
    """Énumère 15 commandes × 4 distances × véhicules, autres corps figés.

    Ce résultat cinématique est un candidat, jamais une preuve dynamique.
    Seule son exécution native contrôlée peut établir une libération physique.
    """
    views = instance.views
    bodies = {
        k: rectangle(
            v.own["position"], v.own["heading_rad"], v.own["length_m"], v.own["width_m"]
        )
        for k, v in views.items()
    }
    spaces = {k: required_space(v) for k, v in views.items()}
    edges = {
        (r, b)
        for r in views
        for b in views
        if r != b
        and (relevant_requesters is None or r in relevant_requesters)
        and spaces[r].intersection(bodies[b]).area > 0.01
    }
    candidates = []
    for ego, view in views.items():
        obstacles = unary_union([b for k, b in bodies.items() if k != ego]).buffer(0.15)
        for action in option_bank()[1:]:
            if action.speed_mps == 0:
                continue
            for distance in [0.5, 1.0, 2.0, 4.5]:
                if (
                    action.speed_mps < 0
                    and instance.reverse_used[ego] + distance
                    > instance.contract.reverse_budget_m
                ):
                    continue
                sweep = sweep_predictor(view.own, action, distance)
                if not view.road.buffer(1e-5).covers(sweep) or sweep.intersects(
                    obstacles
                ):
                    continue
                pose = pose_predictor(view.own, action, distance)
                end = rectangle(
                    pose[:2], pose[2], view.own["length_m"], view.own["width_m"]
                )
                released = [
                    (r, b)
                    for r, b in edges
                    if b == ego and spaces[r].intersection(end).area < 0.01
                ]
                gain = sum(
                    spaces[r].intersection(bodies[ego]).area
                    - spaces[r].intersection(end).area
                    for r, b in edges
                    if b == ego
                )
                progress = view.route.line.project(
                    Point(pose[:2])
                ) - view.route.line.project(Point(view.own["position"]))
                if released or gain > 0.1:
                    candidates.append(
                        dict(
                            ego=ego,
                            speed_mps=action.speed_mps,
                            steering=action.steering,
                            distance_m=distance,
                            released_edges=released,
                            overlap_gain_m2=gain,
                            score=10 * len(released)
                            + gain
                            + 0.1 * progress
                            - 0.05 * distance,
                            locally_admitted=instance.guard.check(
                                view, action, require_agreement=False
                            )["accepted"],
                        )
                    )
    # Deux étapes bornées : un suiveur peut devoir libérer le recul du CAV
    # qui occupe réellement la ressource. Ne pas exiger un gain immédiat.
    if not any(c["locally_admitted"] and c["released_edges"] for c in candidates):
        blockers = {b for _, b in edges}
        for first, view in views.items():
            for distance in [2.0, 4.5]:
                action = Action(-0.6)
                first_sweep = sweep_predictor(view.own, action, distance)
                obstacles = unary_union(
                    [b for k, b in bodies.items() if k != first]
                ).buffer(0.15)
                if not view.road.covers(first_sweep) or first_sweep.intersects(
                    obstacles
                ):
                    continue
                pose = pose_predictor(view.own, action, distance)
                first_end = rectangle(
                    pose[:2], pose[2], view.own["length_m"], view.own["width_m"]
                )
                altered = {**bodies, first: first_end}
                for second in blockers - {first}:
                    if (
                        math.dist(view.own["position"], views[second].own["position"])
                        > 12.0
                    ):
                        continue
                    for secondary in [Action(-0.6, s) for s in [-0.3, 0.0, 0.3]]:
                        for extent in [2.0, 4.5]:
                            sweep = sweep_predictor(
                                views[second].own, secondary, extent
                            )
                            obstacles2 = unary_union(
                                [b for k, b in altered.items() if k != second]
                            ).buffer(0.15)
                            if not view.road.covers(sweep) or sweep.intersects(
                                obstacles2
                            ):
                                continue
                            endpoint = pose_predictor(
                                views[second].own, secondary, extent
                            )
                            other = views[second].own
                            end = rectangle(
                                endpoint[:2],
                                endpoint[2],
                                other["length_m"],
                                other["width_m"],
                            )
                            released = [
                                (r, b)
                                for r, b in edges
                                if b == second
                                and spaces[r].intersection(end).area < 0.01
                            ]
                            if released:
                                candidates.append(
                                    dict(
                                        ego=first,
                                        speed_mps=-0.6,
                                        steering=0.0,
                                        distance_m=distance,
                                        released_edges=[],
                                        future_released_edges=released,
                                        lookahead_second=dict(
                                            ego=second,
                                            steering=secondary.steering,
                                            distance_m=extent,
                                        ),
                                        overlap_gain_m2=0.0,
                                        score=8 * len(released) - 0.05 * distance,
                                        locally_admitted=instance.guard.check(
                                            view, action, require_agreement=False
                                        )["accepted"],
                                    )
                                )
    return dict(
        at_s=instance.at_s,
        privileged_global=True,
        physical_candidate_edges=sorted(edges),
        candidates=sorted(candidates, key=lambda c: c["score"], reverse=True),
    )


class GlobalCoordinator:
    """Choisit globalement une manœuvre ; le garde local reste obligatoire."""

    def __init__(self):
        self.plan = None
        self.anchor = None
        self.started = 0.0
        self.attempt = None
        self.log = []
        self.count = 0
        self.next_search = 30.0
        self.searcher = search

    def decide(self, instance, nominal_actions):
        now = instance.at_s
        outgoing = []
        if self.plan is None and now >= self.next_search and self.count < 4:
            from ..dataset.qualify import qualify
            from ..dataset.contract import Contract
            from ..experiments.evaluation import active_events

            annotation = qualify(
                instance.audit_frames,
                instance.metadata,
                dict(errors=instance.errors, complete_measurements=True),
                Contract(),
            )
            relevant = {e["requester"] for e in active_events(annotation, now)}
            if not relevant:
                self.next_search = now + 1.0
                return nominal_actions, outgoing
            report = self.searcher(instance, relevant)
            self.log.append(report)
            best = next(
                (c for c in report["candidates"] if c["locally_admitted"]), None
            )
            if best:
                self.plan = best
                self.count += 1
                self.started = now
                self.anchor = list(instance.views[best["ego"]].own["position"])
                self.attempt = f"global:{self.count}"
            else:
                self.next_search = now + 5.0
        if self.plan is None:
            return nominal_actions, outgoing
        ego = self.plan["ego"]
        if (
            ego not in instance.views
            or math.dist(self.anchor, instance.views[ego].own["position"])
            >= self.plan["distance_m"]
            or now - self.started > 25.0
        ):
            self.log.append(dict(event="native_option_end", at_s=now, plan=self.plan))
            self.plan = None
            self.next_search = now + 8.0
            return {
                k: Action(0.0, reason="global_option_end") for k in instance.views
            }, outgoing
        actions = {
            k: Action(0.0, reason="global_feasibility_hold") for k in instance.views
        }
        for key, view in instance.views.items():
            if (
                key != ego
                and abs(view.own["speed_mps"]) < instance.contract.stationary_speed_mps
            ):
                outgoing.append(
                    instance.endpoints[key].send(
                        "AGREE_HOLD",
                        self.attempt,
                        view.own["position"],
                        now,
                        target=ego,
                        payload={"until_s": now + instance.contract.message_lifetime_s},
                    )
                )
        if now - self.started >= 0.4:
            actions[ego] = Action(
                self.plan["speed_mps"],
                self.plan["steering"],
                "global_feasibility_option",
                self.attempt,
            )
        return actions, outgoing


class GlobalBodyGuard(LocalGuard):
    """Vérificateur privilégié de faisabilité, jamais un garde opérationnel."""

    def __init__(self, instance):
        super().__init__(instance.contract)
        self.instance = instance

    def check(self, view, action, *, require_agreement=True):
        receipt = super().check(view, action, require_agreement=require_agreement)
        if action.speed_mps == 0:
            return receipt
        if not receipt["accepted"] and receipt["reason"] not in {
            "RAY_OCCUPANCY_OR_UNKNOWN",
            "RECEIVED_OCCUPANCY",
        }:
            return receipt
        speed = abs(view.own["speed_mps"])
        distance = max(
            0.25,
            speed * self.contract.step_s
            + speed**2 / 2
            + abs(action.speed_mps) * self.contract.step_s
            + 0.15,
        )
        sweep = trajectory(view.own, action, distance)
        for ego, other in self.instance.views.items():
            if ego == view.ego:
                continue
            state = other.own
            v = abs(state["speed_mps"])
            body = rectangle(
                state["position"],
                state["heading_rad"],
                state["length_m"],
                state["width_m"],
            )
            if sweep.intersects(body.buffer(0.15 + v * 0.2 + v**2 / 2 + 0.12)):
                return self._refuse(action, "GLOBAL_AUDIT_OCCUPANCY")
        if action.speed_mps < 0 and require_agreement:
            peers = {
                m["sender"]
                for m in view.messages
                if m["kind"] == "STATE"
                and math.dist(m["position"], view.own["position"]) <= 10.0
            }
            agreed = {
                m["sender"]
                for m in view.messages
                if m["kind"] == "AGREE_HOLD"
                and m["attempt"] == action.attempt
                and m["payload"]["until_s"] >= view.at_s
            }
            if not peers <= agreed:
                return self._refuse(action, "COOPERATIVE_AGREEMENT_MISSING")
        return dict(
            accepted=True,
            reason="GLOBAL_AUDIT_BODY_ENVELOPE",
            requested=action.speed_mps,
            accepted_speed_mps=action.speed_mps,
            scope="privileged_global_upper_bound_not_operational",
        )
