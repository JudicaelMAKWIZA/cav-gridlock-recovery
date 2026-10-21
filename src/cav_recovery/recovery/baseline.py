"""Baseline à informations locales : invitations, maintien et retrait borné."""

import math

from .contracts import Action


def gentle_forward_options():
    """Options publiques de dégagement, également proposées au futur acteur."""
    return [
        Action(0.4, steering, "local_forward_probe") for steering in [-0.3, 0.0, 0.3]
    ]


class DeterministicPolicy:
    """Aucune liste de groupe physique ; le retrait est une hypothèse locale."""

    def __init__(self, ego, endpoint, contract, *, enabled=True):
        self.ego, self.endpoint, self.contract = ego, endpoint, contract
        self.enabled = enabled
        self.stopped_since = None
        self.attempt = None
        self.started = None
        self.retreat_start = None
        self.cooldown_until = 0.0
        self.outbox = []
        self.hold_until = 0.0
        self.hold_for = None
        self.required_partners = set()
        self.yield_until = 0.0
        self.forward_probe_until = 0.0
        self.forward_steering = 0.0

    def emit(self, kind, view, *, attempt=None, target=None, payload=None):
        self.outbox.append(
            self.endpoint.send(
                kind,
                attempt or self.attempt or "state",
                view.own["position"],
                view.at_s,
                target=target,
                payload=payload,
            )
        )

    def decide(self, view, nominal, guard):
        self.outbox = []
        now = view.at_s
        if abs(view.own["speed_mps"]) < self.contract.stationary_speed_mps:
            if self.stopped_since is None:
                self.stopped_since = now
        else:
            self.stopped_since = None
        if not self.enabled:
            return nominal
        if now < self.yield_until:
            # Le créneau libéré ne doit pas être repris aussitôt par celui
            # qui vient de reculer ; les autres CAV reprennent leurs missions.
            for invitation in (m for m in view.messages if m["kind"] == "INTENT"):
                if abs(view.own["speed_mps"]) <= self.contract.stationary_speed_mps:
                    self.emit(
                        "AGREE_HOLD",
                        view,
                        attempt=invitation["attempt"],
                        target=invitation["sender"],
                        payload={"until_s": now + self.contract.message_lifetime_s},
                    )
            return Action(0.0, reason="post_retreat_yield")
        intents = [m for m in view.messages if m["kind"] == "INTENT"]
        contenders = {m["sender"] for m in intents}
        if self.attempt:
            contenders.add(self.ego)
        winner = min(contenders) if contenders else None
        if now < self.hold_until and self.hold_for != winner:
            return Action(0.0, reason="honour_unexpired_commitment")
        if winner and winner != self.ego:
            invitation = next(m for m in intents if m["sender"] == winner)
            if abs(view.own["speed_mps"]) <= self.contract.stationary_speed_mps:
                self.hold_until = now + self.contract.message_lifetime_s
                self.hold_for = winner
                self.emit(
                    "AGREE_HOLD",
                    view,
                    attempt=invitation["attempt"],
                    target=winner,
                    payload={"until_s": self.hold_until},
                )
            else:
                self.emit(
                    "REFUSE",
                    view,
                    attempt=invitation["attempt"],
                    target=winner,
                    payload={"reason": "braking_not_stopped"},
                )
            if self.attempt:
                self.emit("CANCEL", view, payload={"reason": "competing_intent"})
                self.attempt = None
            return Action(0.0, reason="cooperative_hold")
        stalled = (
            self.stopped_since is not None
            and now - self.stopped_since >= self.contract.stall_s
        )
        if not self.attempt and stalled and now >= self.cooldown_until:
            # Un braquage nominal saturé peut être refusé alors qu'une avance
            # douce reste libre. La politique teste ces options explicites ;
            # le garde ne choisit jamais lui-même une autre trajectoire.
            if not guard.check(view, nominal)["accepted"]:
                options = sorted(
                    gentle_forward_options(),
                    key=lambda a: abs(a.steering - nominal.steering),
                )
                selected = next(
                    (a for a in options if guard.check(view, a)["accepted"]), None
                )
                if selected is not None:
                    self.forward_probe_until = now + 5.0
                    self.forward_steering = selected.steering
                    self.cooldown_until = now + 6.0
                    return selected
            reverse = Action(
                -self.contract.maximum_reverse_mps, reason="local_retreat_probe"
            )
            forward_blocked = not guard.check(view, nominal)["accepted"]
            if (
                forward_blocked
                and guard.check(view, reverse, require_agreement=False)["accepted"]
            ):
                self.attempt = f"{self.ego}:{now:.1f}"
                self.started = now
                self.retreat_start = view.reverse_used_m
                self.required_partners = set()
        if self.attempt:
            if (
                now - self.started > 25
                or view.reverse_used_m - self.retreat_start
                > min(6.0, self.contract.reverse_budget_m - 0.1)
            ):
                self.emit("DONE", view, payload={"reason": "bounded_attempt_ended"})
                self.attempt = None
                self.cooldown_until = now + 8.0
                self.yield_until = now + 10.0
                return Action(0.0, reason="end_retreat")
            self.emit(
                "INTENT",
                view,
                payload={
                    "speed_mps": -self.contract.maximum_reverse_mps,
                    "steering": 0.0,
                    "distance_m": min(
                        6.0, self.contract.reverse_budget_m - view.reverse_used_m
                    ),
                },
            )
            required = {
                m["sender"]
                for m in view.messages
                if m["kind"] == "STATE"
                and math.dist(m["position"], view.own["position"]) <= 10.0
            }
            self.required_partners.update(required)
            agreements = {
                m["sender"]
                for m in view.messages
                if m["kind"] == "AGREE_HOLD"
                and m["attempt"] == self.attempt
                and m["payload"]["until_s"] >= now
            }
            if self.required_partners <= agreements and now - self.started >= 0.4:
                return Action(
                    -self.contract.maximum_reverse_mps,
                    reason="agreed_retreat",
                    attempt=self.attempt,
                )
            return Action(0.0, reason="await_hold_agreements")
        if now < self.forward_probe_until:
            return Action(0.4, self.forward_steering, "local_forward_probe")
        return nominal
