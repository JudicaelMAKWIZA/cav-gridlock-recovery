"""Engagement local de passage après une manœuvre de dégagement.

Les bénéficiaires proviennent uniquement des demandes V2V reçues. Leur passage
est vérifié avec les messages d'état récents, jamais avec l'annotation oracle.
"""

from dataclasses import dataclass

from shapely import wkt

from ..metadrive.observation import rectangle
from .clearance_policy import ClearancePolicy
from .contracts import Action


@dataclass
class Passage:
    beneficiary: str
    space: object
    initial_progress_m: float
    last_progress_m: float
    last_progress_at_s: float
    entered: bool = False
    crossed: bool = False


class PassagePolicy(ClearancePolicy):
    """Retient un cédeur tant que son bénéficiaire utilise l'espace dégagé.

    Le bail de 20 s n'est renouvelé que si un état frais montre une progression
    nouvelle. Une attente sans progrès s'achève au plus tard à 40 s. La reprise
    reste ensuite soumise au garde local, y compris en cas de perte de messages.
    """

    lease_s = 20.0
    maximum_lease_s = 40.0

    def __init__(self, ego, endpoint, contract):
        super().__init__(ego, endpoint, contract)
        self.requesters = {}
        self.passages = {}
        self.lease_started_s = None
        self.lease_until_s = None
        self.lease_attempt = None

    def report_option_end(self, view, reason):
        if (
            reason == "announced_space_clear"
            and self.action.speed_mps < 0
            and self.requesters
        ):
            return
        super().report_option_end(view, reason)

    def _fresh_state(self, view, beneficiary):
        return next(
            (
                message
                for message in view.messages
                if message["kind"] == "STATE"
                and message["sender"] == beneficiary
                and 0
                <= view.at_s - message["sent_s"]
                <= self.contract.perception_max_age_s
            ),
            None,
        )

    def _passage_action(self, view):
        now = view.at_s
        unresolved = []
        for passage in self.passages.values():
            state = self._fresh_state(view, passage.beneficiary)
            if state is not None:
                progress = state["payload"]["progress_m"]
                if progress > passage.last_progress_m + 0.2:
                    passage.last_progress_m = progress
                    passage.last_progress_at_s = now
                vehicle = rectangle(
                    state["position"],
                    state["payload"]["heading_rad"],
                    state["payload"]["length_m"],
                    state["payload"]["width_m"],
                )
                passage.entered |= vehicle.intersects(passage.space)
                passage.crossed |= (
                    passage.entered
                    and progress
                    >= passage.initial_progress_m + state["payload"]["length_m"] + 0.5
                    and not vehicle.buffer(0.15).intersects(passage.space)
                )
            if not passage.crossed:
                unresolved.append(passage)

        if not unresolved:
            reason = "beneficiaries_passed"
        elif now >= self.lease_until_s - 1e-7:
            advancing = any(
                now - passage.last_progress_at_s <= 5.0 for passage in unresolved
            )
            if advancing and now < self.lease_started_s + self.maximum_lease_s - 1e-7:
                self.lease_until_s = min(
                    now + self.lease_s,
                    self.lease_started_s + self.maximum_lease_s,
                )
                reason = None
            else:
                reason = "bounded_passage_timeout"
        else:
            reason = None

        if reason is not None:
            self.emit("DONE", view, {"reason": reason}, attempt=self.lease_attempt)
            self.passages = {}
            self.lease_started_s = None
            self.lease_until_s = None
            self.lease_attempt = None
            return None

        for passage in unresolved:
            current_intent = next(
                (
                    message
                    for message in view.messages
                    if message["kind"] == "INTENT"
                    and message["sender"] == passage.beneficiary
                    and message["expires_s"] >= now
                ),
                None,
            )
            self.emit(
                "AGREE_HOLD",
                view,
                {"until_s": now + self.contract.message_lifetime_s},
                target=passage.beneficiary,
                attempt=(
                    current_intent["attempt"]
                    if current_intent is not None
                    else self.lease_attempt
                ),
            )
        return Action(0.0, reason="passage_commitment")

    def decide(self, view, nominal, guard):
        completed_messages = []
        if self.passages:
            self.outbox = []
            action = self._passage_action(view)
            if action is not None:
                return action
            completed_messages = self.outbox

        prior_attempt = self.attempt
        prior_action = self.action
        prior_requesters = self.requesters
        action = super().decide(view, nominal, guard)
        self.outbox = completed_messages + self.outbox

        if prior_attempt is None and self.attempt is not None:
            self.requesters = {
                message["sender"]: (
                    wkt.loads(message["payload"]["space_wkt"]),
                    message["payload"]["progress_m"],
                )
                for message in view.messages
                if message["kind"] == "CLEAR_REQUEST" and message["target"] == self.ego
            }
        elif prior_attempt is not None and self.attempt is None:
            self.requesters = {}
            if (
                action.reason == "option_end"
                and prior_action.speed_mps < 0
                and prior_requesters
                and self.release_space is not None
            ):
                own = view.own
                present = rectangle(
                    own["position"],
                    own["heading_rad"],
                    own["length_m"],
                    own["width_m"],
                )
                if not present.buffer(0.15).intersects(self.release_space):
                    self.passages = {}
                    for beneficiary, (space, progress) in prior_requesters.items():
                        passage = Passage(
                            beneficiary, space, progress, progress, view.at_s
                        )
                        state = self._fresh_state(view, beneficiary)
                        if state is not None:
                            vehicle = rectangle(
                                state["position"],
                                state["payload"]["heading_rad"],
                                state["payload"]["length_m"],
                                state["payload"]["width_m"],
                            )
                            passage.entered = vehicle.intersects(space)
                        self.passages[beneficiary] = passage
                    self.lease_started_s = view.at_s
                    self.lease_until_s = view.at_s + self.lease_s
                    self.lease_attempt = prior_attempt
                    self.yield_until = view.at_s
        return action
