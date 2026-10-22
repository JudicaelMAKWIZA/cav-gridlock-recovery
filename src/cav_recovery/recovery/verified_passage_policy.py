"""Variantes expérimentales du passage, séparées de la référence B3."""

import copy
from dataclasses import dataclass
import math

from shapely import wkt

from ..metadrive.observation import rectangle
from .passage_policy import PassagePolicy


class FreshPartnerPolicy(PassagePolicy):
    """Attend les accords des voisins courants, puis laisse le garde trancher.

    La référence B3 gardait l'union de tous les voisins vus depuis le début
    d'une option. Un CAV sorti des 10 m pouvait ainsi retenir l'option sans
    pouvoir lui transmettre un nouvel accord. Le garde exige toujours les
    accords des voisins présents au moment de la commande de recul.
    """

    def decide(self, view, nominal, guard):
        if self.attempt is not None:
            self.partners = {
                message["sender"]
                for message in view.messages
                if message["kind"] == "STATE"
                and math.dist(message["position"], view.own["position"]) <= 10.0
            }
        return super().decide(view, nominal, guard)


class RevalidatedPassagePolicy(PassagePolicy):
    """Abandonne une option ancienne si sa demande a disparu et la route est sûre.

    L'annulation est annoncée ; elle ne signifie pas qu'un passage a réussi.
    Le garde conserve la décision finale sur l'action nominale retrouvée.
    """

    request_grace_s = 2.0

    def __init__(self, ego, endpoint, contract):
        super().__init__(ego, endpoint, contract)
        self.last_request_s = None

    def decide(self, view, nominal, guard):
        if any(
            message["kind"] == "CLEAR_REQUEST" and message["target"] == self.ego
            for message in view.messages
        ):
            self.last_request_s = view.at_s
        if self.attempt is not None and self.action.speed_mps < 0:
            last_request = (
                self.last_request_s if self.last_request_s is not None else self.started
            )
            if (
                view.at_s - last_request >= self.request_grace_s
                and abs(view.own["speed_mps"]) <= self.contract.direction_change_speed_mps
                and guard.limit(view, nominal)[1]["accepted"]
            ):
                self.outbox = []
                self.emit(
                    "CANCEL", view, {"reason": "request_expired_nominal_safe"},
                    attempt=self.attempt,
                )
                self.attempt = None
                self.partners = set()
                self.requesters = {}
                self.cooldown = view.at_s + 2.0
                self.yield_until = view.at_s
                return nominal
        return super().decide(view, nominal, guard)


@dataclass
class OfferedPassage:
    """Mémoire propre du bénéficiaire, liée à une seule tentative reçue."""

    space: object
    initial_progress_m: float
    purpose: str
    entered: bool = False
    cleared: bool = False


class VerifiedPassagePolicy(PassagePolicy):
    """Demande la preuve locale du bénéficiaire et la corrobore par V2V STATE.

    Le rapport annonce séparément le refus du garde, la décision de la
    politique et le franchissement réel. Il ne contient aucun témoin oracle.
    Une expiration conserve son motif distinct et ne devient jamais un succès.
    """

    def __init__(self, ego, endpoint, contract):
        super().__init__(ego, endpoint, contract)
        self.offers = {}
        self.request_purposes = {}
        self._request_guard = None
        self._option_cancelled = False

    def request_clearance(self, view, space, *, target, purpose, action):
        """Transmet la mission et l'enveloppe locale effectivement testée."""
        _, _, required = self._request_guard.action_envelope(view, action)
        receipt = self._request_guard.check(
            view, action, require_agreement=False,
        )
        self.emit(
            "CLEAR_REQUEST", view,
            dict(
                space_wkt=space.wkt, progress_m=view.own["progress_m"],
                purpose=purpose, movement_id=view.route.agent["movement_id"],
                required_wkt=required.wkt, guard_reason=receipt["reason"],
            ),
            target=target, attempt=f"{self.ego}:clear:{purpose}",
        )

    def passage_crossed(self, passage, vehicle, progress, length_m):
        if passage.purpose == "REVERSE_OPTION":
            return (
                passage.entered
                and abs(progress - passage.initial_progress_m) >= 0.2
                and not vehicle.buffer(0.15).intersects(passage.space)
            )
        return super().passage_crossed(passage, vehicle, progress, length_m)

    def emit(self, kind, view, payload, *, target=None, attempt=None):
        if kind == "DONE" and payload.get("reason") == "bounded_passage_timeout":
            kind = "CANCEL"
            payload = {"reason": "passage_unconfirmed_timeout"}
        return super().emit(
            kind, view, payload, target=target, attempt=attempt,
        )

    def report_option_end(self, view, reason):
        if (
            reason != "announced_space_clear"
            and self.action.speed_mps < 0
            and self.requesters
        ):
            self._option_cancelled = True
            self.emit(
                "CANCEL", view, {"reason": "option_ended_without_clearance"},
            )
            return
        super().report_option_end(view, reason)

    def passage_complete(self, view, passage):
        if not passage.crossed:
            return False
        return any(
            message["kind"] == "PASSAGE_REPORT"
            and message["sender"] == passage.beneficiary
            and message["target"] == self.ego
            and message["attempt"] == self.lease_attempt
            and message["payload"]["phase"] == "CLEARED"
            and message["payload"]["resource_status"] == "CLEAR"
            and message["payload"]["purpose"] == passage.purpose
            and 0 <= view.at_s - message["sent_s"] <= self.contract.perception_max_age_s
            for message in view.messages
        )

    def _report_offers(self, view, nominal, action, guard):
        """Rapporte la sortie du cédeur, pas une vacance globale de la ressource."""
        live = {}
        for message in view.messages:
            if message["kind"] != "PASSAGE_OFFER" or message["target"] != self.ego:
                continue
            if not 0 <= view.at_s - message["sent_s"] <= self.contract.perception_max_age_s:
                continue
            key = (message["sender"], message["attempt"])
            offer = self.offers.get(key)
            if offer is None:
                offer = OfferedPassage(
                    wkt.loads(message["payload"]["space_wkt"]),
                    message["payload"]["initial_progress_m"],
                    message["payload"]["purpose"],
                )
            live[key] = offer
            own = view.own
            body = rectangle(
                own["position"], own["heading_rad"],
                own["length_m"], own["width_m"],
            )
            offer.entered |= body.intersects(offer.space)
            progressed = (
                abs(own["progress_m"] - offer.initial_progress_m) >= 0.2
                if offer.purpose == "REVERSE_OPTION"
                else own["progress_m"]
                >= offer.initial_progress_m + own["length_m"] + 0.5
            )
            offer.cleared |= (
                offer.entered and progressed
                and not body.buffer(0.15).intersects(offer.space)
            )
            receipt = guard.check(view, nominal)
            _, _, required = guard.action_envelope(view, nominal)
            yielder_state = self._fresh_state(view, message["sender"])
            if yielder_state is None:
                resource_status = "UNKNOWN"
            else:
                state = yielder_state["payload"]
                yielder_body = rectangle(
                    yielder_state["position"], state["heading_rad"],
                    state["length_m"], state["width_m"],
                )
                resource_status = (
                    "OCCUPIED" if yielder_body.buffer(0.15).intersects(offer.space)
                    else "CLEAR"
                )
            if offer.cleared:
                phase = "CLEARED"
            elif resource_status == "UNKNOWN":
                phase = "UNKNOWN"
            elif resource_status == "OCCUPIED":
                phase = "BLOCKED"
            elif offer.entered and own["progress_m"] > offer.initial_progress_m + 0.2:
                phase = "ENTERED"
            elif receipt["accepted"]:
                phase = "ADMISSIBLE"
            else:
                phase = "BLOCKED"
            self.emit(
                "PASSAGE_REPORT", view,
                dict(
                    phase=phase, guard_reason=receipt["reason"],
                    resource_status=resource_status,
                    policy_reason=action.reason,
                    required_wkt=required.wkt,
                    progress_m=own["progress_m"],
                    purpose=offer.purpose,
                ),
                target=message["sender"], attempt=message["attempt"],
            )
        self.offers = live

    def decide(self, view, nominal, guard):
        self._request_guard = guard
        self._option_cancelled = False
        previous_attempt = self.attempt
        action = super().decide(view, nominal, guard)
        if self._option_cancelled:
            self.passages = {}
            self.lease_started_s = None
            self.lease_until_s = None
            self.lease_attempt = None
        if previous_attempt is None and self.attempt is not None:
            self.request_purposes = {
                message["sender"]: message["payload"].get(
                    "purpose", "NOMINAL_PASSAGE"
                )
                for message in view.messages
                if message["kind"] == "CLEAR_REQUEST"
                and message["target"] == self.ego
            }
        for passage in self.passages.values():
            passage.purpose = self.request_purposes.get(
                passage.beneficiary, "NOMINAL_PASSAGE"
            )
        self._report_offers(view, nominal, action, guard)
        for passage in self.passages.values():
            self.emit(
                "PASSAGE_OFFER", view,
                dict(
                    space_wkt=passage.space.wkt,
                    initial_progress_m=passage.initial_progress_m,
                    purpose=passage.purpose,
                ),
                target=passage.beneficiary, attempt=self.lease_attempt,
            )
        return action


class PrioritizedPassagePolicy(VerifiedPassagePolicy):
    """Ne renouvelle pas un accord circulaire quand son passage est recevable.

    Le CAV laisse expirer tout accord déjà donné avant de bouger. Le garde
    ordinaire vérifie ensuite la prochaine commande ; aucun passage n'est
    autorisé par la seule simulation contrefactuelle locale.
    """

    def yield_to_intent(self, view, nominal, guard, message):
        if self.attempt is not None or message["payload"]["speed_mps"] >= 0:
            return True
        offers = [
            packet for packet in view.messages
            if packet["kind"] == "PASSAGE_OFFER"
            and packet["target"] == self.ego
            and packet["payload"]["purpose"] == "NOMINAL_PASSAGE"
            and 0 <= view.at_s - packet["sent_s"]
            <= self.contract.perception_max_age_s
        ]
        peer = self._fresh_state(view, message["sender"])
        if not offers or peer is None:
            return True
        if abs(peer["payload"]["speed_mps"]) >= self.contract.stationary_speed_mps:
            return True
        probe = copy.copy(view)
        probe.own_commitment_until_s = view.at_s
        if not guard.check(probe, nominal)["accepted"]:
            return True
        self.emit(
            "REFUSE", view, {"reason": "upstream_passage_priority"},
            target=message["sender"], attempt=message["attempt"],
        )
        return False


class ContinuousPassagePolicy(PrioritizedPassagePolicy):
    """Ne rétablit pas un accord adverse pendant un franchissement engagé.

L'offre et l'entrée sont toutes deux observées localement. Refuser un nouvel
accord ne révoque pas un accord déjà valable et n'autorise aucune commande :
le garde conserve toutes ses vérifications physiques à chaque pas.
    """

    def _nominal_passage_in_flight(self, view):
        for packet in view.messages:
            if packet["kind"] != "PASSAGE_OFFER" or packet["target"] != self.ego:
                continue
            if packet["payload"]["purpose"] != "NOMINAL_PASSAGE":
                continue
            if not 0 <= view.at_s - packet["sent_s"] <= self.contract.perception_max_age_s:
                continue
            memory = self.offers.get((packet["sender"], packet["attempt"]))
            if memory is not None and memory.entered and not memory.cleared:
                return True
        return False

    def yield_to_intent(self, view, nominal, guard, message):
        if (
            message["payload"]["speed_mps"] < 0
            and self._nominal_passage_in_flight(view)
        ):
            self.emit(
                "REFUSE", view, {"reason": "in_flight_passage_priority"},
                target=message["sender"], attempt=message["attempt"],
            )
            return False
        return super().yield_to_intent(view, nominal, guard, message)
