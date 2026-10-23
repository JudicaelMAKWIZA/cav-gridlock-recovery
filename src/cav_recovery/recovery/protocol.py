"""Intentions et engagements propres à portée bornée, sans arbitre global."""

import copy
import math


KINDS = {
    "INTENT", "AGREE_HOLD", "REFUSE", "CANCEL", "DONE", "STATE",
    "CLEAR_REQUEST", "PASSAGE_OFFER", "PASSAGE_REPORT",
}

# Les enveloppes balayées d'un mouvement tournant peuvent produire un WKT
# détaillé, même lorsque leur surface reste locale. Cette limite borne la
# taille du message sans simplifier la géométrie transmise.
MAX_SPACE_WKT_CHARS = 65536


def validate(message, now):
    fields = {
        "version",
        "sender",
        "sequence",
        "kind",
        "attempt",
        "target",
        "sent_s",
        "expires_s",
        "position",
        "payload",
    }
    if (
        not isinstance(message, dict)
        or set(message) != fields
        or message["version"] not in {"REC-V2V-1", "REC-V2V-2", "REC-V2V-3"}
    ):
        raise ValueError("Schéma coopératif invalide.")
    if (
        message["kind"] not in KINDS
        or type(message["sequence"]) is not int
        or message["sequence"] < 1
    ):
        raise ValueError("Type ou séquence invalide.")
    v3 = message["version"] == "REC-V2V-3"
    if (
        message["kind"] in {"PASSAGE_OFFER", "PASSAGE_REPORT"} and not v3
    ) or (v3 and message["kind"] not in {
        "CLEAR_REQUEST", "PASSAGE_OFFER", "PASSAGE_REPORT"
    }):
        raise ValueError("Version de passage incompatible.")
    if not all(
        isinstance(message[k], str) and message[k] for k in ["sender", "attempt"]
    ):
        raise ValueError("Identité de message manquante.")
    if any(len(message[k]) > 256 for k in ["sender", "attempt"]) or (
        message["target"] is not None
        and (
            not isinstance(message["target"], str)
            or not 0 < len(message["target"]) <= 256
        )
    ):
        raise ValueError("Identité coopérative hors domaine.")
    if not all(
        math.isfinite(v) for v in [message["sent_s"], message["expires_s"], now]
    ):
        raise ValueError("Date non finie.")
    if (
        not 0 <= message["sent_s"] <= now <= message["expires_s"]
        or message["expires_s"] <= message["sent_s"]
    ):
        raise ValueError("Message futur ou expiré.")
    if len(message["position"]) != 2 or not all(
        math.isfinite(v) for v in message["position"]
    ):
        raise ValueError("Position propre invalide.")
    if not isinstance(message["payload"], dict):
        raise ValueError("Contenu coopératif invalide.")
    fields_by_kind = dict(
        CLEAR_REQUEST=(
            {"space_wkt", "progress_m", "purpose", "movement_id",
             "required_wkt", "guard_reason"}
            if v3 else {"space_wkt", "progress_m"}
        ),
        PASSAGE_OFFER={"space_wkt", "initial_progress_m", "purpose"},
        PASSAGE_REPORT={
            "phase", "resource_status", "guard_reason", "policy_reason",
            "required_wkt", "progress_m", "purpose",
        },
        STATE={"speed_mps", "heading_rad", "length_m", "width_m", "progress_m"},
        INTENT={"speed_mps", "steering", "distance_m"},
        AGREE_HOLD={"until_s"},
        REFUSE={"reason"},
        CANCEL={"reason"},
        DONE={"reason"},
    )
    payload = message["payload"]
    if set(payload) != fields_by_kind[message["kind"]]:
        raise ValueError("Champs non recevables dans le contenu.")
    for key, value in payload.items():
        if key in {"space_wkt", "required_wkt"}:
            from shapely import wkt

            if not isinstance(value, str) or len(value) > MAX_SPACE_WKT_CHARS:
                raise ValueError("Espace demandé invalide.")
            polygon = wkt.loads(value)
            if polygon.is_empty or not polygon.is_valid or polygon.area > 200:
                raise ValueError("Espace demandé hors domaine.")
        elif key in {"reason", "guard_reason", "policy_reason"}:
            if not isinstance(value, str) or not 0 < len(value) < 256:
                raise ValueError("Motif invalide.")
        elif key == "phase":
            if not isinstance(value, str) or value not in {
                "BLOCKED", "ADMISSIBLE", "ENTERED", "CLEARED", "UNKNOWN",
                "ABANDONED",
            }:
                raise ValueError("Phase de passage invalide.")
        elif key == "resource_status":
            if not isinstance(value, str) or value not in {
                "CLEAR", "OCCUPIED", "UNKNOWN"
            }:
                raise ValueError("Occupation de ressource invalide.")
        elif key == "purpose":
            if not isinstance(value, str) or value not in {
                "NOMINAL_PASSAGE", "REVERSE_OPTION"
            }:
                raise ValueError("Objet de dégagement invalide.")
        elif key == "movement_id":
            if not isinstance(value, str) or not 0 < len(value) < 256:
                raise ValueError("Mouvement demandé invalide.")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError("Mesure de message invalide.")
    if (
        message["kind"] == "AGREE_HOLD"
        and not message["sent_s"] < payload["until_s"] <= message["expires_s"]
    ):
        raise ValueError("Engagement hors validité.")
    if message["kind"] == "STATE" and any(
        not 0 < payload[k] <= 30 for k in ["length_m", "width_m"]
    ):
        raise ValueError("Gabarit communiqué invalide.")
    if message["kind"] == "INTENT" and (
        abs(payload["steering"]) > 1 or payload["distance_m"] <= 0
    ):
        raise ValueError("Intention géométrique invalide.")
    return message


class Endpoint:
    """État d'un CAV : seuls les messages livrés rejoignent sa boîte."""

    def __init__(self, ego, lifetime_s=0.6):
        self.ego = ego
        self.lifetime = lifetime_s
        self.sequence = 0
        self.received = {}
        self.last_sequence = {}
        self.events = []
        self.committed_until_s = 0.0

    def send(self, kind, attempt, position, now, *, target=None, payload=None):
        self.sequence += 1
        message = dict(
            version=(
                "REC-V2V-3" if kind in {"PASSAGE_OFFER", "PASSAGE_REPORT"}
                or (kind == "CLEAR_REQUEST" and "purpose" in (payload or {}))
                else "REC-V2V-2" if kind == "CLEAR_REQUEST" else "REC-V2V-1"
            ),
            sender=self.ego,
            sequence=self.sequence,
            kind=kind,
            attempt=attempt,
            target=target,
            sent_s=now,
            expires_s=now + self.lifetime,
            position=list(position),
            payload=copy.deepcopy(payload or {}),
        )
        validate(message, now)
        if kind == "AGREE_HOLD":
            self.committed_until_s = max(
                self.committed_until_s, message["payload"]["until_s"]
            )
        self.events.append(dict(event="SENT", at_s=now, message=message))
        return message

    def receive(self, message, now):
        validate(message, now)
        sender = message["sender"]
        if sender == self.ego or message["target"] not in (None, self.ego):
            return False
        if message["sequence"] <= self.last_sequence.get(sender, 0):
            return False
        self.last_sequence[sender] = message["sequence"]
        key = (sender, message["kind"], message["attempt"])
        if message["kind"] in {"CANCEL", "DONE"}:
            self.received = {
                k: v
                for k, v in self.received.items()
                if not (k[0] == sender and k[2] == message["attempt"])
            }
        self.received[key] = copy.deepcopy(message)
        self.events.append(
            dict(event="RECEIVED", at_s=now, message=copy.deepcopy(message))
        )
        return True

    def view(self, now):
        expired = [k for k, v in self.received.items() if v["expires_s"] < now - 1e-7]
        for key in expired:
            self.events.append(dict(event="EXPIRED", at_s=now, identity=list(key)))
            del self.received[key]
        return copy.deepcopy(list(self.received.values()))


def deliver(messages, endpoints, positions, now, range_m):
    """Infrastructure de transport uniquement : aucune intention n'est décidée ici."""
    events = []
    for message in messages:
        validate(message, now)
        sender = message["sender"]
        if (
            sender not in positions
            or math.dist(message["position"], positions[sender]) > 1e-6
        ):
            raise ValueError("L'émetteur ne fournit pas sa position propre.")
        for recipient in sorted(endpoints):
            if recipient == sender or recipient not in positions:
                continue
            distance = math.dist(positions[sender], positions[recipient])
            if distance <= range_m and endpoints[recipient].receive(message, now):
                events.append(
                    dict(
                        sender=sender,
                        recipient=recipient,
                        distance_m=distance,
                        at_s=now,
                        sequence=message["sequence"],
                        kind=message["kind"],
                    )
                )
    return events
