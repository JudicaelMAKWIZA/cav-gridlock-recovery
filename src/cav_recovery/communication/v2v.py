"""Transport applicatif idéal, indépendant du moteur de simulation."""

import copy
import math


PROTOCOL_VERSION = 1


class V2VExchange:
    """Livre les états propres sans donner aux CAV l’état global du simulateur."""

    def __init__(self, *, range_m=120.0, interval_s=1.0, lifetime_s=2.0, step_s=0.5):
        for name, value in (
            ("range_m", range_m),
            ("interval_s", interval_s),
            ("lifetime_s", lifetime_s),
            ("step_s", step_s),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Paramètre V2V invalide : {name}.")
        if abs(interval_s / step_s - round(interval_s / step_s)) > 1e-9:
            raise ValueError(
                "La cadence V2V doit être un multiple du pas de simulation."
            )
        self.range_m = float(range_m)
        self.interval_s = float(interval_s)
        self.lifetime_s = float(lifetime_s)
        self.step_s = float(step_s)
        self.time_s = 0.0
        self.local: dict[str, dict] = {}
        self.previous_distance: dict[str, tuple[float, float]] = {}
        self.sequence: dict[str, int] = {}
        self.neighbors: dict[str, list[str]] = {}
        self.distances: dict[tuple[str, str], float] = {}
        self.latest_events: list[dict] = []
        self.recent_emissions: list[dict] = []
        self.max_active = 0
        self.max_pairs = 0
        self.counts = {
            "emitted": 0,
            "delivered": 0,
            "out_of_range_pairs": 0,
            "expired": 0,
            "duplicate_rejected": 0,
            "joined": 0,
            "left": 0,
        }

    def _own_state(self, item, row, time_s):
        """L’adaptateur fournit seulement l’état propre de cet émetteur."""
        raise NotImplementedError("Un adaptateur d’état propre est nécessaire.")

    def _connectivity(self):
        self.neighbors = {item: [] for item in sorted(self.local)}
        self.distances = {}
        items = sorted(self.local)
        for index, left in enumerate(items):
            a = self.local[left]["own"]["position_xy_m"]
            for right in items[index + 1 :]:
                b = self.local[right]["own"]["position_xy_m"]
                distance = math.dist(a, b)
                self.distances[left, right] = distance
                if distance <= self.range_m:
                    self.neighbors[left].append(right)
                    self.neighbors[right].append(left)

    def _deliver(self, recipient, message, time_s):
        state = self.local[recipient]
        sender = message["sender_id"]
        if message["sequence_number"] <= state["last_sequence"].get(sender, 0):
            self.counts["duplicate_rejected"] += 1
            return False
        if (
            not message["observed_at_s"]
            <= message["sent_at_s"]
            <= time_s
            <= message["valid_until_s"]
        ):
            raise ValueError("Message V2V reçu hors de sa période de validité.")
        state["last_sequence"][sender] = message["sequence_number"]
        state["received"][sender] = {"message": message, "received_at_s": time_s}
        state["expired"].pop(sender, None)
        self.counts["delivered"] += 1
        return True

    def observe(self, time_s, readings):
        """Après le pas de simulation, observe les présents puis livre sans latence simulée."""
        if not math.isfinite(time_s) or abs(time_s - self.time_s - self.step_s) > 1e-7:
            raise ValueError(
                "Le temps V2V doit suivre les pas de simulation sans avance ni retour."
            )
        self.time_s = time_s
        events = []
        active = set(readings)
        for item in sorted(set(self.local) - active):
            del self.local[item]
            self.previous_distance.pop(item, None)
            self.counts["left"] += 1
            events.append({"event": "left", "time_s": time_s, "vehicle_id": item})
        for item in sorted(active):
            if item not in self.local:
                self.local[item] = {
                    "own": None,
                    "received": {},
                    "expired": {},
                    "last_sequence": {},
                    "last_sent": None,
                }
                self.counts["joined"] += 1
                events.append({"event": "joined", "time_s": time_s, "vehicle_id": item})
            self.local[item]["own"] = self._own_state(item, readings[item], time_s)
            for sender, entry in sorted(list(self.local[item]["received"].items())):
                if time_s > entry["message"]["valid_until_s"]:
                    self.local[item]["expired"][sender] = {
                        "message_id": entry["message"]["message_id"],
                        "last_received_at_s": entry["received_at_s"],
                        "valid_until_s": entry["message"]["valid_until_s"],
                    }
                    del self.local[item]["received"][sender]
                    self.counts["expired"] += 1
                    events.append(
                        {
                            "event": "expired",
                            "time_s": time_s,
                            "vehicle_id": item,
                            "sender_id": sender,
                        }
                    )
        self._connectivity()
        self.max_active = max(self.max_active, len(self.local))
        self.max_pairs = max(
            self.max_pairs, sum(map(len, self.neighbors.values())) // 2
        )
        if abs(time_s / self.interval_s - round(time_s / self.interval_s)) < 1e-7:
            for item in sorted(active):
                self.sequence[item] = self.sequence.get(item, 0) + 1
                message = {
                    "protocol_version": PROTOCOL_VERSION,
                    "message_id": f"{item}:{self.sequence[item]}",
                    "sender_id": item,
                    "sequence_number": self.sequence[item],
                    "observed_at_s": time_s,
                    "sent_at_s": time_s,
                    "valid_until_s": time_s + self.lifetime_s,
                    **{
                        key: value
                        for key, value in self.local[item]["own"].items()
                        if key not in ("vehicle_id", "observed_at_s")
                    },
                }
                recipients = []
                for recipient in self.neighbors[item]:
                    if self._deliver(recipient, message, time_s):
                        recipients.append(recipient)
                self.local[item]["last_sent"] = {
                    "message_id": message["message_id"],
                    "sent_at_s": time_s,
                }
                self.counts["emitted"] += 1
                missed = len(active) - 1 - len(self.neighbors[item])
                self.counts["out_of_range_pairs"] += missed
                events.append(
                    {
                        "event": "emitted",
                        "time_s": time_s,
                        "message": message,
                        "delivered_to": recipients,
                        "out_of_range_count": missed,
                    }
                )
                self.recent_emissions.append(events[-1])
            self.recent_emissions = self.recent_emissions[-30:]
        self.latest_events = events
        return events

    def snapshot(self):
        """Donne à l'instrumentation sa vue globale, pas à un CAV."""
        return {
            "protocol_version": PROTOCOL_VERSION,
            "time_s": self.time_s,
            "range_m": self.range_m,
            "interval_s": self.interval_s,
            "lifetime_s": self.lifetime_s,
            "active_cav": len(self.local),
            "communicating_pairs": sum(map(len, self.neighbors.values())) // 2,
            "max_active_cav": self.max_active,
            "max_communicating_pairs": self.max_pairs,
            "counts": dict(self.counts),
            "participants": copy.deepcopy(self.local),
            "reachable_neighbors": copy.deepcopy(self.neighbors),
            "transport_distances_m": {
                f"{a}|{b}": value for (a, b), value in self.distances.items()
            },
            "latest_events": copy.deepcopy(self.latest_events),
            "recent_emissions": copy.deepcopy(self.recent_emissions),
        }

    def knowledge(self, vehicle_id):
        """Rend seulement l'état propre et les messages reçus par ce CAV."""
        if vehicle_id not in self.local:
            return None
        return copy.deepcopy(self.local[vehicle_id])
