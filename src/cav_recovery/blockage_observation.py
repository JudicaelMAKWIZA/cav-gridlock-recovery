"""Suit les attentes et leur évolution physique, sans diagnostiquer un gridlock."""

from collections import Counter, defaultdict
import math

from .crdg import junction_movements


def _number(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else None


def _position(item, readings):
    row = readings.get(item)
    if row is None:
        return None
    return {"vehicle_id": item, "lane": row.get("lane"), "edge": row.get("road_id"),
            "lane_position_m": _number(row.get("lane_position")),
            "distance_m": _number(row.get("distance")), "speed_m_per_s": _number(row.get("speed"))}


class BlockageObservation:
    """Garde des épisodes de faible progression et leurs preuves.

    Un arrêt lance le suivi. Une longueur de véhicule parcourue termine cet
    épisode, sans prouver que toute la file est libérée. Un changement de cause
    ne remet pas l'âge physique à zéro. Aucune connexion SUMO n'est reçue ici.
    """

    def __init__(self, *, halting_speed=0.1):
        if not math.isfinite(halting_speed) or halting_speed <= 0:
            raise ValueError("Le seuil de faible vitesse doit être positif.")
        self.halting_speed = halting_speed
        self.active = {}
        self.groups = {}
        self.last_episode = {}
        self.counts = Counter()
        self.sequence = 0
        self.last_time = None
        self.longest = None

    def _start(self, item, row, time_s, previous_time):
        self.sequence += 1
        distance = _number(row.get("distance"))
        length = _number(row.get("length"))
        episode = {"episode_id": f"vehicle:{self.sequence:06d}", "vehicle_id": item,
                   "first_seen_s": time_s, "last_seen_s": time_s, "observed_age_s": 0,
                   "start_interval_s": [previous_time, time_s], "left_censored": previous_time is None,
                   "end_s": None, "right_censored": False, "state": "open",
                   "first_distance_m": distance, "last_distance_m": distance,
                   "progress_m": 0 if distance is not None else None,
                   "significant_progress_distance_m": length if length else None,
                   "progress_measurement_since_s": time_s if distance is not None else None,
                   "without_vehicle_length_progress_s": 0 if distance is not None and length else None,
                   "cause_changes": 0, "max_dependency_age_s": None,
                   "_causes": None, "_passage": None, "_motion": None,
                   "_participants": {}, "_measurement_continuous": distance is not None}
        self.active[item] = episode
        self.last_episode[item] = episode["episode_id"]
        self.counts["vehicle_episodes"] += 1
        return episode

    @staticmethod
    def _public(episode):
        return {key: value for key, value in episode.items() if not key.startswith("_")}

    def _finish(self, item, time_s, reason, state="ended"):
        row = self.active.pop(item)
        previous = row["last_seen_s"] if time_s > row["last_seen_s"] else row.get("_previous_seen_s", row["last_seen_s"])
        row.update(state=state, end_reason=reason, right_censored=state == "right_censored",
                   end_s=time_s if state == "ended" else None, observation_end_s=time_s,
                   end_interval_s=[previous, time_s] if state == "ended" else None,
                   last_causes=row["_causes"])
        self.counts[state] += 1
        self.counts[reason] += 1
        if self.longest is None or row["observed_age_s"] > self.longest["observed_age_s"]:
            self.longest = self._public(row)
        return {"event": "episode_censored" if state == "right_censored" else
                         "observation_interrupted" if state == "interrupted" else "situation_ended",
                "scope": "vehicle", "time_s": time_s, **self._public(row)}

    def _evidence(self, item, row, episode, snapshot, nodes, outgoing, receiving, lanes, movements, readings):
        source = "vehicle:" + item
        dependencies = outgoing.get(source, [])
        causes = [edge["dependency_id"] for edge in dependencies]
        reason = nodes.get(source, {}).get("waiting_reason", snapshot["waiting_states"].get(item, "unknown"))
        if not causes:
            causes = [reason]
        internal = lanes[row["lane"]]["internal"]
        service = []
        for connection in junction_movements(row, lanes, movements):
            state = None
            tls = connection.get("tls")
            if tls:
                states = snapshot.get("tls_states", {}).get(tls)
                index = connection["link_index"]
                state = states[index] if states is not None and 0 <= index < len(states) else "unknown"
            service.append({"from_lane": connection["from_lane"], "to_lane": connection["lane"],
                            "via_lane": connection.get("via_lane"), "tls": tls,
                            "signal_state": None if internal else state, "entry_signal_state": state if internal else None})
        # Le feu d'entrée ne commande pas un véhicule déjà dans le carrefour.
        permission = ("not_applicable" if internal else
                      "permitted" if any(r["signal_state"] in (None, "G", "g") for r in service) else
                      "prohibited" if service and all(r["signal_state"] in ("r", "y", "u") for r in service) else "unknown")
        lane_states = receiving.get(item, {})
        # Un espace libre et une continuation légale ne prouvent pas la manœuvre.
        passage = "constrained" if dependencies or permission == "prohibited" else "unknown"
        participants = set()
        resources = []
        for edge in dependencies:
            target = edge["target"]
            if target.startswith("vehicle:"):
                participants.add(target.removeprefix("vehicle:"))
            elif target in nodes:
                resource = nodes[target]
                resources.append({key: resource[key] for key in
                                  ("id", "resource_type", "release_mode", "candidate_lanes", "lane_states",
                                   "required_space_m", "via_lane", "foe_lanes") if key in resource})
                participants.update(e["target"].removeprefix("vehicle:") for e in outgoing.get(target, [])
                                    if e["target"].startswith("vehicle:"))
        related = []
        for participant in sorted(participants | episode["_participants"].keys()):
            position = _position(participant, readings)
            if position is None:
                related.append({"vehicle_id": participant, "observation": "unknown",
                                "currently_related": participant in participants})
                episode["_participants"].pop(participant, None)
                continue
            first = episode["_participants"].setdefault(participant, {"first": position["distance_m"],
                                                                     "reported": position["distance_m"]})
            distance = position["distance_m"]
            progress = distance - first["first"] if distance is not None and first["first"] is not None and distance >= first["first"] else None
            length = _number(readings[participant].get("length"))
            advanced = bool(length and distance is not None and first["reported"] is not None
                            and distance - first["reported"] >= length)
            related.append({**position, "currently_related": participant in participants,
                            "progress_since_first_related_observation_m": progress,
                            "vehicle_length_progress_observed": advanced})
            if advanced:
                first["reported"] = distance
            if participant not in participants:
                episode["_participants"].pop(participant, None)
        return {"causes": sorted(causes), "waiting_reason": reason, "signal_permission": permission,
                "passage_state": passage, "passage_basis": "observed_constraints" if passage == "constrained" else
                "material_access_not_verified", "service": service, "receiving_lane_states": lane_states,
                "distance_to_lane_end_m": max(0, lanes[row["lane"]]["length_m"] - row["lane_position"]),
                "dependencies": [{key: edge[key] for key in
                                  ("dependency_id", "edge_type", "target", "evidence", "dependency_first_seen_s",
                                   "dependency_last_seen_s", "dependency_age_s", "signal_state", "ego_response",
                                   "foe_response", "link_state", "link_has_priority", "link_is_open", "link_has_foe") if key in edge}
                                 for edge in dependencies],
                "resources": resources, "vehicle": _position(item, readings), "related_vehicles": related}

    def observe(self, time_s, readings, snapshot, lanes, movements, *, arrivals=(), dependency_events=()):
        """Consomme le calcul C-RDG déjà fait et les lectures physiques du même pas."""
        if not math.isfinite(time_s) or (self.last_time is not None and time_s <= self.last_time):
            raise ValueError("Les observations doivent suivre le temps simulé.")
        previous_time = self.last_time
        self.last_time = time_s
        self.counts["observations"] += 1
        events = []
        observed_episodes = self.last_episode.copy()
        nodes = {row["id"]: row for row in snapshot["nodes"]}
        outgoing = defaultdict(list)
        for edge in snapshot["edges"]:
            outgoing[edge["source"]].append(edge)
        receiving = {r["vehicle_id"]: r["lane_states"] for r in snapshot.get("receiving_observations", [])}
        arrivals = set(arrivals)
        for item in sorted(set(self.active) - set(readings)):
            events.append(self._finish(item, time_s, "arrival_observed" if item in arrivals else
                                       "missing_observation", "ended" if item in arrivals else "interrupted"))
        for item in sorted(arrivals & self.last_episode.keys()):
            events.append({"event": "arrival_observed", "scope": "vehicle", "time_s": time_s,
                           "vehicle_id": item, "episode_id": self.last_episode[item]})
            self.counts["tracked_vehicles_arrived"] += 1
            del self.last_episode[item]
        for item in sorted(readings):
            row = readings[item]
            speed = _number(row.get("speed"))
            if item not in self.active and (speed is None or speed >= self.halting_speed):
                continue
            started = item not in self.active
            episode = self._start(item, row, time_s, previous_time) if started else self.active[item]
            old_distance = episode["last_distance_m"]
            distance = _number(row.get("distance"))
            if speed is None or (old_distance is not None and distance is not None and distance < old_distance):
                events.append(self._finish(item, time_s, "invalid_physical_observation", "interrupted"))
                continue
            episode.update(_previous_seen_s=episode["last_seen_s"],
                           last_seen_s=time_s, observed_age_s=time_s - episode["first_seen_s"],
                           last_distance_m=distance)
            if distance is None:
                episode["_measurement_continuous"] = False
                episode["without_vehicle_length_progress_s"] = None
            elif not episode["_measurement_continuous"]:
                episode["progress_measurement_since_s"] = time_s
                episode["_measurement_continuous"] = True
                episode["_measurement_baseline_m"] = distance
            first_distance = episode["first_distance_m"]
            episode["progress_m"] = distance - first_distance if distance is not None and first_distance is not None else None
            length = episode["significant_progress_distance_m"]
            baseline = episode.get("_measurement_baseline_m", first_distance)
            gain = distance - baseline if distance is not None and baseline is not None else None
            progressed = length is not None and gain is not None and gain >= length
            if distance is not None and length is not None:
                episode["without_vehicle_length_progress_s"] = time_s - episode["progress_measurement_since_s"]
            evidence = self._evidence(item, row, episode, snapshot, nodes, outgoing, receiving, lanes, movements, readings)
            ages = [r["dependency_age_s"] for r in evidence["dependencies"] if "dependency_age_s" in r]
            if ages:
                episode["max_dependency_age_s"] = max(ages + [episode["max_dependency_age_s"] or 0])
            cause = evidence["causes"]
            if progressed:
                episode["without_vehicle_length_progress_s"] = 0
                evidence.update(passage_state="progress_observed", passage_basis="vehicle_length_displacement")
            passage = (evidence["signal_permission"], evidence["passage_state"],
                       tuple(sorted(evidence["receiving_lane_states"].items())),
                       tuple((r["from_lane"], r["to_lane"], r["via_lane"], r["tls"], r["signal_state"])
                             for r in evidence["service"]))
            if passage[0] == "permitted" and (episode["_passage"] is None or episode["_passage"][0] != "permitted"):
                self.counts["nominal_permitted_intervals"] += 1
            motion = speed >= self.halting_speed
            changed = []
            if started:
                changed.append("situation_started")
            else:
                if cause != episode["_causes"]:
                    changed.append("cause_changed")
                    episode["cause_changes"] += 1
                if passage != episode["_passage"]:
                    changed.append("passage_changed")
                if motion != episode["_motion"]:
                    changed.append("motion_observed" if motion else "low_speed_observed")
            if distance is None and old_distance is not None:
                changed.append("progress_measurement_unavailable")
            if progressed:
                changed.append("progress_observed")
            if any(r.get("vehicle_length_progress_observed") for r in evidence["related_vehicles"]):
                changed.append("related_progress_observed")
            for name in changed:
                events.append({"event": name, "scope": "vehicle", "time_s": time_s,
                               "episode_id": episode["episode_id"], "vehicle_id": item,
                               "observed_age_s": episode["observed_age_s"], "progress_m": episode["progress_m"],
                               "previous_causes": episode["_causes"] if name == "cause_changed" else None,
                               "evidence": evidence})
                self.counts[name] += 1
            episode.update(_causes=cause, _passage=passage, _motion=motion)
            if progressed:
                events.append(self._finish(item, time_s, "vehicle_length_progress"))
        # Les changements natifs restent datés, sans leur prêter une résolution physique.
        observed_episodes.update(self.last_episode)
        for event in dependency_events:
            item = event.get("source", "").removeprefix("vehicle:")
            if item in observed_episodes and event["event"] == "disappeared":
                events.append({"event": "dependency_no_longer_represented", "scope": "vehicle", "time_s": time_s,
                               "episode_id": observed_episodes[item], "dependency": event})
        events.extend(self._observe_groups(time_s, readings, snapshot))
        return events

    def _observe_groups(self, time_s, readings, snapshot):
        events, current = [], set()
        candidates = {tuple(sorted(candidate["nodes"])): kind
                      for kind in ("cycle_candidates", "closed_cycle_candidates") for candidate in snapshot[kind]}
        for key, kind in sorted(candidates.items()):
            members = sorted(n.removeprefix("vehicle:") for n in key if n.startswith("vehicle:"))
            if not all(item in readings for item in members):
                continue
            current.add(key)
            if key not in self.groups:
                self.sequence += 1
                self.groups[key] = {"episode_id": f"group:{self.sequence:06d}", "kind": kind,
                                    "nodes": list(key), "members": members, "first_seen_s": time_s,
                                    "last_seen_s": time_s, "observed_age_s": 0, "first_distances_m":
                                    {item: _number(readings[item].get("distance")) for item in members}}
                self.counts["group_episodes"] += 1
                events.append({"event": "candidate_group_started", "scope": "group", "time_s": time_s,
                               **self.groups[key], "vehicles": [_position(item, readings) for item in members],
                               "edges": [edge for edge in snapshot["edges"]
                                         if edge["source"] in key and edge["target"] in key]})
            group = self.groups[key]
            if group["kind"] != kind:
                events.append({"event": "candidate_kind_changed", "scope": "group", "time_s": time_s,
                               "episode_id": group["episode_id"], "previous_kind": group["kind"], "kind": kind})
            group.update(kind=kind, last_seen_s=time_s, observed_age_s=time_s - group["first_seen_s"])
            group["progress_m"] = {item: distance - group["first_distances_m"][item]
                                   if distance is not None and group["first_distances_m"][item] is not None
                                   and distance >= group["first_distances_m"][item] else None
                                   for item in members for distance in [_number(readings[item].get("distance"))]}
            reported = group.setdefault("_reported_progress_m", {item: 0 for item in members})
            advanced = [item for item in members if group["progress_m"][item] is not None
                        and _number(readings[item].get("length"))
                        and group["progress_m"][item] - reported[item] >= readings[item]["length"]]
            if advanced:
                events.append({"event": "candidate_member_progress", "scope": "group", "time_s": time_s,
                               "advanced_members": advanced, **self._public(group)})
                for item in advanced:
                    reported[item] = group["progress_m"][item]
        for key in sorted(set(self.groups) - current):
            group = self.groups.pop(key)
            missing = key in candidates
            events.append({"event": "candidate_group_no_longer_observed", "scope": "group", "time_s": time_s,
                           "reason": "member_observation_missing" if missing else "current_structure_absent",
                           "physical_resolution": "unknown", **self._public(group)})
        return events

    def finish(self, time_s, status):
        """Censure les observations encore ouvertes ; l'horizon ne prouve aucune permanence."""
        events = []
        for item in sorted(self.active):
            events.append(self._finish(item, time_s, "horizon_reached" if status == "horizon_reached" else
                                       "run_ended_without_resolution", "right_censored" if status == "horizon_reached"
                                       else "interrupted"))
        for key in sorted(self.groups):
            group = self.groups[key]
            events.append({"event": "candidate_group_observation_ended", "scope": "group", "time_s": time_s,
                           "end_s": None, "right_censored": status == "horizon_reached", "run_status": status,
                           **self._public(group)})
        self.groups.clear()
        return events

    def summary(self):
        return {"counts": dict(sorted(self.counts.items())), "last_observation_s": self.last_time,
                "progress_basis": "one_observed_vehicle_length", "halting_speed_m_per_s": self.halting_speed,
                "passage_states": ["progress_observed", "constrained", "unknown"],
                "longest_vehicle_episode": self.longest, "gridlock": "not_evaluated"}
