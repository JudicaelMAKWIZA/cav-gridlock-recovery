"""Caractérise les obstructions observées, sans décider d'une intervention."""

from collections import Counter, defaultdict
from copy import deepcopy
import math
import xml.etree.ElementTree as ET


def _measurement(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else None


def service_cycles(path, lanes, movements):
    """Retrouve les cycles statiques ; un programme ambigu reste inconnu."""
    programs = defaultdict(list)
    for logic in ET.parse(path).findall("tlLogic"):
        duration = sum(float(phase.get("duration")) for phase in logic.findall("phase"))
        programs[logic.get("id")].append(duration if logic.get("type") == "static" and duration > 0 else None)
    controllers = defaultdict(set)
    for rows in movements.values():
        for movement in rows:
            if movement.get("tls"):
                controllers[lanes[movement["from_lane"]]["junction"]].add(movement["tls"])
    cycles = {}
    for junction, ids in sorted(controllers.items()):
        values = [programs[tls][0] if len(programs[tls]) == 1 else None for tls in sorted(ids)]
        cycles[junction] = max(values) if values and all(value is not None for value in values) else None
    return cycles


class ObstructionDetector:
    """Suit des cohortes par carrefour, pas des groupes causalement fermés.

    Une durée et une occupation ne suffisent pas à confirmer une cause.
    Le cycle sert à décrire la persistance et à vérifier une reprise durable.
    Aucune connexion SUMO n'est reçue ici.
    """

    def __init__(self, cycles, *, observation_interval_s=0.5, halting_speed=0.1):
        if (not math.isfinite(observation_interval_s) or observation_interval_s <= 0
                or not math.isfinite(halting_speed) or halting_speed <= 0
                or any(value is not None and (not math.isfinite(value) or value <= 0) for value in cycles.values())):
            raise ValueError("Les cadences, cycles et seuil de faible vitesse doivent être positifs.")
        self.cycles = dict(cycles)
        self.interval = observation_interval_s
        self.halting_speed = halting_speed
        self.active, self.finished = {}, []
        self.arrivals, self.counts = {}, Counter()
        self.last_time, self.sequence = None, 0

    @staticmethod
    def _public(episode):
        return deepcopy({key: value for key, value in episode.items() if not key.startswith("_")})

    def record_arrivals(self, arrivals):
        """Garde les dates validées sans inventer une observation du carrefour."""
        events = []
        for item in sorted(set(arrivals) - set(self.arrivals), key=lambda v: (arrivals[v], v)):
            time_s = arrivals[item]
            if not math.isfinite(time_s) or time_s < 0:
                raise ValueError("Une arrivée doit avoir une date valide.")
            self.arrivals[item] = time_s
            for episode in [*self.active.values(), *self.finished]:
                if item not in episode["members"]:
                    continue
                if "_released" in episode:
                    episode["_released"].add(item)
                episode["arrivals_s"][item] = time_s
                episode["member_observations"][item]["arrival_s"] = time_s
                events.append({"event": "member_arrived", "time_s": time_s,
                               "episode_id": episode["episode_id"], "vehicle_id": item,
                               "physical_resolution": "unknown", "last_physical_observation_s": episode["last_seen_s"]})
        return events

    def _start(self, junction, members, readings, time_s, previous):
        self.sequence += 1
        episode = {"episode_id": f"obstruction:{self.sequence:06d}", "junction": junction,
                   "members": sorted(members), "first_seen_s": time_s, "last_seen_s": time_s,
                   "start_interval_s": [previous, time_s], "left_censored": previous is None,
                   "observed_age_s": 0, "cycle_s": self.cycles.get(junction),
                   "state": "indeterminate", "physical_resolution": "unknown",
                   "right_censored": False, "end_s": None, "arrivals_s": {},
                   "member_observations": {item: self._member(readings[item], time_s) for item in sorted(members)},
                   "_baseline": {item: readings[item].get("distance") for item in members},
                   "_lengths": {item: readings[item].get("length") for item in members},
                   "_previous_distance": {},
                   "_advanced": set(), "_released": set(), "_lost": set(),
                   "_continuous": True, "_clear_since": None, "_basis": None, "_cause_since": None,
                   "_signature": None, "_visited_states": set()}
        self.active[junction] = episode
        self.counts["episodes"] += 1
        return episode

    @staticmethod
    def _member(row, time_s):
        return {"joined_s": time_s, "measurement_since_s": time_s,
                "baseline_distance_m": _measurement(row.get("distance")), "length_m": _measurement(row.get("length")),
                "first_progress_s": None, "first_progress_m": None,
                "left_passage_s": None, "arrival_s": None, "last_observed_s": time_s,
                "progress_m": 0 if _measurement(row.get("distance")) is not None else None,
                "step_progress_m": None}

    def observe(self, time_s, readings, snapshot, physical_evidence, lanes, *, footprints=None):
        """Réutilise les lectures et les preuves du suivi physique du même pas."""
        if not math.isfinite(time_s) or (self.last_time is not None and time_s <= self.last_time):
            raise ValueError("Les observations du détecteur doivent suivre le temps simulé.")
        previous = self.last_time
        gap = previous is not None and time_s - previous > self.interval + 1e-9
        self.last_time = time_s
        current = defaultdict(set)
        present = defaultdict(set)
        uncertain_junctions = set()
        for item, row in readings.items():
            body, unknown = (footprints or {}).get(item, ([], []))
            uncertain_junctions.update(lanes[lane]["junction"] for lane in unknown)
            locations = {lanes[row["lane"]]["junction"]}
            locations.update(lanes[lane]["junction"] for lane, _ in body)
            for junction in locations:
                present[junction].add(item)
            if row.get("speed") is not None and row["speed"] < self.halting_speed:
                for junction in locations:
                    current[junction].add(item)
        events = []
        for junction in sorted(set(current) | set(self.active)):
            members = current[junction]
            episode = self.active.get(junction)
            if episode is None:
                episode = self._start(junction, members, readings, time_s, previous)
                events.append({"event": "episode_started", "time_s": time_s, **self._public(episode)})
            if gap:
                episode["_continuous"] = False
                episode["_clear_since"] = episode["_cause_since"] = None
            for item in sorted(members - set(episode["members"])):
                episode["members"].append(item)
                episode["_baseline"][item] = readings[item].get("distance")
                episode["_lengths"][item] = readings[item].get("length")
                episode["member_observations"][item] = self._member(readings[item], time_s)
                events.append({"event": "member_added", "time_s": time_s,
                               "episode_id": episode["episode_id"], "vehicle_id": item})
            episode["members"].sort()
            for item in sorted(members & episode["_released"]):
                # Un véhicule peut revenir au même passage sur une mission de boucle.
                episode["_released"].remove(item)
                episode["_advanced"].discard(item)
                episode["_baseline"][item] = readings[item].get("distance")
                detail = self._member(readings[item], time_s)
                detail["joined_s"] = episode["member_observations"][item]["joined_s"]
                episode["member_observations"][item] = detail
                episode["_previous_distance"].pop(item, None)
                events.append({"event": "member_returned", "time_s": time_s,
                               "episode_id": episode["episode_id"], "vehicle_id": item})
            progress = {}
            advancing = []
            for item in episode["members"]:
                row, baseline = readings.get(item), episode["_baseline"][item]
                detail = episode["member_observations"][item]
                if row is None:
                    if item not in self.arrivals:
                        episode["_lost"].add(item)
                    progress[item] = None
                    detail.update(progress_m=None, step_progress_m=None)
                    episode["_previous_distance"].pop(item, None)
                    continue
                distance, length = row.get("distance"), episode["_lengths"][item]
                valid = (distance is not None and baseline is not None and length is not None
                         and all(math.isfinite(v) for v in (distance, baseline, length))
                         and distance >= baseline and length > 0)
                progress[item] = distance - baseline if valid else None
                old = episode["_previous_distance"].get(item)
                delta = (distance - old if valid and old is not None and not gap and distance >= old else None)
                detail.update(last_observed_s=time_s, progress_m=progress[item], step_progress_m=delta,
                              vehicle={"lane": row["lane"], "lane_position_m": _measurement(row.get("lane_position")),
                                       "speed_m_per_s": _measurement(row.get("speed"))})
                if delta is not None and delta > 0 and item in present[junction]:
                    advancing.append(item)
                if valid:
                    episode["_previous_distance"][item] = distance
                else:
                    episode["_previous_distance"].pop(item, None)
                if not valid:
                    episode["_continuous"] = False
                elif progress[item] >= length:
                    if item not in episode["_advanced"]:
                        detail.update(first_progress_s=time_s, first_progress_m=progress[item])
                        events.append({"event": "member_progress", "time_s": time_s,
                                       "episode_id": episode["episode_id"], "vehicle_id": item,
                                       "progress_m": progress[item], "measurement_since_s": detail["measurement_since_s"],
                                       "physical_resolution": "unknown"})
                    episode["_advanced"].add(item)
                    body, unknown = (footprints or {}).get(item, ([], []))
                    body_clear = (bool(body) and not unknown and
                                  all(lanes[lane]["junction"] != junction for lane, _ in body))
                    if not body and not unknown and row.get("lane_position") is not None:
                        body_clear = row["lane_position"] >= length and lanes[row["lane"]]["junction"] != junction
                    if body_clear:
                        if item not in episode["_released"]:
                            detail["left_passage_s"] = time_s
                        episode["_released"].add(item)
                    else:
                        episode["_released"].discard(item)
                        detail["left_passage_s"] = None
            evidence = {item: physical_evidence.get(item, {}) for item in sorted(members)}
            qualified = [{"source_vehicle_id": item, **edge} for item, proof in evidence.items() if proof.get("signal_permission") != "prohibited"
                         for edge in proof.get("dependencies", [])
                         if edge["edge_type"] in ("connection_obstacle", "waits_for")]
            occupied = []
            for observation in snapshot.get("junction_observations", []):
                if observation["vehicle_id"] not in members:
                    continue
                if evidence[observation["vehicle_id"]].get("signal_permission") == "prohibited":
                    continue
                occupied.extend({"ego_id": observation["vehicle_id"], **foe} for foe in observation["foes"]
                                if foe["state"] == "occupied" and foe["vehicle_id"] in readings)
            basis = tuple(sorted(edge["dependency_id"] for edge in qualified))
            if basis != episode["_basis"] or gap:
                episode["_basis"], episode["_cause_since"] = basis, time_s if basis else None
            cause = "qualified" if qualified else "occupation_only" if occupied else "insufficient"
            normal = bool(members) and all(proof.get("signal_permission") == "prohibited" for proof in evidence.values())
            cycle = episode["cycle_s"]
            persistent_members = sorted(item for item, proof in evidence.items()
                                        if cycle is not None and proof.get("low_progress_age_s") is not None
                                        and proof["low_progress_age_s"] >= cycle)
            persistent = (bool(persistent_members) and cycle is not None
                          and time_s - episode["first_seen_s"] >= cycle
                          and episode["_continuous"] and not episode["_lost"])
            persistent_cause = (cycle is not None and episode["_cause_since"] is not None
                                and time_s - episode["_cause_since"] >= cycle
                                and any(edge["source_vehicle_id"] in persistent_members for edge in qualified))
            clear = (set(episode["members"]) <= episode["_released"] and not members
                     and not qualified and not occupied and not episode["_lost"] and not gap
                     and junction not in uncertain_junctions)
            episode["_clear_since"] = (episode["_clear_since"] if episode["_clear_since"] is not None else time_s) if clear else None
            identified = sorted({edge["source_vehicle_id"] for edge in qualified})
            state = ("interrupted" if episode["_lost"] else
                     "resolution_pending" if clear else
                     "persistent_obstruction" if persistent and persistent_cause else
                     "persistent_immobility" if persistent else
                     "normal_wait" if normal else
                     "observed_constraint" if qualified else
                     "stopped" if members else "advancing" if advancing else "indeterminate")
            resolved = (clear and cycle is not None and time_s - episode["_clear_since"] >= cycle)
            if resolved:
                state = "resolved"
            episode.update(last_seen_s=time_s, observed_age_s=time_s - episode["first_seen_s"],
                           state=state, progress_m=progress, advanced_members=sorted(episode["_advanced"]),
                           remaining_members=sorted(set(episode["members"]) - episode["_released"]),
                           current_halted_members=sorted(members), cause_status=cause,
                           identified_members=identified,
                           signal_wait_members=sorted(item for item, proof in evidence.items()
                                                      if proof.get("signal_permission") == "prohibited"),
                           current_present_members=sorted(set(episode["members"]) & present[junction]),
                           advancing_members=sorted(advancing), progress_interval_s=[previous, time_s],
                           obstruction_verdict="verified" if qualified else "candidate" if occupied else "not_confirmed" if normal else "indeterminate",
                           qualified_cause_age_s=time_s - episode["_cause_since"] if episode["_cause_since"] is not None else None,
                           persistent=persistent, continuous_measurement=episode["_continuous"],
                           persistent_members=persistent_members,
                           partial_recovery=bool(episode["_advanced"] or episode["_released"]),
                           historical_progress_observed=bool(episode["_advanced"]),
                           actor_scope=("multiple_identified_vehicles" if len(identified) > 1 else
                                        "individual_identified_vehicle" if identified else "co_located_only"),
                           physical_resolution="verified" if resolved else "unknown",
                           evidence={"qualified_dependencies": qualified, "native_occupations": occupied,
                                     "footprint_uncertainty": junction in uncertain_junctions,
                                     "vehicles": evidence, "movement_lanes": sorted({readings[item]["lane"] for item in members}),
                                     "known_movements": sorted({(lanes[service["from_lane"]].get("edge"),
                                                                lanes[service["to_lane"]].get("edge"))
                                                               for proof in evidence.values() for service in proof.get("service", [])
                                                               if service["from_lane"] in lanes and service["to_lane"] in lanes
                                                               and lanes[service["from_lane"]].get("edge") is not None
                                                               and lanes[service["to_lane"]].get("edge") is not None}),
                                     "basis": "co_located_cohort_not_closed_causal_group"},
                           uncertainties=(["cause_not_fully_qualified"] if cause != "qualified" else []) +
                           (["service_cycle_unknown"] if cycle is None else []) +
                           (["observation_gap_or_invalid_progress"] if not episode["_continuous"] else []) +
                           ["material_alternatives_not_fully_verified", "no_intervention_decision", "no_gridlock_confirmation"])
            if resolved or state == "interrupted":
                episode["end_s"] = time_s
            signature = (state, cause, basis, tuple(episode["remaining_members"]), tuple(sorted(members)), tuple(persistent_members),
                         tuple(advancing), tuple(episode["advanced_members"]))
            if signature != episode["_signature"]:
                events.append({"event": "assessment_changed", "time_s": time_s, **self._public(episode)})
                self.counts["assessment_changes"] += 1
            if state not in episode["_visited_states"]:
                self.counts["episodes_with_" + state] += 1
                episode["_visited_states"].add(state)
            episode["observed_states"] = sorted(episode["_visited_states"])
            episode["_signature"] = signature
            if resolved or state == "interrupted":
                self.finished.append(self._public(episode))
                del self.active[junction]
        return events

    def snapshot(self):
        return {"time_s": self.last_time, "episodes": [self._public(e) for _, e in sorted(self.active.items())],
                "gridlock": "not_evaluated", "intervention": "not_evaluated"}

    def finish(self, time_s, status, *, complete_drain=False):
        """Une fin à l'horizon ne transforme pas une attente en permanence."""
        events = []
        for _, episode in sorted(self.active.items()):
            drained = status == "completed" and complete_drain and set(episode["members"]) <= set(self.arrivals)
            episode.update(right_censored=status == "horizon_reached", observation_end_s=time_s,
                           end_s=time_s if drained else None, run_status=status,
                           physical_resolution="verified" if drained else "unknown")
            if drained:
                episode["state"] = "resolved_by_complete_drain"
                episode["completion_proof"] = "all_missions_arrived_and_verified_no_future_or_present_traffic"
            self.counts["resolved_by_complete_drain" if drained else "right_censored" if status == "horizon_reached" else "unfinished"] += 1
            events.append({"event": "complete_drain_verified" if drained else
                                    "episode_censored" if status == "horizon_reached" else "observation_ended",
                           "time_s": time_s, **self._public(episode)})
            self.finished.append(self._public(episode))
        self.active.clear()
        return events

    def summary(self):
        return {"counts": dict(sorted(self.counts.items())), "episodes": deepcopy(self.finished),
                "counts_basis": "episodes_may_visit_multiple_states_assessment_changes_are_events",
                "last_physical_observation_s": self.last_time, "service_cycles_s": self.cycles,
                "calculation_interval_s": self.interval, "persistence_basis": "one_known_service_cycle",
                "resolution_basis": "cohort_left_passage_and_one_cycle_without_local_constraint",
                "causal_scope": "qualified_sources_only_not_entire_cohort",
                "state_basis": "current_observations_not_historical_progress",
                "progress_basis": "displacement_between_consecutive_observations_no_recency_threshold",
                "gridlock": "not_evaluated", "intervention": "not_evaluated"}
