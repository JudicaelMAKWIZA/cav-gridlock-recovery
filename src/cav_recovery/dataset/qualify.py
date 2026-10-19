"""C1-C7 references from measured footprints and motion, not guard/D01/C-RDG labels."""

import math

POSITIVE = "POSITIVE_CONFIRMED"
NEGATIVE = "NEGATIVE_CONFIRMED"
UNKNOWN = "UNKNOWN_OR_CENSORED"


def qualify(frames, metadata, integrity, contract):
    episodes, active, candidates = [], {}, []
    last_t = -1.
    for frame in frames:
        t = frame["time_s"]
        if t <= last_t:
            return dict(label=UNKNOWN, reason="invalid_timestamps", events=[], dependencies=[], scopes=[])
        last_t = t
        seen = set()
        grouped = {}
        for witness in frame["witnesses"]:
            ego, other = witness["requester"], witness["blocker"]
            movement = witness.get("movement", {}).get("id", "legacy_pair")
            resource = witness["resource"]["id"]
            key = (ego, movement, resource)
            if not witness["c1_expected"] or not witness["road_admissible"] or not witness["body_intersection"]:
                continue
            grouped.setdefault(key, []).append(witness)
        for key, witnesses in grouped.items():
            witness = witnesses[0]
            ego, other = witness["requester"], witness["blocker"]
            seen.add(key)
            state = frame["states"][ego]
            episode = active.setdefault(key, dict(start_s=t, last_s=t, requester=ego, blocker=other,
                movement_id=key[1], resource_id=key[2], occupant_intervals=[], open_occupants={}, mechanisms=set(),
                samples=[], evidence=witness, source="independent_native_body_sweep", causal_scope="necessary_space_for_declared_movement"))
            episode["last_s"] = t
            episode["samples"].append([t, state["progress_m"], state["speed_mps"]])
            episode["mechanisms"].update(w.get("mechanism","unspecified") for w in witnesses)
            occupants={w["blocker"]:w for w in witnesses}
            for blocker in list(episode["open_occupants"]):
                if blocker not in occupants:
                    episode["occupant_intervals"].append(episode["open_occupants"].pop(blocker))
            for blocker, proof in occupants.items():
                interval=episode["open_occupants"].setdefault(blocker,dict(blocker=blocker,start_s=t,end_s=t,proof=proof))
                interval["end_s"]=t
        for key in list(active):
            if key not in seen:
                episodes.append(active.pop(key))
    episodes += list(active.values())
    events = []
    for item in episodes:
        samples = item.pop("samples")
        item["occupant_intervals"] += list(item.pop("open_occupants").values())
        item["blockers"] = sorted({o["blocker"] for o in item["occupant_intervals"]})
        item["mechanisms"] = sorted(item["mechanisms"])
        duration = item["last_s"]-item["start_s"]
        length = metadata["vehicles"][item["requester"]]["length_m"]
        windows = []
        for i, (t, progress, speed) in enumerate(samples):
            if t-samples[0][0] < contract.persistence_s-1e-7:
                continue
            start = next((j for j in range(i+1) if samples[j][0] >= t-contract.persistence_s-1e-7), i)
            window = samples[start:i+1]
            low = sum(abs(r[2]) < contract.low_speed_mps for r in window)/len(window)
            if progress-window[0][1] < length*contract.progress_vehicle_lengths and low >= contract.low_speed_fraction:
                windows.append(t)
        persistent = bool(windows)
        followup = duration >= contract.nominal_followup_s-1e-7
        runs=[]
        for moment in windows:
            if not runs or moment-runs[-1][-1]>contract.sample_s+1e-7:
                runs.append([])
            runs[-1].append(moment)
        effect_duration=max((run[-1]-run[0]+contract.persistence_s for run in runs),default=0.)
        successful=next((run for run in runs if run[-1]-run[0]+contract.persistence_s>=contract.nominal_followup_s-1e-7),None)
        effects=[dict(start_s=max(item["start_s"],run[0]-contract.persistence_s),end_s=run[-1],
            confirmed=run[-1]-run[0]+contract.persistence_s>=contract.nominal_followup_s-1e-7,
            confirmation_at_s=run[0]+contract.nominal_followup_s-contract.persistence_s
                if run[-1]-run[0]+contract.persistence_s>=contract.nominal_followup_s-1e-7 else None)
            for run in runs]
        confirmed = persistent and followup and successful is not None
        censored = item["last_s"]>=last_t-contract.sample_s-1e-7 and (item["requester"],item["movement_id"],item["resource_id"]) in active
        open_candidate = (censored and not confirmed and abs(samples[-1][2])<contract.low_speed_mps
                          and (duration<contract.nominal_followup_s or (windows and windows[-1]>=last_t-contract.sample_s-1e-7)))
        item.update(end_s=item.pop("last_s"), duration_s=duration, persistent=persistent,
                    nominal_followup_observed=followup, confirmed=confirmed,
                    movement_effect_duration_s=effect_duration,
                    effect_intervals=effects,
                    confirmation_at_s=(successful[0]+contract.nominal_followup_s-contract.persistence_s) if confirmed else None,
                    right_censored=censored,end_kind="last_observed_right_censored" if censored else "release_observed",
                    status="persistent_obstruction" if confirmed else "open_candidate" if open_candidate else "resolved_or_short_constraint",
                    alternatives=dict(forward_declared="obstructed", other_maneuvers="not_evaluated", intervention_necessity="unknown"))
        events.append(item)
        if open_candidate:
            candidates.append(item)
    bad = bool(integrity.get("errors")) or not integrity.get("complete_measurements", False)
    if bad:
        for event in events:
            event["confirmed"] = False
            event["confirmation_at_s"] = None
            event["status"] = "invalid_or_incomplete_evidence"
    confirmed = [e for e in events if e["confirmed"]]
    sufficient = last_t >= contract.minimum_observation_s or integrity.get("all_arrived", False)
    if integrity.get("introduced"):
        sufficient = sufficient and not integrity.get("not_introduced") and all(
            key in integrity.get("arrivals",{}) or last_t-introduced>=contract.minimum_observation_s
            for key,introduced in integrity["introduced"].items())
    if bad:
        label, reason = UNKNOWN, "physical_or_technical_integrity_failure"
    elif confirmed:
        label, reason = POSITIVE, "independent_persistent_movement_constraint_C1_C7"
    elif not sufficient or candidates:
        label, reason = UNKNOWN, "insufficient_followup_or_open_candidate"
    else:
        label, reason = NEGATIVE, "no_qualifying_persistent_obstruction_in_verified_window"
    dependencies=[]
    for event in events:
        for occupant in event["occupant_intervals"]:
            effects=[effect for effect in event["effect_intervals"] if effect["confirmed"] and event["confirmed"]]
            cuts=sorted({occupant["start_s"],occupant["end_s"],*[max(occupant["start_s"],min(occupant["end_s"],x))
                for effect in effects for x in (effect["start_s"],effect["end_s"])]})
            parts=list(zip(cuts,cuts[1:])) or [(occupant["start_s"],occupant["end_s"])]
            for lo,hi in parts:
                effect=next((effect for effect in effects if effect["start_s"]<=(lo+hi)/2<=effect["end_s"]),None)
                censored=event["right_censored"] and hi>=last_t-contract.sample_s-1e-7
                dependencies.append(dict(requester=event["requester"],blocker=occupant["blocker"],resource=occupant["proof"]["resource"],
                    movement_id=event["movement_id"],start_s=lo,end_s=hi,relation="direct_physical_movement_constraint",
                    confirmation_at_s=effect["confirmation_at_s"] if effect and not bad else None,
                    right_censored=censored,end_kind="last_observed_right_censored" if censored else "observed_relation_interval_end",
                    relevance="confirmed_for_requested_movement" if effect and not bad else "candidate_not_confirmed",
                    movement_interval_start_s=event["start_s"],movement_interval_end_s=event["end_s"],
                    occupancy_start_s=occupant["start_s"],occupancy_end_s=occupant["end_s"],
                    proof=occupant["proof"],scope=event["causal_scope"]))
    scopes = ["LOCAL"] if confirmed and not bad else []
    # Indirect references require contemporaneous links, not just graph reachability across time.
    indirect = []
    relevant = [d for d in dependencies if d["relevance"]=="confirmed_for_requested_movement"]
    for a in relevant:
        for b in relevant:
            lo, hi = max(a["start_s"], b["start_s"]), min(a["end_s"], b["end_s"])
            if a["blocker"] == b["requester"] and a["requester"] != b["blocker"] and hi-lo >= contract.persistence_s:
                indirect.append(dict(requester=a["requester"], via=a["blocker"], blocker=b["blocker"],
                    start_s=lo, end_s=hi, proof="two_overlapping_confirmed_direct_constraints", relevance="indirect_spatial_dependency_not_unique_causal_root"))
    if indirect and not bad:
        scopes.append("INDIRECT")
    collective = any(a["requester"] != b["requester"] and
                     {a["requester"], a["blocker"]} & {b["requester"], b["blocker"]} and
                     min(a["end_s"], b["end_s"])-max(a["start_s"], b["start_s"]) >= contract.persistence_s
                     for a in relevant for b in relevant)
    if collective and not bad:
        scopes.append("COLLECTIVE")
    return dict(label=label, reason=reason, scopes=scopes, events=events, dependencies=dependencies,
                indirect_dependencies=indirect, observed_until_s=last_t, contract_sha256=contract.sha256,
                c6="expected_movement_required; moving_queues_and_short_resolved_constraints_excluded; changing_occupants_do_not_reset_movement_clock",
                observation_end_reason="integrity_failure" if bad else "all_missions_arrived" if integrity.get("all_arrived") else "horizon_reached",
                impossibility_of_all_alternatives="not_established", future_information_for_labels_only=True)
