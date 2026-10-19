"""Causal K04 individual observer. No engine, files, oracle, graph or actions."""

from collections import deque, OrderedDict
from dataclasses import dataclass, asdict
import copy
import math

from .inputs import adapt, plain, message_age, InputError, EPS_TIME
from .geometry import MovementGeometry, GeometryError, body_evidence, anonymous_returns


@dataclass(frozen=True)
class Config:
    sample_s: float = .2
    proof_s: float = .4
    proof_origin_s: float = 0.
    effect_s: float = 5.
    progression_lengths: float = 1.
    low_speed_mps: float = .15
    low_speed_fraction: float = .8
    persistence_s: float = 30.
    movement_lengths: float = 1.2
    history_s: float = 35.

    def __post_init__(self):
        for key,value in asdict(self).items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f"invalid {key}")
            if key!='proof_origin_s' and value<=0:raise ValueError(f"invalid {key}")
        if self.sample_s!=.2 or self.proof_s!=.4 or self.low_speed_fraction>1:
            raise ValueError("K04 requires 5 Hz updates and 0.4 s proof appointments")
        if self.persistence_s<self.effect_s or self.history_s<self.persistence_s+self.effect_s:
            raise ValueError("history must cover persistence plus effect window")
        if abs(self.effect_s/.2-round(self.effect_s/.2))>EPS_TIME:
            raise ValueError("effect window must align with the 5 Hz observation grid")


class EgoObserver:
    """One independent instance per ego; every result is a detached snapshot."""
    def __init__(self, config=None):
        self.config=config or Config()
        self.geometry=MovementGeometry()
        self.history=deque(maxlen=math.ceil(self.config.history_s/.2)+1)
        self.proofs=deque(maxlen=math.ceil(self.config.history_s/.4)+1)
        self.events=deque(maxlen=256)
        self.confirmations=deque(maxlen=64)
        self.seen=OrderedDict()
        self.time_s=self.ego_id=self.mission_signature=None
        self.series_start=self.effect_start=self.anchor=None
        self.last_proof=None
        self.occupants=[]
        self.lost=False
        self.confirmed_at=None
        self.last_output=None
        self.resumed=False
        self.finished=False

    def _event(self, kind, t, **fields):
        event=dict(kind=kind,available_at_s=t,**fields)
        self.events.append(event)
        return event

    def _break(self,t,reason,kind="OBSERVATION_LOST"):
        if self.series_start is not None:
            closed=copy.deepcopy(self.occupants)
            for row in closed:
                if row["end_s"] is None:row["end_s"]=row["proof_times_s"][-1]
            self._event(kind,t,reason=reason,proof_start_s=self.series_start,last_covered_s=self.last_proof,
                        resource_bounds_xy_m=list(self.anchor.bounds),occupant_intervals=closed)
        self.series_start=self.effect_start=self.anchor=self.last_proof=None
        self.proofs.clear();self.occupants=[];self.confirmed_at=None
        self.lost=kind=="OBSERVATION_LOST"

    def _effect(self,t,length):
        start=t-self.config.effect_s
        rows=[row for row in self.history if row[0]>=start-EPS_TIME]
        expected=round(self.config.effect_s/.2)+1
        if len(rows)!=expected or abs(rows[0][0]-start)>EPS_TIME:
            return None
        if any(abs(b[0]-a[0]-.2)>EPS_TIME or b[1]<a[1]-1e-7 for a,b in zip(rows,rows[1:])):
            return None
        progress=rows[-1][1]-rows[0][1]
        low=sum(abs(row[2])<self.config.low_speed_mps for row in rows)/len(rows)
        return dict(start_s=start,end_s=t,samples=len(rows),progress_m=progress,low_speed_fraction=low,
                    qualifies=0<=progress<length*self.config.progression_lengths and low>=self.config.low_speed_fraction)

    def _output(self,data,status,reason,*,geometry=None,anonymous=(),current=(),freshness=(),effect=None):
        t=data["decision_at_s"]
        proof_rows=list(self.proofs)
        result=dict(schema_version="F01-diagnostic-1",contract_id="K04_APPROVED",experimental_thresholds=True,
            ego_id=self.ego_id,movement_id=data["mission"]["movement_id"],destination=data["mission"]["destination"],
            evidence_through_s=t,emitted_at_s=t,status=status,reason=reason,scope="individual_requested_movement",
            temporal_scope="sampled_0.4s",proof_origin_s=self.config.proof_origin_s,
            current_CAV_occupancy_proven=bool(current),current_occupants=plain(current),anonymous_obstacles=plain(anonymous),
            freshness=plain(freshness),effect_window=copy.deepcopy(effect),
            own_state=plain(data["ego"]),own_controls=plain(data["own_controls"]),
            own_effect_samples=[dict(observed_at_s=q,mission_progress_m=p,signed_speed_mps=v) for q,p,v in self.history],
            coverage=dict(series_start_s=self.series_start,last_covered_s=self.last_proof,
                effect_start_s=self.effect_start,proof_times_s=[r["time_s"] for r in proof_rows],
                appointments=copy.deepcopy(proof_rows),continuous_between_samples=False,
                intermediate_pose_hold=False,history_truncated=bool(proof_rows and self.series_start is not None and proof_rows[0]["time_s"]>self.series_start+EPS_TIME)),
            occupant_intervals=copy.deepcopy(self.occupants),historical_confirmations=copy.deepcopy(list(self.confirmations)),
            events=copy.deepcopy(list(self.events)),config=asdict(self.config),
            intervention_necessity="UNKNOWN",full_mission_impasse="NOT_ESTABLISHED",action=None)
        if geometry is not None:
            result["movement"]=dict(station_m=geometry["station_m"],distance_m=geometry["distance_m"],
                                    required_bounds_xy_m=list(geometry["required"].bounds))
        self.last_output=copy.deepcopy(result)
        return result

    def update(self, packet):
        # Always validate the public boundary; LocalInput cannot bypass it.
        if self.finished:raise InputError("observer is censored; use a new instance for another run")
        data=adapt(packet).data;t=data["decision_at_s"];ego=data["ego"];mission=data["mission"]
        if self.ego_id is not None and data["ego_id"]!=self.ego_id:
            raise InputError("one observer cannot consume another ego")
        if self.time_s is not None and t<=self.time_s+EPS_TIME:
            raise InputError("decision times must increase")
        previous_t=self.time_s;previous_progress=self.history[-1][1] if self.history else None
        self.time_s=t;self.ego_id=data["ego_id"]
        signature=(mission["version"],mission["origin"],mission["destination"],mission["own_route_xy_m"],mission["movement_id"])
        if self.mission_signature is None:self.mission_signature=signature
        if signature!=self.mission_signature:
            self._break(t,"MISSION_CHANGED");self._event("MISSION_CHANGED",t)
            return self._output(data,"UNKNOWN","MISSION_CHANGED")
        self.history.append((t,ego["mission_progress_m"],ego["signed_speed_mps"]))
        if previous_t is not None and abs(t-previous_t-.2)>EPS_TIME:
            self._break(t,"EGO_OBSERVATION_GAP")
            return self._output(data,"UNKNOWN","EGO_OBSERVATION_GAP")
        if abs(ego["observed_at_s"]-t)>EPS_TIME or abs(data["own_controls"]["observed_at_s"]-t)>EPS_TIME:
            self._break(t,"OLD_OWN_MEASUREMENT")
            return self._output(data,"UNKNOWN","OLD_OWN_MEASUREMENT")
        if ego["signed_speed_mps"]<-.001 or (previous_progress is not None and ego["mission_progress_m"]<previous_progress-1e-7):
            self._break(t,"NON_NOMINAL_REVERSE")
            return self._output(data,"UNKNOWN","NON_NOMINAL_REVERSE")
        if data["map"]["rule_context"]=="unknown":
            self._break(t,"UNKNOWN_RULE_CONTEXT")
            return self._output(data,"UNKNOWN","UNKNOWN_RULE_CONTEXT")
        if not mission["demand_forward"] or data["own_controls"]["requested_speed_mps"]<=0 or data["map"]["rule_context"]=="ordinary_wait":
            self._break(t,"ORDINARY_WAIT",kind="DEMAND_WITHDRAWN")
            return self._output(data,"NO_ALERT","ORDINARY_WAIT")
        try:
            geometry=self.geometry.build(data,self.config.movement_lengths)
        except GeometryError as exc:
            self._break(t,str(exc))
            return self._output(data,"UNKNOWN",str(exc))
        anonymous=anonymous_returns(data,geometry)
        freshness=[];current=[];overlaps=[];contradiction=False
        box=data["v2v"]
        for entry in (() if box is None else box["received"]):
            m=entry["message"];sender=m["sender_id"];sequence=m["sequence_number"]
            previous=self.seen.get(sender)
            fingerprint=plain(m)
            if previous and (sequence<previous[0] or (sequence==previous[0] and fingerprint!=previous[1])):
                contradiction=True
                freshness.append(dict(sender_id=sender,state="contradictory",message_id=m["message_id"]))
                continue
            self.seen[sender]=(sequence,fingerprint);self.seen.move_to_end(sender)
            if len(self.seen)>256:self.seen.popitem(last=False)
            age=message_age(entry,t)
            freshness.append(dict(sender_id=sender,state=age,message_id=m["message_id"],age_s=t-m["observed_at_s"],
                                  observed_at_s=m["observed_at_s"],received_at_s=entry["received_at_s"],valid_until_s=m["valid_until_s"]))
            if age!="current":continue
            try: evidence=body_evidence(m,geometry)
            except GeometryError:
                contradiction=True;continue
            if evidence is not None:
                row,overlap=evidence
                row.update(received_at_s=entry["received_at_s"],available_at_s=box["available_at_s"])
                current.append(row);overlaps.append(overlap)
        if box is not None:
            freshness.extend(dict(sender_id=e["sender_id"],message_id=e["message_id"],state="expired",valid_until_s=e["valid_until_s"]) for e in box["expired"])
        if contradiction:
            self._break(t,"CONTRADICTORY_CURRENT_EVIDENCE")
            return self._output(data,"UNKNOWN","CONTRADICTORY_CURRENT_EVIDENCE",geometry=geometry,anonymous=anonymous,freshness=freshness)
        appointment=abs((t-self.config.proof_origin_s)/.4-round((t-self.config.proof_origin_s)/.4))<EPS_TIME
        if appointment:
            if current:
                if self.anchor is not None and not any(p.intersection(self.anchor).area>0 for p in overlaps):
                    self._break(t,"RESOURCE_CHANGED",kind="RESOURCE_CHANGED")
                if self.series_start is None:
                    self.series_start=t;self.anchor=geometry["required"];self.lost=False
                    self._event("CONSTRAINT_OBSERVED",t,movement_id=mission["movement_id"])
                self.last_proof=t
                self.proofs.append(dict(time_s=t,occupants=plain(current)))
                ids={row["sender_id"] for row in current}
                for item in self.occupants:
                    if item["end_s"] is None and item["sender_id"] not in ids:item["end_s"]=item["proof_times_s"][-1]
                for sender in sorted(ids):
                    item=next((r for r in reversed(self.occupants) if r["sender_id"]==sender and r["end_s"] is None),None)
                    if item is None:
                        item=dict(sender_id=sender,start_s=t,end_s=None,proof_times_s=[]);self.occupants.append(item)
                    item["proof_times_s"].append(t)
                # Bound memory independently of total simulation duration.
                cutoff=t-self.config.history_s
                self.occupants=[r for r in self.occupants if r["end_s"] is None or r["end_s"]>=cutoff]
                for r in self.occupants:r["proof_times_s"]=[q for q in r["proof_times_s"] if q>=cutoff]
            else:
                # A fresh non-intersecting known occupant can establish its release.
                known={r["sender_id"] for r in self.occupants if r["end_s"] is None}
                fresh={r["sender_id"] for r in freshness if r["state"]=="current"}
                released=bool(known) and known<=fresh
                if self.series_start is not None:
                    self._break(t,"KNOWN_RELATION_RELEASED" if released else "MISSING_CURRENT_CAV_OCCUPANCY",
                                kind="RELEASE_OBSERVED" if released else "OBSERVATION_LOST")
        effect=self._effect(t,ego["length_m"])
        if self.series_start is not None and t-self.series_start>=self.config.effect_s-EPS_TIME:
            if effect and effect["qualifies"]:
                if self.effect_start is None:self.effect_start=effect["start_s"]
            else:self.effect_start=None
        progressed=previous_progress is not None and ego["mission_progress_m"]>previous_progress+1e-6
        if progressed and not self.resumed:
            self._event("MOVEMENT_RESUMED",t);self.resumed=True
        if not progressed:self.resumed=False
        if current and appointment and self.effect_start is not None and t-self.effect_start>=self.config.persistence_s-EPS_TIME:
            if self.confirmed_at is None:
                self.confirmed_at=t
                event=self._event("OBSTRUCTION_CONFIRMED",t,proof_start_s=self.series_start,effect_start_s=self.effect_start,
                                  temporal_scope="sampled_0.4s")
                self.confirmations.append(copy.deepcopy(event))
            status,reason="CONFIRMED_OBSTRUCTION","PERSISTENT_MOVEMENT_OBSTRUCTION"
        elif self.lost and not current:
            status,reason="UNKNOWN","INSUFFICIENT_INFORMATION"
        elif current or anonymous:
            status,reason="SUSPECTED","CANDIDATE_PHYSICAL_CONSTRAINT"
        elif progressed or abs(ego["signed_speed_mps"])>=self.config.low_speed_mps:
            status,reason="NO_ALERT","PROGRESSING"
        else:status,reason="UNKNOWN","STOPPED_UNEXPLAINED"
        return self._output(data,status,reason,geometry=geometry,anonymous=anonymous,current=current,freshness=freshness,effect=effect)

    def finish(self, at_s, reason="horizon"):
        if self.finished or self.last_output is None or not math.isfinite(at_s) or at_s<self.time_s-EPS_TIME:
            raise InputError("invalid censor time")
        had_series=self.series_start is not None
        self._break(at_s,reason,kind="CENSORED")
        if not had_series:self._event("CENSORED",at_s,reason=reason)
        result=copy.deepcopy(self.last_output)
        result.update(status="UNKNOWN",reason="CENSORED",emitted_at_s=at_s,current_CAV_occupancy_proven=False,
                      current_occupants=[],events=copy.deepcopy(list(self.events)),
                      freshness=[dict(row,state="not_current_at_censor") for row in result["freshness"]])
        self.finished=True
        self.last_output=copy.deepcopy(result)
        return result
