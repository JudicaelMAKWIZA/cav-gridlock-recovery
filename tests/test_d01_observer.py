import copy
import json
import math
import pytest

from cav_recovery.d01.observer import EgoObserver, Config
from cav_recovery.d01.inputs import InputError
from d01_fixtures import packet, fixture_a, fixture_b, fixture_c


def run(rows, config=None):
    observer=EgoObserver(config)
    return observer,[observer.update(p) for p in rows]


def test_A_positive_at_30_4_only_and_no_pose_hold():
    rows=fixture_a();before=copy.deepcopy(rows);observer,out=run(rows)
    assert rows==before
    assert all(r["status"]!="CONFIRMED_OBSTRUCTION" for r in out[:-1])
    last=out[-1]
    assert last["status"]=="CONFIRMED_OBSTRUCTION" and last["emitted_at_s"]==30.4
    assert len(last["coverage"]["proof_times_s"])==76
    assert last["effect_window"]["samples"]==26
    assert last["coverage"]["effect_start_s"]==pytest.approx(.4)
    assert all(not r["current_CAV_occupancy_proven"] for r in out[1::2])
    assert all(r["status"] in ("SUSPECTED","UNKNOWN") for r in out[1::2])
    old=copy.deepcopy(last);after=observer.update(packet(30.6))
    assert after["status"]=="SUSPECTED" and len(after["historical_confirmations"])==1
    assert last==old
    json.dumps(last,allow_nan=False)


def test_B_transient_release_and_resume_are_separate():
    _,out=run(fixture_b())
    assert not any(r["status"]=="CONFIRMED_OBSTRUCTION" for r in out)
    four=next(r for r in out if r["evidence_through_s"]==4.)
    assert four["status"]=="NO_ALERT" and four["reason"]=="PROGRESSING"
    kinds={e["kind"] for e in four["events"]}
    assert {"MOVEMENT_RESUMED","RELEASE_OBSERVED"}<=kinds
    assert all(r["status"]=="NO_ALERT" for r in out if r["evidence_through_s"]>=4.)


def test_C_missing_appointment_expiry_and_censor_are_not_negative():
    observer,out=run(fixture_c())
    assert not any(r["status"]=="CONFIRMED_OBSTRUCTION" for r in out)
    assert all(r["status"]=="UNKNOWN" for r in out[2:])
    assert out[-1]["freshness"][0]["state"]=="expired"
    final=observer.finish(30.4)
    assert final["status"]=="UNKNOWN" and final["reason"]=="CENSORED"
    assert final["events"][-1]["kind"]=="CENSORED"


def test_missed_appointment_restarts_not_accumulates():
    rows=fixture_a()
    rows[50]["v2v"]["received"]=[] # t=10.4, no current CAV proof
    _,out=run(rows)
    assert out[50]["status"]=="UNKNOWN"
    assert not any(r["status"]=="CONFIRMED_OBSTRUCTION" for r in out)
    assert out[-1]["coverage"]["series_start_s"]==10.8
    assert any(e["kind"]=="OBSERVATION_LOST" for e in out[-1]["events"])


def test_successive_occupants_share_only_movement_clock():
    rows=fixture_a()
    for i,p in enumerate(rows):
        if i>=76:
            m=p["v2v"]["received"][0]["message"];m["sender_id"]="V2";m["message_id"]=f'V2:{m["sequence_number"]}'
    _,out=run(rows)
    assert out[-1]["status"]=="CONFIRMED_OBSTRUCTION"
    intervals=out[-1]["occupant_intervals"]
    assert [i["sender_id"] for i in intervals]==["V1","V2"]
    assert intervals[0]["end_s"]==15.2
    assert intervals[1]["start_s"]==15.6 and intervals[1]["proof_times_s"][0]==15.6


@pytest.mark.parametrize("missing",["v2v","lidar","both"])
def test_stop_or_brake_alone_never_confirms(missing):
    rows=fixture_a()
    for p in rows:
        p["v2v"]["received"]=[]
        if missing in ("lidar","both"):p["lidar"]=None
        if missing in ("v2v","both"):p["v2v"]=None
    _,out=run(rows)
    assert all(r["status"]!="CONFIRMED_OBSTRUCTION" for r in out)


def test_known_ordinary_wait_vs_unknown_context():
    p=packet();p["map"]["rule_context"]="ordinary_wait"
    assert EgoObserver().update(p)["reason"]=="ORDINARY_WAIT"
    p["map"]["rule_context"]="unknown"
    assert EgoObserver().update(p)["status"]=="UNKNOWN"


def test_guard_reserve_without_actual_body_intersection_is_not_proof():
    rows=[packet(round(.4+.2*n,7),other_x=11.) for n in range(151)]
    _,out=run(rows)
    assert out[-1]["status"]!="CONFIRMED_OBSTRUCTION"
    assert not out[-1]["current_CAV_occupancy_proven"]


def test_same_sequence_with_changed_payload_is_quarantined():
    observer=EgoObserver();observer.update(packet(.4))
    p=packet(.6);p["v2v"]["received"][0]["message"]["position_xy_m"]=[8.,0.]
    out=observer.update(p)
    assert out["status"]=="UNKNOWN" and out["reason"]=="CONTRADICTORY_CURRENT_EVIDENCE"


def test_mission_mutation_is_visible_and_no_age_inheritance_between_egos():
    observer=EgoObserver();observer.update(packet())
    p=packet(.6);p["mission"]["destination"]="north"
    assert observer.update(p)["reason"]=="MISSION_CHANGED"
    p=packet(.8);p["ego_id"]="new"
    with pytest.raises(InputError):observer.update(p)


def test_gap_and_backwards_times_are_not_hidden():
    observer=EgoObserver();observer.update(packet(.4))
    assert observer.update(packet(.8))["reason"]=="EGO_OBSERVATION_GAP"
    with pytest.raises(InputError):observer.update(packet(.6))


def test_prefix_future_invariance_order_and_no_file_access(monkeypatch):
    # Two independent local streams and hidden suffixes; no reader is admitted.
    rows=fixture_a()
    import builtins
    def forbidden(*a,**k):raise AssertionError("oracle/global access")
    monkeypatch.setattr(builtins,"open",forbidden)
    a=EgoObserver();b=EgoObserver()
    for row in rows[:80]:
        assert a.update(row)==b.update(dict(reversed(list(row.items()))))
    old=copy.deepcopy(a.last_output)
    suffix=copy.deepcopy(rows[80]);suffix["mission"]["demand_forward"]=False
    a.update(suffix);b.update(rows[80])
    assert old["evidence_through_s"]==16.2


def test_id_and_time_origin_invariance():
    a=fixture_a();b=copy.deepcopy(a);offset=10.2
    for p in b:
        p["ego_id"]="ego-renamed";p["decision_at_s"]+=offset
        for k in ("ego","mission","map","lidar","own_controls"):
            p[k]["observed_at_s"]+=offset;p[k]["available_at_s"]+=offset
        p["v2v"]["available_at_s"]+=offset
        for entry in p["v2v"]["received"]:
            entry["received_at_s"]+=offset
            m=entry["message"];m["sender_id"]="peer-renamed";m["message_id"]=f'peer-renamed:{m["sequence_number"]}'
            for k in ("observed_at_s","sent_at_s","valid_until_s"):m[k]+=offset
    _,x=run(a);_,y=run(b,Config(proof_origin_s=offset))
    assert [r["status"] for r in x]==[r["status"] for r in y]
    assert y[-1]["emitted_at_s"]-x[-1]["emitted_at_s"]==pytest.approx(offset)


def test_anonymous_rays_are_not_full_vehicle_contours_or_free_space():
    p=packet();p["v2v"]["received"]=[]
    out=EgoObserver().update(p)
    assert out["status"]=="SUSPECTED" and out["anonymous_obstacles"][0]["identity"] is None
    p["lidar"]["normalized_ranges"]=[1.]*16
    assert EgoObserver().update(p)["status"]=="UNKNOWN"


def test_invalid_route_and_overlap_abstain():
    p=packet();p["mission"]["own_route_xy_m"]=[[0.,0.],[1.,0.]]
    assert EgoObserver().update(p)["reason"]=="INCOMPLETE_OWN_MOVEMENT"
    p=packet(other_x=3.)
    assert EgoObserver().update(p)["reason"]=="CONTRADICTORY_CURRENT_EVIDENCE"


def test_dense_progressing_queue_and_low_speed_fraction_boundary():
    rows=[]
    for n in range(151):
        p=packet(round(.4+.2*n,7),ego_x=.2*n,other_x=.2*n+5,speed=1.)
        rows.append(p)
    _,out=run(rows)
    assert not any(r["status"]=="CONFIRMED_OBSTRUCTION" for r in out)
    rows=fixture_a()
    for p in rows:p["ego"]["signed_speed_mps"]=.15
    assert run(rows)[1][-1]["status"]!="CONFIRMED_OBSTRUCTION"


def test_incomplete_own_history_is_not_padded_with_future_or_other_ego():
    observer=EgoObserver();p=packet();observer.update(p)
    assert len(observer.history)==1 and observer.last_output["effect_window"] is None
    assert observer.last_output["action"] is None
