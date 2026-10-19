"""Scientific labels, native integrity, grouped splits and resumable production."""

from dataclasses import replace
import gzip
import json
import os
from pathlib import Path

import pytest

from cav_recovery.dataset.contract import Contract
from cav_recovery.dataset.generation import candidate,FAMILIES
from cav_recovery.dataset.qualify import qualify,POSITIVE,NEGATIVE,UNKNOWN


def evidence(duration=40, *, moving=False, held=False, blocked_until=100, links=(("a","b"),)):
    frames=[]
    ids={v for pair in links for v in pair}
    for tick in range(int(duration*5)+1):
        t=tick*.2
        states={key:dict(progress_m=t*.6 if moving else 0.,speed_mps=.6 if moving else 0.) for key in ids}
        frames.append(dict(time_s=t,states=states,witnesses=[dict(requester=a,blocker=b,c1_expected=not held,
            road_admissible=True,body_intersection=True,resource=dict(id=a+":"+b),
            alternatives="not_evaluated") for a,b in links] if t<=blocked_until else []))
    metadata=dict(vehicles={key:dict(length_m=4.515) for key in ids})
    return frames,metadata,dict(errors=[],complete_measurements=True,all_arrived=False)


@pytest.mark.parametrize("case,expected", [("persistent",POSITIVE),("short",NEGATIVE),("open",UNKNOWN),
    ("held",NEGATIVE),("moving_queue",NEGATIVE),("bad_physics",UNKNOWN),("stopped_free",NEGATIVE)])
def test_C1_C7_labels_do_not_require_a_cycle_and_do_not_label_waiting(case,expected):
    args={}
    if case=="short":args["blocked_until"]=3
    if case=="open":args["duration"]=10
    if case=="held":args["held"]=True
    if case=="moving_queue":args["moving"]=True
    frames,metadata,integrity=evidence(**args)
    if case=="bad_physics":integrity["errors"]=[dict(kind="collision")]
    if case=="stopped_free":
        for frame in frames:frame["witnesses"]=[]
    result=qualify(frames,metadata,integrity,Contract())
    assert result["label"]==expected
    if expected==POSITIVE:
        assert result["scopes"]==["LOCAL"]
        assert result["events"][0]["alternatives"]["other_maneuvers"]=="not_evaluated"
    if case=="bad_physics":assert not any(e["confirmed"] for e in result["events"])


def test_acyclic_chain_has_indirect_reference_with_overlapping_validity():
    frames,metadata,integrity=evidence(links=(("a","b"),("b","c")))
    result=qualify(frames,metadata,integrity,Contract())
    assert result["label"]==POSITIVE
    assert set(result["scopes"])=={"LOCAL","INDIRECT","COLLECTIVE"}
    assert any(d["requester"]=="a" and d["blocker"]=="c" for d in result["indirect_dependencies"])


def test_timestamp_error_is_unknown():
    frames,metadata,integrity=evidence()
    frames[2]["time_s"]=frames[1]["time_s"]
    assert qualify(frames,metadata,integrity,Contract())["label"]==UNKNOWN


def test_short_stop_followed_by_a_progressing_queue_is_not_positive():
    frames,metadata,integrity=evidence()
    for frame in frames:
        t=frame["time_s"]
        if t>8:
            frame["states"]["a"].update(progress_m=(t-8)*.6,speed_mps=.6)
    result=qualify(frames,metadata,integrity,Contract())
    assert result["label"]==NEGATIVE


def test_persistent_movement_event_records_confirmation_and_censoring():
    frames,metadata,integrity=evidence()
    result=qualify(frames,metadata,integrity,Contract())
    event=result["events"][0]
    assert event["confirmed"] and event["right_censored"]
    assert event["end_kind"]=="last_observed_right_censored"
    assert event["confirmation_at_s"]>=30
    assert result["dependencies"][0]["right_censored"]


def test_late_unresolved_constraint_is_unknown_instead_of_a_negative():
    frames,metadata,integrity=evidence()
    for frame in frames:
        if frame["time_s"]<38:frame["witnesses"]=[]
    result=qualify(frames,metadata,integrity,Contract())
    assert result["label"]==UNKNOWN
    assert result["events"][0]["right_censored"] and result["events"][0]["status"]=="open_candidate"


def test_ended_constraint_has_a_measured_release():
    frames,metadata,integrity=evidence(blocked_until=3)
    result=qualify(frames,metadata,integrity,Contract())
    assert not result["events"][0]["right_censored"]
    assert result["events"][0]["end_kind"]=="release_observed"


@pytest.mark.parametrize("parameters", [dict(persistence_s=0),dict(nominal_followup_s=2),dict(low_speed_fraction=2)])
def test_invalid_contract_rejected(parameters):
    with pytest.raises(ValueError):Contract(**parameters)


def test_generator_reproducibility_parent_grouping_and_prepared_missions():
    from cav_recovery.dataset.contract import digest
    physical=set()
    for index in range(80):
        spec=candidate(index)
        assert spec==candidate(index)
        assert spec["parent_id"]==candidate(index//4*4)["parent_id"]
        assert spec["id"]!=candidate(index,phase="calibration")["id"]
        assert all(a["target_speed_mps"]>0 for a in spec["agents"])
        assert spec["formation"].startswith("staggered")
        signature=digest({key:spec[key] for key in ("roads","agents","horizon_s")})
        assert signature not in physical
        physical.add(signature)


@pytest.fixture(scope="module")
def native_dataset(tmp_path_factory):
    pytest.importorskip("metadrive")
    from cav_recovery.dataset.pipeline import generate
    cached=os.environ.get("CGR_TEST_DATASET_ROOT")
    root=Path(cached) if cached else tmp_path_factory.mktemp("dataset")
    # Resume still verifies the complete source/contract identity. This optional
    # cache avoids repeating native fixtures during a platform migration.
    result=generate(root,count=16,phase="calibration",workers=2,max_seconds=1800,reserve_gib=0)
    assert result["episodes"]==16
    return root


def test_native_labels_integrity_and_no_imposed_wait_for_positives(native_dataset):
    from cav_recovery.dataset.pipeline import records,read_rows
    labels=set()
    for folder,row in records(native_dataset):
        labels.add(row["label"])
        assert not row["integrity"]["errors"]
        assert row["integrity"]["substeps"]>0
        if row["label"]==POSITIVE:
            assert all(c["requested_speed_mps"]>0 for frame in read_rows(folder/"commands.jsonl.gz") for c in frame["commands"].values())
    assert labels <= {POSITIVE,NEGATIVE,UNKNOWN}


def test_resume_keeps_complete_episodes_and_checks_source(native_dataset):
    from cav_recovery.dataset.pipeline import generate,records
    before={r["id"]:r for _,r in records(native_dataset)}
    result=generate(native_dataset,count=16,phase="calibration",workers=1,reserve_gib=0)
    assert result["episodes"]==16
    assert before=={r["id"]:r for _,r in records(native_dataset)}


def test_validate_recomputes_independent_annotations(native_dataset):
    from cav_recovery.dataset.export import validate
    assert validate(native_dataset,deep=True)["valid"]


def test_calibration_only_freezes_on_calibration_data(native_dataset):
    from cav_recovery.dataset.pipeline import calibrate
    result=calibrate(native_dataset)
    assert result["frozen"] and len(result["sensitivity"])==9
    assert result["contract_sha256"]==Contract().sha256


def test_production_requires_frozen_contract(tmp_path):
    from cav_recovery.dataset.pipeline import generate
    with pytest.raises(ValueError,match="contract"):
        generate(tmp_path,count=1)


def test_resource_stop_can_resume_and_preserves_interrupted_files(tmp_path):
    pytest.importorskip("metadrive")
    from cav_recovery.dataset.pipeline import generate
    root=tmp_path/"resume"
    result=generate(root,count=1,phase="calibration",reserve_gib=10**9)
    assert result["stop_reason"]=="disk_reserve"
    spec=candidate(0,phase="calibration",variants=1)
    partial=root/"episodes"/spec["id"];partial.mkdir();(partial/"partial.txt").write_text("interrupted")
    result=generate(root,count=1,phase="calibration",reserve_gib=0)
    assert result["episodes"]==1
    assert list((root/"interrupted").glob("*/partial.txt"))


def test_partition_before_windows_and_features_are_causal(tmp_path):
    pytest.importorskip("metadrive")
    from cav_recovery.dataset.pipeline import generate,write_json,read_rows
    from cav_recovery.dataset.export import split,export,validate
    contract=tmp_path/"frozen.json"
    write_json(contract,dict(frozen=True,contract=Contract().record(),contract_sha256=Contract().sha256))
    root=tmp_path/"production"
    generate(root,count=8,contract_file=contract,workers=2,reserve_gib=0)
    partition=split(root)
    assert all(len({partition["episodes"][candidate(i)["id"]] for i in range(start,start+4)})==1 for start in (0,4))
    output=tmp_path/"export"
    result=export(root,output)
    assert result["annotations_separate"] and result["windows"]>0
    for file in output.glob("*/features.jsonl.gz"):
        samples={(s["episode_id"],s["sample_index"]):s for s in read_rows(file)}
        for window in read_rows(file.with_name("windows.jsonl.gz")):
            selected=[samples[window["episode_id"],i] for i in window["sample_indices"]]
            assert all(s["time_s"]<=window["end_s"] for s in selected)
            assert all(s["vehicle_id"]==window["vehicle_id"] for s in selected)
            assert not any(key in s for s in selected for key in ("label","blocker","witnesses","guard_blockers"))
    assert validate(root)["valid"]


def test_cli_dataset_routes_without_sumo_runner(capsys):
    from cav_recovery.cli import main
    assert main(["dataset","sources"])==0
    assert "891225821" in capsys.readouterr().out


def test_related_geometry_and_whole_mechanisms_stay_out_of_training(tmp_path):
    from cav_recovery.dataset.pipeline import write_json
    from cav_recovery.dataset.export import split
    write_json(tmp_path/"campaign.json",dict(contract_sha256=Contract().sha256))
    specs=[candidate(i) for i in range(160)]
    for spec in specs:
        folder=tmp_path/"episodes"/spec["id"];folder.mkdir(parents=True)
        write_json(folder/"spec.json",spec)
        write_json(folder/"manifest.json",{})
        write_json(folder/"annotation.json",dict(id=spec["id"],parent_id=spec["parent_id"],
            phase="production",family=spec["family"],label=NEGATIVE))
    partitions=split(tmp_path)
    groups={}
    for spec in specs:
        group=partitions["episodes"][spec["id"]]
        groups.setdefault(spec["lineage_group"],set()).add(group)
        if spec["family"] in ("asymmetric",):
            assert group=="test_ood"
    assert all(len(values)==1 for values in groups.values())


@pytest.mark.parametrize("name", ["max_seconds","max_gib","reserve_gib"])
def test_nonfinite_resource_limits_rejected(name,tmp_path):
    from cav_recovery.dataset.pipeline import generate
    with pytest.raises(ValueError,match="finite"):
        generate(tmp_path,count=1,phase="calibration",**{name:float("nan")})


def test_dead_coordinator_does_not_overwrite_a_live_worker(tmp_path):
    import os
    from cav_recovery.dataset.pipeline import generate
    root=tmp_path/"locked";root.mkdir();locks=root/"worker-locks";locks.mkdir()
    (root/"campaign.lock").write_text("2147483647")
    worker=locks/"active.lock";worker.write_text(str(os.getpid()))
    with pytest.raises(ValueError,match="workers still active"):
        generate(root,count=1,phase="calibration",reserve_gib=0)
    assert worker.exists()


def test_requalification_preserves_raw_evidence_and_records_both_versions(native_dataset,tmp_path):
    from cav_recovery.dataset.pipeline import requalify,records
    from cav_recovery.dataset.export import validate
    revised=tmp_path/"revised"
    result=requalify(native_dataset,revised)
    assert result["episodes"]==16
    for folder,row in records(revised):
        original=native_dataset/"episodes"/row["id"]
        assert (folder/"physics.jsonl.gz").read_bytes()==(original/"physics.jsonl.gz").read_bytes()
        provenance=json.loads((folder/"manifest.json").read_text())
        assert provenance["measurement_code"] and provenance["qualification_code"]
    assert validate(revised,deep=True)["valid"]


def test_unknown_exports_are_separate_and_never_false_negative_targets(tmp_path):
    pytest.importorskip("metadrive")
    from cav_recovery.dataset.contract import digest
    from cav_recovery.dataset.pipeline import generate,execute_one,code_hashes,write_json,read_rows
    from cav_recovery.dataset.export import split,export
    contract=tmp_path/"frozen.json"
    write_json(contract,dict(frozen=True,contract=Contract().record(),contract_sha256=Contract().sha256))
    root=tmp_path/"raw"
    generate(root,count=1,contract_file=contract,workers=1,reserve_gib=0)
    short=candidate(12);short["horizon_s"]=10.
    short["id"]=digest({k:v for k,v in short.items() if k!="id"})[:24]
    result=execute_one((short,str(root),Contract().record(),code_hashes()))
    assert result["label"]==UNKNOWN
    split(root);output=tmp_path/"export";export(root,output,reserve_gib=0)
    targets=list(read_rows(output/"unknown/targets.jsonl.gz"))
    assert targets and all(row["event_obstructed"] is None and not row["supervision_available"] for row in targets)


def test_successive_occupants_preserve_movement_clock_without_stretching_pair_intervals():
    frames,metadata,integrity=evidence(links=(("a","b"),("a","c")))
    for frame in frames:
        t=frame['time_s']
        frame['witnesses']=[w for w in frame['witnesses'] if w['blocker']==('b' if t<20 else 'c')]
        for witness in frame['witnesses']:
            witness['movement']={'id':'left_turn'}
            witness['resource']={'id':'shared_centre'}
    result=qualify(frames,metadata,integrity,Contract())
    assert result['label']==POSITIVE and len(result['events'])==1
    event=result['events'][0]
    assert event['blockers']==['b','c'] and event['confirmation_at_s']==pytest.approx(30)
    intervals={d['blocker']:(d['start_s'],d['end_s']) for d in result['dependencies']}
    assert intervals['b'][1]<20 and intervals['c'][0]==20


def test_resource_change_and_real_evidence_gap_reset_movement_clock():
    for interruption in ('resource','gap'):
        frames,metadata,integrity=evidence()
        for frame in frames:
            t=frame['time_s']
            for witness in frame['witnesses']:
                witness['movement']={'id':'straight'}
                witness['resource']={'id':'first' if t<20 or interruption=='gap' else 'second'}
            if interruption=='gap' and 19.8<t<20.2:
                frame['witnesses']=[]
        result=qualify(frames,metadata,integrity,Contract())
        assert not any(e['confirmed'] for e in result['events'])


def test_simultaneous_blockers_do_not_double_temporal_samples():
    frames,metadata,integrity=evidence(links=(("a","b"),("a","c")))
    for frame in frames:
        for witness in frame['witnesses']:
            witness.update(movement={'id':'right'},resource={'id':'exit'})
    result=qualify(frames,metadata,integrity,Contract())
    assert len(result['events'])==1 and result['events'][0]['duration_s']==40
    assert result['events'][0]['confirmation_at_s']==pytest.approx(30)


def test_dense_topologies_dimensions_and_staggered_arrivals():
    specs=[candidate(i,phase='calibration',variants=1) for i in range(16)]
    assert {s['family'] for s in specs}==set(FAMILIES)
    assert {s['active_limit'] for s in specs}=={8,16,24,30}
    for spec in specs:
        assert spec['intersection_count']==1 and spec['topology']['lane_num'] in (2,3)
        assert 3.3<=spec['topology']['lane_width_m']<=3.7
        assert len(spec['agents'])>spec['active_limit']
        assert len({a['departure_s'] for a in spec['agents']})==len(spec['agents'])
        assert not any('hold_until_s' in a for a in spec['agents'])
        assert all(a['target_speed_mps']>=3.2 for a in spec['agents'])
        incoming={(r['start'],r['end'],r['lane']):r for r in spec['roads'] if r['kind']=='in'}
        outgoing={(r['start'],r['end'],r['lane']):r for r in spec['roads'] if r['kind']=='out'}
        for turn in (r for r in spec['roads'] if r['kind'] not in ('in','out')):
            origin=int(turn['start'].split('_')[0][1:]);destination=int(turn['end'].split('_')[0][1:])
            assert turn['points'][0]==incoming[(f'a{origin}_source',f'a{origin}_entry',turn['lane'])]['points'][-1]
            assert turn['points'][-1]==outgoing[(f'a{destination}_exit',f'a{destination}_sink',turn['lane'])]['points'][0]

def test_earlier_occupant_and_moving_tail_are_not_confirmed_dependencies():
    frames,metadata,integrity=evidence(duration=65,links=(("a","b"),("a","c")))
    for frame in frames:
        t=frame['time_s']
        frame['witnesses']=[w for w in frame['witnesses'] if w['blocker']==('b' if t<8 else 'c')]
        for w in frame['witnesses']:
            w.update(movement={'id':'turn'},resource={'id':'centre'})
        if t<10 or t>50:
            frame['states']['a'].update(progress_m=t*.6 if t<10 else 6+(t-50)*.6,speed_mps=.6)
        else:
            frame['states']['a'].update(progress_m=6,speed_mps=0.)
    result=qualify(frames,metadata,integrity,Contract())
    assert result['label']==POSITIVE
    assert all(d['relevance']=='candidate_not_confirmed' for d in result['dependencies'] if d['blocker']=='b')
    assert not any(d['relevance']=='confirmed_for_requested_movement' and d['end_s']>55 for d in result['dependencies'])


def test_native_admission_is_visible_in_same_time_physics_snapshot(native_dataset):
    from cav_recovery.dataset.pipeline import records,read_rows
    for folder,row in records(native_dataset):
        frames={f['time_s']:f for f in read_rows(folder/'physics.jsonl.gz')}
        assert len(frames)==row['integrity']['substeps']+1
        for event in read_rows(folder/'native_events.jsonl.gz'):
            if event.get('event')=='introduced':
                assert frames[event['time_s']]['states'][event['vehicle_id']]==event['initial_pose']

@pytest.mark.parametrize('heading',[0.,.6,1.57079632679,-1.2])
def test_forward_clearance_releases_rear_only_buffer_and_preserves_front_side_and_contact(heading):
    pytest.importorskip('metadrive')
    import math
    import numpy as np
    import shapely
    from types import SimpleNamespace
    from cav_recovery.dataset.native import forward_clearance_hits
    from cav_recovery.metadrive.observation import rectangle
    length,width=4.515,1.852
    def xy(x,y):return (x*math.cos(heading)-y*math.sin(heading),x*math.sin(heading)+y*math.cos(heading))
    positions=np.array([xy(s,0) for s in (0,.5,1.)]);headings=np.array([heading]*3)
    bodies=np.array([rectangle(p,heading,length,width) for p in positions],dtype=object)
    route=SimpleNamespace(grid_positions=positions,grid_headings=headings,guard_grid=shapely.buffer(bodies,.9))
    predicted={name:rectangle(xy(x,y),heading,length,width) for name,x,y in (
        ('rear',-5.2,0),('front',5.2,0),('side',0,2.3),('rear_contact',-4.,0))}
    assert shapely.intersects(route.guard_grid[0],predicted['rear'])
    hits=forward_clearance_hits(route,0,3,predicted,length)
    assert not hits[:,0].any()
    assert hits[0,1] and hits[0,2] and hits[0,3]
    physical=shapely.intersects(bodies[:,None],np.array(list(predicted.values()),dtype=object)[None,:])
    assert np.all(~physical | hits)


def test_forward_clearance_never_discards_an_actual_future_body_intersection():
    pytest.importorskip('metadrive')
    import random
    import numpy as np
    import shapely
    from types import SimpleNamespace
    from cav_recovery.dataset.native import forward_clearance_hits
    from cav_recovery.metadrive.observation import rectangle
    rng=random.Random(17);length,width=4.515,1.852
    positions=np.array([[0.,0.],[1.,.1],[2.,.4]]);headings=np.array([0.,.2,.4])
    bodies=np.array([rectangle(p,h,length,width) for p,h in zip(positions,headings)],dtype=object)
    route=SimpleNamespace(grid_positions=positions,grid_headings=headings,guard_grid=shapely.buffer(bodies,.7))
    predicted={str(i):rectangle((rng.uniform(-8,8),rng.uniform(-6,6)),rng.uniform(-3,3),length,width) for i in range(100)}
    # The moving rear envelope must also remain protected if it enters a body.
    predicted['closing_rear']=shapely.union_all([rectangle((-5.2,0),0,length,width),rectangle((-3.,0),0,length,width)]).convex_hull
    physical=shapely.intersects(bodies[:,None],np.array(list(predicted.values()),dtype=object)[None,:])
    hits=forward_clearance_hits(route,0,3,predicted,length)
    assert np.all(~physical | hits)
    assert hits[0,-1]
