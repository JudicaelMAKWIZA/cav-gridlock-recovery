"""Optional passive Windows integration and small reproducible smoke experiments.

Run with python -m cav_recovery.d01.native. Observer is OFF unless --observer.
Historical MetaDrive/dataset modules are imported, never patched or rewritten.
Global integrity instrumentation stays in this harness, outside the D01 input.
"""

import argparse
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def project_mailbox(knowledge, time_s):
    """Project only the ego's actual received/expired entries, never snapshot()."""
    if knowledge is None:return None
    return dict(available_at_s=time_s,
        received=[copy.deepcopy(row) for _,row in sorted(knowledge["received"].items())],
        expired=[dict(sender_id=sender,**copy.deepcopy(row)) for sender,row in sorted(knowledge["expired"].items())])


def local_packet(key, time_s, measurement, mission, map_context, rays, controls, knowledge):
    """Pure composition: no environment, native vehicle, neighbour list or files.

    measurement is the selected ego's scalar state, not an all-actor record.
    mission/map are whitelisted static projections known at admission.
    """
    return dict(schema_version="F01-input-1",contract_id="K04_APPROVED",ego_id=key,decision_at_s=time_s,
        ego=dict(observed_at_s=time_s,available_at_s=time_s,**copy.deepcopy(measurement)),
        mission=copy.deepcopy(mission),map=copy.deepcopy(map_context),
        lidar=dict(observed_at_s=time_s,available_at_s=time_s,source="native_lidar",num_lasers=16,
                   range_m=30.,height_m=1.2,phase_offset_rad=0.,normalized_ranges=list(rays),missing=False),
        own_controls=dict(observed_at_s=time_s,available_at_s=time_s,**copy.deepcopy(controls)),
        v2v=project_mailbox(knowledge,time_s))


def _write(stream,value):
    stream.write(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")


def _percentile(values, q):
    values=sorted(values)
    if not values:return None
    index=(len(values)-1)*q;lo=math.floor(index);hi=math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def _source_fingerprints():
    from ..dataset.pipeline import code_hashes
    return dict(historical=code_hashes(),d01={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(Path(__file__).parent.glob("*.py"))})


def smoke_spec(count,index):
    """Small native validation workload, not a production campaign or dataset.

    Reuse an existing single-junction geometry and native admissions. Vehicles
    start on separate unoccupied approach slots; no pose changes after creation.
    """
    from ..dataset.generation import candidate
    spec=candidate(index)
    spec["geometry_template_id"]=spec.pop("id")
    spec["phase"]="F01_native_validation"
    spec["nominal_controller"]="existing_M01_courtesy_wait"
    spec["formation"]="occupancy_checked_native_approach_slots_for_passive_validation"
    arms=spec["topology"]["arms"];lanes=spec["topology"]["lane_num"]
    spec["agents"]=spec["agents"][:count]
    if len(spec["agents"])!=count:raise ValueError("not enough native mission templates")
    for i,a in enumerate(spec["agents"]):
        a["lane"]=(i//arms)%lanes
        slot=i//(arms*lanes)
        a["longitude"]=spec["topology"]["approach_length_m"]-12.-8.*slot
        a["departure_s"]=0.
        a["movement_id"]=f'a{a["origin"]}_a{a["destination"]}_l{a["lane"]}'
        if a["longitude"]<4:raise ValueError("workload does not fit these approaches")
    return spec


def run_native(output, *, observer=False, vehicles=4, horizon_s=12., index=48):
    """Run a small separate experiment; diagnostics cannot return an action."""
    if vehicles not in (4,8,16,24,30) or not math.isfinite(horizon_s) or horizon_s<=0 or abs(horizon_s/.2-round(horizon_s/.2))>1e-7:
        raise ValueError("4/8/16/24/30 vehicles; positive horizon multiple of 0.2 s")
    if type(observer) is not bool or type(index) is not int or index<0:
        raise ValueError("invalid observer/index")
    output=Path(output).resolve()
    if output.exists():raise ValueError("choose a new experimental directory")
    if any((p/"campaign.json").exists() or (p/"manifest.json").exists() for p in output.parents):
        raise ValueError("experiments cannot be written inside an existing dataset/export")
    # Lazy imports: the scientific core and --help do not initialize MetaDrive.
    from contextlib import ExitStack
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union
    import psutil
    from metadrive.policy.idm_policy import IDMPolicy
    from ..dataset.native import environment, introduce, Route
    from ..metadrive.native import missions_for, SubstepEvidence, configuration_record
    from ..metadrive.run import commands_for
    from ..metadrive.observation import body, signed_speed
    from ..metadrive.cooperation import MetaDriveExchange
    from .observer import EgoObserver

    spec=smoke_spec(vehicles,index)
    spec.update(horizon_s=horizon_s,active_limit=vehicles,density_level=vehicles)
    spec["id"]="f01-smoke-"+hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:20]
    source_fingerprints=_source_fingerprints()
    output.mkdir(parents=True)
    (output/"workload.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
    env=proof=None;costs=[];composition=[];batch_composition=[];status_counts={};arrivals={};departures=[]
    memory=[];logical=hashlib.sha256();physical=hashlib.sha256();count_physics=0
    process=psutil.Process();start=time.perf_counter();initial_memory=process.memory_info().rss
    def physics_record(row):
        nonlocal count_physics
        raw=json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False)
        physical.update(raw.encode());count_physics+=1
        physics_stream.write(raw+"\n")
    try:
        env=environment(spec);env.reset(seed=spec["seed"])
        for a in spec["agents"][1:]:
            lane=env.current_map.road_network.get_lane((f'a{a["origin"]}_source',f'a{a["origin"]}_entry',a["lane"]))
            from .geometry import footprint
            spawn=footprint(lane.position(a["longitude"],0),lane.heading_theta_at(a["longitude"]),4.515,1.852)
            if any(spawn.buffer(.5).intersects(body(v)) for v in env.agents.values()):
                raise ValueError("occupied experimental admission; no relocation allowed")
            introduce(env,a)
        missions=missions_for(env)
        identities={k:v.id for k,v in env.agents.items()}
        routes={a["id"]:Route(env,a) for a in spec["agents"]}
        initial_progress={k:r.line.project(Point(env.agents[k].position)) for k,r in routes.items()}
        surfaces=[Polygon(l.polygon).buffer(0) for a in env.current_map.road_network.graph.values() for ls in a.values() for l in ls]
        surface=unary_union(surfaces).buffer(1e-5)
        road=dict(observed_at_s=0.,available_at_s=0.,version="native-static-map",coordinate_frame="metadrive_local_xy_m",
                  drivable_polygons_xy_m=[[[float(x),float(y)] for x,y in p.exterior.coords] for p in surfaces],
                  lane_width_m=spec["topology"]["lane_width_m"],rule_context="not_modelled")
        own_missions={k:dict(observed_at_s=0.,available_at_s=0.,version="native-initial-mission",
            origin=str(m.origin_lane),destination=m.destination,own_route_xy_m=[list(p) for p in routes[k].line.coords],
            movement_id=f"{k}:initial:forward",demand_forward=True) for k,m in missions.items()}
        observers={k:EgoObserver() for k in missions} if observer else {}
        followers={k:IDMPolicy(v,spec["seed"]) for k,v in env.agents.items()}
        exchange=MetaDriveExchange(range_m=30,interval_s=.4,lifetime_s=1.2,step_s=.2)
        previous_velocity={};priority=None
        with ExitStack() as stack:
            streams={name:stack.enter_context(gzip.open(output/(name+".jsonl.gz"),"wt",encoding="utf-8"))
                     for name in ("physics","execution","local_inputs","diagnostics")}
            physics_stream=streams["physics"]
            proof=SubstepEvidence(env,missions,surface,physics_record);proof.record(0.)
            for step in range(round(horizon_s/.2)+1):
                t=round(step*.2,7)
                actions,_,commands,priority=commands_for(env,followers,t,priority,"courtesy_wait")
                readings={}
                for k,v in sorted(env.agents.items()):
                    if v.id!=identities[k] or v.config["destination"]!=missions[k].destination or tuple(v.navigation.checkpoints)!=missions[k].checkpoints:
                        raise RuntimeError("native mission/identity changed")
                    readings[k]=dict(position_xy_m=list(map(float,v.position)),heading_rad=float(v.heading_theta),
                        signed_speed_mps=signed_speed(v),length_m=float(v.LENGTH),width_m=float(v.WIDTH),mission=dict(destination=missions[k].destination))
                if step:exchange.observe(t,readings)
                before=set(env.agents)
                _write(streams["execution"],dict(time_s=t,states=readings,commands=commands))
                logical.update(json.dumps(dict(time_s=t,states=readings,commands=commands),sort_keys=True).encode())
                # Sensor/transport acquisition occurs in both arms. Only D01 is switched.
                packets=[];composition_start=len(composition)
                for k,v in sorted(env.agents.items()):
                    velocity=list(map(float,v.velocity));old=previous_velocity.get(k)
                    previous_velocity[k]=velocity
                    measurement=dict(position_xy_m=list(map(float,v.position)),heading_rad=float(v.heading_theta),
                        signed_speed_mps=signed_speed(v),velocity_xy_mps=velocity,
                        acceleration_xy_mps2=None if old is None else [(velocity[i]-old[i])/.2 for i in (0,1)],
                        mission_progress_m=float(routes[k].line.project(Point(v.position))-initial_progress[k]),
                        length_m=float(v.LENGTH),width_m=float(v.WIDTH))
                    rays=env.engine.get_sensor("lidar").perceive(v,env.engine.physics_world.dynamic_world,num_lasers=16,distance=30,show=False)[0]
                    c=commands[k];own_controls=dict(requested_speed_mps=c["requested"]["target_speed_mps"],
                        accepted_speed_mps=c["accepted"]["target_speed_mps"],native_action=c["native_action"])
                    begin=time.perf_counter()
                    p=local_packet(k,t,measurement,own_missions[k],road,rays,own_controls,exchange.knowledge(k))
                    composition.append((time.perf_counter()-begin)*1000)
                    _write(streams["local_inputs"],p);packets.append((k,p))
                batch_composition.append(sum(composition[composition_start:]))
                begin=time.perf_counter()
                captured_elapsed=begin-start
                diagnostics=[observers[k].update(p) for k,p in packets] if observer else []
                elapsed=(time.perf_counter()-begin)*1000
                if observer:costs.append(elapsed)
                for d in diagnostics:
                    d["runtime"]=dict(wall_captured_elapsed_s=captured_elapsed,
                                      wall_available_elapsed_s=time.perf_counter()-start,batch_compute_ms=elapsed)
                    status_counts[d["status"]]=status_counts.get(d["status"],0)+1
                    _write(streams["diagnostics"],d)
                memory.append(process.memory_info().rss)
                if step==round(horizon_s/.2):break
                proof.commands=commands
                _,_,terminated,truncated,info=env.step(actions)
                for k in sorted(before-set(env.agents)):
                    terminal=dict(time_s=round(t+.2,7),vehicle_id=k,terminated=bool(terminated.get(k)),
                                  truncated=bool(truncated.get(k)),info=info.get(k,{}))
                    # Serialize native JSON-compatible flags only, not engine handles.
                    row=dict(time_s=terminal["time_s"],vehicle_id=k,arrive_dest=bool(terminal["info"].get("arrive_dest",False)))
                    departures.append(row)
                    if row["arrive_dest"]:arrivals[k]=row["time_s"]
                    if observer:
                        _write(streams["diagnostics"],observers[k].finish(row["time_s"],
                            reason="own_native_arrival" if row["arrive_dest"] else "own_native_departure"))
            for o in observers.values():
                if o.last_output is not None and not o.finished:_write(streams["diagnostics"],o.finish(horizon_s))
        if _source_fingerprints()!=source_fingerprints:
            raise RuntimeError("sources changed during native validation; preserve evidence and rerun")
        result=dict(schema="F01-native-validation-1",observer=observer,vehicles=vehicles,index=index,horizon_s=horizon_s,
            platform="Windows" if os.name=='nt' else sys.platform,source_fingerprints=source_fingerprints,
            logical_sha256=logical.hexdigest(),physics_sha256=physical.hexdigest(),physics_samples=count_physics,
            arrivals=arrivals,departures=departures,missions={k:asdict(m) for k,m in missions.items()},
            integrity=dict(contacts=proof.contacts,overlaps=proof.overlaps,outside=proof.outside),
            status_counts=status_counts,observer_updates=len(costs),
            observer_ms=dict(p50=_percentile(costs,.5),p95=_percentile(costs,.95),p99=_percentile(costs,.99),maximum=max(costs,default=0),
                             over_20ms=sum(x>20 for x in costs),over_50ms=sum(x>50 for x in costs),over_200ms=sum(x>200 for x in costs)),
            composition_ms_per_ego=dict(p95=_percentile(composition,.95)),
            composition_ms_per_batch=dict(p95=_percentile(batch_composition,.95),p99=_percentile(batch_composition,.99)),
            observer_with_composition_ms=dict(p95=_percentile([a+b for a,b in zip(costs,batch_composition)],.95),
                p99=_percentile([a+b for a,b in zip(costs,batch_composition)],.99),
                maximum=max((a+b for a,b in zip(costs,batch_composition)),default=0),
                over_200ms=sum(a+b>200 for a,b in zip(costs,batch_composition))),
            rss_peak_bytes=max(memory,default=initial_memory),rss_start_bytes=initial_memory,
            process_peak_rss_bytes=getattr(process.memory_info(),"peak_wset",max(memory,default=initial_memory)),
            wall_s=time.perf_counter()-start,configuration=configuration_record(env,spec["seed"]),
            controller="existing_M01_courtesy_wait_on_v4_geometry_not_M02_production_replay")
        (output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
        return result
    finally:
        if proof is not None:proof.close()
        if env is not None:env.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description="Small native F01 validation; observer disabled by default")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--observer",action="store_true")
    parser.add_argument("--vehicles",type=int,choices=(4,8,16,24,30),default=4)
    parser.add_argument("--horizon",type=float,default=12.)
    parser.add_argument("--index",type=int,default=48)
    a=parser.parse_args(argv)
    result=run_native(a.output,observer=a.observer,vehicles=a.vehicles,horizon_s=a.horizon,index=a.index)
    print(json.dumps({k:result[k] for k in ("observer","vehicles","observer_ms","logical_sha256","physics_sha256")},indent=2))


if __name__=="__main__":main()
