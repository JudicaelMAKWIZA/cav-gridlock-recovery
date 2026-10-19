"""Grouped partitions and causal feature windows, physically separate from targets."""

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path

from .contract import Contract, digest
from .pipeline import read_rows, records, write_json, file_sha256
from .qualify import qualify, POSITIVE, NEGATIVE, UNKNOWN

FEATURE_KEYS = ("time_s","vehicle_id","position_xy_m","heading_rad","signed_speed_mps",
                "velocity_xy_mps","acceleration_xy_mps2","mission_progress_m","lidar_normalized",
                "received_v2v","missing_mask")


def validate(root, *, deep=False, report_path=None):
    root=Path(root);issues=[];seen=set();parents={};lineages={};counts=Counter();physical_configs={}
    campaign=json.loads((root/"campaign.json").read_text());contract=Contract(**campaign["contract"])
    for folder,row in records(root):
        counts[row["label"]]+=1
        if row["id"] in seen:
            issues.append(dict(id=row["id"],kind="duplicate_identity"))
        seen.add(row["id"])
        manifest=json.loads((folder/"manifest.json").read_text())
        spec=json.loads((folder/"spec.json").read_text());metadata=json.loads((folder/"metadata.json").read_text())
        physical_signature=digest({key:spec[key] for key in ("roads","agents","horizon_s")})
        if physical_signature in physical_configs:
            issues.append(dict(id=row["id"],kind="duplicate_physical_configuration",other=physical_configs[physical_signature]))
        physical_configs[physical_signature]=row["id"]
        if digest(spec)!=manifest["spec_sha256"] or contract.sha256!=manifest["contract_sha256"] or manifest["code"]!=campaign["code"]:
            issues.append(dict(id=row["id"],kind="provenance_mismatch"))
        for name,expected in manifest["files"].items():
            path=folder/name
            if not path.is_file() or file_sha256(path)!=expected:
                issues.append(dict(id=row["id"],kind="file_hash",file=name))
        if row["label"]==POSITIVE and (row["integrity"]["errors"] or not any(e["confirmed"] for e in row["events"])):
            issues.append(dict(id=row["id"],kind="positive_without_valid_proof"))
        if deep:
            reference=list(read_rows(folder/"reference.jsonl.gz"))
            by_time={frame["time_s"]:frame for frame in reference}
            from shapely.geometry import Point, LineString
            from shapely import wkt
            from shapely.ops import unary_union
            from ..metadrive.observation import rectangle
            surface=wkt.loads(metadata["surface_wkt"]).buffer(1e-6) if metadata.get("surface_wkt") else None
            commands={frame["time_s"]:frame["commands"] for frame in read_rows(folder/"commands.jsonl.gz")}
            routes={key:LineString([point for lane in lanes for point in lane["points"]])
                    for key,lanes in metadata.get("route_lanes",{}).items()}
            recomputed=qualify(reference,metadata,row["integrity"],contract)
            if recomputed["label"]!=row["label"] or recomputed["dependencies"]!=row["dependencies"]:
                issues.append(dict(id=row["id"],kind="qualification_not_reproducible"))
            previous=-1.;last_positions={}
            for frame in read_rows(folder/"physics.jsonl.gz"):
                t=frame["time_s"]
                if t<=previous:
                    issues.append(dict(id=row["id"],kind="physics_time"));break
                for key,state in frame["states"].items():
                    if not all(math.isfinite(v) for v in state):
                        issues.append(dict(id=row["id"],kind="nonfinite_state"))
                    if key in last_positions and math.dist(state[:2],last_positions[key])>.15:
                        issues.append(dict(id=row["id"],kind="pose_jump"))
                    last_positions[key]=state[:2]
                if t in by_time and surface is not None:
                    witnesses=by_time[t]["witnesses"]
                    for witness in witnesses:
                        ego,other=witness["requester"],witness["blocker"]
                        if ego not in frame["states"] or other not in frame["states"]:
                            issues.append(dict(id=row["id"],kind="witness_missing_same_time_native_pose",time_s=t))
                            continue
                        a,b=frame["states"][ego],frame["states"][other]
                        dims=metadata["vehicles"][ego];other_dims=metadata["vehicles"][other]
                        heading=a[2];distance=witness["movement"]["distance_m"]
                        start=rectangle(a[:2],heading,dims["length_m"],dims["width_m"])
                        finish=rectangle((a[0]+distance*math.cos(heading),a[1]+distance*math.sin(heading)),heading,dims["length_m"],dims["width_m"])
                        poses=witness["movement"].get("reference_poses")
                        if poses and ego in routes:
                            route=routes[ego];station=route.project(Point(a[:2]))
                            if any(route.distance(Point(p[:2]))>1e-4 or not station-1e-4<=route.project(Point(p[:2]))<=station+distance+1e-4 for p in poses):
                                issues.append(dict(id=row["id"],kind="reference_not_on_assigned_native_route",time_s=t))
                        required=(unary_union([start]+[rectangle(p[:2],p[2],dims["length_m"],dims["width_m"]) for p in poses])
                                  if poses else unary_union([start,finish]).convex_hull)
                        blocker=rectangle(b[:2],b[2],other_dims["length_m"],other_dims["width_m"])
                        if required.intersection(blocker).area<=1e-8 or (witness["road_admissible"] and not surface.covers(required)):
                            issues.append(dict(id=row["id"],kind="physical_witness_not_reproduced",time_s=t))
                        intent=commands[t][ego]["requested_speed_mps"]>0
                        if witness["c1_expected"]!=intent:
                            issues.append(dict(id=row["id"],kind="expected_movement_mismatch",time_s=t))
                previous=t
            for sample in read_rows(folder/"observations.jsonl.gz"):
                if set(sample)!=set(FEATURE_KEYS):
                    issues.append(dict(id=row["id"],kind="feature_schema_or_privileged_field"));break
                # Native local measurements are finite; annotations never enter this stream.
                if len(sample["lidar_normalized"])!=16 or not all(0<=v<=1 for v in sample["lidar_normalized"]):
                    issues.append(dict(id=row["id"],kind="sensor_quality"));break
                def future(value):
                    if isinstance(value,dict):
                        return any((key in ("observed_at_s","sent_at_s","received_at_s") and isinstance(item,(int,float))
                                    and item>sample["time_s"]+1e-7) or future(item) for key,item in value.items())
                    return isinstance(value,list) and any(future(item) for item in value)
                if future(sample["received_v2v"]):
                    issues.append(dict(id=row["id"],kind="future_v2v_information"));break
        parents.setdefault(row["parent_id"],[]).append(row["id"])
        lineages.setdefault(spec.get("lineage_group",row["parent_id"]),[]).append(row["id"])
    split_path=root/"splits.json"
    if split_path.exists():
        partition=json.loads(split_path.read_text())
        assigned=partition["episodes"]
        if set(assigned)!=seen:
            issues.append(dict(kind="split_coverage"))
        for parent,ids in parents.items():
            if len({assigned.get(i) for i in ids})>1:
                issues.append(dict(kind="parent_leakage",parent=parent))
        for group,ids in lineages.items():
            if len({assigned.get(i) for i in ids})>1:
                issues.append(dict(kind="related_geometry_leakage",lineage=group))
        if partition.get("contract_sha256")!=contract.sha256:
            issues.append(dict(kind="split_contract_mismatch"))
    result=dict(valid=not issues,episodes=len(seen),labels=dict(counts),issues=issues,deep=deep,
                contract_sha256=contract.sha256)
    write_json(Path(report_path) if report_path is not None else root/("validation-deep.json" if deep else "validation.json"),result)
    return result


def split(root, *, seed=1741, ood_families=("asymmetric",)):
    root=Path(root);assignments={};parents={};counts=Counter();class_counts={}
    rows=list(records(root));groups=defaultdict(list)
    for folder,row in rows:
        if row["phase"]!="production":
            raise ValueError("Calibration episodes must not enter train/validation/test.")
        spec=json.loads((folder/"spec.json").read_text())
        groups[spec.get("lineage_group",row["parent_id"])].append(row)
    validation_count=0;target_validation=round(.15*len(rows))
    for lineage in sorted(groups,key=lambda key:digest([seed,key])):
        members=groups[lineage]
        if any(row["family"] in ood_families for row in members):
            group="test_ood"
        elif validation_count+len(members)<=target_validation:
            group="validation";validation_count+=len(members)
        else:
            group="train"
        for row in members:
            parent=row["parent_id"];parents[parent]=group
            assignments[row["id"]]=group;counts[group]+=1
            class_counts.setdefault(group,Counter())[row["label"]]+=1
    campaign=json.loads((root/"campaign.json").read_text())
    result=dict(seed=seed,episodes=assignments,parents=parents,counts=dict(counts),classes=class_counts,
        requested_proportions=[.70,.15,.15],group_adjusted_expected_proportions=[.70,.15,.15],
        adjustment="asymmetric junction topology held out; parent and normalized geometry groups indivisible",
        ood_families=list(ood_families),contract_sha256=campaign["contract_sha256"],windows_created_after_partition=True)
    write_json(root/"splits.json",result)
    return result


def export(root, output, *, window_s=5., stride_s=1., max_gib=4., reserve_gib=8.):
    root=Path(root);output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Export directory must be absent or empty.")
    if window_s<=0 or stride_s<=0 or max_gib<=0 or reserve_gib<0 or not all(math.isfinite(v) for v in (window_s,stride_s,max_gib,reserve_gib)):
        raise ValueError("Window and stride must be positive.")
    partition=json.loads((root/"splits.json").read_text())
    if not validate(root)["valid"]:
        raise ValueError("Dataset QC must pass before export.")
    output.mkdir(parents=True,exist_ok=True);window_count=0;episode_count=0;review=[];review_counts=Counter()
    from contextlib import ExitStack
    with ExitStack() as stack:
        streams={}
        for group in set(partition["episodes"].values())|{"unknown"}:
            for name in ("features","windows","targets","episodes","dependencies"):
                path=output/group;path.mkdir(exist_ok=True)
                streams[(group,name)]=stack.enter_context(gzip.open(path/(name+".jsonl.gz"),"wt",encoding="utf-8",compresslevel=3))
        for folder,row in records(root):
            group="unknown" if row["label"]==UNKNOWN else partition["episodes"][row["id"]];episode_count+=1
            if episode_count%20==0:
                for stream in streams.values():stream.flush()
                if (shutil.disk_usage(output).free<reserve_gib*1024**3
                        or sum(p.stat().st_size for p in output.rglob("*") if p.is_file())>max_gib*1024**3):
                    raise ValueError("Export resource limit reached; original episodes preserved, partial export not certified.")
            def emit(name,value):
                streams[(group,name)].write(json.dumps(value,separators=(",",":"),allow_nan=False)+"\n")
            emit("episodes",dict(episode_id=row["id"],parent_id=row["parent_id"],label=row["label"],family=row["family"],
                scopes=row["scopes"],events=row["events"],integrity=row["integrity"],unknown_alternatives=True))
            for dep in row["dependencies"]+row["indirect_dependencies"]:
                emit("dependencies",dict(episode_id=row["id"],**dep))
            histories=defaultdict(list);last_export={};introduced={}
            for sample_index,observation in enumerate(read_rows(folder/"observations.jsonl.gz")):
                vehicle=observation["vehicle_id"];t=observation["time_s"]
                introduced.setdefault(vehicle,t)
                emit("features",dict(episode_id=row["id"],sample_index=sample_index,
                    **{k:observation[k] for k in FEATURE_KEYS}))
                histories[vehicle].append((t,sample_index))
                histories[vehicle]=[s for s in histories[vehicle] if s[0]>=t-window_s-1e-7]
                if t-introduced[vehicle]<window_s-1e-7 or t-last_export.get(vehicle,-stride_s)<stride_s-1e-7:
                    continue
                last_export[vehicle]=t;window_count+=1
                window_id=digest([row["id"],vehicle,t,window_s])[:28]
                emit("windows",dict(window_id=window_id,episode_id=row["id"],vehicle_id=vehicle,
                    start_s=histories[vehicle][0][0],end_s=t,sample_indices=[s[1] for s in histories[vehicle]],
                    causal=True,feature_file="features.jsonl.gz",available_columns=list(FEATURE_KEYS)))
                current=[e for e in row["events"] if e["requester"]==vehicle and e["start_s"]<=t<=e["end_s"]]
                event_positive=any(e["confirmed"] and any(effect["confirmed"] and effect["start_s"]<=t<=effect["end_s"]
                    for effect in e["effect_intervals"]) for e in current)
                unknown_target=row["label"]==UNKNOWN or (not event_positive and any(e["status"]=="open_candidate" for e in current))
                emit("targets",dict(window_id=window_id,episode_label=row["label"],
                    event_obstructed=None if unknown_target else event_positive,events=current,
                    supervision_available=not unknown_target,
                    label_uses_future_followup=True,feature_join_only_by_window_id=True))
            bucket=(row["label"],tuple(row["scopes"]),row["family"])
            if review_counts[bucket]<2:
                review_counts[bucket]+=1
                review.append(dict(stratum=[bucket[0],list(bucket[1]),bucket[2]],episode_id=row["id"],partition=group,
                    annotation=str(folder/"annotation.json"),reference=str(folder/"reference.jsonl.gz"),
                    inspection="verify C1-C7, body zone, timestamps, cause, alternatives; no automatic human approval"))
    write_json(output/"human_review.json",review)
    files={p.relative_to(output).as_posix():file_sha256(p) for p in output.rglob("*") if p.is_file()}
    result=dict(episodes=episode_count,windows=window_count,window_s=window_s,stride_s=stride_s,
        schema="cgr-causal-indexed-windows-3",window_storage="indices_into_per_episode_feature_samples_no_duplicate_histories",
        feature_columns=list(FEATURE_KEYS),annotations_separate=True,partition_sha256=digest(partition),files=files,
        no_training_or_D01_implemented=True)
    write_json(output/"manifest.json",result)
    return result
