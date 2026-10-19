"""Causal long histories and prefix masks for future TCN tasks; no training."""
import argparse
from collections import defaultdict,deque
import gzip
import hashlib
import json
import math
from pathlib import Path
from cav_recovery.dataset.pipeline import read_rows,write_json


def prepare(exported,horizon_s=35.,sample_s=.2):
    exported=Path(exported)
    if not math.isfinite(horizon_s) or horizon_s<5 or abs(horizon_s/sample_s-round(horizon_s/sample_s))>1e-7:
        raise ValueError('Temporal horizon must cover at least five seconds on the native sampling grid')
    expected=round(horizon_s/sample_s)+1;count=0;prefixes=0;partitions={}
    for feature_path in sorted(exported.glob('*/features.jsonl.gz')):
        group=feature_path.parent;histories=defaultdict(deque);episode=None;local=0
        windows=iter(read_rows(group/'windows.jsonl.gz'));targets=iter(read_rows(group/'targets.jsonl.gz'))
        upcoming=next(windows,None);target=next(targets,None)
        with gzip.open(group/'tcn_windows.jsonl.gz','wt',compresslevel=6) as out,gzip.open(group/'temporal_targets.jsonl.gz','wt',compresslevel=6) as truth:
            for sample in read_rows(feature_path):
                if sample['episode_id']!=episode:histories.clear();episode=sample['episode_id']
                key=sample['vehicle_id'];t=sample['time_s'];history=histories[key];history.append((t,sample['sample_index']))
                while history and history[0][0]<t-horizon_s-1e-7:history.popleft()
                if upcoming is None or (upcoming['episode_id'],upcoming['sample_indices'][-1])!=(episode,sample['sample_index']):continue
                if target is None or target['window_id']!=upcoming['window_id']:raise ValueError('Temporal endpoint target mismatch')
                if key!=upcoming['vehicle_id'] or t!=upcoming['end_s']:raise ValueError('Temporal endpoint identity mismatch')
                indices=[i for _,i in history];padding=expected-len(indices)
                if padding<0 or any(s>t+1e-7 for s,_ in history):raise ValueError('Noncausal or oversized temporal history')
                row=dict(window_id=upcoming['window_id'],episode_id=episode,vehicle_id=key,end_s=t,
                    horizon_s=horizon_s,sample_s=sample_s,sample_indices=indices,left_padding_samples=padding,
                    expected_samples=expected,causal=True,padding='left only; false mask for padding, true for referenced samples',
                    relative_time_origin='window end; absolute timestamps are join metadata')
                out.write(json.dumps(row,separators=(',',':'))+'\n');count+=1;local+=1;prefixes+=padding>0
                effects=[effect for e in target['events'] if e['confirmed'] for effect in e['effect_intervals']
                         if effect['confirmed'] and effect['start_s']<=t<=effect['end_s']]
                as_of=any(e['confirmation_at_s']<=t for e in effects)
                known=target['supervision_available']
                annotation=dict(window_id=upcoming['window_id'],retrospective_persistent_event=target['event_obstructed'],
                    persistent_confirmation_available_at_end=as_of if known else None,
                    future_followup_required=bool(target['event_obstructed'] and not as_of) if known else None,
                    geometric_candidate_active=bool(target['events']) if known else None,
                    transient_effect=any(not e['confirmed'] and e['start_s']<=t<=e['end_s']
                        for event in target['events'] for e in event['effect_intervals']) if known else None,
                    supervision_available=known,source='offline_physical_reference_never_model_input')
                truth.write(json.dumps(annotation,separators=(',',':'))+'\n')
                upcoming=next(windows,None);target=next(targets,None)
        windows.close();targets.close()
        if upcoming is not None or target is not None:raise ValueError('Unconsumed temporal endpoints')
        partitions[group.name]=local
    manifest=json.loads((exported/'manifest.json').read_text())
    for path in list(exported.glob('*/tcn_windows.jsonl.gz'))+list(exported.glob('*/temporal_targets.jsonl.gz')):
        manifest['files'][path.relative_to(exported).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['temporal_context']=dict(horizon_s=horizon_s,sample_s=sample_s,expected_samples=expected,
        windows=count,prefix_windows=prefixes,partitions=partitions,
        input='tcn_windows.jsonl.gz references past/present feature indices with left padding masks',
        targets='temporal_targets.jsonl.gz separates retrospective event membership from confirmation available at this endpoint',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest['model_input_files']=['features.jsonl.gz','windows.jsonl.gz','tcn_windows.jsonl.gz','mission_context.jsonl.gz','ego_controls.jsonl.gz','map_context.jsonl.gz']
    manifest['privileged_target_files']=['targets.jsonl.gz','temporal_targets.jsonl.gz','episodes.jsonl.gz','dependencies.jsonl.gz']
    write_json(exported/'manifest.json',manifest)
    return manifest['temporal_context']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('export',type=Path)
    parser.add_argument('--horizon-s',type=float,default=35.)
    args=parser.parse_args();print(json.dumps(prepare(args.export,args.horizon_s),indent=2))
