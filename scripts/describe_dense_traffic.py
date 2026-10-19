"""Offline native traffic distributions; no simulation and no learned model.

Counts distinct requesters with confirmed movement effects, never blocker
cardinality or mere geometric candidates. Retrospective and as-of-confirmation
counts are separate, and all oracle-derived outputs stay outside model inputs.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from cav_recovery.dataset.pipeline import records,read_rows,write_json,code_hashes

def zone(resource):
    return 'central' if resource=='junction:central' else 'approach' if resource.endswith(':approach') else 'exit'

def effect_changes(events,*,confirmed_only=True,asof=False):
    changes=defaultdict(Counter)
    for event in events:
        if confirmed_only and not event['confirmed']:continue
        for effect in event['effect_intervals']:
            if confirmed_only and not effect['confirmed']:continue
            start=effect['start_s']
            if asof:
                if not effect['confirmed'] or not event['confirmed']:continue
                start=max(start,effect['confirmation_at_s'])
            if start>effect['end_s']:continue
            key=(event['requester'],zone(event['resource_id']))
            changes[round(start,8)][key]+=1
            # The native .2-second observation includes its final sample.
            changes[round(effect['end_s']+.2,8)][key]-=1
    return sorted(changes.items())

def summary(hist):
    hist=Counter({int(k):v for k,v in hist.items() if v});n=sum(hist.values())
    if not n:return dict(n=0)
    def quantile(q):
        target=math.ceil(q*n);seen=0
        for value in sorted(hist):
            seen+=hist[value]
            if seen>=max(1,target):return value
    return dict(n=n,minimum=min(hist),q25=quantile(.25),median=quantile(.5),q75=quantile(.75),
        q90=quantile(.9),q95=quantile(.95),q99=quantile(.99),maximum=max(hist),
        mean=sum(k*v for k,v in hist.items())/n,histogram=dict(sorted(hist.items())))

def describe(root,*,verify_ledger=False):
    root=Path(root);groups={};episodes=[];issues=[];labels=Counter();sources={};costs=defaultdict(list)
    campaign=json.loads((root/'campaign.json').read_text());begun=time.perf_counter()
    for folder,row in records(root):
        label=row['label'];family=row['family'];labels[label]+=1
        spec=json.loads((folder/'spec.json').read_text());timeline=row['traffic']['timeline']
        sources[row['id']]=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()
        keys=[f'{label}|{family}',f'{label}|ALL',f'ALL|{family}','ALL|ALL']
        for key in keys:
            groups.setdefault(key,dict(episodes=0,samples=defaultdict(Counter),episode_peaks=defaultdict(Counter),ever_affected=Counter()))['episodes']+=1
        reference=iter(read_rows(folder/'reference.jsonl.gz')) if verify_ledger else None
        schedules=[effect_changes(row['events']),effect_changes(row['events'],asof=True),effect_changes(row['events'],confirmed_only=False)]
        indices=[0,0,0];active=[Counter(),Counter(),Counter()];peaks=Counter();ever=set();exemplar=None
        for sample in timeline:
            t=sample['time_s'];sets=[]
            for j,changes in enumerate(schedules):
                while indices[j]<len(changes) and changes[indices[j]][0]<=t+1e-7:
                    active[j].update(changes[indices[j]][1]);indices[j]+=1
                sets.append({pair for pair,n in active[j].items() if n>0})
            confirmed,asof,effect=sets;actors={p[0] for p in confirmed};ever.update(actors)
            metrics={k:int(sample[k]) for k in ('active','central','approach','exit')}
            metrics['geometric_candidates']=int(sample['physically_constrained'])
            metrics['affected_confirmed']=len(actors)
            metrics['confirmation_available']=len({p[0] for p in asof})
            metrics['effect_5s_including_transient']=len({p[0] for p in effect}) if not row['integrity']['errors'] else 0
            for region in ('central','approach','exit'):
                metrics['affected_'+region]=len({vehicle for vehicle,z in confirmed if z==region})
            if len(confirmed)!=len(actors):issues.append(dict(id=row['id'],time_s=t,kind='requester_in_two_regions'))
            if any(metrics['affected_'+r]>metrics[r] for r in ('central','approach','exit')) or len(actors)>metrics['active']:
                issues.append(dict(id=row['id'],time_s=t,kind='affected_exceeds_present'))
            if reference is not None:
                frame=next(reference,None)
                if frame is None or frame['time_s']!=t:issues.append(dict(id=row['id'],time_s=t,kind='ledger_clock'))
                else:
                    counts=Counter(zone(s['resource_id']) for s in frame['states'].values())
                    if len(frame['states'])!=metrics['active'] or any(counts[r]!=metrics[r] for r in ('central','approach','exit')):
                        issues.append(dict(id=row['id'],time_s=t,kind='ledger_not_native_reference'))
                    if not actors<=set(frame['states']):issues.append(dict(id=row['id'],time_s=t,kind='affected_not_present'))
            if exemplar is None or len(actors)>exemplar['affected']:
                exemplar=dict(time_s=t,affected=len(actors),central=metrics['affected_central'],approach=metrics['affected_approach'],
                    exit=metrics['affected_exit'],active=metrics['active'],present_central=metrics['central'],present_approach=metrics['approach'])
            for metric,value in metrics.items():
                peaks[metric]=max(peaks[metric],value)
                for key in keys:groups[key]['samples'][metric][value]+=1
        if reference is not None and next(reference,None) is not None:issues.append(dict(id=row['id'],kind='extra_reference_samples'))
        for key in keys:
            for metric,value in peaks.items():groups[key]['episode_peaks'][metric][value]+=1
            groups[key]['ever_affected'][len(ever)]+=1
        episodes.append(dict(id=row['id'],label=label,family=family,active_limit=spec['active_limit'],peaks=dict(peaks),
            ever_affected=len(ever),peak_affected_snapshot=exemplar,integrity_failed=bool(row['integrity']['errors'])))
        if not row['integrity']['errors']:
            steps=row['integrity']['substeps'];costs[spec['active_limit']].append(dict(cpu_s=row['cpu_s'],wall_s=row['wall_s'],
                simulated_s=steps*.02,normalized_cpu_110s=row['cpu_s']*110/(steps*.02) if steps else None,
                mean_active=statistics.mean(s['active'] for s in timeline),
                peak_rss_kib=row['peak_rss_kib'],bytes=row['storage_bytes'],label=label))
    result=dict(episodes=len(episodes),labels=dict(labels),valid=not issues,issues=issues,
        definitions=dict(sample_weighting='all native 5Hz observations, including warm-up and final observation; episode durations vary',
            region='native assigned-route segment: central connector versus approach and exit; a straddling body is assigned by route progress',
            affected='distinct requesters inside confirmed movement-effect intervals; retrospective oracle reference, never a model input',
            confirmation_available='same actors counted only after their persistent-effect confirmation time',
            geometric_candidates='instantaneous body intersections from native traffic ledger; not automatically qualifying obstruction',
            peaks='each metric has its own episode peak; marginal peaks must not be added as simultaneous counts'),
        groups={key:dict(episodes=g['episodes'],samples={k:summary(v) for k,v in g['samples'].items()},
            episode_peaks={k:summary(v) for k,v in g['episode_peaks'].items()},ever_affected=summary(g['ever_affected'])) for key,g in groups.items()},
        episode_evidence=episodes,reference_ledger_verified=verify_ledger,
        integrity_failures=[dict(id=e['id'],label=e['label']) for e in episodes if e.get('integrity_failed')],
        costs={cap:dict(episodes=len(rows),cpu_s_median=statistics.median(r['cpu_s'] for r in rows),
            normalized_cpu_110s_median=statistics.median(r['normalized_cpu_110s'] for r in rows if r['normalized_cpu_110s'] is not None),
            mean_active_median=statistics.median(r['mean_active'] for r in rows),
            wall_s_median=statistics.median(r['wall_s'] for r in rows),maximum_worker_highwater_mib=max(r['peak_rss_kib'] for r in rows)/1024,
            minimum_worker_highwater_mib=min(r['peak_rss_kib'] for r in rows)/1024,mean_bytes=statistics.mean(r['bytes'] for r in rows)) for cap,rows in costs.items()},
        source_manifest_sha256=sources,source_code_unchanged=campaign['code']==code_hashes(),contract_sha256=campaign['contract_sha256'],
        oracle_only=True,models_developed=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),wall_s=time.perf_counter()-begun)
    families=sorted({e['family'] for e in episodes})
    for label in ('POSITIVE_CONFIRMED','NEGATIVE_CONFIRMED','UNKNOWN_OR_CENSORED'):
        for family in families:
            result['groups'].setdefault(label+'|'+family,dict(episodes=0,samples={},episode_peaks={},ever_affected=dict(n=0)))
    write_json(root/'traffic-distributions.json',result)
    lines=['# Distributions natives v4 — référence hors ligne','','Chaque cellule indique min / médiane / P95 / max des pics par épisode. Les pics marginaux ne sont pas simultanés.',
        'Les affectés sont des demandeurs distincts dans des intervalles d’effet confirmé, pas le nombre de liens ni tous les bloqueurs.','',
        '| Classe | Topologie | Épisodes | Actifs | Centre | Approches | Affectés totaux | Affectés centre | Affectés approches |',
        '|---|---|---:|---|---|---|---|---|---|']
    for key,g in sorted(result['groups'].items()):
        label,family=key.split('|')
        if 'ALL' in (label,family):continue
        def cell(metric):
            d=g['episode_peaks'].get(metric)
            return f"{d['minimum']} / {d['median']} / {d['q95']} / {d['maximum']}" if d and d['n'] else 'non observé'
        lines.append(f"| {label} | {family} | {g['episodes']} | "+' | '.join(cell(k) for k in ('active','central','approach','affected_confirmed','affected_central','affected_approach'))+' |')
    lines+=['','## Distribution des observations à 5 Hz','',
        'Chaque cellule indique médiane / P95 / max sur tous les instants observés. Les épisodes plus longs contribuent davantage ; démarrage et fin sont inclus. Un zéro confirmé dans la classe inconnue ne prouve pas une absence du phénomène.','',
        '| Classe | Topologie | Actifs | Centre | Approches | Affectés confirmés | Effet 5 s, transitoires compris |',
        '|---|---|---|---|---|---|---|']
    for key,g in sorted(result['groups'].items()):
        label,family=key.split('|')
        if 'ALL' in (label,family):continue
        def sample_cell(metric):
            d=g['samples'].get(metric)
            return f"{d['median']} / {d['q95']} / {d['maximum']}" if d and d['n'] else 'non observé'
        lines.append(f'| {label} | {family} | '+' | '.join(sample_cell(k) for k in ('active','central','approach','affected_confirmed','effect_5s_including_transient'))+' |')
    (root/'traffic-distributions.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset',type=Path)
    parser.add_argument('--verify-ledger',action='store_true');args=parser.parse_args()
    r=describe(args.dataset,verify_ledger=args.verify_ledger)
    print(json.dumps(dict(valid=r['valid'],episodes=r['episodes'],labels=r['labels'],issues=r['issues'],wall_s=r['wall_s']),indent=2))
    raise SystemExit(0 if r['valid'] and r['source_code_unchanged'] else 1)
