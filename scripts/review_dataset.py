"""Private interactive gallery of exact native poses; no simulator rerun or video."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import os
from cav_recovery.dataset.pipeline import records, read_rows, write_json

HTML = r'''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Carrefour dense — inspection privée</title>
<style>*{box-sizing:border-box}body{margin:0;background:#edf2f5;color:#182b38;font:16px/1.5 system-ui}header,main{max-width:1650px;margin:auto;padding:18px}h1{margin:4px 0}button,select,input{font:inherit;padding:8px;border:1px solid #8ca3b2;border-radius:5px;background:white}button{cursor:pointer}canvas{display:block;width:100%;height:auto;background:#dfebe4;border-radius:8px}.bar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:10px 0}#seek{flex:1;min-width:250px}#evidence{background:white;padding:18px;border-left:5px solid #367b97}a{color:#075f85}#stats{font-variant-numeric:tabular-nums}small{color:#48606e}details{padding:12px;background:white}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border-bottom:1px solid #ccd7df;text-align:left}</style>
<header><small>CAV RECOVERY · GALERIE PRIVÉE · POSES NATIVES</small><h1>Un carrefour, des arrivées et des interactions</h1><p>Rendu à l'échelle physique. Orange : occupation candidate actuelle ; rouge : effet persistant confirmé à cet instant. Les classes finales sont rétrospectives. Aucun modèle entraîné.</p><div class="bar"><label>Classe <select id="classFilter"><option value="all">Toutes</option><option>POSITIVE_CONFIRMED</option><option>NEGATIVE_CONFIRMED</option><option>UNKNOWN_OR_CENSORED</option></select></label><label>Épisode <select id="episode"></select></label></div></header>
<main><canvas id="scene" width="1600" height="1000" aria-label="Rejeu du carrefour à l'échelle, silhouettes et identifiants des CAV"></canvas><div class="bar"><button id="play">Lire</button><button id="back">−5 s</button><button id="next">+5 s</button><label>Vitesse <select id="rate"><option>1</option><option>2</option><option>5</option><option>10</option></select></label><input type="range" id="seek" min="0" max="110" step="0.1" value="0" aria-label="Instant simulé"><output id="clock"></output></div><div class="bar"><label><input type="checkbox" id="paths" checked> Trajectoires actives</label><label><input type="checkbox" id="relations" checked> Relations physiques actuelles</label></div><p id="stats"></p><section id="evidence"></section><details><summary>Chronologie et preuves de qualification</summary><div id="events"></div></details><p><small>Les carrosseries utilisent leurs longueurs et largeurs natives. Le rendu choisit une pose enregistrée à 10 Hz, sans interpolation ni changement de pose. Le contrôleur observe un voisinage idéal de 30 m, sans occlusion ; l'oracle global est réservé à l'annotation. Une alternative possible ne supprime pas la contrainte sur le mouvement initial. Les alternatives non vérifiées et la nécessité d'intervention restent inconnues.</small></p></main>
<script id="data" type="application/json">__DATA__</script><script>
const scenes=JSON.parse(document.getElementById('data').textContent),canvas=document.getElementById('scene'),ctx=canvas.getContext('2d');let selected=0,time=0,playing=false,last=0;const el=id=>document.getElementById(id),colors=['#218cc8','#e5a02f','#8f60bf','#339471','#d16a91','#26a9b0','#6076c6','#bd7340'];const vid=k=>'V'+k.replace('agent','');
function filter(){let c=el('classFilter').value,old=selected;el('episode').innerHTML='';scenes.forEach((s,i)=>{if(c==='all'||s.label===c){let o=document.createElement('option');o.value=i;o.textContent=`${s.family} · pic ${s.peak_active} CAV · ${s.label} · ${s.id.slice(0,8)}`;el('episode').append(o)}});if([...el('episode').options].some(o=>Number(o.value)===old))el('episode').value=old;if(!el('episode').options.length){playing=false;el('scene').style.visibility='hidden';el('stats').textContent='Aucun épisode dans cette classe.';el('evidence').textContent='Aucun cas disponible dans cet échantillon.';el('events').replaceChildren();return;}el('scene').style.visibility='visible';load();}
function load(){if(!el('episode').options.length)return;selected=Number(el('episode').value);time=0;playing=false;el('play').textContent='Lire';let s=scenes[selected];el('seek').max=s.frames.at(-1).t;el('evidence').textContent=`${s.label} — ${s.reason}. Introduits : ${s.total_introduced}, pic actif : ${s.peak_active}, arrivées : ${s.arrived}. Partition : ${s.partition}. Corps : ${s.dimensions}. Voies : ${s.lane_width} m. ${s.limits}`;let p=document.createElement('p');for(let name of ['annotation.json','metadata.json','spec.json','manifest.json','reference.jsonl.gz','physics.jsonl.gz']){let a=document.createElement('a');a.href=s.proofs+'/'+name;a.textContent=name;a.style.marginRight='14px';p.append(a)}el('evidence').append(p);let table=document.createElement('table');table.innerHTML='<tr><th>Mouvement / ressource</th><th>Demandeur</th><th>Occupants successifs</th><th>Début</th><th>Fin observée</th><th>Confirmation</th><th>Statut</th></tr>';s.events.forEach(e=>{let tr=document.createElement('tr');[e.resource_id,vid(e.requester),e.blockers.map(vid).join(', '),e.start_s.toFixed(1),e.end_s.toFixed(1),e.confirmation_at_s?.toFixed(1)||'—',e.status+(e.right_censored?' · censuré':'')].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.append(td)});table.append(tr)});el('events').replaceChildren(table);draw();}
function draw(){let s=scenes[selected],frame=s.frames[Math.min(s.frames.length-1,Math.floor(time*10+1e-6))];if(!frame)return;let span=Math.max(s.bounds[2]-s.bounds[0],s.bounds[3]-s.bounds[1]),scale=1040/(span+14),cx=(s.bounds[0]+s.bounds[2])/2,cy=(s.bounds[1]+s.bounds[3])/2;const xy=p=>[565+(p[0]-cx)*scale,500-(p[1]-cy)*scale];function line(points,color,width=1){if(!points.length)return;ctx.beginPath();points.forEach((p,i)=>{let q=xy(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
ctx.clearRect(0,0,1600,1000);ctx.fillStyle='#dfebe4';ctx.fillRect(0,0,1600,1000);s.surface.forEach(rings=>{ctx.beginPath();rings.forEach(r=>{r.forEach((p,i)=>{let q=xy(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.closePath()});ctx.fillStyle='#687581';ctx.fill('evenodd')});ctx.setLineDash([8,9]);s.lanes.forEach(l=>line(l,'#b8c6cd',1));ctx.setLineDash([]);
if(el('paths').checked)Object.keys(frame.states).forEach(k=>{let n=Number(k.replace('agent',''));line(s.routes[k],colors[n%colors.length]+'88',1.5)});
let ref=s.references[Math.min(s.references.length-1,Math.floor(frame.t*5+1e-6))],relevant=[];if(el('relations').checked&&ref)ref.witnesses.forEach(w=>{let e=s.events.find(e=>e.requester===w.a&&e.resource_id===w.resource&&e.start_s<=frame.t&&e.end_s>=frame.t);let effect=e?.effect_intervals.find(f=>f.confirmed&&f.start_s<=frame.t&&f.end_s>=frame.t);let confirmed=effect&&frame.t>=effect.confirmation_at_s;let color=confirmed?'#ff616d':'#ffd25c';w.polygons.forEach(poly=>line([...poly,poly[0]],color,2));if(frame.states[w.a]&&frame.states[w.b]){ctx.setLineDash(confirmed?[]:[4,5]);line([frame.states[w.a],frame.states[w.b]],color,2);ctx.setLineDash([]);relevant.push(`${vid(w.a)} → ${vid(w.b)} ${confirmed?'confirmée':'candidate'}`)}});
const boxes=[],labels=[];Object.entries(frame.states).forEach(([k,p])=>{let [x,y]=xy(p),d=s.vehicles[k],n=Number(k.replace('agent','')),length=d[0]*scale,width=d[1]*scale;ctx.save();ctx.translate(x,y);ctx.rotate(-p[2]);ctx.fillStyle=colors[n%colors.length];ctx.fillRect(-length/2,-width/2,length,width);ctx.strokeStyle='#182b38';ctx.lineWidth=1;ctx.strokeRect(-length/2,-width/2,length,width);ctx.fillStyle='#b4e0ed';ctx.fillRect(length*.12,-width*.34,length*.15,width*.68);ctx.fillStyle='#243644';ctx.fillRect(-length*.23,-width*.32,length*.12,width*.64);ctx.fillStyle='#ffbfbe';ctx.fillRect(-length*.47,-width*.42,length*.06,width*.18);ctx.fillRect(-length*.47,width*.24,length*.06,width*.18);ctx.restore();let bw=Math.abs(Math.cos(p[2])*length)+Math.abs(Math.sin(p[2])*width),bh=Math.abs(Math.sin(p[2])*length)+Math.abs(Math.cos(p[2])*width);boxes.push([x-bw/2,y-bh/2,x+bw/2,y+bh/2]);labels.push({k,x,y,bw,bh})});
let drawnLabels=0;ctx.font='bold 15px system-ui';function overlap(a,b){return a[0]<b[2]&&a[2]>b[0]&&a[1]<b[3]&&a[3]>b[1]}labels.forEach(l=>{let text=vid(l.k),w=ctx.measureText(text).width+8,chosen=null;for(let radius of [0,16,32,48,64,80,100,125]){for(let [dx,dy] of [[0,-1],[1,0],[-1,0],[0,1],[1,-1],[-1,-1],[1,1],[-1,1]]){let x=l.x+dx*(l.bw/2+w/2+5+radius),y=l.y+dy*(l.bh/2+16+radius),box=[x-w/2,y-10,x+w/2,y+10];if(box[0]>5&&box[2]<1115&&box[1]>5&&box[3]<995&&!boxes.some(b=>overlap(box,b))){chosen=box;break}}if(chosen)break}if(!chosen)return;drawnLabels++;boxes.push(chosen);let x=(chosen[0]+chosen[2])/2,y=(chosen[1]+chosen[3])/2;ctx.strokeStyle='#1d364a';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(l.x,l.y);ctx.lineTo(x,y);ctx.stroke();ctx.fillStyle='#f8fcff';ctx.fillRect(...[chosen[0],chosen[1],w,20]);ctx.fillStyle='#142f41';ctx.textAlign='center';ctx.fillText(text,x,y+5)});
canvas.dataset.visibleLabels=drawnLabels;canvas.dataset.visibleVehicles=Object.keys(frame.states).length;ctx.fillStyle='#f9fcff';ctx.fillRect(1130,0,470,1000);ctx.textAlign='left';ctx.fillStyle='#183044';ctx.font='bold 25px system-ui';ctx.fillText(`${frame.t.toFixed(1)} s · ${Object.keys(frame.states).length} CAV`,1150,45);ctx.font='16px system-ui';ctx.fillText(s.family+' · '+s.lane_width+' m/voie',1150,78);ctx.fillText('Courbes, carrosseries et route à la même échelle',1150,110);ctx.fillText('Orange : candidate · rouge : confirmée',1150,145);ctx.fillText('Dépendances physiques actuelles',1150,192);relevant.slice(0,28).forEach((v,i)=>ctx.fillText(v,1150,225+i*24));if(relevant.length>28)ctx.fillText(`+ ${relevant.length-28} autres relations`,1150,925);ctx.fillStyle='#203e50';ctx.fillRect(30,950,10*scale,4);ctx.fillText('10 m',30,940);el('clock').value=frame.t.toFixed(1)+' s';el('seek').value=time;el('stats').textContent=`Épisode ${s.id} — ${Object.keys(frame.states).length} actifs visibles — ${relevant.length} relations géométriques actuelles — ${s.scopes.join(', ')||'aucune portée persistante qualifiée'}`;}
function tick(stamp){if(playing){time+=Math.min(.2,(stamp-last)/1000)*Number(el('rate').value);if(time>=Number(el('seek').max)){time=Number(el('seek').max);playing=false;el('play').textContent='Lire'}draw()}last=stamp;requestAnimationFrame(tick)}
el('classFilter').onchange=filter;el('episode').onchange=load;el('play').onclick=()=>{if(time>=Number(el('seek').max))time=0;playing=!playing;el('play').textContent=playing?'Pause':'Lire'};el('seek').oninput=e=>{time=Number(e.target.value);draw()};el('back').onclick=()=>{time=Math.max(0,time-5);draw()};el('next').onclick=()=>{time=Math.min(Number(el('seek').max),time+5);draw()};el('paths').onchange=draw;el('relations').onchange=draw;filter();requestAnimationFrame(tick);
</script></html>'''


def polygons(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type=='Polygon':
        return [list(map(list,geometry.exterior.coords))]
    return [ring for g in geometry.geoms for ring in polygons(g)] if hasattr(geometry,'geoms') else []


def build(root, output, count=16):
    from shapely import wkt
    root=Path(root);output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Gallery output must be empty')
    output.mkdir(parents=True,exist_ok=True)
    splits=json.loads((root/'splits.json').read_text())['episodes'] if (root/'splits.json').exists() else {}
    remaining=[]
    enriched={}
    if (root/'scientific-summary.json').exists():
        enriched={r['id']:r for r in json.loads((root/'scientific-summary.json').read_text())['episode_evidence']}
    for folder,row in records(root):
        if row['integrity']['errors']:
            continue
        spec=json.loads((folder/'spec.json').read_text())
        coverage={row['label'],row['family'],f"density:{spec['density_level']}",*[f'scope:{s}' for s in row['scopes']]}
        if any(len(e['blockers'])>1 and len({(o['start_s'],o['end_s']) for o in e['occupant_intervals']})>1
               for e in row['events']):coverage.add('successive_occupants')
        if row['traffic']['peak_active']>=24:coverage.add('dense:'+row['label'])
        if any(not e['right_censored'] for e in row['events']):coverage.add('resolved_constraint')
        if row['integrity']['all_arrived']:coverage.add('all_arrived')
        evidence=enriched.get(row['id'],{})
        if evidence.get('acyclic_confirmed_component'):coverage.add('acyclic_confirmed_dependencies')
        if evidence.get('acyclic_union_of_confirmed_dependencies'):coverage.add('entire_confirmed_graph_acyclic')
        if len(evidence.get('simultaneous_confirmed_mechanisms',[]))>=2:coverage.add('coexisting_confirmed_mechanisms')
        if evidence.get('peak_unaffected_moving_while_confirmed',0)>0:coverage.add('unaffected_traffic_during_obstruction')
        remaining.append((folder,row,spec,coverage))
    selected=[];covered=set()
    while remaining and len(selected)<count:
        entry=max(remaining,key=lambda x:(len(x[3]-covered),x[1]['traffic']['peak_active']))
        selected.append(entry);remaining.remove(entry);covered.update(entry[3])
    scenes=[];sources={}
    for folder,row,spec,_ in selected:
        meta=json.loads((folder/'metadata.json').read_text());surface=wkt.loads(meta['surface_wkt'])
        ps=surface.geoms if hasattr(surface,'geoms') else [surface]
        native=[f for i,f in enumerate(read_rows(folder/'physics.jsonl.gz')) if i%5==0]
        refs=[]
        for f in read_rows(folder/'reference.jsonl.gz'):
            refs.append(dict(t=f['time_s'],witnesses=[dict(a=w['requester'],b=w['blocker'],resource=w['resource']['id'],
                mechanism=w.get('mechanism'),polygons=polygons(wkt.loads(w['resource']['polygon_wkt'])))
                for w in f['witnesses'] if w['road_admissible']]))
        scenes.append(dict(id=row['id'],label=row['label'],family=row['family'],reason=row['reason'],
            scopes=row['scopes'],events=row['events'],partition=splits.get(row['id'],'development'),
            surface=[[list(map(list,p.exterior.coords)),*[list(map(list,r.coords)) for r in p.interiors]] for p in ps],
            bounds=list(surface.bounds),lane_width=spec['topology']['lane_width_m'],
            lanes=[r['points'] for r in spec['roads'] if r['kind'] in ('in','out')],
            vehicles={k:[v['length_m'],v['width_m']] for k,v in meta['vehicles'].items()},
            dimensions=', '.join(f'{a:.3f} × {b:.3f} m' for a,b in sorted({(v['length_m'],v['width_m']) for v in meta['vehicles'].values()})),
            routes={k:[p for lane in lanes for p in lane['points']] for k,lanes in meta['route_lanes'].items()},
            frames=[dict(t=f['time_s'],states={k:v[:3] for k,v in f['states'].items()}) for f in native],references=refs,
            proofs=os.path.relpath(folder,output).replace(os.sep,'/'),
            limits='Référence spatiale du mouvement ; alternatives non évaluées. Validation scientifique humaine encore requise.',
            **{k:row['traffic'][k] for k in ('peak_active','total_introduced','arrived')}))
        sources[row['id']]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
    payload=json.dumps(scenes,separators=(',',':'),ensure_ascii=False).replace('<','\\u003c')
    (output/'index.html').write_text(HTML.replace('__DATA__',payload),encoding='utf-8')
    report=dict(episodes=len(scenes),coverage=sorted(covered),labels=dict(Counter(s['label'] for s in scenes)),
        render_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_files_sha256=sources,native_sample_s=.1,interpolation=False,physical_graphical_scale_equal=True,
        index_sha256=hashlib.sha256((output/'index.html').read_bytes()).hexdigest())
    write_json(output/'manifest.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path);parser.add_argument('--count',type=int,default=16)
    args=parser.parse_args()
    if args.count<1:parser.error('Count must be positive')
    result=build(args.dataset,args.output_dir,args.count)
    print(json.dumps({k:v for k,v in result.items() if k!='source_files_sha256'},indent=2))
