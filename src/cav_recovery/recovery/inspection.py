"""Figures de récupération tirées des traces natives, sans relancer MetaDrive."""

from collections import Counter
from pathlib import Path
import json
import os

import numpy as np

from ..laboratory.storage import file_hash, json_file
from ..metadrive.observation import rectangle
from .evaluation import rows


class ConstructiveStudy:
    """Lecture seule des essais, y compris les échecs et fragments conservés."""

    def __init__(self, root=None):
        self.root = Path(root or os.environ["CGR_CONSTRUCTIVE_ROOT"])

    def inventory(self):
        records = []
        for path in sorted(self.root.rglob("summary.json")):
            data = json_file(path)
            if not isinstance(data, dict) or "reverse_m" not in data:
                continue
            folder = path.parent
            record = dict(
                run=folder.relative_to(self.root).as_posix(),
                method=data["method"],
                seed=data["seed"],
                scenario=data["scenario"],
                observed_s=data["at_s"],
                introduced=data["introduced"],
                arrivals=len(data["arrivals"]),
                peak_active=data["peak_active"],
                errors=len(data["errors"]),
                minimum_gap_m=data["minimum_gap_m"],
                reverse_max_m=max(data["reverse_m"].values(), default=0),
                label=data["physical_label"],
                global_actor=data["global_audit_in_actor"],
                cpu_s=data["cpu_s"],
                wall_s=data["wall_s"],
            )
            records.append(record)
        return records

    def verify(self):
        """Vérifie les fichiers déclarés, pas une intégrité physique implicite."""
        checked, errors = 0, []
        for path in sorted(self.root.rglob("manifest.json")):
            manifest = json_file(path)
            for name, expected in manifest.get("files", {}).items():
                checked += 1
                target = path.parent / name
                if not target.is_file() or file_hash(target) != expected:
                    errors.append(str(target))
        return dict(checked=checked, errors=errors)

    def command_counts(self, run):
        commands = list(rows(self.root / run / "commands.jsonl.gz"))
        return dict(
            total=len(commands),
            accepted=sum(r["guard"]["accepted"] for r in commands),
            refusals=dict(
                Counter(
                    r["guard"]["reason"] for r in commands if not r["guard"]["accepted"]
                )
            ),
            applied=dict(Counter(r["applied"]["reason"] for r in commands)),
        )

    def plot_sequence(self, run, times=(30, 69, 80, 110, 140, 160)):
        """Corps à leur taille réelle, positions natives ; aucune image inventée."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from shapely import wkt

        folder = self.root / run
        metadata = json_file(folder / "metadata.json")
        surface = wkt.loads(metadata["surface_wkt"])
        frames = list(rows(folder / "physics.jsonl.gz"))
        selected = [min(frames, key=lambda f: abs(f["at_s"] - t)) for t in times]
        figure, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
        vehicles = metadata["vehicles"]
        for axis, frame in zip(axes.flat, selected):
            for polygon in getattr(surface, "geoms", [surface]):
                axis.add_patch(Polygon(np.array(polygon.exterior.coords), color="0.88"))
            for ego, state in frame["states"].items():
                dimensions = vehicles[ego]
                footprint = rectangle(
                    state[:2], state[2], dimensions["length_m"], dimensions["width_m"]
                )
                color = plt.get_cmap("tab10")(int(ego.removeprefix("agent")) % 10)
                axis.add_patch(
                    Polygon(np.array(footprint.exterior.coords), color=color)
                )
                axis.text(
                    state[0],
                    state[1] + 1.6,
                    ego.replace("agent", "V"),
                    fontsize=9,
                    ha="center",
                    clip_on=True,
                )
            axis.set(
                xlim=(-35, 35),
                ylim=(-30, 30),
                title=f"t = {frame['at_s']:.1f} s",
                xlabel="x (m)",
                ylabel="y (m)",
            )
            axis.set_aspect("equal")
        summary = json_file(folder / "summary.json")
        figure.suptitle(
            f"{run} · graine {summary['seed']}\n"
            "Reconstruction des positions et empreintes natives enregistrées"
        )
        return figure

    def plot_movement(self, run):
        import matplotlib.pyplot as plt

        folder = self.root / run
        reference = list(rows(folder / "reference.jsonl.gz"))
        commands = list(rows(folder / "commands.jsonl.gz"))
        figure, axes = plt.subplots(
            3, 1, figsize=(12, 9), sharex=True, layout="constrained"
        )
        for ego in sorted({ego for frame in reference for ego in frame["states"]}):
            values = [
                (frame["time_s"], frame["states"][ego])
                for frame in reference
                if ego in frame["states"]
            ]
            axes[0].plot(
                [t for t, _ in values], [s["progress_m"] for _, s in values], label=ego
            )
            axes[1].plot(
                [t for t, _ in values], [s["speed_mps"] for _, s in values], label=ego
            )
            selected = [r for r in commands if r["ego"] == ego]
            axes[2].step(
                [r["at_s"] for r in selected],
                [r["applied"]["speed_mps"] for r in selected],
                label=ego,
            )
        for axis, label in zip(
            axes,
            [
                "Progression nette (m)",
                "Vitesse native (m/s)",
                "Consigne réellement appliquée (m/s)",
            ],
        ):
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            axis.legend(ncol=4)
        axes[-1].set_xlabel("Temps simulé (s)")
        figure.suptitle(run)
        return figure

    def plot_comparison(self, comparison):
        import matplotlib.pyplot as plt

        result = json_file(self.root / comparison)
        figure, axis = plt.subplots(figsize=(11, 4), layout="constrained")
        for name, label in [("nominal", "Nominal"), ("intervention", "Intervention")]:
            timeline = result[name].get("timeline", [])
            axis.plot(
                [r["at_s"] for r in timeline],
                [r["fraction"] for r in timeline],
                label=f"{label} : {result[name]['status']}",
            )
        if result["t_act_s"] is not None:
            axis.axvline(
                result["t_act_s"], color="gray", ls=":", label="Première intervention"
            )
        confirmed = result["intervention"]["t_conf_s"]
        if confirmed is not None:
            axis.axvline(
                confirmed, color="green", ls="--", label="Progression qualifiante"
            )
        axis.set(
            xlabel="Temps simulé (s)",
            ylabel="Fraction de A0 ayant progressé",
            ylim=(-0.03, 1.06),
            title="Une fraction égale à 1 ne suffit pas à prouver la durabilité",
        )
        axis.legend(fontsize=8)
        return figure

    def plot_protocol(self, run):
        import matplotlib.pyplot as plt

        messages = [
            r
            for r in rows(self.root / run / "messages.jsonl.gz")
            if r.get("kind") not in {None, "STATE"}
        ]
        identities = sorted({r["sender"] for r in messages})
        kinds = sorted({r["kind"] for r in messages})
        figure, axis = plt.subplots(figsize=(12, 4), layout="constrained")
        for kind in kinds:
            selected = [r for r in messages if r["kind"] == kind]
            axis.scatter(
                [r.get("sent_s", r.get("at_s")) for r in selected],
                [identities.index(r["sender"]) for r in selected],
                s=14,
                label=kind,
            )
        axis.set_yticks(range(len(identities)), identities)
        axis.set(
            xlabel="Temps simulé (s)",
            ylabel="Émetteur",
            title="Livraisons V2V enregistrées ; un accord ne commande pas le voisin",
        )
        axis.legend(ncol=4, fontsize=8)
        return figure

    def plot_pilot(self, pilot="graph-pilot-v2"):
        import matplotlib.pyplot as plt

        history = json_file(self.root / pilot / "history.json")
        figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for entry in history:
            decisions = json_file(
                self.root / pilot / entry["run"] / "neural-decisions.json"
            )
            axes[0].scatter(
                [d["at_s"] for d in decisions],
                [d["action"] for d in decisions],
                s=24,
                label=f"Épisode {entry['episode']}",
            )
        axes[0].set(
            xlabel="Temps simulé (s)",
            ylabel="Option neuronale choisie",
            title="Choix effectifs, dont accord/refus",
        )
        axes[0].legend()
        axes[1].plot(
            [e["episode"] for e in history],
            [e["learning"]["reward_sum"] for e in history],
            "o-",
        )
        axes[1].set(
            xlabel="Épisode de développement",
            ylabel="Récompense partagée",
            title="Deux épisodes : aucune tendance\nd'apprentissage établie",
        )
        return figure

    def plot_dependencies(self, run, times=(69, 80, 140)):
        """Référence physique hors ligne, explicitement distincte du C-RDG acteur."""
        import matplotlib.pyplot as plt
        import networkx as nx
        from ..experiments.evaluation import active_events

        annotation = json_file(self.root / run / "physical-annotation.json")
        figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        for axis, at in zip(axes, times):
            graph = nx.DiGraph()
            for event in active_events(annotation, at):
                resource = (event["requester"], event["resource_id"])
                graph.add_node(resource, resource=True)
                graph.add_edge(event["requester"], resource)
                graph.add_edge(resource, event["blocker"])
            if not graph:
                axis.text(
                    0.5,
                    0.5,
                    "Aucune relation persistante\nqualifiée à cet instant",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            else:
                positions = nx.spring_layout(graph, seed=17)
                for is_resource, shape, color in [
                    (False, "o", "#8ac5ee"),
                    (True, "s", "#ffd58c"),
                ]:
                    nodes = [n for n in graph if isinstance(n, tuple) == is_resource]
                    nx.draw_networkx_nodes(
                        graph,
                        positions,
                        nodelist=nodes,
                        node_shape=shape,
                        node_color=color,
                        node_size=900,
                        ax=axis,
                    )
                labels = {
                    n: "espace requis\n" + n[0]
                    if isinstance(n, tuple)
                    else n.replace("agent", "V")
                    for n in graph
                }
                nx.draw_networkx_edges(graph, positions, arrows=True, ax=axis)
                nx.draw_networkx_labels(graph, positions, labels, font_size=8, ax=axis)
            axis.set_title(f"t = {at:.1f} s")
            axis.axis("off")
        figure.suptitle(
            "Véhicule → ressource → occupant : référence physique globale hors acteur"
        )
        return figure

    def write_player(self, runs, output):
        """Lecteur autonome de traces, sans simulateur, téléportation ou recalcul."""
        from shapely import wkt
        from ..laboratory.storage import safe_destination

        output = Path(output)
        for run in runs:
            safe_destination(self.root / run, output)
        data = []
        for run in runs:
            folder = self.root / run
            metadata = json_file(folder / "metadata.json")
            surface = wkt.loads(metadata["surface_wkt"])
            data.append(
                dict(
                    name=run,
                    vehicles=metadata["vehicles"],
                    roads=[
                        list(p.exterior.coords)
                        for p in getattr(surface, "geoms", [surface])
                    ],
                    frames=list(rows(folder / "physics.jsonl.gz")),
                    summary=json_file(folder / "summary.json"),
                )
            )
        template = """<!doctype html><html lang="fr"><meta charset="utf-8">
<title>Récupération — traces physiques natives</title>
<style>body{font:16px system-ui;margin:20px;background:#f7f8fa;color:#172434}
canvas{background:white;width:min(94vw,1000px);border:1px solid #aaa}
button,select,input{font:inherit;margin:5px;padding:5px}#time{width:45vw}
pre{white-space:pre-wrap;max-width:1000px}</style>
<h1>Lecture des traces physiques enregistrées</h1>
<p>Référence globale hors ligne pour inspection. Aucun état global n'est ajouté aux acteurs.
Les dimensions et positions viennent des mesures natives, en mètres.</p>
<select id="run"></select><button id="play">Lecture</button>
<select id="speed"><option>1</option><option selected>4</option><option>10</option></select> ×
<input id="time" type="range" min="0" step="1" value="0"><b id="stamp"></b><br>
<canvas id="scene" width="1000" height="760"></canvas><pre id="details"></pre>
<script>const data=__DATA__, select=document.getElementById('run'), slider=document.getElementById('time');
const canvas=document.getElementById('scene'), ctx=canvas.getContext('2d');
let running=false,last=0,t=0; const colors=['#146bb1','#bd563d','#35834a','#893ca0','#986d1c'];
data.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.text=r.name;select.add(o)});
function draw(){let r=data[select.value],f=r.frames[+slider.value];
ctx.clearRect(0,0,1000,760);ctx.save();ctx.translate(500,380);ctx.scale(6,-6);
for(let road of r.roads){ctx.beginPath();road.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));
ctx.closePath();ctx.fillStyle='#dadde1';ctx.fill()}
for(let [id,s] of Object.entries(f.states)){let v=r.vehicles[id];ctx.save();ctx.translate(s[0],s[1]);
ctx.rotate(s[2]);ctx.fillStyle=colors[+id.replace('agent','')%colors.length];
ctx.fillRect(-v.length_m/2,-v.width_m/2,v.length_m,v.width_m);ctx.strokeStyle='white';ctx.lineWidth=.1;
ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(v.length_m/2,0);ctx.stroke();ctx.restore();
ctx.save();ctx.translate(s[0],s[1]+2);ctx.scale(1,-1);ctx.fillStyle='#111';ctx.font='2px sans-serif';
ctx.textAlign='center';ctx.fillText(id.replace('agent','V'),0,0);ctx.restore()}
ctx.strokeStyle='#172434';ctx.lineWidth=.2;ctx.beginPath();ctx.moveTo(-75,-55);ctx.lineTo(-65,-55);ctx.stroke();
ctx.restore();document.getElementById('stamp').textContent=f.at_s.toFixed(1)+' s';
document.getElementById('details').textContent='Graine : '+r.summary.seed+' | '+r.summary.decision_scope+
'\\nArrivées natives (temps simulé en s) : '+JSON.stringify(r.summary.arrivals)+
'\\nErreurs natives : '+JSON.stringify(r.summary.errors)+' | Échelle : segment de 10 m';}
function reset(){running=false;document.getElementById('play').textContent='Lecture';t=0;
slider.max=data[select.value].frames.length-1;slider.value=0;draw()}
select.onchange=reset;slider.oninput=()=>{t=data[select.value].frames[+slider.value].at_s;draw()};
document.getElementById('play').onclick=()=>{running=!running;document.getElementById('play').textContent=running?'Pause':'Lecture'};
function tick(now){let dt=last?(now-last)/1000:0;last=now;if(running){t+=dt*Number(document.getElementById('speed').value);
let frames=data[select.value].frames;while(+slider.value<frames.length-1&&frames[+slider.value+1].at_s<=t)slider.value=+slider.value+1;
if(+slider.value===frames.length-1)running=false;draw()}requestAnimationFrame(tick)}reset();requestAnimationFrame(tick);
</script></html>"""
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            template.replace("__DATA__", json.dumps(data).replace("<", "\\u003c")),
            encoding="utf-8",
        )
        return output
