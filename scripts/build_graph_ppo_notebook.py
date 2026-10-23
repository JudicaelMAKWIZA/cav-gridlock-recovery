"""Construit le notebook Graph-PPO sans démarrer de simulation ni d'entraînement."""

from pathlib import Path

import nbformat as nbf


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "notebooks/12_ENTRAINEMENT_GRAPH_PPO_COMPLET.ipynb"
ROOT = Path("D:/cav-recovery-data/graph-ppo-training-20261023")
notebook = nbf.v4.new_notebook()
cells = []


def section(title, explanation, code):
    cells.append(nbf.v4.new_markdown_cell(f"## {title}\n\n{explanation}"))
    cells.append(nbf.v4.new_code_cell(code))


cells.append(nbf.v4.new_markdown_cell("""# Graph-PPO multi-CAV — campagne de développement

Ce notebook prépare et suit un apprentissage personnel sur des scènes MetaDrive **physiquement requalifiées**. L'acteur voit uniquement ses observations locales et les messages V2V reçus. P4 négocie et maintient les engagements ; REC-PHYS-3 vérifie les commandes avant la physique native. Les annotations de l'oracle sont réservées à la qualification et aux mesures après l'épisode.

**Ne lancer la cellule « Démarrer ou reprendre » qu'après le verdict `TRAIN_READY` et un précontrôle réussi.** L'entraînement long n'a pas été exécuté lors de la préparation. Le test historique réservé reste fermé. Une amélioration de récompense ne prouve pas à elle seule une récupération physique.

Sous Windows, utiliser le noyau `D:/cav-recovery-venvs/metadrive/Scripts/python.exe`. Après redémarrage du noyau, réexécuter d'abord la cellule d'imports. La campagne se déroule dans un processus distinct : relancer la cellule de démarrage pendant qu'il vit ne crée pas de second producteur."""))

section("01 — Imports et chemins", "Torch doit être importé avant tout module susceptible de charger MetaDrive sous Windows.", f"""import sys, time, json, shutil
from pathlib import Path
PROJECT = Path(r'{PROJECT}')
ROOT = Path(r'{ROOT}')
CONFIG = ROOT / 'experiment-config.json'
if str(PROJECT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT / 'src'))
import torch
from IPython.display import display
from cav_recovery.recovery.graph_marl_core import load_config, scene_plan
from cav_recovery.recovery.graph_ppo_campaign import readiness_gate
from cav_recovery.recovery.graph_marl_notebook import (
    environment_check, process_status, launch, request_stop,
    recent_history, render_figures, final_manifest)
print('Interpréteur :', sys.executable)
print('Projet :', PROJECT, '\\nCampagne :', ROOT)
""")

section("02 — Identité et porte scientifique", "La configuration, les sources et la qualification doivent avoir exactement les empreintes approuvées.", """from cav_recovery.recovery.graph_marl_training import _prepare
report = readiness_gate(ROOT)
config, _ = _prepare(CONFIG, ROOT)
display({'verdict': report['verdict'], 'commit': report.get('git_commit'),
         'scènes entraînement': len(config['scenes']['train']),
         'scènes validation': len(config['scenes']['validation']),
         'validation périodique': config['periodic_validation_ids']})
""")

section("03 — Environnement et ressources", "Vérifie Python, Torch, MetaDrive, les dépendances, la mémoire et le stockage sur D:.", """diagnostic = environment_check(PROJECT, ROOT)
display(diagnostic)
display({'RAM disponible Gio': round(__import__('psutil').virtual_memory().available / 1024**3, 2),
         'D: libre Gio': round(shutil.disk_usage(ROOT.anchor).free / 1024**3, 2)})
""")

section("04 — Catalogue figé", "Les labels viennent du nominal requalifié. Les épisodes répétés sont comptés séparément des configurations uniques.", """config = load_config(CONFIG)
for split in ('preflight', 'train', 'validation'):
    rows = config['scenes'][split]
    counts = {label: len({row['source_id'] for row in rows
                          if row['qualified_label'] == label})
              for label in ('POSITIVE_CONFIRMED', 'NEGATIVE_CONFIRMED')}
    print(split, 'épisodes', len(rows), 'configurations uniques', counts)
for row in config['scenes']['validation']:
    print('validation', row['source_id'], row['source_family'], row['qualified_label'],
          'graine', row['seed'], 'affectés', len(row['qualified_affected']))
""")

section("05 — Acteur, options et masque", "Le réseau reçoit un graphe local. Le masque est calculé par l'agent coopératif ; le garde et les missions demeurent actifs.", """from cav_recovery.recovery.graph_policy import FEATURES, GraphActorCritic
from cav_recovery.recovery.cooperative import option_bank
from cav_recovery.recovery.graph_marl_policy import P4GraphPolicy
from cav_recovery.recovery.verified_passage_policy import GuardEnvelopePassagePolicy
model = GraphActorCritic(config['hidden'])
assert issubclass(P4GraphPolicy, GuardEnvelopePassagePolicy)
print('Caractéristiques par nœud :', FEATURES)
print('Paramètres :', sum(p.numel() for p in model.parameters()))
for index, option in enumerate(option_bank()):
    print(index, option)
""")

section("06 — Récompense et PPO", "Le bonus collectif dépend d'une preuve physique après l'épisode. Les options sont actualisées selon leur durée réelle ; les cas censurés utilisent une valeur locale de continuation.", """display({'récompense': config['reward'],
         'gamma par pas': config['gamma_per_step'],
         'lambda GAE': config['gae_lambda'],
         'taille de lot': config['ppo_batch_size'],
         'minibatch': config['minibatch_size'],
         'époques PPO': config['ppo_epochs'],
         'budget total minutes': config['maximum_wall_minutes']})
print('Les annotations globales ne font pas partie des entrées du réseau.')
""")

section("07 — Précontrôle physique", "Un seul épisode qualifié vérifie les alternatives réelles, les commandes et les traces. Il peut durer plusieurs minutes. Un rapport existant est réutilisé.", """from cav_recovery.recovery.graph_marl_training import preflight
gate = ROOT / 'reports/preflight.json'
if gate.exists():
    result = json.loads(gate.read_text(encoding='utf-8'))
else:
    result = preflight(CONFIG, ROOT)
display(result)
assert result['proceed'], 'Précontrôle non concluant : ne pas démarrer.'
""")

section("08 — Estimation avant lancement", "Le budget de six heures inclut les épisodes et les validations périodiques. Les estimations reposent sur les temps de qualification mesurés.", """qualification = json.loads((Path(config['qualification_root']) / 'qualification.json').read_text(encoding='utf-8'))['scenes']
ids = {row['source_id'] for row in config['scenes']['train']}
durations = [row['summary']['wall_s'] for row in qualification
             if row['split'] == 'train' and row['source_id'] in ids]
estimate = sum(durations) / len(durations)
print('Temps nominal moyen mesuré (s) :', round(estimate, 1))
print('Ordre de grandeur sans validation périodique :',
      int(360 * 60 / estimate), 'épisodes en six heures')
print('Cette projection ne garantit ni le nombre de lots PPO ni la récupération.')
""")

section("09 — État et absence de doublon", "Consulter le vrai processus, le journal et la progression avant toute relance.", """state = process_status(ROOT)
display(state)
if state['process']:
    print('Journal :', state['process']['log'])
""")

section("10 — Démarrer ou reprendre personnellement", "Cette cellule est la seule qui lance l'entraînement long. La porte `TRAIN_READY` est revérifiée dans le notebook et dans le producteur.", """readiness_gate(ROOT)
state = process_status(ROOT)
if state['alive']:
    print('Producteur déjà actif ; aucun second lancement. PID :', state['worker_pid'] or state['process']['pid'])
else:
    record = launch('train', PROJECT, ROOT)
    display({'PID': record['pid'], 'journal': record['log'], 'début': time.ctime(record['started_at'])})
""")

section("11 — Suivi des épisodes et de la sécurité", "Relancer cette cellule pendant l'entraînement : elle lit les checkpoints, arrivées, décisions utiles, refus et coûts sans démarrer de producteur.", """state = process_status(ROOT)
history = recent_history(ROOT)
display({'état': state['effective_state'], 'épisodes terminés': len(history),
         'processus actif': state['alive'], 'progression': state['experiment']})
for row in history[-8:]:
    print(row['index'] + 1, row['name'], row['qualified_label'],
          'arrivées', row['validity']['arrived'],
          'choix libres', row['useful_choices'],
          'updates', row['learning']['updates'],
          'reward', round(row['reward'], 3),
          'récupération', row.get('physical', {}).get('recovery_status'),
          'refus garde', row.get('physical', {}).get('guard_refusals'),
          'recul total', round(row.get('physical', {}).get('reverse_total_m', 0), 2),
          'valide', row['validity']['valid'])
""")

section("12 — Courbes de récompense et pertes PPO", "Les courbes utilisent exclusivement les mesures déjà enregistrées ; une absence de mise à jour reste visible.", """import matplotlib.pyplot as plt
history = recent_history(ROOT)
if not history:
    print('Aucun épisode terminé.')
else:
    x = [row['index'] + 1 for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(x, [row['reward'] for row in history], '.-')
    axes[0].set(title='Récompense observée', xlabel='Épisode', ylabel='Somme')
    points = [(row['index'] + 1, loss) for row in history
              for loss in row['learning'].get('losses', [])]
    for key in ('policy', 'value', 'total', 'entropy'):
        axes[1].plot([p[0] for p in points], [p[1][key] for p in points],
                     '.', label=key)
    axes[1].set(title='Pertes à chaque mise à jour PPO', xlabel='Épisode')
    axes[1].legend()
    display(fig)
    plt.close(fig)
""")

section("13 — Récupérations et validation périodique", "La récupération durable et les comparaisons sont des mesures physiques, distinctes de la récompense.", """history = recent_history(ROOT)
for row in history[-10:]:
    receipt = ROOT / row['folder'] / 'reward-summary.json'
    recovery = json.loads(receipt.read_text(encoding='utf-8')).get('physical_recovery') if receipt.exists() else None
    print(row['index'] + 1, row['name'], 'récupération',
          None if recovery is None else recovery['status'],
          'sécurité', row['validity']['problems'])
for path in sorted((ROOT / 'validation/periodic').glob('after-*/summary.json')):
    record = json.loads(path.read_text(encoding='utf-8'))
    print('Validation après épisode', record['episode_index'],
          [(entry['source_id'], entry['paired']['improvement_not_spontaneous'])
           for entry in record['outcomes']])
""")

section("14 — Checkpoints et arrêt propre", "Une demande d'arrêt attend la frontière d'épisode ; aucune interruption brutale ni suppression de trace.", """latest = ROOT / 'checkpoints/latest.json'
print('Dernier checkpoint :', latest.read_text(encoding='utf-8') if latest.exists() else 'aucun')
STOP_NOW = False  # Mettre True et relancer cette cellule pour demander l'arrêt.
if STOP_NOW:
    print(request_stop(ROOT))
""")

section("15 — Reprise après arrêt", "Après arrêt à la frontière, relancer la cellule 10 : elle recharge poids, optimiseur, lot PPO en attente et états aléatoires.", """state = process_status(ROOT)
display({'état': state['effective_state'], 'actif': state['alive'],
         'checkpoint': state['experiment'].get('checkpoint') if state['experiment'] else None})
RESUME_NOW = False  # Mettre True et relancer cette cellule après un arrêt propre.
if RESUME_NOW and not state['alive']:
    readiness_gate(ROOT)
    display(launch('train', PROJECT, ROOT))
""")

section("16 — Validation finale appariée", "À la fin ou au budget de six heures, compare sur les scènes de validation fixes le nominal, B3, P4 et deux checkpoints Graph-PPO. Cette phase peut durer longtemps et ne met pas à jour les poids.", """state = process_status(ROOT)
if state['alive']:
    print('Attendre la fin du producteur avant la validation.')
elif state['experiment'] and state['experiment'].get('state') in {'training_complete', 'wall_budget_at_boundary'}:
    if (ROOT / 'reports/validation_summary.json').exists():
        print('Validation finale déjà terminée.')
    else:
        display(launch('validate', PROJECT, ROOT))
else:
    print('Entraînement non terminé ou aucun checkpoint sélectionnable.')
""")

section("17 — Figures, tableaux et manifeste", "Exporter uniquement les mesures effectivement disponibles, puis conserver les preuves et les échecs.", """if recent_history(ROOT):
    display(render_figures(ROOT))
    display(final_manifest(ROOT))
else:
    print('Aucun résultat à exporter.')
print('Dossier privé :', ROOT)
""")

notebook.cells = cells
notebook.metadata = {"kernelspec": {"display_name": "CGR MetaDrive (Python 3.12 sur D:)",
                                     "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}}
nbf.validate(notebook)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
