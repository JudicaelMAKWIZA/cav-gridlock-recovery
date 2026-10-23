"""Construit le notebook exécutable sans lancer la moindre expérience."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/11_ENTRAINEMENT_GRAPH_MARL_INTERACTIF.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def section(number, title, explanation, code):
    cells.append(nbf.v4.new_markdown_cell(
        f"## {number:02d} — {title}\n\n{explanation}"))
    cells.append(nbf.v4.new_code_cell(code))


cells.append(nbf.v4.new_markdown_cell("""# Graph-MARL local + P4 + REC-PHYS-3

**Expérience de développement à exécuter personnellement.** MetaDrive simule la physique ; chaque CAV décide à partir de son état, de son LiDAR, des messages V2V reçus et des diagnostics locaux admissibles. Un réseau de graphes partagé choisit une option de dégagement parmi les options masquées. P4 garde la négociation, la confirmation de passage et la reprise ; REC-PHYS-3 garde la sécurité. Aucun oracle global n'entre dans l'acteur.

Le précontrôle, l'entraînement, la validation et les figures sont **réservés à l'étudiant**. Seules les étapes 01–02 ont été exécutées pendant la préparation. Les scènes de développement peuvent échouer ou rester sans obstruction. Une fin de processus ne signifie jamais récupération scientifique. Le test réservé reste fermé.

Exécuter 01–09 dans l'ordre, puis relancer 10 pour suivre. Après l'entraînement, 11–16. La cellule 09 démarre un processus externe unique pour laisser le noyau libre ; un arrêt se fait à la frontière d'épisode. Après un redémarrage de noyau, reprendre à 01 puis à la cellule voulue. **Importer Torch avant MetaDrive** sur Windows."""))

BOOT = """import sys
from pathlib import Path
PROJECT = Path(r'D:\\cav-recovery-graph-marl')
ROOT = Path(r'D:\\cav-recovery-data\\graph-marl-training-20261023')
CONFIG = ROOT / 'experiment-config.json'
if str(PROJECT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT / 'src'))
import torch  # ordre impératif sous Windows : Torch avant MetaDrive
"""

section(1, "Imports sans simulation",
        "Cette cellule initialise un noyau vierge. La relancer après tout redémarrage.",
        BOOT + """import json, time
from IPython.display import display, HTML
from cav_recovery.recovery.graph_marl_notebook import (
    environment_check, process_status, launch, request_stop,
    recent_history, render_figures, final_manifest)
from cav_recovery.recovery.graph_marl_core import load_config, scene_plan
print('Imports prêts ; aucune simulation lancée.')""")

section(2, "Environnement Windows",
        "Vérifie l'interpréteur sur D:, les bibliothèques, la mémoire et l'espace libre. Une erreur arrête la préparation.",
        BOOT + """from cav_recovery.recovery.graph_marl_notebook import environment_check
diagnostic = environment_check(PROJECT, ROOT)
display(diagnostic)
if diagnostic['memory_available_gib'] < 4:
    print('Réserve : moins de 4 Gio de RAM libres ; fermer les applications lourdes'
          ' avant les épisodes à huit CAV. Le producteur refuse chaque épisode'
          ' si moins de 2,0 Gio (2–4 CAV) ou 2,5 Gio (8 CAV) sont disponibles.')""")

section(3, "Configuration versionnée",
        "Un seul fichier fixe algorithme, scènes, graines, horizons, budget et récompenses.",
        BOOT + """from cav_recovery.recovery.graph_marl_core import load_config
config = load_config(CONFIG)
display({key: value for key, value in config.items() if key != 'scenes'})
print('Checkpoints :', ROOT / 'checkpoints')""")

section(4, "Scènes et partitions",
        "Les descriptions sont déterministes ; leurs noms ne sont pas des étiquettes physiques. Les parents de validation sont distincts de ceux d'entraînement.",
        BOOT + """from cav_recovery.recovery.graph_marl_core import load_config, scene_plan
config = load_config(CONFIG)
for scene in scene_plan(config):
    topology = scene['topology']
    print(scene['split'], scene['name'], 'graine', scene['seed'],
          'CAV', scene['vehicles'], 'horizon', scene['horizon_s'],
          'bras', topology['arms'], 'voies', topology['lane_num'],
          'parent', scene['parent_id'])
print('Aucun label positif présumé ; aucune scène du test réservé.')""")

section(5, "Agent, graphe et masque",
        "L'entrée a 24 caractéristiques par nœud et une matrice d'adjacence variable. Le masque contient l'attente et les options locales admissibles ; le réseau choisit l'option, P4 garde les accords et REC-PHYS-3 vérifie la commande.",
        BOOT + """from cav_recovery.recovery.graph_policy import FEATURES, GraphActorCritic
from cav_recovery.recovery.cooperative import option_bank
from cav_recovery.recovery.graph_marl_policy import P4GraphPolicy
from cav_recovery.recovery.verified_passage_policy import GuardEnvelopePassagePolicy
config = load_config(CONFIG)
agent = GraphActorCritic(config['hidden'])
print(agent)
print('Caractéristiques/nœud :', FEATURES,
      'options :', len(option_bank()),
      'paramètres :', sum(p.numel() for p in agent.parameters()))
assert issubclass(P4GraphPolicy, GuardEnvelopePassagePolicy)
for index, option in enumerate(option_bank()):
    print(index, option)
print('Pas de décision physique déduite sans LocalView réelle.')""")

section(6, "Récompense exacte",
        "La récompense est calculée hors acteur après le pas natif. Le bonus collectif est nul tant qu'une récupération n'est pas établie par comparaison appariée. Un message DONE `beneficiaries_passed` n'est compté qu'une fois par tentative.",
        BOOT + """weights = load_config(CONFIG)['reward']
print('r_t = progression / longueur + passage confirmé + arrivée - coût du pas'
      ' - recul cumulé - attente improductive - événement invalide')
display(weights)
assert weights['collective_recovery'] == 0.0""")

section(7, "Modèle et checkpoint",
        "Initialise des poids ou lit une génération vérifiée. Cette opération ne lance ni entraînement ni simulateur.",
        BOOT + """from cav_recovery.recovery.graph_marl_training import initialize
from cav_recovery.recovery.graph_marl_core import weight_hash
config, model, optimizer, store, saved, receipt = initialize(CONFIG, ROOT)
display({'architecture': str(model), 'parameters': sum(p.numel() for p in model.parameters()),
         'training_seed': config['training_seed'], 'weights_sha256': weight_hash(model),
         'resume_receipt': receipt, 'completed_episodes':
         len(saved['history']) if saved else 0})""")

section(8, "Précontrôle physique — exécution par l'étudiant",
        "**Cette cellule lance le premier épisode MetaDrive.** Elle vérifie le garde, les décisions neuronales, la présence d'au moins une alternative physique admissible, les récompenses et les sorties. Elle indique séparément les commandes coopératives réellement appliquées. Un verdict défavorable bloque 09. Ce contrôle n'est pas une validation scientifique.",
        BOOT + """from cav_recovery.recovery.graph_marl_training import preflight
verdict = preflight(CONFIG, ROOT)
display(verdict)
if not verdict['proceed']:
    raise RuntimeError('Précontrôle défavorable : entraînement bloqué. Lire le dossier preflight et le rapport.')""")

section(9, "Démarrer l'entraînement — exécution par l'étudiant",
        "Démarre un seul processus de fond. La réexécution refuse un second producteur. Les épisodes se suivent sans rendu permanent ; le budget est vérifié entre épisodes.",
        BOOT + """from cav_recovery.recovery.graph_marl_notebook import launch
record = launch('train', PROJECT, ROOT)
display({'démarrage': time.ctime(record['started_at']), 'PID': record['pid'],
         'campagne': ROOT.name, 'épisodes prévus': len(load_config(CONFIG)['scenes']['train']),
         'journal': record['log']})""")

section(10, "Suivre la campagne",
        "Relancer cette cellule pendant le processus. Récompense et arrivées restent deux mesures distinctes ; l'ETA dépend des épisodes réellement achevés.",
        BOOT + """from cav_recovery.recovery.graph_marl_notebook import process_status, recent_history
state = process_status(ROOT)
history = recent_history(ROOT)
display(state)
for row in history:
    losses = row['learning'].get('losses', [])
    recent = [item['reward'] for item in history[max(0, row['index']-2):row['index']+1]]
    print('Épisode', row['index'] + 1, row['name'], 'r=', round(row['reward'], 3),
          'r moyen 3=', round(sum(recent)/len(recent), 3),
          'arrivées=', row['validity']['arrived'],
          'valide=', row['validity']['valid'],
          'décisions=', row['learning']['decisions'],
          'MAJ=', row['learning']['updates'],
          'perte=', losses[-1] if losses else None,
          'recul max=', row['validity']['reverse_max_m'])
    print('  erreurs/collisions natives :', row['summary']['errors'],
          'entropie PPO :', losses[-1]['entropy'] if losses else None)
if not history: print('Aucun épisode terminé : aucune tendance mesurable.')
if history:
    from cav_recovery.recovery.inspection import ConstructiveStudy
    print('Dernier épisode — garde et commandes :',
          ConstructiveStudy(ROOT).command_counts(history[-1]['folder']))
print('Checkpoints :', [p.name for p in (ROOT / 'checkpoints').glob('checkpoint-*.pt')])
if state['process']:
    log = Path(state['process']['log'])
    if log.exists(): print('Journal récent :', '\\n'.join(log.read_text(errors='replace').splitlines()[-12:]))""")

section(11, "Relire les trajectoires réelles",
        "Construit un lecteur HTML autonome à partir des positions physiques natives, y compris les échecs. Les décisions, demandes V2V et refus sont affichés à leur instant enregistré ; leurs journaux complets restent disponibles.",
        BOOT + """from cav_recovery.recovery.inspection import ConstructiveStudy
from cav_recovery.recovery.graph_marl_notebook import write_review
history = recent_history(ROOT)
if not history:
    print('Aucune trace complète à lire.')
else:
    runs = [row['folder'] for row in history]
    target = ROOT / 'figures' / 'LECTEUR_EPISODES.html'
    write_review(ROOT, runs, target)
    print('Lecteur des positions et dimensions natives :', target)
    print('Décisions, commandes et messages :', ROOT / runs[-1])
    display(HTML('<a href="' + target.as_uri() + '">Ouvrir le lecteur</a>'))""")

section(12, "Arrêter ou reprendre sans écraser",
        "Pour arrêter, exécuter la ligne commentée `request_stop(ROOT)` puis attendre la fin de l'épisode. Reprise : relancer 09 après l'arrêt. Le checkpoint vérifié et l'état aléatoire reprennent à la prochaine frontière ; un fragment reste conservé.",
        BOOT + """from cav_recovery.recovery.graph_marl_notebook import process_status, request_stop
display(process_status(ROOT))
print('Arrêt volontaire : exécuter request_stop(ROOT) dans cette cellule.')
# request_stop(ROOT)  # retirer le # seulement pour demander un arrêt sûr
print('Reprise : relancer la cellule 09 après que alive soit False.')""")

section(13, "Validation distincte — exécution par l'étudiant",
        "À lancer après l'entraînement. Deux derniers checkpoints sont comparés sans gradient sur des parents de validation distincts. La sélection privilégie les gains REC-EVAL-1 face au nominal, puis les arrivées valides. Le checkpoint retenu est comparé au nominal, à B3 et à P4 avec les mêmes graines et horizons. Le test réservé reste interdit.",
        BOOT + """from cav_recovery.recovery.graph_marl_notebook import launch, process_status
state = process_status(ROOT)
if state['alive']:
    raise RuntimeError('Attendre la fin de la tâche active.')
if not state['experiment'] or state['experiment']['state'] != 'training_complete':
    raise RuntimeError('Entraînement incomplet ou arrêté : ne pas valider comme modèle final.')
record = launch('validate', PROJECT, ROOT)
display({'PID': record['pid'], 'journal': record['log'],
         'sortie': str(ROOT / 'reports' / 'validation_summary.json')})""")

section(14, "Tableaux et figures observés — exécution par l'étudiant",
        "À lancer après la fin de la validation. Les graphiques proviennent exclusivement des fichiers produits ; aucune courbe attendue n'est substituée.",
        BOOT + """from cav_recovery.recovery.graph_marl_notebook import render_figures, recent_history, process_status
if process_status(ROOT)['alive']:
    raise RuntimeError('Attendre la fin du processus actif.')
display([{'épisode': r['index']+1, 'scène': r['name'], 'récompense': r['reward'],
          'arrivées': r['validity']['arrived'], 'valide': r['validity']['valid'],
          'décisions': r['learning']['decisions']} for r in recent_history(ROOT)])
print('Figure réelle :', render_figures(ROOT))
validation = ROOT / 'reports' / 'validation_summary.json'
print('Validation :', json.loads(validation.read_text(encoding='utf-8')) if validation.exists()
      else 'absente ou incomplète')""")

section(15, "Bilan et manifeste — exécution par l'étudiant",
        "Inventorie les épisodes terminés, la validation réellement achevée et les empreintes SHA-256. Les incidents et épisodes incomplets restent visibles ; aucun succès scientifique n'est présumé.",
        BOOT + """from cav_recovery.recovery.graph_marl_notebook import final_manifest, recent_history, process_status
if process_status(ROOT)['alive']:
    raise RuntimeError('Attendre la fermeture du processus avant le manifeste.')
manifest = final_manifest(ROOT)
history = recent_history(ROOT)
display({'épisodes terminés': len(history),
         'épisodes valides': sum(row['validity']['valid'] for row in history),
         'épisodes invalides': sum(not row['validity']['valid'] for row in history),
         'mises à jour': sum(row['learning']['updates'] for row in history),
         'validation présente': manifest['validation_present'],
         'fichiers empreintés': len(manifest['files']),
         'anomalies': [row['validity']['problems'] for row in history
                        if not row['validity']['valid']]})""")

section(16, "Fin et transmission",
        "Quand toutes les étapes choisies sont achevées, transmettre seulement « L'entraînement est terminé. Analyse les résultats et prépare le rapport scientifique du chapitre 3. » Les fichiers locaux suffisent pour l'analyse ultérieure.",
        BOOT + """from cav_recovery.recovery.graph_marl_notebook import process_status
state = process_status(ROOT)
validation = ROOT / 'reports' / 'validation_summary.json'
print('Expérience terminée. Les résultats et checkpoints ont été enregistrés dans :', ROOT)
print('Processus actif :', state['alive'])
print('Entraînement :', state['experiment']['state'] if state['experiment'] else 'non lancé')
print('Validation :', 'effectuée' if validation.exists() else 'absente ou incomplète')
print('Dossier complet pour analyse :', bool(validation.exists() and
                                            (ROOT / 'final/artifact_manifest.json').exists()))
print('Aucune réussite scientifique n’est déduite automatiquement.')""")

notebook.cells = cells
notebook.metadata = dict(kernelspec=dict(
    display_name="CGR MetaDrive (Python 3.12 sur D:)",
    language="python", name="cgr-metadrive-d"),
    language_info=dict(name="python", version="3.12.10"))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
