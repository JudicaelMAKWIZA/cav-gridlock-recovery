"""Crée les cinq cahiers légers ; leurs sorties réelles restent dans l'espace privé."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)
COMMON = """from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from cav_recovery.laboratory.workbench import Laboratory
from cav_recovery.laboratory.storage import json_file

lab = Laboratory.from_environment()
display(HTML(lab.table_html([{'résultats': str(lab.root), 'M02': str(lab.dataset)}])))
"""


def write(name, parts):
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = dict(
        display_name="Python (environnement CAV Recovery)",
        language="python",
        name="python3",
    )
    notebook.metadata.language_info = dict(name="python", version="3.12")
    notebook.cells = [
        nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text)
        for kind, text in parts
    ]
    nbf.write(notebook, OUT / name)


write(
    "01_donnees_et_qualification.ipynb",
    [
        (
            "md",
            """# 1. Données et qualification physique

**Question.** Que contient réellement M02 et que signifie une cible temporelle ?

Le corpus gelé précède l'entraînement. Ses observations locales et ses annotations physiques ont des rôles séparés. Nous consultons les manifestes, sans simuler ni requalifier les épisodes. Les chemins se configurent par `CGR_DATA_ROOT`, `CGR_LAB_ROOT`, `CGR_M02_ROOT` et `CGR_HISTORICAL_ROOT`. Choisir l'interpréteur Windows du projet dans VS Code.
""",
        ),
        ("code", COMMON),
        (
            "md",
            """## Effectifs et partitions

La partition est attribuée au parent géométrique, avant les fenêtres. Les trois inconnus sont retirés de la supervision et exportés séparément. Les effectifs initiaux de `splits.json` et les effectifs supervisés ne doivent pas être additionnés.
""",
        ),
        (
            "code",
            """partitions = lab.dataset_overview()
display(HTML(lab.table_html(partitions)))
fig = lab.plot_dataset()
plt.show()
""",
        ),
        (
            "md",
            """## Cible disponible maintenant, pas étiquette rétrospective comme entrée

Le TCN estime `persistent_confirmation_available_at_end` : la confirmation physique est-elle déjà disponible à la fin de cette fenêtre ? Il n'apprend pas à prédire une intervention obligatoire et ne remplace pas une preuve K04. Les labels `retrospective_persistent_event` et `future_followup_required` restent des annotations.

Les 176 mesures potentielles couvrent 35 s à 5 Hz. Le préfixe manquant est complété à gauche avec un masque, sans inventer de mesures. Chaque fenêtre appartient à un seul CAV et à son mouvement ; les identifiants servent aux jointures, jamais aux canaux du modèle.
""",
        ),
        (
            "code",
            """manifest = json_file(lab.dataset / 'export-final' / 'manifest.json')
display(HTML(lab.table_html([
    {'rôle': 'Entrées autorisées', 'fichiers': ', '.join(manifest['model_input_files'])},
    {'rôle': 'Cibles privilégiées', 'fichiers': ', '.join(manifest['privileged_target_files'])},
])))
cache = json_file(lab.cache / 'complete.json')
display(HTML(lab.table_html([{'cible': cache['target'], 'oracle comme entrée': cache['oracle_as_input'],
                           'identité du corpus': cache['source_id']}])) )
print('Canaux réellement utilisés :', ', '.join(cache['features']))
""",
        ),
        (
            "md",
            """## Normalisation et biais

La normalisation est ajustée sur les seules observations de train. Les positions absolues, topologies, graines, identifiants et labels ne sont pas des variables. La vitesse et la perception peuvent néanmoins refléter des biais du générateur ; une bonne mesure sur ce corpus ne suffit pas à valider un système routier réel.
""",
        ),
        (
            "code",
            """normalisation = json_file(lab.cache / 'normalization.json')
display(HTML(lab.table_html([{'partition ajustée': normalisation['fitted_partition'],
                            'observations': normalisation['observations'],
                            'canaux': len(normalisation['features'])}])))
limites = json_file(lab.dataset / 'production-final' / 'coverage-limits.json')
display(HTML(lab.table_html([
    {'constat': 'Négatifs avec effet établi ≥ 5 s', 'nombre': limites['negatives_with_effect_at_least_5s']},
    {'constat': 'Changements d’occupants', 'nombre': limites['changing_occupant_episodes']},
    {'constat': 'Confirmations nécessitant la succession de bloqueurs', 'nombre': limites['confirmations_requiring_successive_blockers']},
])))
""",
        ),
        (
            "md",
            """**Interprétation.** Les partitions sont intactes ; le cache est une dérivation contrôlée, pas un nouveau dataset physique. L'asymétrique reste le test OOD M02. Les négatifs difficiles et certaines successions de bloqueurs restent insuffisamment représentés. Le champ historique M02 `persistence_s=5` désigne la fenêtre d'effet, et `nominal_followup_s=30` la durée qualifiante ; K04 nomme ces mêmes durées `effect_s=5` et `persistence_s=30`.
""",
        ),
    ],
)

write(
    "02_evaluation_d01.ipynb",
    [
        (
            "md",
            """# 2. Évaluer D01 sans confondre abstention et absence

**Question.** Que signifient précision élevée et rappel faible dans les 32 essais historiques ?

A utilise les informations propres et le LiDAR ; B ajoute les messages réellement reçus. C conserve le diagnostic individuel B et ajoute le graphe. Les 32 essais ont été choisis pour le développement et la démonstration ; ils ne constituent pas un test indépendant de généralisation.
""",
        ),
        ("code", COMMON),
        (
            "md",
            """## Comptes, tâches et formules

Une vraie confirmation prédite est TP ; une confirmation prédite sans référence est FP ; une confirmation de référence non détectée, y compris avec abstention, est FN. Précision = TP/(TP+FP), rappel = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN). Une précision sans prédiction positive reste non définie.

Les valeurs suivantes sont agrégées par la même fonction d'évaluation que la campagne, à partir de ses 32 fichiers de résultats. Les nombreux rendez-vous sont corrélés : l'unité expérimentale reste l'épisode.
""",
        ),
        (
            "code",
            """mesures = lab.historical_metrics()
display(HTML(lab.table_html([m for m in mesures if m['task'] in ('A', 'B')])))
display(HTML(lab.table_html(lab.historical_abstentions())))
""",
        ),
        (
            "md",
            """## Pourquoi le rappel reste limité

Les reculs physiques, même faibles, peuvent sortir du domaine nominal K04. Une absence d'identification ou de preuve géométrique recevable impose également l'abstention. Le contrat n'est pas abaissé pour augmenter le score. Les 62 faux positifs de B restent dans les mesures, même si l'audit les situe sous le seuil surfacique M02.

Le correctif de l'horloge C-RDG refuse une preuve propre antidatée par le graphe. Il ne fournit pas de gain de rappel sur les flux historiques correctement datés. Les fixtures F01 A/B/C restent des tests de contrat distincts des configurations de comparaison A/B/C.
""",
        ),
        (
            "code",
            """evenements = [dict(essai=r['name'], **e) for r in lab.historical_episodes() for e in r['event_results']]
retrouves = [e for e in evenements if e['mode']=='B' and not e['missed']]
manques = [e for e in evenements if e['mode']=='B' and e['missed']]
display(HTML(lab.table_html([{'retrouvés B': len(retrouves), 'manqués B': len(manques),
    'délai médian conditionnel (s)': float(np.median([e['delay_s'] for e in retrouves])),
    'délai maximal conditionnel (s)': max(e['delay_s'] for e in retrouves)}])))
""",
        ),
        (
            "md",
            """## Comparaison appariée avec les modèles temporels

La nouvelle sous-étude utilise exactement les mêmes endpoints, cibles disponibles maintenant et groupes tenus à part pour D01 et le TCN. La cadence est de 2 s sur des rendez-vous de preuve valides. Ces chiffres ne remplacent donc pas les anciens scores à 0,4 s sur les 32 essais.
""",
        ),
        (
            "code",
            """display(HTML(lab.table_html(lab.model_results('integrated'))))
fig = lab.plot_quality('integrated')
plt.show()
display(HTML(lab.table_html(lab.transfer_results())))
display(HTML(lab.table_html(lab.replay_costs())))
""",
        ),
        (
            "md",
            """**Interprétation.** Une alerte statistique TCN peut couvrir des abstentions ou produire de nouvelles erreurs. Elle ne transforme pas une preuve K04 manquante en confirmation physique. La fusion « D01 B ou TCN » est une règle d'alerte statistique étudiée séparément, sans action de récupération.
""",
        ),
    ],
)

write(
    "03_evaluation_crdg.ipynb",
    [
        (
            "md",
            """# 3. Dépendances et groupes C-RDG

**Question.** Reconstruit-on des liens pertinents, des sous-groupes valides ou des groupes complets ?

Ces tâches ont des dénominateurs différents. Une vue locale à un saut peut être correcte et incomplète. L'absence de groupe établi localement n'affirme pas son absence globale. Le graphe consomme les diagnostics propres et réellement reçus ; l'oracle reste un évaluateur hors ligne.
""",
        ),
        ("code", COMMON),
        (
            "md",
            """## Liens et groupes : séparer validité et complétude

B est ici le comptage simple de demandeurs confirmés. C ajoute leur cohérence spatiale. Les métriques de groupes exacts sanctionnent un sous-groupe correct mais incomplet ; la validité partielle répond à une autre question.
""",
        ),
        (
            "code",
            """mesures = lab.historical_metrics()
display(HTML(lab.table_html([m for m in mesures if m['task'] in ('links_global','links_observable','B_groups','C_groups')])))
fig = lab.plot_historical_graphs()
plt.show()
""",
        ),
        (
            "md",
            """## Un groupe collectif peut être acyclique

Dans `collective-1`, graine 21019, à 56,4 s, la vue de V1 contient V3 → ressource → V1 → ressource → V4. Deux demandeurs sont contraints, trois CAV sont liés et aucun cycle n'est nécessaire. Les ressources sont des espaces requis par les mouvements, pas une seconde classe de véhicules.

La preuve est lue ci-dessous dans le résultat indépendant enregistré ; aucune arête n'est inventée pour le dessin.
""",
        ),
        (
            "code",
            """collectif = next(r for r in lab.historical_episodes() if r['name']=='collective-1')
display(HTML(lab.table_html(collectif['acyclic_collective_appointments'])))
audit = json_file(lab.historical / 'final-audits' / 'additional-metrics.json')
display(HTML(lab.table_html([{'référence': k, **v} for k,v in audit['indirect_metrics'].items()])))
from IPython.display import Image
capture = lab.root / 'evidence' / 'interface_0512af2' / 'graph.png'
if capture.exists():
    display(Image(filename=str(capture)))
    print('Vue acceptée à 60 s, collective-1, graine 21019, commit 0512af2. '
          'La première preuve acyclique ci-dessus est à 56,4 s.')
""",
        ),
        (
            "md",
            """## Apport au TCN : une ablation explicite

La variante TCN brute n'a aucun canal D01/C-RDG. La variante D01 ajoute quatre statuts locaux. La variante D01+C-RDG ajoute le nombre de relations actuelles et de groupes observés localement. Ce sont des connaissances accessibles au CAV à cet instant, pas les liens de référence globaux.
""",
        ),
        (
            "code",
            """from cav_recovery.laboratory.features import BASE_FEATURES, DIAGNOSTIC_FEATURES, GRAPH_FEATURES
display(HTML(lab.table_html([
    {'variante': 'TCN brut', 'canaux': len(BASE_FEATURES), 'extension': 'aucune'},
    {'variante': 'TCN + D01', 'canaux': len(BASE_FEATURES)+4, 'extension': ', '.join(DIAGNOSTIC_FEATURES)},
    {'variante': 'TCN + D01 + C-RDG', 'canaux': len(BASE_FEATURES)+6, 'extension': ', '.join(GRAPH_FEATURES)},
])))
display(HTML(lab.table_html(lab.model_results('integrated'))))
""",
        ),
        (
            "md",
            """**Interprétation.** Un gain éventuel du modèle temporel grâce à ces deux compteurs ne démontre pas une reconstruction complète du graphe ni une racine causale dynamique unique. Les scores historiques exacts et le faible rappel indirect restent conservés.
""",
        ),
    ],
)

write(
    "04_entrainement_tcn.ipynb",
    [
        (
            "md",
            """# 4. Entraînement réel d'un TCN causal sur CPU

**Question.** Le passé local permet-il d'estimer qu'une confirmation physique est disponible maintenant ?

Ce notebook charge les résultats effectivement obtenus. Son exécution normale ne lance aucun entraînement. Le TCN ne prédit ni une étiquette future fournie en entrée, ni une action de récupération. Il utilise des convolutions causales dilatées et un masque de préfixe ; les 7 dilatations donnent un champ réceptif de 255 mesures, suffisant pour couvrir les 176 mesures possibles.
""",
        ),
        ("code", COMMON),
        (
            "md",
            """## Protocole gelé avant l'accès au test

Les poids sont ajustés sur train ; les statistiques de normalisation sont ajustées sur train uniquement. Deux tailles TCN sont comparées sur la BCE de validation, avec une baseline instantanée. Le seuil statistique est calibré sur validation. Le test OOD M02 n'intervient ni dans la sélection des poids, ni dans celle de l'architecture ou du seuil.
""",
        ),
        (
            "code",
            """protocole = json_file(lab.campaign / 'protocol.json')
selection = json_file(lab.campaign / 'selected-model.json')
display(HTML(lab.table_html(protocole['trials'])))
display(HTML(lab.table_html([
    {'paramètre': cle, 'valeur': protocole[cle]}
    for cle in ['history_samples','sample_s','seed','threads','batch_size','epochs',
                'train_stride','validation_selection_stride','learning_rate',
                'dilations','kernel_size','checkpoint_steps','patience']
])))
display(HTML(lab.table_html([selection])))
fig = lab.plot_losses()
plt.show()
""",
        ),
        (
            "md",
            """## Architecture, normalisation et modèle réellement sauvegardé

Les mêmes fonctions construisent le modèle dans la CLI et dans le notebook. Les canaux bruts n'incluent ni identifiants, ni coordonnées absolues, ni statut D01/C-RDG. La baseline instantanée reçoit les mêmes variables au dernier instant ; elle permet de mesurer l'intérêt de l'historique, sans changer la tâche.
""",
        ),
        (
            "code",
            """from cav_recovery.laboratory.model import create_model
modele_dir = lab.campaign / 'm02' / selection['selected']
configuration = json_file(modele_dir / 'config.json')
resultat = json_file(modele_dir / 'complete.json')
modele = create_model(configuration, resultat['compatibility']['features'])
from cav_recovery.laboratory.checkpoints import CheckpointStore
etat, preuve = CheckpointStore(modele_dir / 'checkpoints', resultat['compatibility']).load(best=True)
modele.load_state_dict(etat['model'])
modele.eval()
print(modele)
display(HTML(lab.table_html([{'paramètres': resultat['parameters'], 'époques terminées': resultat['completed_epochs'],
                            'fenêtres train utilisées': resultat['train_windows'],
                            'identité données': resultat['compatibility']['dataset']}])) )
print('Empreinte du checkpoint chargé :', preuve['receipt']['sha256'])
print('Durée murale enregistrée (s) :', resultat['wall_s'])
anomalies = lab.root / 'timing-anomalies.json'
if anomalies.exists():
    display(HTML(lab.table_html([json_file(anomalies)])))
from cav_recovery.laboratory.training import WindowData, predict
validation = WindowData(lab.cache, 'validation')
indices_exemples = validation.indices()[:8]
probabilites, _ = predict(modele, validation, indices_exemples)
display(HTML(lab.table_html([
    {'instant (s)': float(validation.windows['time_s'][i]),
     'probabilité calculée': float(p), 'cible indépendante': int(validation.windows['y'][i])}
    for i,p in zip(indices_exemples, probabilites)
])))
""",
        ),
        (
            "md",
            """## Coupure et reprise réellement exercées

Les checkpoints contiennent les poids, l'optimiseur, les RNG, l'époque, le curseur du mélange, les métriques et l'identité des données. Chaque génération est validée avant publication atomique du pointeur ; une précédente génération valide reste disponible. Le premier entraînement réel a subi une sortie brutale du processus après trois lots, puis a repris au curseur enregistré.

La garantie est une reprise au dernier lot sauvegardé. Les lots non sauvegardés peuvent être recalculés. Cette infrastructure pourra conserver des états supplémentaires Graph-MARL ; elle ne promet pas une restauration d'un épisode physique en cours.
""",
        ),
        (
            "code",
            """preuve_reprise = json_file(lab.campaign / 'power-cut-proof.json')
display(HTML(lab.table_html([preuve_reprise])))
assert preuve_reprise['completed'] and preuve_reprise['actual_process_terminated']
assert preuve_reprise['resume_records'] and preuve_reprise['stale_locks_preserved']
""",
        ),
        (
            "md",
            """## Résultats sur le test OOD réservé

Les fenêtres corrélées ne sont pas autant d'expériences indépendantes. Consulter aussi les métriques par épisode, les inconnus masqués et les événements manqués. Une estimation correcte de confirmation maintenant ne démontre pas une anticipation future.
""",
        ),
        (
            "code",
            """display(HTML(lab.table_html(lab.model_results('m02'))))
fig = lab.plot_confusion('m02')
plt.show()
""",
        ),
        (
            "md",
            """## Mode pédagogique facultatif

Le code suivant appelle le même entraîneur, dans un nouveau dossier, avec une seule époque et un échantillonnage plus léger. Laisser ce mode désactivé pour une présentation de consultation. Il ne remplace pas le modèle sélectionné et ne change aucun résultat historique.
""",
        ),
        (
            "code",
            """LANCER_DEMONSTRATION = False
if LANCER_DEMONSTRATION:
    from datetime import datetime
    from cav_recovery.laboratory.training import train
    configuration_legere = dict(configuration, epochs=1, train_stride=100,
                                validation_selection_stride=100, checkpoint_steps=10)
    sortie_distincte = lab.root / ('pedagogie-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    resultat_leger = train(lab.cache, sortie_distincte, configuration_legere)
    display(HTML(lab.table_html([{'dossier distinct': str(sortie_distincte),
                                'époques': resultat_leger['completed_epochs']}])) )
""",
        ),
    ],
)

write(
    "05_comparaisons_systeme.ipynb",
    [
        (
            "md",
            """# 5. Comparaisons du système et preuves pour le chapitre 3

**Question.** Quel apport est mesuré sur chaque tâche, avec quel périmètre ?

Deux expériences restent distinctes : le TCN et sa baseline sur le test OOD M02 réservé, puis les ablations appariées sur des groupes tenus à part dans la collection choisie de démonstrations. On ne classe pas des méthodes en comparant le F1 d'un groupe exact avec celui d'une confirmation individuelle.
""",
        ),
        ("code", COMMON),
        (
            "md",
            """## TCN et baseline : même cible, mêmes fenêtres M02

Les seules différences sont l'architecture et l'utilisation du passé. Les sorties ci-dessous sont lues dans les résultats sauvegardés ; les fonctions de graphique sont aussi celles de l'export de preuves.
""",
        ),
        (
            "code",
            """display(HTML(lab.table_html(lab.model_results('m02'))))
fig = lab.plot_quality('m02')
plt.show()
""",
        ),
        (
            "md",
            """## Apport des diagnostics et des graphes, sur les mêmes observations

Les variantes apparentées restent dans une même partition de démonstration. Le découpage est stratifié au niveau des parents sur leurs références physiques établies, sans utiliser les prédictions. Le nom interne `test_ood` de ce cache désigne seulement des groupes de démonstration tenus à part ; ce n'est pas le test OOD M02.
""",
        ),
        (
            "code",
            """display(HTML(lab.table_html(lab.model_results('integrated'))))
fig = lab.plot_quality('integrated')
plt.show()
""",
        ),
        (
            "md",
            """## Délais et échecs temporels

Le délai est mesuré après l'instant où la confirmation physique est disponible. Les événements sans endpoint recevable sont censurés ; ceux observables mais non détectés restent manqués. Les préalertes statistiques sont dénombrées séparément et restent des faux positifs pour la tâche « confirmation maintenant » avant cet instant.
""",
        ),
        (
            "code",
            """evenements = lab.event_analysis('m02')
display(HTML(lab.table_html(evenements[:12])))
fig = lab.plot_event_delays(evenements)
plt.show()
fig = lab.plot_motion()
plt.show()
""",
        ),
        (
            "md",
            """## Limites et mise à jour du mémoire

Maintenir : les résultats historiques défavorables, les abstentions K04, l'incomplétude des groupes, les biais du générateur, les masques de supervision et la différence entre alerte statistique et preuve physique. L'étude ne développe pas Graph-MARL et ne démontre aucune récupération obligatoire.

Le dossier privé `evidence` contient les figures, CSV, sorties réelles des notebooks et `MISE_A_JOUR_CHAPITRE3_DETECTION_TCN.md`. Les commandes appartiennent aux cahiers et au dépôt, pas au texte scientifique du chapitre 3.
""",
        ),
        (
            "code",
            """dossier_preuves = lab.root / 'evidence'
print('Preuves sauvegardées :', dossier_preuves)
print('Mise à jour académique :', dossier_preuves / 'MISE_A_JOUR_CHAPITRE3_DETECTION_TCN.md')
print('Notebooks exécutés :', dossier_preuves / 'notebooks_executed')
""",
        ),
    ],
)
print("Cinq notebooks créés, sans sortie privée dans le dépôt.")
