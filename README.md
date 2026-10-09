# CAV Gridlock Recovery

Ce projet étudie les blocages collectifs entre véhicules connectés et automatisés.
Son objectif principal est de rétablir la circulation après leur formation ;
la prévention reste secondaire.

## Réseau et trafic

La topologie vient d'OpenStreetMap autour de Kintambo Magasin, à Kinshasa.
Elle conserve plusieurs avenues, des chaussées séparées et de nombreuses
intersections proches. Le réseau actif se limite au noyau, à ses approches et
aux liaisons locales utiles ; l'extrait source reste complet. Les voies,
vitesses et feux sont des hypothèses de simulation.

© [OpenStreetMap contributors — ODbL 1.0](https://www.openstreetmap.org/copyright).
La petite géométrie nécessaire est fournie avec le projet ; la préparation
fonctionne hors ligne. Sa provenance figure dans
[scenario.json](src/cav_recovery/scenarios/kintambo/scenario.json).

Python génère les arrivées par intervalles exponentiels (processus de Poisson)
avec `random.Random(seed)`. Les missions, routes et destinations sont fixées
avant le départ. Même configuration et même seed reproduisent ces entrées.

`LOW`, `MEDIUM`, `HIGH` et `STRESS` désignent une intensité demandée, pas un
diagnostic. La configuration indique la durée, les intensités par entrée, les
destinations et leurs poids. Le temps tiré reste dans `sampled_s` ; sa programmation
SUMO `scheduled_s` est arrondie vers le haut à la milliseconde. L'insertion réelle
peut être retardée.

## Installation

Python 3.12 ou ultérieur est nécessaire. L'installation apporte SUMO et TraCI
1.27.1 ; sumolib est fourni par leurs dépendances.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements.txt
```

Sous PowerShell : `.venv\Scripts\Activate.ps1`.

## Utilisation

Depuis le dépôt, dans l'environnement activé :

```bash
python scripts/run_traffic.py --demand LOW --seed 1
python scripts/run_traffic.py --demand HIGH --seed 1 --gui --gui-delay-ms 100
python scripts/run_traffic.py --demand HIGH --seed 1 --gui --street-names
```

Une seule commande prépare le réseau, génère les missions et lance SUMO.
Chaque exécution conserve ses fichiers dans un nouveau dossier.
`--output-dir <nouveau-dossier>` permet de choisir l'emplacement ;
un dossier non vide est refusé.

`--rate` règle l'intensité en véhicules/heure/par entrée.
`--duration-s` fixe la période d'injection et `--drain-horizon-s`
l'attente après injection. `--config <configuration.json>` fournit d'autres
paramètres de simulation. Consulter `--help` pour les options.

La GUI utilise réellement `sumo-gui`, démarre automatiquement et se centre sur
le noyau. Les noms OSM sont masqués par défaut ; `--street-names` les affiche.
Une interface compatible est nécessaire, par exemple WSLg.
Les voitures utilisent une forme native de SUMO. Le délai graphique ne change
pas le pas simulé de 0,5 s. Dans un terminal interactif, appuyer sur Entrée
après observation pour fermer la vue à la fin.

## Sorties et contrôles

Les sorties comprennent `summary.json`, `vehicles.csv`, `timeline.csv`,
`lanes.csv`, `observations.jsonl`, `tripinfo.xml` et `sumo.log`.
Les configurations, les corrections de connexions et les journaux de conversion
sont aussi conservés.

Le bilan distingue programmés, départs réels, insertions retardées, actifs et
arrivées à destination. Téléportation, collision, disparition ou destination
modifiée invalident l'exécution. Aucun véhicule n'est retiré pour obtenir un succès.

Une fin à l'horizon est distincte d'une vidange complète. Codes de sortie :
0 pour une vidange, 3 pour l'horizon atteint, 1 pour un échec technique ou
d'intégrité, 2 pour une entrée refusée. Les files et occupations sont des
observations ; elles ne constituent pas à elles seules une preuve de gridlock.

## C-RDG

Le C-RDG représente les dépendances observées entre véhicules et espace aval.
Il représente aussi certains conflits de carrefour à partir des observations
natives de SUMO. Il lit la simulation sans agir sur le trafic. Pour l'activer :

```bash
python scripts/run_traffic.py --demand HIGH --seed 1 --crdg
```

Il produit `crdg.jsonl` (graphes successifs), `crdg_events.jsonl` (apparitions,
disparitions et changements de cause), `crdg_summary.json` (bilan) et
`crdg_peak.json` (snapshot retenu). Il distingue les cycles structurels des
candidats fermés sans alternative de réception observable hors du groupe.
Aucun de ces candidats n'est une preuve de gridlock. L'âge d'une dépendance
est distinct de la durée d'arrêt du véhicule. Le calcul et l'export ont des
cadences séparées, réglables dans la partie `crdg` de la configuration.

Pour suivre aussi l'évolution physique des attentes :

```bash
python scripts/run_traffic.py --demand LOW --seed 1 --blockage-evidence
```

Cette option active le C-RDG et ajoute `blockage_events.jsonl` et
`blockage_summary.json`. Le suivi conserve les déplacements, les changements
de cause, les permissions de passage et les arrivées observées. Une longueur
de véhicule parcourue termine un épisode de faible progression, sans prouver
que toute la file est libérée. Les situations encore ouvertes à l'horizon sont
censurées, pas déclarées permanentes. Un vert ou une voie légalement accessible
ne prouve pas que le passage est matériellement possible ; les inconnues restent
explicites. Aucun gridlock confirmé n'est déclaré.

## Circulation ciblée sur Kintambo

Kintambo reste le réseau principal. Trois variantes de demande utilisent ses
routes, voies et feux canoniques sans les modifier : `clearance` vise une
vidange, `crossing` concentre des mouvements croisés et `spillback` vise des
contraintes entre plusieurs carrefours. Les mécanismes et preuves attendues
figurent dans [cases.json](src/cav_recovery/scenarios/kintambo/cases.json).
Ces variantes ont des missions finies et des départs explicites seedés, pas
une augmentation uniforme du Poisson. Leur nom ne présume aucun résultat.

```bash
python scripts/run_traffic.py --kintambo-case clearance --seed 1 --blockage-evidence
python scripts/run_traffic.py --kintambo-case crossing --seed 1 --blockage-evidence
python scripts/run_traffic.py --kintambo-case spillback --seed 1 --blockage-evidence
```

Ne pas combiner une variante avec `--demand`, `--rate`, `--duration-s` ou
`--config`. `--drain-horizon-s` peut prolonger l'observation des mêmes missions.
Sans `--kintambo-case`, les expériences Poisson restent inchangées.

## Dépendances dans SUMO-GUI

```bash
python scripts/run_traffic.py --kintambo-case crossing --seed 1 \
  --crdg-scene-at 81.5 --crdg-focus montagne_000012 --crdg-gui
```

Cette commande termine l'expérience puis ouvre son **état natif figé à
81,5 s** dans SUMO-GUI, avec les annotations du même calcul C-RDG. Elle ne
dessine pas une relation ancienne sur un trafic qui a déjà avancé.
Répéter `--crdg-scene-at` pour enregistrer plusieurs instants et examiner
l'évolution ; choisir des multiples de la cadence de calcul (0,5 s par défaut).
Les snapshots habituels `crdg.jsonl` restent exportés à 5 s.

`--crdg-focus <ID>` choisit la voiture ; `--crdg-depth 1|2|3` règle son voisinage.
Un anneau bleu marque cette voiture. Les repères orange et les flèches montrent
ses dépendances proches, sur les vraies voies et les véhicules de l'état SUMO.
Les ressources sont repérées sur leurs voies natives, pas par une zone de
conflit recalculée. Les textes français expliquent les arcs et les paramètres
des POI conservent leur preuve. Les repères V/R ont leurs IDs complets dans
les paramètres et les fichiers de scène. La vue est limitée à 16 nœuds ; toute
limitation ou cause inconnue est signalée, sans inventer de flèche.

On peut cliquer sur les vrais véhicules dans SUMO pour examiner leurs données,
zoomer et déplacer la vue. **Ne pas lancer le trafic de cette vue figée** :
elle représente uniquement l'instant enregistré. Fermer la fenêtre termine
sa consultation. Pour changer le véhicule mis en évidence, rouvrir la même
scène avec `--focus` ; aucun graphe ni trafic n'est recalculé :

```bash
python scripts/view_sumo_crdg.py <sortie>/crdg_scenes/81.5 --list-vehicles
python scripts/view_sumo_crdg.py <sortie>/crdg_scenes/81.5 --focus montagne_000012
```

Sans `--crdg-gui`, l'enregistrement reste sans fenêtre ; le rejeu s'ouvre plus
tard avec `view_sumo_crdg.py`. Les scènes nécessitent les fichiers de leur
exécution d'origine, dont le réseau. Les empreintes de l'état et du réseau
sont vérifiées. Ce mode est une inspection fidèle d'instants choisis, pas
encore une animation synchronisée en direct ni un diagnostic de gridlock.

## Croisements secondaires

Des scénarios synthétiques séparés servent de tests secondaires pour l'attente à une
priorité, la rétention d'approches secondaires, des arrivées simultanées et
le spillback entre deux croisements. Les paramètres et le mécanisme visé sont
dans [cases.json](src/cav_recovery/scenarios/intersections/cases.json).
Ils utilisent des missions finies et des départs explicites reproductibles,
pas le Poisson de Kintambo. Le type de véhicule et les contrôles d'intégrité
restent les mêmes. Un nom de scénario n'est jamais une preuve de son résultat.

```bash
python scripts/run_traffic.py --scenario priority_wait --seed 1 --crdg
python scripts/run_traffic.py --scenario priority_starvation --seed 1 --crdg
python scripts/run_traffic.py --scenario mutual_yield --seed 1 --crdg
python scripts/run_traffic.py --scenario junction_spillback --seed 1 --crdg
```

Ajouter `--gui` pour voir la circulation ou `--blockage-evidence` pour suivre
les épisodes. `--drain-horizon-s` peut prolonger l'observation sans changer les
missions. `--demand`, `--rate`, `--duration-s` et `--config` sont réservés à
Kintambo, qui reste le choix par défaut. `priority_wait_holdout` est une variante
réservée à une future évaluation : ne pas l'utiliser pour régler un détecteur.

## Vue schématique secondaire du C-RDG

Après une exécution, remplacer le chemin ci-dessous par le dossier indiqué :

```bash
python scripts/view_crdg.py outputs/simulation/mon-essai/crdg.jsonl
```

Ouvrir le fichier `crdg_view.html` créé près de la source dans Chrome, Edge ou
un autre navigateur. La page est autonome et fonctionne hors connexion.
Choisir un instant exporté, saisir l'ID d'une voiture puis cliquer sur
« Centrer ». Cliquer sur un nœud ou une flèche pour lire ses attributs.
La molette zoome ; glisser le fond déplace la vue. Pour un gros fichier,
`--time 285` garde uniquement cet instant s'il a réellement été exporté.
`--output <nouveau-fichier.html>` choisit la destination sans écraser un fichier.

Les véhicules, ressources, arcs et candidats viennent du C-RDG source.
La disposition est schématique, pas une carte SUMO. Les grands graphes demandent
une focalisation explicite ; aucune relation n'est inventée ou supprimée du
fichier. Les exports ne représentent pas tous les calculs internes et les
cycles restent des candidats, pas des diagnostics confirmés.

## Tests

```bash
python -m compileall -q src scripts tests
python -m pytest -q
python scripts/check_sumo.py
```

Les intégrations nécessitent les exécutables SUMO dans l'environnement activé.
Un test ignoré ne valide pas l'intégration. Le contrôle technique court utilise
une fixture synthétique indépendante du réseau de Kintambo.

## Limites et suite

Les paramètres de trafic sont simulés et ne constituent pas une calibration
du trafic réel de Kintambo. La calibration sur des données réelles constitue
une perspective.

Le réseau, les arrivées et le suivi des missions sont disponibles.
Les phénomènes observés doivent être distingués d'un gridlock confirmé.
Le diagnostic final de gridlock, la récupération autonome, l'environnement RL
et Graph-MARL ne sont pas encore implémentés.
