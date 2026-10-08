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
