# CAV Gridlock Recovery

Ce projet étudie les blocages collectifs entre véhicules connectés et automatisés.
L'objectif principal est de rétablir la circulation après leur formation ;
la prévention reste secondaire.

## Banc de simulation

Le benchmark utilise une topologie OpenStreetMap autour de **Kintambo Magasin,
à Kinshasa**, avec plusieurs avenues, des chaussées séparées et de nombreuses
intersections voisines. Les voies, vitesses, feux et volumes sont synthétiques :
ce n'est pas une reproduction calibrée du trafic réel de Kintambo.

Python tire des inter-arrivées exponentielles (processus de Poisson) avec une
seed explicite, puis affecte des routes et destinations fixes. Même configuration
et même seed reproduisent les missions. Aucune donnée pNEUMA n'est nécessaire.

L'extrait routier est distribué avec sa provenance dans
[SOURCE.md](src/cav_recovery/scenarios/kintambo/SOURCE.md).
© [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), ODbL 1.0.
La préparation fonctionne hors ligne, sans télécharger une nouvelle carte
à chaque lancement.

## Installation

Python 3.12 ou ultérieur est nécessaire. SUMO et TraCI 1.27.1 sont installés avec
le projet ; sumolib est apporté par leurs dépendances.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements.txt
```

Sous PowerShell : `.venv\Scripts\Activate.ps1`.

## Lancer une expérience

Depuis le dépôt, dans l'environnement activé :

```bash
python scripts/run_experiment.py --demand LOW --seed 1
python scripts/run_experiment.py --demand HIGH --seed 1 --gui --gui-delay-ms 100
```

La commande prépare le réseau et les missions, lance SUMO et conserve un bilan
dans un nouveau dossier. `--output-dir <nouveau-dossier>` permet de choisir
l'emplacement ; un dossier non vide est refusé.

`LOW`, `MEDIUM`, `HIGH` et `STRESS` sont des intensités demandées, pas des
diagnostics de congestion ou de gridlock. La configuration se trouve dans
[experiment.json](src/cav_recovery/scenarios/kintambo/experiment.json) :
réseau, huit entrées, destinations et poids de chaque mouvement.

Pour un essai explicite :

```bash
python scripts/run_experiment.py --seed 2 --rate 450 --duration-s 600
```

`--rate` est en véhicules/heure/**par entrée**, pas pour tout le réseau.
`--config <configuration.json>` permet de modifier les paramètres expérimentaux.
Consulter `--help` pour les autres options.

Le tirage continu reste dans `sampled_s`. `scheduled_s` est son arrondi supérieur
à la milliseconde pour SUMO ; l'insertion réelle peut être retardée. Le pas simulé
reste 0,5 s. Les feux sont statiques et identiques entre niveaux ; aucun véhicule
n'est artificiellement immobilisé.

Le mode graphique utilise réellement `sumo-gui` avec démarrage automatique.
Une interface compatible est nécessaire, par exemple WSLg sous Windows.
Les véhicules utilisent une forme automobile native. Le délai graphique ne
modifie que la vitesse d'affichage, pas la dynamique simulée.

## Résultats

Le bilan comprend `summary.json`, `vehicles.csv`, `timeline.csv`, `lanes.csv`,
`observations.jsonl`, `tripinfo.xml` et `sumo.log`. Les fichiers du scénario
et les journaux de conversion restent disponibles.

Programmés, insertions retardées, actifs et arrivées à destination sont distingués.
Les observations donnent les vitesses, progressions, files et occupations des
voies (longueur occupée / longueur de voie). Une file atteignant le début d'une
voie reste un indice : elle ne prouve pas à elle seule un gridlock.

Téléportations, collisions, disparitions et destinations modifiées font échouer
l'expérience. L'horizon atteint sans vidange est signalé séparément ; aucun
véhicule n'est supprimé pour obtenir un succès. `--drain-horizon-s` fixe
explicitement l'attente après injection. Codes de sortie : 0 pour une vidange
complète, 3 pour un horizon atteint, 1 pour un échec technique ou d'intégrité,
2 pour une entrée refusée.

## Tests et contrôle technique

```bash
python -m compileall -q src scripts tests
python -m pytest -q
python scripts/check_sumo.py
```

Les tests couvrent Poisson, les routes, destinations, bilans d'échec et fermeture.
Les intégrations réelles nécessitent les exécutables SUMO dans l'environnement
activé ; un test ignoré ne valide pas l'intégration. Le petit contrôle technique
utilise une fixture synthétique indépendante de Kintambo.

## État et limites

Le dépôt fournit le réseau expérimental, la demande reproductible, la GUI et les
observations. Un blocage collectif doit encore être confirmé par une preuve
physique indépendante.

Les essais LOW (seeds 1 et 2) se vident ; HIGH/seed 1 reste chargé à l'horizon.
MEDIUM/seed 1 a détecté une collision de jonction et cet essai est invalidé.
Le réseau n'est donc pas certifié pour toutes les configurations ou seeds.

Ces paramètres ne décrivent ni les débits historiques ni la capacité réelle
de Kintambo. Les proportions de destinations sont synthétiques ; les feux ne
reconstituent pas des plans observés. Le fond graphique est cosmétique.

C-RDG, récupération autonome et Graph-MARL ne sont pas encore implémentés.
La suite vérifiera les dépendances physiques puis des méthodes de récupération
préservant les missions. Calibration empirique et validation externe restent
des perspectives.
