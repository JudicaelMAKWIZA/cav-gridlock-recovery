# CAV Gridlock Recovery

Ce projet étudie la circulation de véhicules connectés et automatisés dans des
situations où plusieurs véhicules peuvent se bloquer mutuellement. Il prépare
d'abord des données de trafic réelles, puis servira à construire et évaluer des
méthodes de détection et de récupération.

## Objectif du projet

Le projet vise à :

- détecter les blocages formés par des dépendances entre véhicules et zones
  routières ;
- représenter ces dépendances sous une forme exploitable par un algorithme ;
- choisir des actions capables de rétablir la circulation après la formation
  d'un blocage ;
- étudier ensuite une approche d'apprentissage multi-agent.

Le projet se concentre principalement sur la récupération après un blocage
déjà formé. La prévention reste secondaire.
Ces fonctions ne sont pas encore toutes implémentées.

## État actuel

Le dépôt permet actuellement de :

- lire et vérifier progressivement des trajectoires pNEUMA sans modifier le
  fichier source ;
- étudier un secteur routier réel à partir d'une géométrie et de portes
  virtuelles ;
- extraire les passages, les visites et les mouvements observés ;
- calculer des comptages, des proportions et des flux lorsque la couverture
  temporelle est connue ;
- répartir les fenêtres complètes entre trois niveaux relatifs de charge ;
- construire un profil de trafic destiné aux futurs scénarios ;
- vérifier un petit trajet synthétique dans SUMO avec TraCI ;
- construire un réseau depuis la carte historique validée et simuler trois
  demandes de trafic, avec ou sans interface graphique ;
- suivre chaque véhicule jusqu'à sa destination et vérifier les bilans.

Aucune simulation de blocage ni méthode de récupération n'est encore incluse.

## Technologies

- Python 3.12 ou version ultérieure ;
- bibliothèque standard Python pour les traitements principaux ;
- pytest pour les tests automatisés ;
- SUMO pour simuler les déplacements et TraCI pour avancer la simulation et
  lire ses événements depuis Python ;
- fichiers CSV, JSON, GeoJSON et Markdown pour les entrées et les résultats.

SUMO et TraCI servent à la simulation de trafic nominal, sans incident ni
simulation de blocage.

## Structure du dépôt

```text
src/cav_recovery/empirical/   lecture, validation et préparation du trafic
src/cav_recovery/simulation/  réseau, demandes et suivi des simulations SUMO
scripts/                      commandes utilisables depuis le dépôt
tests/                        tests automatisés et petites données synthétiques
```

## Installation

Créer un environnement Python puis installer le projet et les outils de test.
L'installation du projet inclut SUMO et TraCI en version 1.27.1 :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements.txt
```

Sous Windows PowerShell, l'activation s'effectue avec :

```powershell
.venv\Scripts\Activate.ps1
```

## Vérifier l'installation

```bash
python -m compileall -q src scripts tests
python -m pytest -q
```

## Préparer les données de trafic

### Vérifier un fichier pNEUMA

La première commande lit le fichier progressivement, signale ses anomalies et
produit des exports structurés sans modifier la source :

```bash
python scripts/qualify_pneuma.py \
  --input <fichier-pneuma.csv> \
  --output-dir <dossier-de-sortie>
```

### Construire les profils d'un secteur

Cette commande utilise plusieurs fichiers de géométrie et de configuration.
Afficher son aide pour connaître les arguments attendus :

```bash
python scripts/profile_pneuma.py --help
```

Elle produit notamment les passages par porte, les visites, les mouvements et
les profils temporels du secteur étudié.

### Construire le contrat de trafic

Une fois les profils et leur couverture validés :

```bash
python scripts/build_empirical_contract.py \
  --profile-dir <dossier-des-profils> \
  --coverage <fichier-de-couverture.json> \
  --output-dir <dossier-de-sortie>
```

Le dossier de sortie contient :

- `regime_profile.csv` ;
- `empirical_contract.json` ;
- `quality_summary.json` ;
- `regime_report.md`.

Chaque dossier de sortie doit être absent ou vide. Les commandes refusent
d'écraser silencieusement des résultats existants.

## Tests

Lancer toute la suite avec :

```bash
python -m pytest -q
```

Les tests utilisent uniquement de petites données synthétiques.

## Vérifier SUMO et TraCI

Le contrôle utilise un réseau synthétique de deux arêtes et un seul véhicule.
Il vérifie le départ, les positions, la route et l'arrivée à la destination
assignée. Une téléportation, une disparition sans arrivée ou un dépassement
de l'horizon fait échouer le contrôle. Aucune commande ne déplace ni ne retire
artificiellement le véhicule.

SUMO et TraCI sont installés par `python -m pip install -e .`, comme indiqué
plus haut. Leurs dépendances `sumo-data` et `sumolib` sont installées
automatiquement. Le contrôle a été exécuté sous Ubuntu via WSL avec Python
3.12.3, SUMO 1.27.1 et TraCI 1.27.1. Activer le même environnement Python
avant de vérifier l'installation :

```bash
sumo --version
python --version
python -c "import traci; print(traci.__file__)"
```

Lancer depuis le dépôt, dans cet environnement :

```bash
python scripts/check_sumo.py
```

Le bilan affiché indique les versions, les événements, les positions relevées
en mètres, le nombre de pas et la fermeture de SUMO/TraCI. L'horizon par défaut
est de 60 secondes simulées. `--horizon` permet de le changer et `--sumo-binary`
de choisir un binaire qui n'est pas dans `PATH`. Le code de sortie est `0` pour
un succès, `1` pour un échec du contrôle et `2` pour une entrée refusée.

Lancer aussi `python -m pytest -q` dans cet environnement pour exécuter le
test d'intégration réel. Ailleurs, ce test est explicitement ignoré si SUMO ou
TraCI manque ; un test ignoré ne valide pas l'installation.

La fixture publiée dans `tests/fixtures/sumo_smoke/` ne contient aucune donnée
réelle. Son réseau est fourni prêt à utiliser ; `netconvert` n'est nécessaire
que pour le reconstruire à partir des petits fichiers de nœuds et d'arêtes :

```bash
netconvert \
  --node-files tests/fixtures/sumo_smoke/nodes.nod.xml \
  --edge-files tests/fixtures/sumo_smoke/edges.edg.xml \
  --output-file tests/fixtures/sumo_smoke/network.net.xml \
  --no-internal-links --no-turnarounds
```

## Simuler le trafic

L'expérience empirique complète nécessite des entrées validées qui ne sont pas
distribuées avec le dépôt public. Leur identité est contrôlée avant la simulation.
Le dépôt permet de reproduire l'installation, les tests synthétiques et les
contrôles techniques, mais ne suffit pas à lui seul à reconstruire cette
expérience empirique complète.

La préparation utilise la carte historique et le contrat de trafic validés.
Leur identité est contrôlée ; une autre carte ou un autre contrat est refusé.
Elle produit un réseau, les routes, les missions LOW/MID/HIGH, les réglages
graphiques et un manifeste décrivant les choix de conversion :

```bash
python scripts/prepare_traffic.py \
  --osm <carte-historique.osm> \
  --contract <contrat-de-trafic.json> \
  --output-dir <dossier-du-scenario>
```

Lancer ensuite un niveau de trafic :

```bash
python scripts/run_traffic.py \
  --scenario-dir <dossier-du-scenario> \
  --regime LOW \
  --output-dir <nouveau-dossier-du-bilan>
```

Remplacer `LOW` par `MID` ou `HIGH` pour les autres niveaux. Ajouter `--gui`
pour lancer `sumo-gui` avec démarrage automatique. `--gui-delay-ms 100`
ralentit seulement l'affichage ; le pas simulé reste de 0,5 s. Les véhicules
sont représentés par des berlines natives de SUMO. Une interface graphique
compatible est nécessaire, par exemple WSLg sous Windows.

Les départs sont réguliers et déterministes, sans tirage aléatoire. Les feux
restent statiques, avec un cycle de 90 s issu de la conversion. Ce programme
et les paramètres des véhicules sont des hypothèses de simulation, pas des
mesures historiques. Le décor n'utilise que les objets présents dans la carte ;
le fond graphique n'a aucune signification géographique.

Le bilan contient `summary.json`, `vehicles.csv`, `timeline.csv`, `tripinfo.xml`
et `sumo.log`. Il distingue les départs programmés, les insertions retardées,
les véhicules actifs et les arrivées normales à la destination assignée.
Une téléportation, une disparition ou une fermeture incomplète fait échouer
l'exécution. Après l'injection, l'attente est limitée à 600 s ;
`--drain-horizon-s` permet de définir explicitement une autre limite.
Chaque préparation et chaque exécution exigent un dossier de sortie absent ou
vide. Consulter `--help` pour tous les arguments.

## Limites actuelles

- Les données étudiées viennent actuellement d'un seul secteur pNEUMA.
- Seules les catégories Car et Taxi servent à construire la population
  principale de véhicules autonomes.
- Motorcycle, Bus, Medium Vehicle et Heavy Vehicle restent décrits, mais ne
  sont pas encore simulés.
- LOW, MID et HIGH sont seulement des niveaux relatifs de trafic ; ils ne
  représentent ni la congestion, ni la capacité de la route.
- Les volumes observés servent à programmer de nouvelles missions ; ils ne
  garantissent pas un taux d'insertion réel dans le réseau simulé.
- Les voies et vitesses absentes de la carte utilisent les règles de conversion
  SUMO. Elles ne constituent pas des mesures de capacité routière.
- Aucune simulation de blocage et aucune stratégie de récupération ne sont
  intégrées dans cette branche.

## Suite du projet

Les prochaines étapes consisteront à représenter les conflits entre véhicules
et zones routières, détecter les
blocages, puis comparer des stratégies de récupération. L'apprentissage
multi-agent sera étudié après la mise en place de cette base expérimentale.
