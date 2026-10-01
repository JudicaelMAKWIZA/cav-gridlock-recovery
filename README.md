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

La récupération après un blocage est donc aussi importante que sa prévention.
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
- construire un profil de trafic destiné aux futurs scénarios.

Aucune simulation de blocage ni méthode de récupération n'est encore incluse.

## Technologies

- Python 3.12 ou version ultérieure ;
- bibliothèque standard Python pour les traitements principaux ;
- pytest pour les tests automatisés ;
- fichiers CSV, JSON, GeoJSON et Markdown pour les entrées et les résultats.

Le projet n'utilise pas encore SUMO ni TraCI.

## Structure du dépôt

```text
src/cav_recovery/empirical/   lecture, validation et préparation du trafic
scripts/                      commandes utilisables depuis le dépôt
tests/                        tests automatisés et petites données synthétiques
configs/                      configurations publiques du projet
outputs/                      résultats générés, ignorés par Git
```

## Installation

Créer un environnement Python puis installer le projet et les outils de test :

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

## Limites actuelles

- Les données étudiées viennent actuellement d'un seul secteur pNEUMA.
- Seules les catégories Car et Taxi servent à construire la population
  principale de véhicules autonomes.
- Motorcycle, Bus, Medium Vehicle et Heavy Vehicle restent décrits, mais ne
  sont pas encore simulés.
- LOW, MID et HIGH sont seulement des niveaux relatifs de trafic ; ils ne
  représentent ni la congestion, ni la capacité de la route.
- Les flux observés ne sont pas encore des taux d'insertion dans une
  simulation.
- Aucune simulation de blocage et aucune stratégie de récupération ne sont
  intégrées dans cette branche.

## Suite du projet

Les prochaines étapes consisteront à construire les scénarios de simulation,
représenter les conflits entre véhicules et zones routières, détecter les
blocages, puis comparer des stratégies de récupération. L'apprentissage
multi-agent sera étudié après la mise en place de cette base expérimentale.
