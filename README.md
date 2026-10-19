# CAV Gridlock Recovery

Ce projet étudie la détection et la récupération de blocages collectifs de
véhicules connectés et automatisés. La récupération après formation du blocage
est l'objectif principal ; la prévention reste secondaire.

## Candidat M02 pour audit indépendant

M02 fournit des expériences physiques et un pipeline de carrefour dense :
génération native, qualification hors ligne C1–C7, partitions groupées, exports
causaux et contrôles. La vérité terrain annotée utilise un oracle global ;
elle ne constitue pas un détecteur autonome. D01 complet, le TCN causal,
le C-RDG de l'architecture hybride et Graph-MARL restent des étapes futures.
Le C-RDG minimal historique présent dans ce dépôt est une instrumentation de
développement distincte. Aucune récupération autonome n'est validée ici.

La campagne privée v4 comporte 2 194 épisodes (1 001 positifs, 1 190 négatifs,
3 inconnus). Les effectifs supervisés sont 1 316 train, 327 validation et 548
test OOD ; les inconnus sont exclus de la supervision. Les proportions effectives
sont environ 60/15/25, ajustées aux groupes, et non les proportions nominales.
Les annotations et les exports ne sont pas distribués avec ce dépôt.

Les limites à prendre en compte pour l'audit sont les suivantes :

- La densité diffère entre classes : pics médians de 24 actifs pour les positifs
  et 16 pour les négatifs. Un seul négatif présente un effet physique d'au moins
  5 s ; la couverture des négatifs difficiles est très faible.
- Des occupants successifs sont observés, mais aucune confirmation n'exige
  exclusivement leur succession. Seuls 53 positifs atteignent 20 demandeurs
  simultanément affectés ; le maximum observé est 28, pour 30 actifs au plus.
- L'asymétrique est séparée des partitions train/validation, mais elle a servi
  au développement et au calibrage, puis aux inspections et rejeux. Ce test
  mesure une séparation de partitions ; il ne peut pas être revendiqué comme
  une topologie inconnue de tout le processus de développement. Les partitions
  et critères de v4 restent figés. Un futur test réellement inédit reste à établir.
- La perception locale est idéalisée et le conducteur n'est pas calibré sur des
  véhicules réels. Les ablations spatiales ne prouvent ni une cause dynamique
  unique, ni l'impossibilité de toute alternative, ni la nécessité d'intervenir.
- Les cinq rejeux Windows/Linux ont conservé les classes, sans identité numérique
  bit à bit ; l'écart maximal observé est d'environ 1,0456 m dans un rejeu. Sa
  cause précise n'est pas isolée. Aucun seuil de tolérance interplateforme n'est
  validé et le corpus Linux n'est pas réécrit.
- Le contrôle strict historique signale deux épisodes avec ambiguïtés de
  projection de coordonnées arrondies. Une preuve complémentaire borne ces
  ambiguïtés par la précision de stockage, sans effacer les alertes initiales.

La livraison technique ne vaut donc pas acceptation scientifique définitive.
Les preuves détaillées, rapports et résultats restent privés ; l'audit du code
public peut reproduire de petites fixtures indépendantes.

## Exécution native Windows (Python 3.12)

Le moteur MetaDrive reste épinglé au commit
`fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1`. Les dépendances Windows sont
décrites dans `requirements-metadrive-windows.txt` ; les environnements Linux
ne doivent pas être recopiés dans un environnement Windows. Le moteur et ses
assets peuvent être conservés sur D: et installés en mode éditable depuis cette
source Git propre. Les dépendances SUMO sont désormais optionnelles (`.[sumo]`).

Depuis la racine du checkout, dans PowerShell, créer un seul environnement et
une source moteur au SHA exact (Git est nécessaire). Les chemins D: sont des
exemples configurables ; ne pas écraser une source ou un environnement existant :

```powershell
New-Item -ItemType Directory -Force -Path 'D:\cav-recovery-cache\pip', 'D:\cav-recovery-temp', 'D:\cav-recovery-venvs', 'D:\cav-recovery-engines', 'D:\cav-recovery-data' | Out-Null
py -3.12 -m venv 'D:\cav-recovery-venvs\metadrive'
git clone https://github.com/metadriverse/metadrive.git 'D:\cav-recovery-engines\metadrive-fe54b033'
git -C 'D:\cav-recovery-engines\metadrive-fe54b033' checkout --detach fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1
$env:PIP_CACHE_DIR = 'D:\cav-recovery-cache\pip'
$env:TEMP = 'D:\cav-recovery-temp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
& 'D:\cav-recovery-venvs\metadrive\Scripts\python.exe' -m pip install -r requirements-metadrive-windows.txt
& 'D:\cav-recovery-venvs\metadrive\Scripts\python.exe' -m pip install --no-deps --no-build-isolation -e 'D:\cav-recovery-engines\metadrive-fe54b033' -e .
& 'D:\cav-recovery-venvs\metadrive\Scripts\python.exe' -m metadrive.pull_asset
& 'D:\cav-recovery-venvs\metadrive\Scripts\python.exe' -m pip check
.\scripts\cgr-windows.cmd metadrive scenarios
.\scripts\cgr-windows.cmd dataset --help
```

`cgr-windows.cmd` configure uniquement son processus et ses enfants : environnement,
caches, temporaires et sorties sur D:, UTF-8 et un thread pour les bibliothèques
numériques. Il ne change ni les réglages globaux Windows ni la politique PowerShell.
`CGR_WINDOWS_VENV`, `CGR_CACHE_ROOT`, `CGR_TEMP_ROOT` et `CGR_OUTPUT_ROOT` permettent d'adapter ces
emplacements. Les commandes dataset exigent un chemin de sortie explicite.

```powershell
.\scripts\cgr-windows.cmd metadrive run crossing_three --headless --horizon-s 100 --output-dir 'D:\cav-recovery-data\demo-m02'
.\scripts\cgr-windows.cmd dataset generate --phase calibration --count 16 --workers 2 --output-dir 'D:\cav-recovery-data\development-m02'
.\scripts\cgr-windows.cmd dataset calibrate 'D:\cav-recovery-data\development-m02'
```

Ces commandes créent des essais de développement, jamais une copie du corpus
scientifique privé. Choisir des dossiers de sortie nouveaux. L'installation
nécessite les assets du moteur ; `pull_asset` les place auprès de la source
moteur, sur D: dans cet exemple. Aucun Ubuntu n'est nécessaire.

Un contrôle d'un corpus existant peut écrire son rapport séparément :

```powershell
.\scripts\cgr-windows.cmd dataset validate 'D:\chemin\corpus' --report 'D:\chemin\audit-windows.json'
& 'D:\cav-recovery-venvs\metadrive\Scripts\python.exe' -X utf8 -m pytest -q -ra -o cache_dir=D:/cav-recovery-cache/pytest --basetemp=D:/cav-recovery-temp/pytest
```

Les tests historiques SUMO nécessitent leurs outils optionnels. Certains tests
de l'ancien panneau Tk nécessitent un bureau assez haut pour leur géométrie.
Ces prérequis sont déclarés dans les tests ; un test ignoré ne constitue pas une
validation de la fonctionnalité correspondante. Les différences physiques entre
builds Bullet Windows et Linux doivent être mesurées, sans réétiqueter un corpus
historique ni supposer une identité numérique entre plateformes.

Les clients Python légers pour les tests historiques simulant l'API SUMO peuvent
être installés séparément, sans moteur SUMO :

```powershell
& 'D:\cav-recovery-venvs\metadrive\Scripts\python.exe' -m pip install --target 'D:\cav-recovery-cache\legacy-test-clients' -r requirements-test-sumo-clients.txt
$env:PYTHONPATH = 'D:\cav-recovery-cache\legacy-test-clients'
```

Cette variable ne concerne que le terminal courant. Sans ces clients, les tests
historiques qui les exigent sont explicitement ignorés ; MetaDrive n'en dépend pas.

## MetaDrive : socle principal

Le nouveau socle utilise MetaDrive avec une vue **2D top-down uniquement**.
Le banc SUMO/Kintambo et ses commandes restent disponibles indépendamment.
Il n'y a pas de co-simulation. D01 complet et la récupération autonome ne sont
pas encore portés vers MetaDrive. Les expériences physiques ci-dessous séparent
les restrictions mesurées, la faisabilité des manœuvres et les inconnues.

Installation Linux historique sous Ubuntu 24.04/WSL2, Python 3.12, dans un environnement séparé
de l'environnement SUMO historique :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-metadrive.txt
python -m pip install -e '.[metadrive]'
python -m pip install -r requirements.txt
```

MetaDrive est épinglé au commit
`fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1` (version déclarée 0.4.3), avec
Panda3D 1.10.16. Ne pas utiliser le paquet PyPI 0.4.3 à la place de cette
source. Une source locale propre à ce SHA peut être installée avec
`pip install --no-deps -e /chemin/source-metadrive`, puis
`pip install --no-deps -e .` ; conserver ses assets natifs. Sur une installation
neuve sans assets, l'outil natif `python -m metadrive.pull_asset` les prépare.
Aucun fichier du moteur installé n'est modifié par le projet.

```bash
cgr metadrive scenarios
cgr metadrive run crossing --headless --seed 1
cgr metadrive run crossing_three --gui --seed 1 --horizon-s 100 --gui-delay-ms 100
cgr metadrive run convergence --headless --output-dir outputs/metadrive/mon-essai
cgr metadrive run crossing --policy courtesy_wait --horizon-s 20
```

`crossing` et `convergence` reprennent deux dispositions préparées sur une
intersection native non signalée. `crossing_three` ajoute un suiveur : trois
CAV, destinations initiales explicites et stables. La politique par défaut
`scheduled_passage` utilise une courtoisie et un passage central programmé
après 10 s, protégés par un garde physique ; ce n'est ni une décision D01,
ni une récupération coopérative autonome. `courtesy_wait` garde la courtoisie
symétrique, qui peut immobiliser des véhicules capables de passer : ce n'est
pas une vérité terrain de deadlock.

Le pas physique est **0,02 s**, la décision **0,1 s**. `--horizon-s` termine
l'observation sans changer les missions ; les présents restent censurés.
`--gui-delay-ms` ralentit seulement l'affichage. La fenêtre montre les voies,
des voitures orientées à l'échelle des corps physiques et les libellés stables
V0/V1/V2 ; Échap ou sa fermeture interrompt le run
proprement. Elle ferme aussi à la fin du run. Il n'existe pas encore de panneau
INFO/D01 MetaDrive. Sans affichage WSLg/X11, utiliser `--headless`.

Les sorties vont dans un nouveau dossier `outputs/metadrive/` : configuration,
missions, géométrie, correspondance `vehicle_labels.json`, `observations.jsonl`,
`physics.jsonl` (empreintes, distances, pénétrations et contacts aux sous-pas),
commandes demandées/acceptées, capteurs, événements, messages V2V, C-RDG,
`summary.json` et empreintes `manifest.json`. Collision, sortie de chaussée,
changement de destination ou disparition injustifiée font échouer le run.
Une arrivée native est corroborée dans le corridor de destination initial.

La garde de développement réserve une marge entre carrosseries de 1 m, un pas
de réaction et une distance de freinage, y compris pour les autres CAV en
mouvement. Elle reste globale et échantillonnée ; elle ne certifie pas toutes
les manœuvres. Les traversées commencent avec un espacement corrigé. Pour
`crossing_three`, seed 1, prévoir `--horizon-s 100` : les trois arrivées se
terminent à 90,1 s ; l'horizon par défaut de 90 s conserve le dernier présent
comme censuré. Le rendu ne modifie pas la physique.

Les rayons LiDAR et de chaussée restent des mesures locales sans identité
automatique des voisins. V2V livre seulement des états propres datés à portée
(30 m, période 1 s, validité 2 s), sans simulation radio. Le garde et le C-RDG
minimal utilisent explicitement une instrumentation **globale de développement**,
pas une connaissance distribuée des CAV. Le graphe représente un chemin assigné
occupé dans les 0,5 m vérifiés, pas toutes les possibilités de manœuvre ni la
cause certaine de tout arrêt. Les alternatives non vérifiées restent inconnues.
Il calcule à 0,1 s, exporte les graphes à 1 s et conserve les événements entre
exports. Les SCC/candidats restent structurels, pas des diagnostics.

`--no-sensors`, `--no-v2v` et `--no-crdg` permettent les témoins séparés.
Pour vérifier le socle natif : `python -m pytest -q tests/test_metadrive.py
tests/test_metadrive_cli.py`. Sans MetaDrive, ses tests natifs sont ignorés,
pas validés. La suite SUMO existante est conservée sans changement de sens.

### Expériences sur des contraintes physiques

Trois scènes supplémentaires utilisent des géométries et des missions fixes :

| Scène | Route et missions | Formation déclarée |
| --- | --- | --- |
| `intersection_approaches` | Intersection native non signalée, trois CAV venant de l'ouest, du nord et de l'est ; traversées vers l'est, le sud et l'ouest. | Départs préparés en amont, puis conduite et interactions. |
| `lane_merge` | Rétrécissement natif de deux voies vers une, puis élargissement ; deux CAV vers la sortie est. | Départs préparés côte à côte en amont, changement de voie obligatoire natif. |
| `shared_passage` | Passage partagé de 3,4 m de large, deux directions sur la même surface, aire de dégagement à l'ouest ; missions opposées ouest–est et est–ouest. | Deux entrants préparés face à face à l'intérieur du passage, puis approche physique. |

Les origines, longitudes initiales et destinations exactes sont définies dans
[`scenes.py`](src/cav_recovery/metadrive/scenes.py) avant le reset. Il n'y a
ni trafic humain, ni changement de destination, ni réinsertion. Le passage est
un bloc routier construit avec les voies et la physique natives de MetaDrive ;
ses deux directions partagent délibérément une chaussée unique. L'aire latérale
est autorisée pour les manœuvres dans les deux sens. Ses bords sont des limites
de circulation vérifiées sur les corps, pas des murs infranchissables.

Pour comparer les deux contrôles, garder la même scène, seed et durée :

```bash
cgr metadrive run intersection_approaches --policy road_following --horizon-s 240
cgr metadrive run intersection_approaches --policy global_clearance --horizon-s 240
cgr metadrive run lane_merge --policy road_following --horizon-s 400
cgr metadrive run lane_merge --policy global_clearance --horizon-s 400
cgr metadrive run shared_passage --policy road_following --horizon-s 180
cgr metadrive run shared_passage --policy global_clearance --horizon-s 180 --gui
```

`road_following` est le contrôle par défaut de ces trois scènes, sans courtoisie
symétrique. Le suivi et son garde restent une instrumentation globale de
développement. `global_clearance` teste un ordre central explicite dans
l'intersection et le rétrécissement ; dans le passage, il tente après 15 s un
recul vers l'aire, un stationnement, le passage opposé et un retour vers la
destination initiale. Ce contrôleur déterministe teste la faisabilité avec
l'état global ; ce n'est pas une politique coopérative distribuée.

Le garde du rétrécissement et du passage utilise une référence cinématique,
une réserve entre corps de 0,7 m et une marge de chaussée de 0,15 m ;
l'intersection conserve le garde avant à 1 m. Ces paramètres ne constituent
pas une garantie universelle. Le passage vérifie l'union exacte des voies ;
les autres cartes utilisent l'enveloppe de voies échantillonnée avec une
tolérance géométrique de 5 cm, en complément des événements natifs.

`physical_reference.jsonl` est indépendant des décisions du garde et du C-RDG.
Il mesure les occupations des recouvrements de trajectoires demandées, les
projections vers les voies finales et des sondes géométriques sur 4 m : chemin
avant assigné, recul rectiligne, arcs avant gauche et droit. Les autres corps
sont figés pour ces sondes. Une empreinte en conflit prouve une restriction de
ce chemin échantillonné ; une sonde libre reste un candidat sans validation
dynamique. Les autres braquages, mouvements conjoints et possibilités non
explorées restent inconnus. Un refus conservateur du garde ne prouve jamais à
lui seul une impossibilité physique ou un deadlock.

Le passage exporte aussi `forward_only_certificate.json` : un argument d'ordre
longitudinal, avec ses hypothèses et vérifications. Lorsque deux voitures ne
peuvent pas se croiser dans la largeur disponible, leur échange d'ordre exige
d'atteindre l'aire à l'ouest et donc un retrait vers l'ouest. La nécessité de la
**marche arrière signée** suppose en plus un modèle non holonome sans dérapage ;
ce n'est pas une preuve exhaustive de l'espace dynamique Bullet. La réalisation
d'un recul admissible et cet argument de nécessité ont des portées distinctes.

Les commandes, capteurs, V2V, C-RDG minimal et contrôles aux sous-pas restent
disponibles. Le bilan distingue distance parcourue, distance en recul, temps
immobile, arrivées natives et présents à l'horizon. Toute violation d'intégrité
invalide l'essai. La vue 2D conserve les voitures et V0/V1/V2, avec un cadrage
adapté à la carte. Tests spécifiques :
`python -m pytest -q tests/test_physical_scenes.py tests/test_metadrive_cli.py`.
Les sorties, captures et comptes rendus restent dans les dossiers privés ignorés.

## Banc SUMO historique — réseau et trafic

Le banc SUMO utilise une topologie dérivée d'OpenStreetMap autour de
Kintambo Magasin, Kinshasa. Le réseau actif conserve les avenues, approches
et liaisons proches du noyau. L'extrait source est fourni, utilisable hors ligne.
Les voies, vitesses et feux sont des hypothèses de simulation, pas une calibration
du trafic réel. La calibration empirique reste une perspective.

© [OpenStreetMap contributors — ODbL 1.0](https://www.openstreetmap.org/copyright).
La provenance figure dans [scenario.json](src/cav_recovery/scenarios/kintambo/scenario.json).

Python génère les arrivées Poisson par intervalles exponentiels avec
`random.Random(seed)`. Routes et destinations sont fixées avant le départ.
Même configuration et même seed reproduisent les missions. `LOW`, `MEDIUM`,
`HIGH` et `STRESS` sont des niveaux de demande, pas des diagnostics.
`sampled_s` garde le tirage ; `scheduled_s` l'arrondit vers le haut à la
milliseconde. SUMO peut retarder l'insertion réelle.

## Installation SUMO

Python ≥ 3.12, SUMO/TraCI 1.27.1 et NetworkX sont nécessaires.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[sumo]'
python -m pip install -r requirements.txt
```

Sous PowerShell : `.venv\Scripts\Activate.ps1`. Pour la démonstration graphique,
il faut un affichage compatible avec SUMO-GUI et Tkinter ; sous Ubuntu/WSLg,
installer `python3-tk` si nécessaire. Aucun navigateur ni serveur n'est requis.

## Commandes SUMO conservées

Pour utiliser `cgr` depuis n'importe quel dossier d'un terminal Ubuntu/WSL,
faire une seule fois depuis le dépôt, après l'installation :

```bash
./.venv/bin/cgr install --shell-path
```

Cette commande installe un relais dans `~/.local/bin/cgr` vers le Python de
ce projet et ajoute explicitement ce dossier au PATH dans `.bashrc` et dans
le profil de connexion actif (`.bash_profile`, `.bash_login` ou `.profile`). Sans
`--shell-path`, aucun fichier de démarrage n'est modifié. Une autre commande
`cgr` n'est jamais écrasée. Ouvrir ensuite un nouveau terminal Ubuntu :

```bash
cgr
cgr crossing
cgr junction_adverse
cgr clearance
cgr --help
cgr scenarios
cgr run crossing --gui --seed 1
cgr run junction_adverse --gui --seed 1
cgr run kintambo --demand LOW --seed 1
```

`cgr` ouvre un menu en français : simulation, liste des scénarios, aide et
retour/quitter. Les scénarios et l'aide restent affichés jusqu'à Entrée pour
revenir au menu. Il propose seed, affichage et délai explicites ; sans terminal
interactif, il affiche l'aide et n'attend aucune saisie. En mode graphique, le
menu propose d'activer le détecteur d'obstructions et les échanges V2V (oui par défaut). Diagnostic
confirmé, récupération, entraînement et évaluation restent non implémentés.

Les raccourcis `cgr <scénario>` ouvrent SUMO et INFO par défaut, seed 1 et délai
300 ms. Le détecteur y est activé avec la GUI ; `--no-detect-obstructions` le
désactive. Le transport V2V y est aussi actif ; `--no-v2v` le désactive.
Modifier aussi avec `--seed`, `--gui-delay-ms` ou `--no-gui` ; sans GUI,
le détecteur n'est activé que sur demande explicite. Les commandes avancées
`cgr run …` gardent leur comportement : GUI seulement avec `--gui`, détecteur
seulement avec `--detect-obstructions`, V2V seulement avec `--v2v`, délai par défaut 100 ms. Le menu et les
raccourcis utilisent le même pipeline.
Le relais prépare le PATH de `.venv` et travaille depuis le dépôt ; les sorties
par défaut y restent sous `outputs/simulation`. Les chemins relatifs d'options
sont donc relatifs au dépôt lorsqu'on utilise ce relais.
Après déplacement du dépôt, recréer `.venv`, réinstaller le package, puis
réexécuter `./.venv/bin/cgr install --shell-path` au nouvel emplacement.

Une commande prépare le réseau et les missions, lance SUMO, observe le trafic,
calcule le C-RDG et enregistre les résultats. `--gui` ouvre aussi INFO C-RDG.
`--gui-delay-ms 100` règle le délai visuel, pas le pas physique de **0,5 s**.
Les noms OSM sont conservés mais masqués à l'écran ; `--street-names` les affiche.

`--output-dir <dossier>` choisit une sortie absente ou vide ; sinon un dossier
daté est créé. `--output-mode full` garde les traces fines pour un audit.
`--output-mode interactive` omet les traces détaillées de voies, observations
et graphes successifs, mais conserve entrées, manifeste, bilans, missions,
timeline, tripinfo et preuves consultées. Il n'est pas destiné à l'évaluation.
Les campagnes d'évaluation ne sont pas encore implémentées.

Pour Poisson, `--rate` est en véhicules/heure/entrée et `--duration-s` en secondes
d'injection ; `--config` fournit une configuration JSON. Les cas ciblés ont leur
propre demande finie : ne pas combiner ces options avec eux.
`--drain-horizon-s` règle l'observation après injection sans changer les missions.
Consulter `cgr run --help` pour les options.

Les commandes publiées `python scripts/run_traffic.py --demand …` et
`--kintambo-case …` restent des wrappers du même pipeline.
Les anciens carrefours indépendants et visualiseurs HTML/à flèches sont retirés.
Utiliser les cas Kintambo et `--gui` ; une ancienne option incompatible donne
un message de migration, jamais un autre scénario silencieusement.

## Cas ciblés Kintambo

`clearance` vise des attentes temporaires puis une vidange ; `crossing` concentre
quatre mouvements ; `spillback` charge plusieurs liaisons proches. Leur demande
est explicite et seedée, distincte du Poisson canonique.
`junction_adverse` partage les missions de `crossing` et désactive seulement
`keepClear` sur huit mouvements déclarés. `loop_nominal` et `loop_adverse`
éprouvent une boucle réelle Yoseki/Transversale avec les mêmes missions ;
l'adverse modifie cinq mouvements déclarés. Ces variantes sont séparées du
réseau canonique. Un nom ou une forte occupation ne prouve aucun gridlock.
Les mécanismes figurent dans [cases.json](src/cav_recovery/scenarios/kintambo/cases.json).

## SUMO animé et INFO C-RDG

Dans ce banc, SUMO est la vue physique. Le panneau reçoit les groupes du réseau
toutes les **5 secondes simulées** ; l'heure SUMO, l'heure du graphe et son âge
sont affichés. Des relations connues comme disparues sont grisées entre exports.
Tous les groupes sont accessibles, avec filtres de suivi, réception, carrefour
et cycles candidats. Les grands dessins sont paginés ; la liste conserve leurs
relations. Les détails techniques sont disponibles sur demande.

Sélectionner un groupe, une voiture ou une relation, puis **Voir sur la carte**.
**Vue du secteur**, **Vue du réseau** et **Caméra libre** règlent le cadrage.
La sélection seule n'impose pas de déplacement de caméra.

En **pause** ou à la fin, ces commandes visent toujours la caméra de SUMO.
INFO C-RDG conserve le graphe ; aucune carte ne le remplace. Sous Linux/X11
(notamment WSLg), un événement de redessin réveille la vue native après la
commande TraCI, sans nouveau pas physique. Cette aide n'est pas validée pour
les autres environnements graphiques ; utiliser leurs contrôles natifs si
le rendu est différé. La reprise du trafic reste explicitement demandée.

Les alias `V001`, `V002`… sont uniques et stables, sans renommer les missions.
Trois modes sont disponibles : **Sélection** par défaut (groupe, voiture ou
relation inspectée), **Toutes** (priorité aux inspectés, labels secondaires
espacés selon le zoom), **Aucune**. Un label masqué ne retire pas sa voiture.
Choisir l'alias dans INFO C-RDG, ou y saisir l'ID SUMO, puis cadrer pour
l'identifier. Aucun affichage sans chevauchement à toute échelle n'est promis.

**Bleu** : sélection ; **orange** : contrainte sortante observée ; **violet** :
autre participant ; **jaune** : trafic ordinaire dans le schéma utilisé.
Priorité : sélection, puis contrainte, puis participant, puis couleur d'origine.
Le violet ne signifie pas nécessairement tête de file ; aucune couleur n'est
un diagnostic. Les couleurs d'origine sont restaurées lorsque le rôle disparaît.
La légende à points colorés reste uniquement dans INFO C-RDG : aucune petite
fenêtre supplémentaire ne recouvre SUMO. Couleurs et alias y sont conservés ;
les flèches des relations restent dans le graphe INFO, pas sur les chaussées.
INFO peut être agrandi ou réduit avec les bords de fenêtre ou la poignée en
bas à droite. Les barres de défilement donnent accès au contenu sur un petit
écran. Le graphe reste présent, même en pause.

Quand le détecteur est activé, le bouton **Détecteur** montre d'abord les passages
dont une contrainte est identifiée, puis les immobilisations à vérifier ; les
attentes ordinaires restent consultables. Un blocage local identifié concerne les
véhicules indiqués, pas nécessairement toute la file proche. **Afficher** choisit
les **Gênés**, les **Arrêtés**, ceux qui **Ont avancé**, ceux **Encore ici** ou
**Tous** les membres. Choisir une situation cadre les véhicules encore présents
dans SUMO. Les dates, distances, mouvements et incertitudes sont dans
**Détails**, avec les données techniques sur demande. Un progrès ancien n'est
pas une reprise actuelle. Les anciens suivis sont indiqués : la carte montre
toujours les positions actuelles, sans rejouer le passé.
Le graphe conserve uniquement les dépendances qualifiées, sans ajouter de faux
arcs. Les résultats sont actualisés toutes les 5 secondes simulées et restent
datés en pause. Un passage dégagé ne signifie pas que tout le réseau est rétabli.
Une fin à l'horizon garde l'issue inconnue. La vérification tripinfo et le bilan
définitif ont lieu après la fermeture contrôlée, pas dans la fenêtre encore ouverte.

Tous les véhicules présents sont supposés CAV. Avec `--v2v`, chacun annonce son
état propre aux voisins situés à portée ; les véhicules non insérés ne participent
pas. **Communications V2V** dans INFO montre les échanges, les voisins et les
informations reçues ou périmées pour un CAV choisi. Les distances des autres
véhicules n'y sont indiquées que comme diagnostic du transport, pas comme
connaissance embarquée. Par défaut, la portée est de 120 m, l'émission toutes
les 1 s et la validité de 2 s ; `--v2v-range-m`, `--v2v-interval-s` et
`--v2v-lifetime-s` les règlent. Il s'agit d'un échange applicatif idéal sans
perte ni simulation radio, qui n'agit pas sur la conduite.

```bash
cgr run clearance --gui --v2v --seed 1
```

```bash
cgr run junction_adverse --gui --detect-obstructions --drain-horizon-s 900 --gui-delay-ms 100
```

Une fin normale ou à l'horizon reste inspectable. **Fermer la démo** dans INFO
C-RDG ferme proprement les ressources. `--close-on-end` permet une fermeture
automatique explicitement demandée. Les erreurs restent des erreurs avec logs.

## Résultats et sécurité

`manifest.json` conserve configuration, seed, version, empreintes, mode de sortie
et statut. `summary.json`, `vehicles.csv`, `timeline.csv`, `tripinfo.xml` et
`sumo.log` distinguent programmés, départs, arrivées, actifs et insertions retardées.
Avec V2V, `v2v_events.jsonl`, `v2v_snapshots.jsonl`, `v2v_final.json` et
`v2v_summary.json` gardent les échanges et les états informationnels datés.
Une collision, téléportation, disparition ou destination modifiée invalide le run.
Aucun véhicule n'est supprimé pour obtenir un succès.

En mode complet, `lanes.csv` et `observations.jsonl` gardent les mesures physiques.
`crdg.jsonl`, `crdg_events.jsonl`, `crdg_summary.json` et `crdg_peak.json` conservent
les dépendances, leurs âges et les candidats structurels. Une SCC et un candidat
fermé ne sont pas un diagnostic de gridlock. Le C-RDG ne commande pas le trafic.

`--blockage-evidence` ajoute le suivi des épisodes et de leurs preuves physiques.
Une reprise individuelle ne prouve pas la libération du groupe. Les épisodes
ouverts à l'horizon sont censurés. Une voie légale ou un vert ne prouve pas la
possibilité matérielle de passage ; les inconnues restent inconnues.

`cgr run junction_adverse --detect-obstructions --drain-horizon-s 900` ajoute un
suivi déterministe des blocages locaux et des arrêts, sans fenêtre obligatoire ni action
sur le trafic. `obstructions.jsonl`, `obstruction_events.jsonl` et
`obstruction_summary.json` séparent attentes normales, faible progression prolongée,
contraintes identifiées ou inconnues, reprises partielles et résolution locale vérifiée.
Les cohortes sont locales aux carrefours ; elles ne sont pas des groupes causalement
fermés. La persistance est décrite par rapport à un cycle de service connu, pas
comme une preuve de permanence ou de nécessité d'intervention. Ce mode reste une
première caractérisation à valider scientifiquement, pas un diagnostic de gridlock.

`--crdg-scene-at <secondes>` conserve un état SUMO natif au même instant,
sans nouveau pas et sans dessiner de flèches sur les routes.

Codes de sortie : 0 vidange, 3 horizon, 4 fermeture utilisateur anticipée,
1 panne/violation d'intégrité, 2 entrée refusée. Les sorties privées ne doivent
pas être publiées automatiquement.

## Vérification et limites

```bash
python -m compileall -q src scripts tests
python -m pytest -q
python scripts/check_sumo.py
```

Le contrôle court SUMO vérifie séparément l'installation et la fermeture TraCI.
Les fixtures natives protègent les contrats d'observation, sans constituer un
second banc scientifique. Les tests natifs sont ignorés si SUMO manque ; un test
ignoré ne valide pas cette propriété.

Le socle représente des contraintes observées et propose une première
caractérisation déterministe des obstructions. Sa validation scientifique
indépendante reste à compléter. Aucun diagnostic final de gridlock, récupération
autonome ou Graph-MARL n'est validé. Les conflits latéraux, les alternatives
matériellement praticables et certaines causes inconnues restent des limites.


## Dataset MetaDrive : carrefour dense

Le package `cav_recovery.dataset` réutilise l'infrastructure MetaDrive pour un
seul carrefour par épisode : X, T, oblique ou asymétrique, avec deux ou trois
voies par sens. Les raccords courbes sont des voies natives `PointLane`, reliées
aux voies d'approche et de sortie. Les voitures gardent leurs gabarits natifs.
La première campagne de scènes élémentaires est remplacée ; ses résultats ne
valent pas validation scientifique de ce nouveau générateur.

Les demandes d'arrivée sont échelonnées ; une insertion attend lorsque sa place
est occupée ou que le plafond de 8, 16, 24 ou 30 CAV actifs est atteint. Chaque
CAV suit sa mission sur des raccords directs ou tournants, avec un contrôle
réactif à portée locale de 30 m. La perception d'objets locale reste idéale,
sans bruit ni occlusion ; ce modèle ne démontre pas une autonomie routière réelle.
Les vitesses nominales, distances et réactions varient. La courbure limite
l'allure ; aucune vitesse infime ou immobilisation permanente n'est prescrite.

```bash
cgr dataset generate --phase calibration --count 32 --workers 4 --output-dir outputs/dense-development
cgr dataset calibrate outputs/dense-development
cgr dataset generate --positive 1000 --negative 1000 --max-candidates 4000 --workers 4 --max-seconds 43200 --max-gib 32 --reserve-gib 16 --contract outputs/dense-development/frozen_contract.json --output-dir outputs/dense-production
python scripts/check_dataset.py outputs/dense-production --workers 4
python scripts/audit_route_precision.py outputs/dense-production
cgr dataset split outputs/dense-production
cgr dataset export outputs/dense-production --output-dir outputs/dense-export --max-gib 24 --reserve-gib 16
python scripts/summarize_dataset.py outputs/dense-production --export outputs/dense-export
python scripts/prepare_temporal.py outputs/dense-export --horizon-s 35
python scripts/audit_dataset.py outputs/dense-production --export outputs/dense-export
python scripts/describe_dense_traffic.py outputs/dense-production --verify-ledger
python scripts/assess_dense_capacity.py outputs/dense-production
python scripts/review_dataset.py outputs/dense-production --output-dir outputs/dense-review
```

Mesurer d'abord la capacité et le volume d'un lot représentatif. Choisir un volume
disposant de la place nécessaire pour les traces, les exports et la réserve,
puis adapter les budgets donnés en exemple. Les limites de
temps, disque et mémoire arrêtent les soumissions ; les tâches déjà en cours
finissent. Les quotas sont calculés sur les étiquettes observées, jamais sur les
paramètres. La même commande reprend les épisodes complets après interruption ;
les empreintes de code et de contrat empêchent de mélanger des versions. Les
adaptations de portabilité Windows changent les empreintes de sources : ce
checkout refuse donc de reprendre la production Linux v4 sous sa provenance
originale. Le corpus conserve ses sources gelées et ses audits privés ; utiliser
un nouveau dossier pour des essais avec la version Windows.
Un quota manqué est signalé par un code de sortie non nul, avec les chiffres réels.

Le contrat `physical-C1-C7-movement-resource-v3` suit le mouvement et la ressource
occupée, même lorsque les occupants changent. La mesure de progression sur 5 s
et l'effet continu conservateur de 30 s distinguent occupation, contrainte
transitoire et obstruction persistante. Les intervalles de chaque occupant
restent distincts. Une relation candidate, un refus du contrôleur, une simple
immobilisation, une priorité ou une non-arrivée ne suffisent pas à confirmer.
Aucun cycle ou SCC n'est requis. Les alternatives non évaluées restent inconnues.
Les seuils et leur sensibilité sont documentés puis gelés sur le développement,
avant la production. Le calibrage couvre explicitement les quatre topologies,
y compris l'asymétrique : voir la réserve OOD ci-dessus.

Les classes sont `POSITIVE_CONFIRMED`, `NEGATIVE_CONFIRMED` et
`UNKNOWN_OR_CENSORED`. Les preuves globales hors ligne contiennent les empreintes,
missions, mouvements, ressources, occupants successifs, progrès, contacts et
intervalles directs/indirects. Elles ne proviennent pas d'un futur détecteur.
L'ablation spatiale établit une occupation nécessaire au mouvement déclaré,
sans prouver une causalité dynamique unique ni une récupération obligatoire.

Les traces physiques compressées conservent les poses à 50 Hz. Les observations
à 5 Hz exposent seulement l'état propre, les 16 rayons LiDAR natifs et le V2V
reçu. Les références globales et commandes restent séparées. Les fenêtres
causales sont construites après regroupement des parents et géométries apparentées.
La topologie asymétrique est tenue hors des partitions train et validation pour
le test OOD ;
les proportions s'ajustent aux groupes indivisibles. Le contexte de navigation propre et les commandes propres sont fournis dans des flux causaux séparés. Les identifiants servent aux jointures. Les inconnus sont exportés
séparément avec des cibles nulles. Aucun D01, C-RDG, TCN ou Graph-MARL n'est
implémenté ni entraîné par ce pipeline.

La galerie privée rejoue un échantillon des poses natives avec identifiants V0,
V1, etc., pause, vitesse et retour temporel. Les dimensions physiques et le rendu
emploient la même échelle. La confirmation apparaît seulement à son instant
étayé. Le succès des tests et les effectifs ne remplacent pas l'inspection
scientifique des proportions, mécanismes et biais entre classes.

Code, configuration et tests sont publics. Datasets, galerie, rapports, exports
et traces restent dans les dossiers ignorés `outputs/` et `docs/`.

Le contrôle approfondi peut suivre la production depuis un second terminal,
avec un cache validé par les empreintes des manifestes :

```bash
python scripts/check_dataset.py outputs/dense-production --follow --workers 4
```

Il vérifie chaque épisode achevé puis contrôle globalement les identités,
empreintes et doublons. Les références d'immobilisation hors ligne restent
séparées du contexte de navigation propre et des commandes propres accessibles
aux modèles. Les fenêtres contiennent des indices vers les observations ; les
historiques ne sont stockés qu'une fois par export.

Les historiques temporels de 35 s référencent uniquement des observations déjà
disponibles, avec un masque de préfixe pour les véhicules récemment introduits.
Les cibles distinguent l'appartenance rétrospective à un événement persistant
de sa confirmation disponible à cet instant. La carte statique connue pour la
navigation est séparée des états globaux dynamiques réservés aux annotations.
Les temps absolus et identifiants servent aux jointures, pas à la prédiction.

Les éventuels échecs du contrôle strict restent conservés. L'audit complémentaire
de précision résout seulement une projection ambiguë d'une position arrondie,
en reconstruisant toutes les poses depuis l'abscisse native et en bornant
l'erreur par la précision de stockage. Il ne change ni les traces, ni les
étiquettes, ni le contrat. Les autres anomalies restent bloquantes pour cet audit.

Les distributions séparent les actifs, les présents au centre et les demandeurs
distincts réellement affectés pendant les intervalles confirmés. Les pics par
épisode et les distributions des observations sont fournis par classe et
topologie ; les pics de régions différentes ne sont pas nécessairement simultanés.
L'étude de capacité utilise uniquement les traces existantes : elle ne lance
aucun épisode à 80–100 CAV et présente ses projections CPU/RAM comme des
hypothèses à vérifier. Ces analyses issues de l'oracle restent hors des entrées
des futurs modèles.

## Observateur individuel expérimental D01-F01

`cav_recovery.d01.observer.EgoObserver` reçoit un paquet `F01-input-1` pour
un seul CAV : état et commandes propres, mission/route, carte statique,
16 distances LiDAR et messages V2V effectivement reçus. La liste blanche
imbriquée rejette les champs privilégiés. Le noyau fonctionne sans démarrer
MetaDrive et ne dispose d'aucune interface d'action, d'oracle ou de lecture
du dataset. Il produit `NO_ALERT`, `SUSPECTED`, `CONFIRMED_OBSTRUCTION` ou
`UNKNOWN`, avec preuves et événements individuels inspectables.

Le noyau seul s'installe avec `python -m pip install -e ".[d01]"` (Shapely,
sans installer MetaDrive). L'exécuteur natif utilise l'environnement MetaDrive
Windows déjà décrit plus haut ; aucune dépendance CUDA ou PyTorch n'est requise.

Le contrat expérimental `K04_APPROVED` traite l'ego à 5 Hz et exige une
occupation contemporaine à chaque rendez-vous de 0,4 s. Une confirmation porte
sur une persistance **échantillonnée**, sans garantie d'occupation entre les
mesures. Une pose V2V ancienne n'est jamais actualisée par son TTL. Une lacune
décisive rompt la preuve ; les confirmations antérieures restent historiques.
Les seuils initiaux (5 s d'effet, 30 s de persistance, balayage de 1,2 longueur)
sont des paramètres de prototype, pas une calibration embarquée définitive.
Les rayons seuls ne reconstituent ni les contours ni l'identité des véhicules.

Un exécuteur natif distinct permet les petites expériences de non-interférence
sous Windows, avec l'observateur **désactivé par défaut** :

```powershell
python -m cav_recovery.d01.native --help
python -m cav_recovery.d01.native --output D:\cav-recovery-data\d01-off --vehicles 4 --horizon 12
python -m cav_recovery.d01.native --output D:\cav-recovery-data\d01-on --vehicles 4 --horizon 12 --observer
python -m pytest tests/test_d01_inputs.py tests/test_d01_observer.py tests/test_d01_native.py -q
```

Choisir des dossiers nouveaux, séparés du corpus figé. L'exécuteur réutilise
les géométries natives et le conducteur historique M01 pour comparer les
commandes, poses, missions, arrivées et contrôles d'intégrité avec/sans D01.
Il ne constitue pas un rejeu du conducteur de production M02. La charge proposée
va jusqu'à 30 CAV ; les coûts CPU/mémoire et les dépassements sont journalisés.
Les diagnostics et entrées locales sont des traces expérimentales privées.

Cette intégration indépendante ne modifie aucun module inclus dans l'empreinte
historique du pipeline M02, ni ses données, partitions ou annotations.
F01 n'implémente ni détection collective, ni TCN, ni nouveau C-RDG, ni politique
Graph-MARL ou récupération.
