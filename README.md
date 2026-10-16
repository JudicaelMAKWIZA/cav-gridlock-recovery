# CAV Gridlock Recovery

Ce projet étudie la détection et la récupération de blocages collectifs de
véhicules connectés et automatisés. La récupération après formation du blocage
est l'objectif principal ; la prévention reste secondaire.

## Réseau et trafic

Le banc principal utilise une topologie dérivée d'OpenStreetMap autour de
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

## Installation

Python ≥ 3.12, SUMO/TraCI 1.27.1 et NetworkX sont nécessaires.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements.txt
```

Sous PowerShell : `.venv\Scripts\Activate.ps1`. Pour la démonstration graphique,
il faut un affichage compatible avec SUMO-GUI et Tkinter ; sous Ubuntu/WSLg,
installer `python3-tk` si nécessaire. Aucun navigateur ni serveur n'est requis.

## Commande officielle

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

SUMO reste la vue physique principale. Le panneau reçoit les groupes du réseau
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
