# Consulter le laboratoire expérimental

Ouvrir les cinq cahiers numérotés avec l'interpréteur Windows du projet. Le mode normal lit les résultats sauvegardés ; il ne relance ni MetaDrive ni l'entraînement. Les sorties exécutées et privées sont dans le dossier de preuves de la campagne.

Les chemins se règlent par variables d'environnement :

```powershell
$env:CGR_DATA_ROOT = 'D:\cav-recovery-data'
$env:CGR_LAB_ROOT = 'D:\cav-recovery-data\detection-tcn-lab-20261021'
$env:CGR_M02_ROOT = "$env:CGR_DATA_ROOT\metadrive-dense-20261018"
$env:CGR_HISTORICAL_ROOT = "$env:CGR_DATA_ROOT\integrated-metadrive-20261020"
```

Les valeurs par défaut suivent l'organisation sur D: ; aucun nom de compte personnel n'est codé dans les cahiers. En déplacement, modifier ces quatre variables. Les données M02 restent externes au dépôt. Les cahiers publics n'embarquent ni poids ni résultats privés.

## Dépendances Windows CPU

Depuis l'environnement virtuel placé sur D: :

```powershell
$env:PIP_CACHE_DIR = 'D:\cav-recovery-cache\pip'
$env:TEMP = 'D:\cav-recovery-temp'
$env:TMP = $env:TEMP
python -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[laboratory,d01]'
python -m pip check
```

Il n'est pas nécessaire d'installer SUMO, CUDA ou un nouveau moteur MetaDrive pour consulter les résultats ou entraîner le TCN. L'environnement MetaDrive existant sert seulement aux contrôles d'intégration natifs.

## Points d'entrée reproductibles

```powershell
cgr lab --help
cgr lab prepare "$env:CGR_M02_ROOT" "$env:CGR_LAB_ROOT\cache-m02-v1" --workers 2
cgr lab run "$env:CGR_LAB_ROOT\campaign-cpu-v1" --config configs/laboratory/tcn_cpu.json --m02-cache "$env:CGR_LAB_ROOT\cache-m02-v1" --integrated-cache "$env:CGR_LAB_ROOT\cache-integrated-v2"
```

Le cache de démonstrations est préparé par la même fonction publique utilisée lors de la campagne :

```powershell
python -c "from cav_recovery.laboratory.integrated import prepare_integrated; import os; from pathlib import Path; prepare_integrated(Path(os.environ['CGR_HISTORICAL_ROOT']), Path(os.environ['CGR_LAB_ROOT'])/'cache-integrated-v2')"
```

Une campagne terminée est consultée et non réentraînée automatiquement. Les entraînements partiels reprennent depuis la dernière génération valide. En cas de changement de code, données ou paramètres, choisir une nouvelle version de résultat ; ne pas réutiliser un checkpoint incompatible. Le test d'interruption volontaire du premier candidat fait partie du protocole ; seul le processus de test est interrompu.

## Tests

```powershell
python -m pytest
python -m pytest tests/integration -q
```

La première commande exécute les unités et la communication, y compris la causalité et la reprise TCN. La seconde vérifie le moteur natif et les intégrations. Les temporaires peuvent être dirigés sur D: avec `--basetemp`.

## Lecture scientifique

Le test OOD M02 asymétrique est réservé à l'évaluation finale du TCN. La collection des 32 démonstrations reste une étude exploratoire distincte : ses variantes parentes sont groupées et ses trois configurations temporelles utilisent les mêmes endpoints.

Une sortie TCN est une probabilité ou une alerte statistique. Elle ne devient pas une preuve K04 ni une commande de récupération. La normalisation, les masques, les seuils de décision et les origines des références sont expliqués dans les cahiers. Les fonctions de calcul et de tracé sont partagées avec l'export de preuves, dans `cav_recovery.laboratory`.
