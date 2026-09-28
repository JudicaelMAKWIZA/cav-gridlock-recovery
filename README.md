## CGR-E01 — Qualification pNEUMA

CGR-E01 qualifie un fichier pNEUMA à une ligne par trajectoire. Il lit la source
progressivement, ne la modifie jamais et ne fait ni
analyse de flux, ni reconstruction de route/OD, ni map-matching, ni simulation.

```bash
python scripts/qualify_pneuma.py --input <fichier.csv> --output-dir outputs/empirical/CGR-E01/<execution>
```

Le dossier de sortie doit être nouveau ou vide. Il reçoit `manifest.json`,
`trajectories.csv`, `observations.csv.gz`, `issues.csv`, `quality_summary.json` et
`qualification_report.md`. Ces sorties et les données réelles restent privées via
`.gitignore`. Un code de sortie `0` signifie qu'au moins un groupe est minimalement
utilisable ; `2` signale une entrée refusée ou un fichier inutilisable ; `1` une erreur
technique. Les diagnostics signalent les anomalies, sans les corriger.

## CGR-E02 — Profil d'un secteur pNEUMA

CGR-E02 réutilise exclusivement les exports normalisés de CGR-E01. Il détecte
les intersections orientées avec les portes finies du secteur figé, reconstruit
des visites et produit des comptages et proportions avec leur couverture et
leurs dénominateurs. Il ne réalise ni map-matching, ni reconstruction des
origines-destinations réelles, ni simulation.

```bash
python scripts/profile_pneuma.py \
  --source <fichier-pNEUMA-prive.csv> \
  --cgr-e01-dir <dossier-prive-des-exports-CGR-E01> \
  --sector-seed <seed-prive-du-secteur.json> \
  --geometry-source <geometrie-OSM-historique-privee.osm> \
  --runtime-config <configuration-privee-CGR-E02.json> \
  --output-dir outputs/empirical/CGR-E02/<execution>
```

La configuration d'exécution utilise le schéma `CGR-E02-runtime-1`. Elle doit
déclarer explicitement `deduplication_s`, `max_time_gap_s` et
`max_space_gap_m`. La géométrie bornée est celle des segments de portes : aucun
axe de branche n'est prolongé. La couverture est fournie par porte sous forme d'intervalles
`[début, fin]`, ou par la chaîne
`"unknown"`. Une couverture inconnue ne produit jamais un débit nul : le débit
reste indéfini. Dans une fenêtre partiellement couverte, le débit utilise seulement
les franchissements compris dans les sous-intervalles d'exposition ; le comptage brut
reste disponible séparément. Les seuils de diagnostic de présélection ne sont pas repris
automatiquement. La configuration doit aussi contenir une justification des
seuils, confirmer que la référence manuelle n'a pas servi à leur réglage et
documenter la preuve — ou l'absence de preuve — de couverture pour chaque porte.

Les résultats réels sont privés dans `outputs/`. Le manifeste vérifie et
enregistre les empreintes calculées localement de la source, des exports CGR-E01,
du seed, de la géométrie et de la configuration utilisée. L'état technique de
l'exécution, l'admissibilité empirique et la validation scientifique sont publiés
séparément ; une exécution réussie ne vaut pas validation scientifique.
`validation_reference.csv` est créé
comme gabarit vide : son annotation manuelle indépendante reste une étape de
validation scientifique, distincte du pipeline automatique.
