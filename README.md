## CGR-E01 — Qualification pNEUMA

La seule fonction empirique disponible qualifie un fichier pNEUMA à une ligne par
trajectoire. Elle lit la source progressivement, ne la modifie jamais et ne fait ni
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
