"""Export unique des preuves : les notebooks emploient ce même laboratoire."""

import html
import shutil


from .storage import atomic_json, file_hash, json_file
from .workbench import Laboratory


def export_evidence(lab=None):
    lab = lab or Laboratory.from_environment()
    from .storage import safe_destination

    safe_destination(lab.dataset, lab.root)
    if not (lab.campaign / "complete.json").exists():
        raise ValueError("Les résultats réels doivent être terminés avant leur export.")
    output = lab.root / "evidence"
    figures = output / "figures"
    tables = output / "tables"
    sources = output / "sources"
    for folder in [figures, tables, sources]:
        folder.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "cgr-lab-20261021",
        }
    )
    registry = []
    selected = json_file(lab.campaign / "selected-model.json")["selected"]
    plots = [
        ("01_m02", lab.plot_dataset, "Composition M02 gelée"),
        ("02_pertes", lab.plot_losses, "Pertes des trois entraînements M02"),
        (
            "03_tcn_m02",
            lambda: lab.plot_quality("m02"),
            "TCN et baseline instantanée : test OOD M02",
        ),
        (
            "04_confusion_m02",
            lambda: lab.plot_confusion("m02"),
            "Matrice de confusion M02",
        ),
        (
            "05_crdg_historique",
            lab.plot_historical_graphs,
            "Groupes C-RDG : validité et complétude",
        ),
        (
            "06_comparaisons_appariees",
            lambda: lab.plot_quality("integrated"),
            "Comparaisons appariées exploratoires",
        ),
        (
            "07_confusion_integree",
            lambda: lab.plot_confusion("integrated"),
            "Matrice de confusion du TCN augmenté",
        ),
        (
            "08_exemple_echec",
            lab.plot_motion,
            "Exemple temporel du groupe tenu à part le moins bien restitué",
        ),
    ]
    events = lab.event_analysis("m02")
    integrated_events = lab.event_analysis("integrated")
    plots.append(
        (
            "09_delais_m02",
            lambda: lab.plot_event_delays(events),
            "Délais, événements manqués et censure M02",
        )
    )
    plots.append(
        (
            "10_delais_integres",
            lambda: lab.plot_event_delays(integrated_events),
            "Délais exploratoires sur démonstrations",
        )
    )
    for name, create, title in plots:
        fig = create()
        for suffix in ["png", "svg"]:
            fig.savefig(
                figures / (name + "." + suffix),
                dpi=180,
                bbox_inches="tight",
                metadata={"Date": None} if suffix == "svg" else None,
            )
        plt.close(fig)
        registry.append(
            dict(
                file="figures/" + name + ".png",
                vector="figures/" + name + ".svg",
                title=title,
                source="cav_recovery.laboratory.workbench ; données enregistrées, pas de nouvelle simulation",
                context=getattr(fig, "cgr_context", {}),
            )
        )
    gallery = lab.root.parent / "chapter-3/metadrive-ui-0512af2"
    if (gallery / "manifest.json").exists():
        manifest = json_file(gallery / "manifest.json")
        destination = output / "interface_0512af2"
        destination.mkdir(exist_ok=True)
        for row in manifest["figures"]:
            assert file_hash(gallery / row["file"]) == row["sha256"]
            for name in [row["file"], row["metadata"]]:
                shutil.copy2(gallery / name, destination / name)
            registry.append(
                dict(
                    file="interface_0512af2/" + row["file"],
                    title="Illustration acceptée : " + row["file"],
                    source="galerie native conservée, commit " + manifest["commit"],
                    context=dict(
                        scenario="collective-1",
                        seed=21019,
                        time_s=60.0,
                        acquisition=row["acquisition"],
                    ),
                )
            )
        shutil.copy2(gallery / "manifest.json", destination / "manifest.json")
    all_tables = {
        "01_partitions_m02": lab.dataset_overview(),
        "02_mesures_historiques": lab.historical_metrics(),
        "03_abstentions_historiques": lab.historical_abstentions(),
        "04_tcn_test_m02": lab.model_results("m02"),
        "05_comparaisons_appariees": lab.model_results("integrated"),
        "06_evenements_m02": events,
        "07_evenements_integres": integrated_events,
        "08_courbes_apprentissage": [
            dict(model=name, **row)
            for name, history in lab.histories().items()
            for row in history
        ],
    }
    all_tables["09_32_essais_historiques"] = [
        dict(
            scenario=row["name"],
            seed=row["seed"],
            topology=row["topology"],
            label=row["reference_label"],
            physics_valid=row["valid_physics"],
            accepted=row["scenario_acceptance"]["physical_rule_satisfied"],
            tp=row["metrics"]["B"].get("tp", 0),
            fp=row["metrics"]["B"].get("fp", 0),
            fn=row["metrics"]["B"].get("fn", 0),
            recall=row["metrics"]["B"]["recall"],
        )
        for row in lab.historical_episodes()
    ]
    all_tables["10_modeles_entraines"] = []
    for scope in ["m02", "integrated"]:
        for complete in sorted((lab.campaign / scope).glob("*/complete.json")):
            result = json_file(complete)
            all_tables["10_modeles_entraines"].append(
                dict(
                    scope=scope,
                    model=complete.parent.name,
                    parameters=result["parameters"],
                    epochs=result["completed_epochs"],
                    wall_s=result["wall_s"],
                    train_windows=result["train_windows"],
                    validation_loss=result["best_validation_loss"],
                    code_identity=result["compatibility"]["code"],
                    data_identity=result["compatibility"]["dataset"],
                )
            )
    all_tables["11_transfert_m02_integre"] = lab.transfer_results()
    all_tables["12_couts_rejeu_enregistre"] = lab.replay_costs()
    for name, records in all_tables.items():
        lab.export_table(tables, name, records)
    source_records = []
    for receipt in [
        "timing-anomalies.json",
        "audit-final.json",
        "source-format-equivalence.json",
    ]:
        path = lab.root / receipt
        if path.exists():
            shutil.copy2(path, sources / receipt)
    for path in [
        lab.campaign / "protocol.json",
        lab.campaign / "selected-model.json",
        lab.campaign / "power-cut-proof.json",
        lab.campaign / "paired-comparisons.json",
        lab.cache / "complete.json",
        lab.integrated_cache / "complete.json",
        lab.integrated_cache / "split.json",
        lab.dataset / "export-final/manifest.json",
        lab.dataset / "production-final/summary.json",
        lab.historical / "report-summary.json",
        lab.historical / "demonstration-selection.json",
    ]:
        name = (
            "cache_m02_"
            if path.parent == lab.cache
            else "cache_integre_"
            if path.parent == lab.integrated_cache
            else "historique_"
            if path.parent == lab.historical
            else "dataset_"
            if path.is_relative_to(lab.dataset)
            else ""
        ) + path.name
        shutil.copy2(path, sources / name)
        source_records.append(
            dict(source=str(path), file="sources/" + name, sha256=file_hash(path))
        )
    original = lab.root / "core-trained-v1"
    if original.exists():
        shutil.copytree(
            original, sources / "training-source-original", dirs_exist_ok=True
        )
    for complete in sorted(lab.campaign.glob("*/*/complete.json")):
        scope, name = complete.parent.parent.name, complete.parent.name
        destination = sources / "models" / scope / name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in [
            "complete.json",
            "config.json",
            "normalization.json",
            "history.json",
            "calibration.json",
            "validation-metrics.json",
            "test_ood-metrics.json",
        ]:
            path = complete.parent / filename
            if path.exists():
                shutil.copy2(path, destination / filename)
        best = json_file(complete.parent / "checkpoints/best.json")
        weight = complete.parent / "checkpoints" / best["file"]
        assert file_hash(weight) == best["sha256"]
        shutil.copy2(weight, destination / "best.pt")
        atomic_json(
            destination / "best-origin.json",
            dict(source=str(weight), sha256=best["sha256"]),
        )
        # Toutes les métriques par fenêtre sont petites face aux traces physiques.
        for path in complete.parent.glob("*-predictions.npz"):
            shutil.copy2(path, destination / path.name)
    atomic_json(output / "sources.json", source_records)
    atomic_json(output / "figures.json", registry)
    atomic_json(
        output / "events-summary.json",
        {
            key: dict(
                total=len(records),
                censored_no_endpoint=sum(r["censored_no_endpoint"] for r in records),
                missed_eligible=sum(
                    r["missed"] and not r["censored_no_endpoint"] for r in records
                ),
                detected=sum(r["delay_s"] is not None for r in records),
                premature_events=sum(r["premature_statistical_alert"] for r in records),
            )
            for key, records in [("m02", events), ("integrated", integrated_events)]
        },
    )
    index = [
        "# Preuves D01 / C-RDG / TCN — 21 octobre 2026",
        "",
        "Les images et tables sont produites par les mêmes fonctions que les cinq notebooks. Aucun apprentissage ni moteur n’est lancé par la consultation.",
        "",
        "## Figures",
        "",
    ]
    gallery = [
        '<!doctype html><html lang="fr"><meta charset="utf-8"><title>Laboratoire D01 C-RDG TCN</title>',
        "<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:16px}img{max-width:100%}section{margin:40px 0}</style>",
        "<h1>Résultats réels du laboratoire</h1><p>Études M02 et démonstrations séparées. Une alerte statistique ne constitue pas une preuve K04.</p>",
    ]
    for row in registry:
        index += [
            f"### {row['title']}",
            "",
            f"[PNG]({row['file']})"
            + (f" · [SVG]({row['vector']})" if "vector" in row else ""),
            "",
            row["source"],
            str(row.get("context", {})),
            "",
        ]
        gallery += [
            "<section><h2>"
            + html.escape(row["title"])
            + '</h2><img src="'
            + row["file"]
            + '"></section>'
        ]
    index += ["## Tableaux", ""] + [
        f"- [{name}](tables/{name}.html) · [CSV](tables/{name}.csv)"
        for name in all_tables
    ]
    index += [
        "",
        "## Lecture et interprétation",
        "",
        "[Mise à jour du chapitre 3](MISE_A_JOUR_CHAPITRE3_DETECTION_TCN.md)",
        "",
        "Les métriques M02 du TCN sont des mesures sur le test OOD asymétrique réservé. Les ablations intégrées sont exploratoires sur des démonstrations choisies, regroupées par parents.",
        "Les anciens scores de D01/C-RDG ne sont pas remplacés par les scores d’une autre tâche ou d’une autre cadence. Les cellules exécutées sont conservées dans notebooks_executed.",
        f"Modèle M02 sélectionné : `{selected}`. Les poids complets et générations de reprise restent dans le dossier de campagne privé ; une copie du meilleur état de chaque modèle figure dans sources/models.",
    ]
    (output / "INDEX_PREUVES.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    (output / "index.html").write_text("\n".join(gallery) + "</html>", encoding="utf-8")
    write_academic_update(lab, output)
    return dict(directory=str(output), figures=len(registry), tables=len(all_tables))


def write_academic_update(lab, output):
    selected = json_file(lab.campaign / "selected-model.json")
    m02 = lab.model_results("m02")
    paired = lab.model_results("integrated")
    transfer = [
        dict(model=r["partition"], **{k: v for k, v in r.items() if k != "partition"})
        for r in lab.transfer_results()
    ]
    event_summary = json_file(output / "events-summary.json")
    historical = {row["task"]: row for row in lab.historical_metrics()}
    partitions = lab.dataset_overview()
    historical_b = historical["B"]
    timing_file = lab.root / "timing-anomalies.json"
    timing_note = (
        str(json_file(timing_file))
        if timing_file.exists()
        else "Aucun reçu d'anomalie temporelle fourni."
    )

    def table(records):
        lines = [
            "| Modèle | TP | FP | FN | Précision | Rappel | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]

        def number(v):
            return "non défini" if v is None else f"{v:.6f}"

        lines += [
            f"| {r['model']} | {r['tp']} | {r['fp']} | {r['fn']} | {number(r['precision'])} | {number(r['recall'])} | {number(r['f1'])} |"
            for r in records
        ]
        return "\n".join(lines)

    text = f"""# Mise à jour du chapitre 3 — détection et TCN

## Résultats historiques conservés

Base acceptée : `0512af2faec2ea5f1fd9acc4e0fd0ccddbdcb538`. M02 reste gelé. Effectifs relus dans les références : `{partitions}`. Les inconnus sont séparés de la supervision. Le test OOD asymétrique conserve sa partition historique.

La collection intégrée historique conserve tous ses essais et sa sélection d'illustrations. Sur ses rendez-vous corrélés de 0,4 s : D01 B a précision {historical_b['precision']:.6f}, rappel {historical_b['recall']:.6f}, F1 {historical_b['f1']:.6f}, avec {historical_b['fp']} FP. Le F1 exact de groupe C-RDG vaut {historical['C_groups']['f1']:.6f} et reste distinct de la validité des sous-groupes. Les événements manqués, reculs physiques et abstentions restent dans les résultats ; les tableaux 02 et 03 fournissent leurs comptes.

## Correction technique et absence de gain D01 inventé

Le C-RDG refuse maintenant une date de graphe incompatible avec la preuve propre, y compris une preuve future. Aucun seuil K04 ni critère physique n'est changé. Les flux correctement datés conservent leurs mesures ; aucun gain de rappel D01 n'est attribué à ce contrôle de validité. Les informations globales ne deviennent pas des entrées locales.

## Rôle du TCN et protocole réellement exécuté

Le TCN estime **si une confirmation physique persistante est déjà disponible maintenant**, à partir de 35 s maximum d'observations causales du même CAV/mouvement. Ce n'est ni une prédiction d'une étiquette future privilégiée, ni une preuve K04, ni une politique de récupération. Les premiers historiques sont masqués à gauche. Les identifiants et coordonnées absolues ne sont pas des canaux. La normalisation est ajustée sur train seulement.

Deux TCN de taille modeste et une baseline instantanée ont été entraînés sur CPU. Train est échantillonné de façon fixe, une fenêtre sur cinq ; la sélection préliminaire emploie une fenêtre de validation sur cinq. Le calibrage final et les évaluations utilisent toutes les fenêtres supervisées des partitions concernées. Les inconnus de supervision sont exclus explicitement, jamais assimilés à des négatifs. Les deux architectures TCN ont les mêmes durées, canaux et règles d'échantillonnage.

La sélection est `{selected["selected"]}`, sur BCE de validation uniquement. Les seuils statistiques sont calibrés sur la validation. Le test OOD M02 n'ajuste aucun paramètre.

### Confirmation disponible maintenant — test OOD M02

{table(m02)}

Les fenêtres sont corrélées. Les résultats par épisode et dénominateurs sont conservés dans `sources/models/m02/*/test_ood-metrics.json`. Ce test OOD concerne le générateur synthétique M02, avec une topologie asymétrique tenue à part ; il ne valide pas la circulation routière réelle. La baseline instantanée mesure les raccourcis accessibles à la dernière observation ; son score ne doit pas être omis.

## Ablations intégrées appariées

La seconde étude utilise la collection de démonstrations choisies. Les variantes parentes sont indivisibles ; les groupes sont stratifiés sur leurs références physiques déjà établies, avant entraînement, avec graine 1741. Une préparation initiale sans stratification, dont la validation ne contenait aucun endpoint positif, est conservée dans `cache-integrated-v1` et n'a entraîné aucun modèle. La version utilisée est `cache-integrated-v2`. Le nom interne `test_ood` de ce second cache signifie seulement groupes de démonstration tenus à part.

Les configurations sont comparées sur les mêmes endpoints de 2 s, appartenant à la grille de preuve, les mêmes mouvements, cibles et conditions. TCN brut ne reçoit aucun statut D01 ni caractéristique C-RDG. Les ablations ajoutent explicitement les quatre statuts D01, puis deux compteurs de connaissances C-RDG locales actuelles. Les trois variantes apprennent sur les mêmes groupes train et valident sur les mêmes groupes validation.

{table(paired)}

Dans cette comparaison tenue à part, l'ajout des compteurs C-RDG ne fournit pas de gain uniforme : la variante D01+C-RDG augmente les faux positifs par rapport à la variante D01 seule. Ce résultat défavorable est conservé. Les variantes ne sont pas sélectionnées a posteriori en fonction du meilleur score de test ; le modèle opérationnel de référence reste le TCN brut sélectionné sur M02 validation.

La règle « D01 B ou TCN » est une alerte statistique exploratoire. Elle ne modifie pas les confirmations de K04 et ne fournit aucune action de conduite. Les scores des ablations ne remplacent pas les anciens scores sur tous les rendez-vous des 32 essais. Ils ne sont pas une validation indépendante de généralisation. Le transfert descriptif du modèle M02 vers les démonstrations, sans nouvelle normalisation ni recalibrage, est également sauvegardé dans `campaign-cpu-v1/transfer-m02-to-integrated`.

### Transfert descriptif M02 vers les démonstrations

{table(transfer)}

La ligne `test_ood` ci-dessus désigne les quatre groupes de démonstration tenus à part, et non une nouvelle topologie inconnue. Ce transfert ne sert pas à sélectionner une autre variante. Le tableau 11 conserve aussi les dénominateurs et vrais négatifs.

## Événements, manqués et préalertes

Résumé exact : M02 `{event_summary["m02"]}` ; démonstrations `{event_summary["integrated"]}`. Les délais concernent uniquement les événements retrouvés après la confirmation disponible. Les événements sans endpoint recevable sont distingués de ceux qui sont observables et manqués. Les préalertes restent des erreurs pour la tâche « confirmation maintenant » avant l'instant de confirmation de référence ; elles ne sont pas présentées comme des preuves physiques anticipées.

## Interruptions, reproductibilité et limites

Le premier entraînement long a réellement subi une terminaison de processus après trois lots sauvegardés, puis a repris. Le verrou abandonné et les journaux de reprise sont conservés. Un test séparé compare exactement poids, états d'optimiseur et historiques avec un entraînement ininterrompu. Les générations sont écrites temporairement, validées, hachées puis publiées atomiquement ; une génération valide antérieure et un meilleur modèle restent disponibles.

La garantie est le dernier lot sauvegardé, avec réexécution possible des lots non sauvegardés. Les états additionnels pourront être fournis pour Graph-MARL, mais aucun Graph-MARL n'est implémenté et aucune restauration d'un épisode physique en cours n'est promise.

Une seule graine d'apprentissage, une petite recherche d'architecture et le même générateur synthétique limitent l'étude. Les faibles reculs hors domaine K04, les références spatiales sans cause dynamique unique, les groupes incomplets et les biais de densité M02 restent documentés. Les temps d'entraînement sont des temps instrumentés sous charge de développement, pas une mesure de temps réel automobile. Aucun gain statistique du TCN ne l'autorise à lire l'oracle, requalifier M02 ou imposer une récupération.

Reçu d'anomalie de durée : `{timing_note}`. Un éventuel intervalle inexpliqué n'est pas soustrait du temps mural et empêche d'en déduire un coût CPU comparable. Le reçu `sources/timing-anomalies.json`, lorsqu'il existe, conserve les dates exactes. Les coûts du tableau 12 proviennent de frames rejouées : ils excluent physique, rendu et acquisition native, et ne constituent pas une garantie temps réel.

## Figures et sections à actualiser

Ajouter les figures 02 à 04 dans la section modèle temporel et évaluation M02 ; 06 et 07 dans les comparaisons appariées ; 08 dans l'analyse d'erreurs, 09 et 10 dans la temporalité et les événements manqués. Conserver la figure 05 et les anciennes vues MetaDrive pour l'interprétation C-RDG. L'index relie chaque fichier à son origine. Les cinq notebooks exécutés montrent les données, les calculs communs, le modèle chargé et les mesures réelles. Aucun fichier LaTeX n'est modifié.

Références méthodologiques : [TCN, Bai et al.](https://arxiv.org/abs/1803.01271), [états de reprise PyTorch](https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html). Ces références motivent l'architecture et la sauvegarde ; elles ne constituent pas les résultats expérimentaux du projet.
"""
    (output / "MISE_A_JOUR_CHAPITRE3_DETECTION_TCN.md").write_text(
        text, encoding="utf-8"
    )
