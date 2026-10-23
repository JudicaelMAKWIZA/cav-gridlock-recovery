"""Établit un verdict binaire vérifiable avant tout entraînement long."""

import argparse
from collections import Counter
from pathlib import Path
import statistics
import subprocess
import math
import psutil

from cav_recovery.laboratory.storage import atomic_json, file_hash, json_file
from cav_recovery.recovery.graph_marl_core import load_config


PROJECT = Path(__file__).resolve().parents[1]


def estimate_costs(config, qualified, train, probe, validation_probe, source):
    """Projette le budget depuis les scènes choisies et une sonde Graph réelle."""
    identifiers = {row["source_id"] for row in train}
    selected = [item for item in qualified
                if item["split"] == "train" and item["source_id"] in identifiers]
    mean_wall = statistics.mean(item["summary"]["wall_s"] for item in selected)
    mean_cpu = statistics.mean(item["summary"]["cpu_s"] for item in selected)
    nominal_probe = next(item["summary"]["wall_s"] for item in selected
                         if item["source_id"] == probe["source_id"])
    graph_ratio = probe["wall_s"] / nominal_probe
    graph_wall = mean_wall * graph_ratio
    periodic_overhead = (validation_probe["wall_s"] + graph_wall) / config[
        "validation_every_episodes"]
    episodes = int(360 * 60 / (graph_wall + periodic_overhead))
    positive_fraction = (sum(row["qualified_label"] == "POSITIVE_CONFIRMED"
                             for row in train) / len(train))
    useful_choices = round(episodes * positive_fraction * probe["useful_choices"])
    sample = list({row["source_id"]: row for row in train}.values())[:12]
    trace_sizes = [sum(path.stat().st_size for path in
                       (source / "nominal" / "train" / row["source_id"]).iterdir()
                       if path.is_file()) for row in sample]
    mean_bytes = statistics.mean(trace_sizes)
    snapshot = source / "resource-snapshot.json"
    return dict(
        nominal_episode_mean_wall_s=mean_wall,
        graph_probe_over_nominal_wall_ratio=graph_ratio,
        estimated_graph_episode_wall_s=graph_wall,
        nominal_episode_mean_cpu_s=mean_cpu,
        nominal_trace_mean_bytes_sample=mean_bytes,
        trace_size_sample_count=len(trace_sizes),
        estimated_training_trace_gib=episodes * mean_bytes / 2**30,
        resource_snapshot=json_file(snapshot) if snapshot.exists() else None,
        estimated_periodic_overhead_s_per_episode=periodic_overhead,
        estimated_episodes_in_six_hours=episodes,
        estimated_useful_choices_from_one_probe=useful_choices,
        estimated_full_ppo_batches_from_one_probe=
        useful_choices // config["ppo_batch_size"],
    )


def write_report(root, report, qualified_count):
    """Présente le même verdict et ses limites dans un texte consultable."""
    if "capacity_estimate" in report:
        capacity = report["capacity_estimate"]
        lines = ["# Graph-PPO P4-3 — préparation finale", "",
                 f"Verdict logiciel et scientifique : **{report['verdict']}**.",
                 "L'entraînement long reste lancé personnellement dans Jupyter.",
                 "", "| Contrôle | Résultat |", "|---|---|"]
        lines += [f"| {name} | {'PASS' if value else 'FAIL'} |"
                  for name, value in report["checks"].items()]
        lines += ["", f"Commit : `{report['git_commit']}` ; "
                  f"qualification historique : {qualified_count} scènes.",
                  f"Train : {report['training_episodes']} épisodes planifiés ; "
                  "18 scènes positives et 10 négatives distinctes. "
                  "Validation : 5 positives et 5 négatives distinctes.",
                  "", "## Capacité estimée, six heures", ""]
        for key in ("optimistic", "prudent", "adverse"):
            row = capacity[key]
            lines += [f"- {key} : {row['episodes']} épisodes, "
                      f"{row['positives']} positifs, {row['negatives']} négatifs, "
                      f"{row['free_choices_low']}–{row['free_choices_typical']} "
                      f"choix libres supposés, {row['ppo_batches_low']}–"
                      f"{row['ppo_batches_typical']} lots PPO, "
                      f"{row['optimizer_steps_low']}–"
                      f"{row['optimizer_steps_typical']} pas d'optimiseur. "
                      f"Validation périodique : {row['periodic_validation_wall_s']/60:.1f} min."]
        lines += ["", f"Validation finale prioritaire : "
                  f"{capacity['final_priority_wall']['lower_s']/60:.0f}–"
                  f"{capacity['final_priority_wall']['upper_s']/60:.0f} min hors budget.",
                  f"Validation complète : {capacity['final_full_wall']['lower_s']/60:.0f}–"
                  f"{capacity['final_full_wall']['upper_s']/60:.0f} min hors budget.",
                  f"RAM libre relevée : {capacity['free_ram_gib']:.2f} Gio ; "
                  f"seuil huit CAV : {capacity['minimum_eight_cav_gib']:.1f} Gio ; "
                  f"pic RSS observé : {capacity['observed_peak_process_rss_gib']:.2f} Gio.",
                  f"Traces légères estimées : {capacity['projected_trace_gib']:.2f} Gio.",
                  "Trace diagnostique détaillée : précontrôle et premier épisode "
                  "d'entraînement uniquement ; référence, physique, commandes "
                  "et diagnostics restent conservés pour tous les épisodes.",
                  "", "## Limites", ""]
        lines += [f"- {item}" for item in report["limitations"]]
        lines += ["- Les résultats de P4 et du réseau non entraîné ne prouvent pas "
                  "la contribution d'un réseau entraîné.",
                  "- Aucune convergence ni supériorité sur P4 n'est prédite.",
                  "- Le test réservé n'a pas été consulté."]
        (root / "TRAINING_READINESS_REPORT.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        return
    verdict = report["verdict"]
    counts = report["selected_unique_counts"]
    lines = ["# Préparation Graph-PPO — verdict " + verdict, "",
             "Le lancement long est réservé à l'étudiant et contrôlé par les "
             "empreintes ci-dessous.", "",
             "| Critère | Résultat |", "|---|---|"]
    lines += [f"| {name} | {'PASS' if passed else 'FAIL'} |"
              for name, passed in report["checks"].items()]
    lines += ["", f"Commit public : `{report['git_commit']}`.",
              f"Train : {counts['train/POSITIVE_CONFIRMED']} positives et "
              f"{counts['train/NEGATIVE_CONFIRMED']} négatives distinctes ; "
              f"{report['training_episodes']} épisodes planifiés avec répétitions.",
              f"Validation : {counts['validation/POSITIVE_CONFIRMED']} positives et "
              f"{counts['validation/NEGATIVE_CONFIRMED']} négatives distinctes.",
              f"Topologies positives : {', '.join(report['training_positive_families'])}.",
              f"Qualification nominale : {qualified_count} scènes conservées, "
              "y compris les échecs.",
              f"Sonde appariée : {report['probe']['useful_choices']} choix libres, "
              f"{report['probe']['native_longitudinal_differences']} différences "
              "d'action longitudinale ; aucun gain de récupération démontré.",
              f"Validation périodique présondée : "
              f"{report['validation_probe']['decisions']} appels réseau, "
              f"{report['validation_probe']['useful_choices']} choix libres ; "
              "pas de gain avec le réseau aléatoire.",
              "Contrôles récupérables à quatre et huit CAV : "
              f"{report['recoverable_neural_probes']} ; le réseau est non entraîné.",
              f"Mini-PPO : {report['ppo']['transitions']} transitions, "
              f"{report['ppo']['updates']} pas d'optimiseur, poids modifiés et restaurés.",
              f"P4, requêtes V2V hors domaine : {report['p4_domain_rejections']} ; "
              "les refus ne sont pas des accords de passage.",
              "Contrôles P4 actuels à quatre et huit CAV : "
              f"{report['known_recoverable_controls']} ; références de "
              "développement, pas test de généralisation.",
              "Qualité : compilation, AST du notebook, pytest et dépendances vérifiés ; "
              "Sourcery est indiqué séparément selon sa disponibilité locale.",
              f"Temps nominal moyen : {report['nominal_episode_mean_wall_s']:.1f} s ; "
              f"ratio Graph/nominal mesuré sur une scène : "
              f"{report['graph_probe_over_nominal_wall_ratio']:.2f} ; projection "
              f"Graph {report['estimated_graph_episode_wall_s']:.1f} s/épisode "
              f"et environ {report['estimated_episodes_in_six_hours']} "
              "épisodes en six heures avec validation périodique, "
              "hors validation finale et variabilité Windows.",
              f"CPU nominal moyen : {report['nominal_episode_mean_cpu_s']:.1f} s ; "
              f"volume moyen des traces sur {report['trace_size_sample_count']} "
              f"scènes : {report['nominal_trace_mean_bytes_sample'] / 2**20:.1f} "
              f"Mio ; projection entraînement "
              f"{report['estimated_training_trace_gib']:.1f} Gio.",
              "Projection très incertaine depuis une seule sonde : environ "
              f"{report['estimated_useful_choices_from_one_probe']} choix libres "
              f"et {report['estimated_full_ppo_batches_from_one_probe']} "
              "lots PPO complets.",
              "", "## Limites", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    lines += ["", "Les preuves détaillées restent privées dans le dossier de "
              "qualification et le dossier de campagne. Le test réservé, M02 "
              "et P4 historiques sont intacts."]
    (root / "TRAINING_READINESS_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def selection_facts(config, qualified, selection):
    """Vérifie les effectifs sélectionnés et les parents des deux partitions."""
    groups = Counter((item["split"], item["qualified_label"])
                     for item in qualified if item["status"]["valid"])
    train = config["scenes"]["train"]
    validation = config["scenes"]["validation"]
    unique = {split: {label: {row["source_id"] for row in rows
                              if row["qualified_label"] == label}
                      for label in ("POSITIVE_CONFIRMED", "NEGATIVE_CONFIRMED")}
              for split, rows in (("train", train), ("validation", validation))}
    families = {row["source_family"] for row in train
                if row["qualified_label"] == "POSITIVE_CONFIRMED"}
    selected_ids = {(split, identifier)
                    for split, labels in unique.items()
                    for identifiers in labels.values()
                    for identifier in identifiers}
    records = {(item["split"], item["source_id"]): item for item in qualified}
    selected_valid = all(records[key]["status"]["valid"] for key in selected_ids)
    selection_matches = all(
        set(selection["selected"][split][label]) == unique[split][label]
        for split in ("train", "validation")
        for label in ("POSITIVE_CONFIRMED", "NEGATIVE_CONFIRMED")
    )
    train_parents = {row["source_parent"] for row in train}
    validation_parents = {row["source_parent"] for row in validation}
    return groups, unique, families, selected_valid, (
        selection_matches and not bool(train_parents & validation_parents))


def evidence_hashes(root, source, quality_path, train, validation):
    """Scelle la configuration et chaque référence nominale effectivement lue."""
    root_files = ["experiment-config.json", "source-manifest.json",
                  "qualification-selection.json", "scene-plan.json",
                  "environment.json", "reports/preflight.json"]
    external = [source / "qualification.json", source / "candidate-shortlist.json",
                source / "probe-causal-analysis.json",
                source / "p4-control-4-paired.json",
                source / "p4-control-8-paired.json",
                source / "graph-control-4-paired.json",
                source / "graph-control-8-paired.json",
                source / "probes/p4-oblique-4-21021/current-p4/graph-marl-manifest.json",
                source / "probes/p4-oblique-8-21021/current-p4/graph-marl-manifest.json",
                source / "probes/p4-oblique-4-21021/probe-summary.json",
                source / "probes/p4-oblique-4-21021/random-graph-current/graph-marl-manifest.json",
                source / "probes/p4-oblique-8-21021/probe-summary.json",
                source / "probes/p4-oblique-8-21021/random-graph-current/graph-marl-manifest.json",
                source / "probes/d9a6ef14caaf56793d7b7c35/probe-summary.json",
                source / "probes/ad1ba7cbf83eea435c26cf7b/probe-summary.json",
                source / "probes/ad1ba7cbf83eea435c26cf7b/random-graph-validation/graph-marl-manifest.json",
                source / "validation-probe-paired.json",
                source / "probes/2f9b86783aeb09504cc5c400/p4/summary.json",
                source / "probes/eaf24bd9d0a1f42b693b5153/p4-domain-serial/summary.json",
                source / "probes/eaf24bd9d0a1f42b693b5153/p4-domain-serial/policy-domain-summary.json",
                source / "mini-ppo-long-credit/mini-ppo-report.json",
                quality_path / "quality-checks.json",
                PROJECT / "notebooks/12_ENTRAINEMENT_GRAPH_PPO_COMPLET.ipynb"]
    external.extend(quality_path.glob("*.log"))
    external.extend(Path(row["source_spec_path"]) for row in train + validation)
    for split, rows in (("train", train), ("validation", validation)):
        for identifier in {row["source_id"] for row in rows}:
            folder = source / "nominal" / split / identifier
            external.extend(folder / name for name in (
                "spec.json", "summary.json", "metadata.json", "contract.json",
                "physical-annotation.json", "reference.jsonl.gz"))
    return ({name: file_hash(root / name) for name in root_files},
            {str(path): file_hash(path) for path in sorted(set(external))})


def capacity_estimate(config, qualified, preparation):
    """Projette les épisodes dans leur ordre figé avec des sondes de deux classes.

    Les bornes englobent l'incertitude de géométrie et de charge Windows.
    Les choix libres hors scènes sondées restent une hypothèse, non une mesure.
    """
    probes = {kind: json_file(preparation / f"probe-{identifier}.json")
              for kind, identifier in (
                  ("negative", "d5b80c6afbc99d5abbd03867"),
                  ("positive", "d9a6ef14caaf56793d7b7c35"),
                  ("positive_four_arm", "c590be2ff28ee70a2bab7727"))}
    nominal = {(row["split"], row["source_id"]): row["summary"]["wall_s"]
               for row in qualified}
    ratios = {
        kind: probe["wall_s"] / nominal["train", probe["source_id"]]
        for kind, probe in probes.items()
    }
    ratios["positive_4_control"] = (
        json_file(Path(config["qualification_root"]) /
                  "probes/p4-oblique-4-21021/probe-summary.json")["wall_s"] /
        nominal["train", "p4-oblique-4-21021"])

    def projected(scale):
        elapsed = 0.0
        selected = []
        periodic_cost = sum(
            nominal["validation", identifier] * ratios[
                "positive" if next(row for row in config["scenes"]["validation"]
                                   if row["source_id"] == identifier)["qualified_label"]
                == "POSITIVE_CONFIRMED" else "negative"] * scale
            for identifier in config["periodic_validation_ids"])
        for row in config["scenes"]["train"]:
            kind = "positive" if row["qualified_label"] == "POSITIVE_CONFIRMED" else "negative"
            ratio = (ratios["positive_4_control"] if row["vehicles"] == 4 else
                     ratios["positive_four_arm"] if kind == "positive" and
                     row["source_family"] == "four_arm" else ratios[kind])
            episode_cost = nominal["train", row["source_id"]] * ratio * scale
            after = len(selected) + 1
            if after % config["validation_every_episodes"] == 0:
                episode_cost += periodic_cost
            if elapsed + episode_cost > config["maximum_wall_minutes"] * 60:
                break
            elapsed += episode_cost
            selected.append(row)
        positive_count = sum(row["qualified_label"] == "POSITIVE_CONFIRMED"
                             for row in selected)
        negative_count = len(selected) - positive_count
        measured_negative = probes["negative"]["free_choices"]
        choices_low = 3 * positive_count + measured_negative * negative_count
        choices_typical = 5 * positive_count + measured_negative * negative_count
        batches_low = choices_low // config["ppo_batch_size"]
        batches_typical = choices_typical // config["ppo_batch_size"]
        steps_per_batch = (config["ppo_epochs"] *
                           math.ceil(config["ppo_batch_size"] /
                                     config["minibatch_size"]))
        return dict(episodes=len(selected), positives=positive_count,
                    negatives=negative_count, predicted_wall_s=round(elapsed),
                    free_choices_low=choices_low,
                    free_choices_typical=choices_typical,
                    ppo_batches_low=batches_low,
                    ppo_batches_typical=batches_typical,
                    optimizer_steps_low=batches_low * steps_per_batch,
                    optimizer_steps_typical=batches_typical * steps_per_batch,
                    periodic_validation_wall_s=round(
                        len(selected) // config["validation_every_episodes"]
                        * periodic_cost))

    def validation_cost(identifiers):
        rows = {row["source_id"]: row for row in config["scenes"]["validation"]}
        # Deux baselines supplémentaires par scène ; facteur 1 à 2 selon
        # les sondes. Les références nominales sont déjà qualifiées.
        graph = sum(nominal["validation", identifier] * ratios[
            "positive_four_arm" if rows[identifier]["qualified_label"] ==
            "POSITIVE_CONFIRMED" and rows[identifier]["source_family"] ==
            "four_arm" else "positive" if rows[identifier]["qualified_label"] ==
            "POSITIVE_CONFIRMED" else "negative"] for identifier in identifiers)
        baseline = sum(nominal["validation", identifier] for identifier in identifiers)
        return dict(lower_s=round(graph + 2 * baseline),
                    upper_s=round(1.4 * graph + 4 * baseline))

    memory = psutil.virtual_memory()
    return dict(
        measured_probes=probes, measured_graph_over_nominal_ratios=ratios,
        optimistic=projected(1.0), prudent=projected(1.3),
        adverse=projected(1.7),
        final_priority_wall=validation_cost(config["priority_validation_ids"]),
        final_full_wall=validation_cost(
            [row["source_id"] for row in config["scenes"]["validation"]]),
        free_ram_gib=round(memory.available / 2**30, 3),
        minimum_eight_cav_gib=config["minimum_available_gib_eight_cav"],
        observed_peak_process_rss_gib=max(probe["peak_rss_gib"]
                                          for probe in probes.values()),
        observed_minimum_free_gib=min(probe["min_available_gib"]
                                      for probe in probes.values()),
        projected_trace_gib=round(
            statistics.mean(probe["bytes"] for probe in probes.values())
            * projected(1.3)["episodes"] / 2**30, 2),
        warning="Extrapolation depuis trois sondes légères et contrôles P4 ; "
                "les choix des scènes non sondées ne sont pas démontrés.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--preparation", type=Path)
    args = parser.parse_args()
    root, source = args.campaign.resolve(), args.qualification.resolve()
    config = load_config(root / "experiment-config.json")
    qualified = json_file(source / "qualification.json")["scenes"]
    selection = json_file(root / "qualification-selection.json")
    quality = json_file(args.quality / "quality-checks.json")
    probe_root = source / "probes" / "d9a6ef14caaf56793d7b7c35"
    probe = json_file(probe_root / "probe-summary.json")
    validation_probe = json_file(
        source / "probes/ad1ba7cbf83eea435c26cf7b/probe-summary.json")
    causal = json_file(source / "probe-causal-analysis.json")
    p4_control = json_file(source / "p4-control-4-paired.json")
    p4_control_eight = json_file(source / "p4-control-8-paired.json")
    graph_control = json_file(
        source / "probes/p4-oblique-4-21021/probe-summary.json")
    graph_control_paired = json_file(source / "graph-control-4-paired.json")
    graph_control_eight = json_file(
        source / "probes/p4-oblique-8-21021/probe-summary.json")
    graph_control_eight_paired = json_file(source / "graph-control-8-paired.json")
    mini = json_file(source / "mini-ppo-long-credit" / "mini-ppo-report.json")
    preflight_path = root / "reports/preflight.json"
    preflight = json_file(preflight_path) if preflight_path.exists() else {}
    environment_path = root / "environment.json"
    environment = json_file(environment_path) if environment_path.exists() else {}
    manifest = json_file(root / "source-manifest.json")
    source_clean = all(file_hash(PROJECT / name) == expected
                       for name, expected in manifest["files"].items())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=PROJECT, text=True).strip()
    source_clean &= manifest["git_commit"] == commit
    train = config["scenes"]["train"]
    validation = config["scenes"]["validation"]
    groups, unique, families, selected_valid, split_independent = selection_facts(
        config, qualified, selection)
    p4_probes = []
    failed_p4_path = (source / "probes" / "eaf24bd9d0a1f42b693b5153" /
                      "p4" / "failure.json")
    failed_p4 = json_file(failed_p4_path) if failed_p4_path.exists() else None
    first_retry_path = (source / "probes" / "eaf24bd9d0a1f42b693b5153" /
                        "p4-domain-guard" / "failure.json")
    p4_cases = (("2f9b86783aeb09504cc5c400", "p4"),
                ("eaf24bd9d0a1f42b693b5153", "p4-domain-serial"))
    p4_domain = {}
    for family, variant in p4_cases:
        folder = source / "probes" / family / variant
        path = folder / "summary.json"
        if path.exists():
            p4_probes.append(json_file(path))
            domain_path = folder / "policy-domain-summary.json"
            if domain_path.exists():
                p4_domain[family] = json_file(domain_path)[
                    "rejected_clearance_requests"]
    proof = {
        "environment": bool(environment) and environment.get("pip_check") == "OK",
        "source_identity": source_clean,
        "qualified_training": len(unique["train"]["POSITIVE_CONFIRMED"]) >= 12
                              and len(unique["train"]["NEGATIVE_CONFIRMED"]) >= 4,
        "qualified_validation": len(unique["validation"]["POSITIVE_CONFIRMED"]) >= 4
                                and len(unique["validation"]["NEGATIVE_CONFIRMED"]) >= 3,
        "topology_diversity": len(families) >= 2,
        "split_independence": split_independent,
        "native_integrity": selected_valid and not probe["summary"]["errors"]
                            and probe["validity"]["reverse_max_m"] <= 8.0
                            and len(p4_probes) == 2
                            and all(not item["errors"]
                                    and max(item["reverse_m"].values(), default=0) <= 8.0
                                    for item in p4_probes),
        "neural_physical_influence": causal["scenario_identical"]
                                     and causal["longitudinal_action_differences"] > 0
                                     and probe["useful_choices"] > 1,
        "validation_neural_choices": (
            validation_probe["source_id"] == config["periodic_validation_ids"][0]
            and validation_probe["validity"]["valid"]
            and validation_probe["useful_choices"] > 1
        ),
        "known_recoverable_controls": all(
            any(row["source_id"] == f"p4-oblique-{count}-21021"
                for row in train)
            and paired["improvement_not_spontaneous"]
            and paired["intervention"]["status"] == "DURABLE_RECOVERY"
            for count, paired in ((4, p4_control), (8, p4_control_eight))
        ),
        "recoverable_neural_options": all(
            item["validity"]["valid"]
            and item["useful_choices"] > 1
            and item["non_hold"] > 0
            and paired["improvement_not_spontaneous"]
            for item, paired in ((graph_control, graph_control_paired),
                                 (graph_control_eight,
                                  graph_control_eight_paired))
        ),
        "ppo_update": mini["weights_changed"] and mini["checkpoint_loaded"]
                      and mini["useful_transitions"] >= 2
                      and mini["update"]["updates"] > 0,
        "preflight": bool(preflight.get("proceed")),
        "software_checks": quality["passed"],
    }
    capacity = None
    if config["version"] == "REC-GRAPH-PPO-P4-3":
        if args.preparation is None:
            raise ValueError("Les sondes légères sont requises pour P4-3.")
        capacity = capacity_estimate(config, qualified, args.preparation.resolve())
        proof["trace_physical_equivalence"] = all(json_file(
            args.preparation / "light-physical-comparison.json")["matching"].values())
        proof["resources_current"] = (capacity["free_ram_gib"] >=
                                      capacity["minimum_eight_cav_gib"])
        proof["learning_capacity"] = capacity["prudent"]["ppo_batches_low"] >= 5
    verdict = "TRAIN_READY" if all(proof.values()) else "TRAIN_NOT_READY"
    estimates = estimate_costs(config, qualified, train, probe,
                               validation_probe, source)
    fingerprints, external_fingerprints = evidence_hashes(
        root, source, args.quality.resolve(), train, validation)
    if capacity is not None:
        for path in (list(args.preparation.glob("*.json")) +
                     list(args.preparation.glob("*.md")) +
                     list((args.preparation / "scenes").glob("*"))):
            if not path.is_file():
                continue
            external_fingerprints[str(path.resolve())] = file_hash(path)
    report = dict(
        verdict=verdict, checks=proof, git_commit=commit,
        source_commit=manifest["git_commit"],
        qualified_counts={f"{split}/{label}": count
                          for (split, label), count in sorted(groups.items())},
        selected_unique_counts={f"{split}/{label}": len(ids)
                                for split, labels in unique.items()
                                for label, ids in labels.items()},
        training_episodes=len(train), validation_scenes=len(validation),
        training_positive_families=sorted(families),
        positive_affected_counts=sorted(len(item["affected"]) for item in qualified
                                        if item["split"] == "train"
                                        and item["qualified_label"] == "POSITIVE_CONFIRMED"),
        probe=dict(useful_choices=probe["useful_choices"],
                   non_hold=probe["non_hold"],
                   native_longitudinal_differences=causal["longitudinal_action_differences"],
                   graph_gain=causal["paired_graph"]["improvement_not_spontaneous"],
                   p4_gain=causal["paired_p4"]["improvement_not_spontaneous"]),
        validation_probe=dict(
            source_id=validation_probe["source_id"],
            useful_choices=validation_probe["useful_choices"],
            decisions=validation_probe["decisions"],
            wall_s=validation_probe["wall_s"],
            gain=json_file(source / "validation-probe-paired.json")[
                "improvement_not_spontaneous"],
        ),
        ppo=dict(transitions=mini["useful_transitions"],
                 updates=mini["update"]["updates"],
                 update_wall_s=mini["update_wall_s"],
                 weights_changed=mini["weights_changed"]),
        quality_tools_available=quality.get("tools_available", {}),
        initial_p4_domain_failure=failed_p4,
        first_guard_retry_failure=(json_file(first_retry_path)
                                   if first_retry_path.exists() else None),
        p4_domain_rejections=p4_domain,
        known_recoverable_controls={str(count): dict(
            intervention=paired["intervention"]["status"],
            nominal=paired["nominal"]["status"],
            improvement_not_spontaneous=paired["improvement_not_spontaneous"],
            t_act_s=paired["t_act_s"],
            t_conf_s=paired["intervention"]["t_conf_s"],
        ) for count, paired in ((4, p4_control), (8, p4_control_eight))},
        recoverable_neural_probes={str(count): dict(
            useful_choices=item["useful_choices"],
            non_hold=item["non_hold"],
            arrived=item["validity"]["arrived"],
            gain=paired["improvement_not_spontaneous"],
            untrained=True,
        ) for count, item, paired in ((4, graph_control, graph_control_paired),
                                      (8, graph_control_eight,
                                       graph_control_eight_paired))},
        **estimates,
        **({"capacity_estimate": capacity} if capacity is not None else {}),
        fingerprints=fingerprints,
        external_fingerprints=external_fingerprints,
        limitations=[
            "Aucune performance Graph-PPO entraînée n'est encore connue.",
            "Le précontrôle apparié ne démontre pas de gain de récupération.",
            "Les épisodes répétés ne sont pas des configurations uniques.",
            "Une qualification nominale ne prouve pas la récupérabilité de chaque scène.",
            "Le coût final de validation P4/B3 est additionnel au budget d'entraînement.",
            "L'échec P4 initial sur four_arm à 139 s est conservé ; le correctif "
            "abstient les requêtes V2V hors domaine sans changer le seuil de 200 m².",
            "Les comparaisons P4 futures utilisent ce garde d'émission partagé ; "
            "le commit P4 historique demeure inchangé.",
        ],
    )
    atomic_json(root / "TRAINING_READINESS.json", report)
    write_report(root, report, len(qualified))
    print(verdict, proof)
    return 0 if verdict == "TRAIN_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
