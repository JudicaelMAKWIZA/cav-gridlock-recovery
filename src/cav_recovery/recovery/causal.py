"""Expériences ciblées : séparer information, déclenchement, décision et sécurité."""

from dataclasses import replace
from pathlib import Path
import gzip
import math
import time

from ..laboratory.storage import atomic_json, json_file, file_hash
from .cooperative import CooperativePolicy
from .clearance_policy import ClearancePolicy
from .passage_policy import PassagePolicy
from .verified_passage_policy import (
    FreshPartnerPolicy, RevalidatedPassagePolicy, VerifiedPassagePolicy,
    PrioritizedPassagePolicy, ContinuousPassagePolicy, GuardEnvelopePassagePolicy,
)
from .feasibility import GlobalCoordinator
from .temporal import temporal_row


def case_specification(config, name):
    """La recette est fixée avant simulation ; aucun résultat n'est injecté."""
    from ..experiments.catalog import specification
    from .scenarios import crossing

    case = next(c for c in config["cases"] if c["name"] == name)
    if case.get("generator") == "crossing":
        spec = crossing(
            case["vehicles"], seed=case["seed"], horizon_s=config["horizon_s"]
        )
    else:
        spec = specification(
            case["catalogue"], seed=case["seed"], horizon_s=config["horizon_s"]
        )
        spec["agents"] = spec["agents"][: case["vehicles"]]
        spec["active_limit"] = case["vehicles"]
    spec["phase"] = "causal_audit_development_only"
    return spec


class FixedSensorGuard:
    """Le décideur oracle B ne remplace pas les capteurs du garde commun."""

    def __init__(self, guard, original):
        self.guard, self.original = guard, original

    def check(self, view, action, **kwargs):
        return self.guard.check(self.original, action, **kwargs)

    def limit(self, view, action):
        return self.guard.limit(self.original, action)


class AuditPolicy:
    """Une instance par CAV ; les améliorations sont des facteurs déclarés."""

    def __init__(self, ego, endpoint, contract, condition, inputs):
        self.condition, self.inputs = condition, inputs
        cls = {
            "clearance": ClearancePolicy,
            "coordinated_clearance": PassagePolicy,
            "fresh_partner_clearance": FreshPartnerPolicy,
            "revalidated_clearance": RevalidatedPassagePolicy,
            "verified_clearance": VerifiedPassagePolicy,
            "prioritized_clearance": PrioritizedPassagePolicy,
            "continuous_clearance": ContinuousPassagePolicy,
            "guard_envelope_clearance": GuardEnvelopePassagePolicy,
        }.get(condition["strategy"], CooperativePolicy)
        self.policy = cls(ego, endpoint, contract)
        self.outbox = []
        self.recent_progress = []

    def decide(self, view, nominal, guard):
        self.outbox = []
        if self.condition["strategy"] == "nominal":
            return nominal
        if view.at_s < self.condition["not_before_s"] - 1e-7:
            history = self.policy.progress_history
            history.append((view.at_s, view.own["progress_m"]))
            while (
                len(history) > 1
                and history[1][0] <= view.at_s - self.policy.contract.stall_s
            ):
                history.pop(0)
            return nominal
        original = view
        if self.condition["information"] == "oracle_global":
            view = self.inputs[view.ego]
            guard = FixedSensorGuard(guard, original)
        trigger = self.condition["trigger"]
        if trigger in {"d01_raw", "d01_metro"}:
            diagnostic = (
                view.diagnostic
                if trigger == "d01_metro"
                else view.historical_diagnostic
            )
            self.policy.activation_override = (
                diagnostic["status"] == "CONFIRMED_OBSTRUCTION"
            )
        elif trigger == "oracle":
            self.policy.activation_override = view.physical_activation
        elif trigger == "tcn":
            self.recent_progress.append((view.at_s, view.own["progress_m"]))
            while (
                len(self.recent_progress) > 1
                and self.recent_progress[1][0] <= view.at_s - 5
            ):
                self.recent_progress.pop(0)
            slow = view.at_s - self.recent_progress[0][0] >= 5 - 1e-7 and (
                view.own["progress_m"] - self.recent_progress[0][1] < 0.5
            )
            self.policy.activation_override = bool(
                slow
                and view.temporal
                and view.temporal["status"] == "STATISTICAL_ALERT"
                and not guard.limit(view, nominal)[1]["accepted"]
            )
        action = self.policy.decide(view, nominal, guard)
        self.outbox = self.policy.outbox
        return action


class AuditPlanner(GlobalCoordinator):
    """Même garde local que A/B/D ; l'oracle choisit seulement la séquence."""

    uses_privileged_information = True

    def __init__(self, not_before_s):
        super().__init__()
        self.next_search = not_before_s


def global_view(original, views, witnesses, contract):
    """État courant omniscient déclaré ; aucun résultat du planificateur ni futur."""
    messages = [m for m in original.messages if m["kind"] != "STATE"]
    for ego, other in views.items():
        if ego == original.ego:
            continue
        messages.append(
            dict(
                version="REC-V2V-1",
                sender=ego,
                sequence=round(original.at_s / 0.2) + 1,
                kind="STATE",
                attempt="oracle_snapshot",
                target=None,
                sent_s=original.at_s,
                expires_s=original.at_s + contract.message_lifetime_s,
                position=list(other.own["position"]),
                payload={
                    key: other.own[key]
                    for key in [
                        "speed_mps",
                        "heading_rad",
                        "length_m",
                        "width_m",
                        "progress_m",
                    ]
                },
            )
        )
    graph = dict(
        scope="privileged_current_physical_witnesses_not_operational_perception",
        at_s=original.at_s,
        groups=[],
        edges=[
            dict(
                requester=w["requester"],
                blocker=w["blocker"],
                resource_wkt=w["resource"]["polygon_wkt"],
                level="CURRENT_PHYSICAL_OCCUPATION_NOT_PERSISTENCE_CONFIRMATION",
                observed_at_s=original.at_s,
                valid_until_s=original.at_s,
            )
            for w in witnesses
        ],
    )
    return replace(original, messages=messages, graph=graph)


def run_condition(spec, output, condition, *, temporal_model=None, inference_threads=1):
    """Exécution native sans entraînement, avec facteurs et privilèges enregistrés."""
    output = Path(output)
    if condition["trigger"] == "tcn" and temporal_model is None:
        raise ValueError("Le déclenchement TCN exige un artefact gelé explicite.")
    if (
        condition["strategy"] not in {"nominal", "current", "clearance"}
        or condition["information"] not in {"local", "oracle_global"}
        or condition["trigger"]
        not in {"local", "d01_raw", "d01_metro", "tcn", "oracle"}
        or not math.isfinite(condition["not_before_s"])
        or condition["not_before_s"] < 0
    ):
        raise ValueError("Condition causale non prise en charge.")
    if type(inference_threads) is not int or inference_threads < 1:
        raise ValueError("Nombre de threads d'inférence invalide.")
    inputs = {}
    monitor = None
    if temporal_model is not None:
        from ..laboratory.inference import LocalTemporalMonitor

        monitor = LocalTemporalMonitor(temporal_model)
        import torch

        torch.set_num_threads(inference_threads)
    # Sous Windows, charger PyTorch avant Panda3D évite un conflit de DLL.
    # Sans moniteur, aucune dépendance PyTorch n'est imposée par cet audit.
    from .environment import run

    def observe(instance):
        if "causal-observations" not in instance.streams:
            instance.streams["causal-observations"] = instance.stack.enter_context(
                gzip.open(
                    output / "causal-observations.jsonl.gz",
                    "wt",
                    encoding="utf-8",
                    compresslevel=3,
                )
            )
        physical = set()
        if condition["trigger"] == "oracle":
            from ..dataset.qualify import qualify
            from ..dataset.contract import Contract
            from ..experiments.evaluation import active_events

            annotation = qualify(
                instance.audit_frames,
                instance.metadata,
                dict(errors=instance.errors, complete_measurements=True),
                Contract(),
            )
            physical = {
                e["requester"] for e in active_events(annotation, instance.at_s)
            }
        for ego, view in instance.views.items():
            view.physical_activation = ego in physical
            view.historical_diagnostic = instance.diagnostics.service.raw_observers[
                ego
            ].last_output
            elapsed = None
            if monitor is not None:
                packet = instance.diagnostic_packets[ego]
                start = time.perf_counter()
                view.temporal = monitor.observe(
                    temporal_row(packet, view.diagnostic, view.graph)
                )
                elapsed = time.perf_counter() - start
            scope = condition["information"]
            if scope == "oracle_global":
                inputs[ego] = global_view(
                    view,
                    instance.views,
                    instance.audit_frames[-1]["witnesses"],
                    instance.contract,
                )
            instance.write(
                "causal-observations",
                dict(
                    ego=ego,
                    at_s=view.at_s,
                    raw_status=view.historical_diagnostic["status"],
                    raw_reason=view.historical_diagnostic["reason"],
                    metro_status=view.diagnostic["status"],
                    metro_reason=view.diagnostic["reason"],
                    temporal=view.temporal,
                    temporal_wall_s=elapsed,
                    information_scope=scope,
                    oracle_current_states=inputs[ego].messages
                    if scope == "oracle_global"
                    else None,
                    oracle_current_graph=inputs[ego].graph
                    if scope == "oracle_global"
                    else None,
                ),
            )

    def factory(ego, endpoint, contract):
        return AuditPolicy(ego, endpoint, contract, condition, inputs)

    coordinator = (
        (lambda: AuditPlanner(condition["not_before_s"]))
        if condition["planner"]
        else None
    )
    record = dict(
        condition=condition,
        safety="same_local_guard_REC_PHYS_3_for_all_arms",
        privileged_information=condition["information"] == "oracle_global"
        or condition["planner"]
        or condition["trigger"] == "oracle",
        training_performed=False,
        temporal_model_sha256=monitor.model_identity if monitor is not None else None,
        inference_threads=inference_threads if monitor is not None else None,
        runtime_version="REC-CAUSAL-RUNTIME-2",
    )
    result = run(
        spec,
        output,
        method="causal_audit",
        measurement_contract="REC-METRO-1",
        policy_factory=factory,
        observation_hook=observe,
        coordinator_factory=coordinator,
        policy_version=record,
    )
    result["activation"] = (
        "causal_trigger:"
        + condition["trigger"]
        + ";not_before_s="
        + str(condition["not_before_s"])
    )
    result["decision_scope"] = (
        "privileged_decision_common_local_sensor_guard"
        if record["privileged_information"]
        else "local_causal_audit"
    )
    atomic_json(output / "summary.json", result)
    atomic_json(output / "audit-condition.json", record)
    manifest = json_file(output / "manifest.json")
    manifest["files"]["audit-condition.json"] = file_hash(
        output / "audit-condition.json"
    )
    manifest["files"]["summary.json"] = file_hash(output / "summary.json")
    atomic_json(output / "manifest.json", manifest)
    return result
