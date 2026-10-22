"""Exécuter la seconde étape prévue, sans répéter indéfiniment son préalable."""

import math
from pathlib import Path

from ..laboratory.storage import atomic_json, file_hash, json_file
from .causal import AuditPolicy
from .contracts import Action
from .feasibility import GlobalCoordinator


class SequencedCoordinator(GlobalCoordinator):
    """Recherche omnisciente expérimentale, mêmes primitives et garde local.

    Une seconde étape n'est armée qu'après le déplacement mesuré demandé à la
    première. Elle conserve les accords frais et la validation native commune.
    L'anticipation géométrique ne vaut jamais autorisation physique automatique.
    """

    uses_privileged_information = True

    def __init__(self, activation_s=69.0):
        super().__init__()
        self.next_search = activation_s
        self.pending_second = None

    def decide(self, instance, nominal_actions):
        now = instance.at_s
        if self.plan is None and self.pending_second is not None:
            if now < self.next_search:
                return {
                    ego: Action(0.0, reason="sequence_between_stages")
                    for ego in instance.views
                }, []
            second = self.pending_second
            self.pending_second = None
            if second["ego"] in instance.views and self.count < 4:
                self.plan = dict(second, speed_mps=-0.6, stage="committed_second")
                self.count += 1
                self.started = now
                self.anchor = list(instance.views[second["ego"]].own["position"])
                self.attempt = f"global:{self.count}"
                self.log.append(
                    dict(event="committed_second_stage", at_s=now, plan=self.plan)
                )
        previous = self.plan
        completed = (
            previous is not None
            and previous["ego"] in instance.views
            and math.dist(self.anchor, instance.views[previous["ego"]].own["position"])
            >= previous["distance_m"]
        )
        result = super().decide(instance, nominal_actions)
        if (
            previous is not None
            and self.plan is None
            and previous.get("lookahead_second")
        ):
            if completed:
                self.pending_second = previous["lookahead_second"]
                self.next_search = now + 0.4
                self.log.append(
                    dict(
                        event="preparatory_distance_reached",
                        at_s=now,
                        pending_second=self.pending_second,
                    )
                )
            else:
                self.log.append(
                    dict(
                        event="preparatory_step_incomplete",
                        at_s=now,
                        cannot_infer_physical_impossibility=True,
                    )
                )
        return result


def run_sequence(spec, output):
    """Une seule modification de stratégie : honorer le plan à deux étapes."""
    from .environment import run

    output = Path(output)
    condition = dict(
        strategy="nominal", trigger="local", information="local", not_before_s=69.0
    )

    def factory(ego, endpoint, contract):
        return AuditPolicy(ego, endpoint, contract, condition, {})

    record = dict(
        condition="C2_committed_second_step",
        privileged_information=True,
        safety="same_local_guard_REC_PHYS_3",
        action_bank_unchanged=True,
        maximum_physical_options=4,
        training_performed=False,
        factor="second_step_execution_and_holding_between_stages",
    )
    result = run(
        spec,
        output,
        method="causal_audit",
        measurement_contract="REC-METRO-1",
        policy_factory=factory,
        coordinator_factory=SequencedCoordinator,
        policy_version=record,
    )
    atomic_json(output / "audit-condition.json", record)
    manifest = json_file(output / "manifest.json")
    manifest["files"]["audit-condition.json"] = file_hash(
        output / "audit-condition.json"
    )
    atomic_json(output / "manifest.json", manifest)
    return result
