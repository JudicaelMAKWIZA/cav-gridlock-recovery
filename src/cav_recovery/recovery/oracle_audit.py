"""Borne privilégiée séparée : corps globaux dans le garde après armement commun."""

from pathlib import Path

from ..laboratory.storage import atomic_json, file_hash, json_file
from .causal import AuditPolicy, AuditPlanner
from .feasibility import GlobalBodyGuard


def run_ideal_guard(spec, output, *, activation_s=69.0):
    """Même préfixe nominal local ; perception parfaite déclarée après activation.

    Cette condition n'est jamais une perception opérationnelle réaliste. Elle
    change aussi l'approximation d'occupation du garde : pas une ablation pure
    de la portée radio. Les limites, budgets et accords restent obligatoires.
    """
    from .environment import run

    output = Path(output)
    condition = dict(
        strategy="nominal",
        trigger="local",
        information="local",
        not_before_s=activation_s,
    )

    def factory(ego, endpoint, contract):
        return AuditPolicy(ego, endpoint, contract, condition, {})

    def observe(instance):
        if instance.at_s >= activation_s and not isinstance(
            instance.guard, GlobalBodyGuard
        ):
            instance.guard = GlobalBodyGuard(instance)

    record = dict(
        condition="I_privileged_guard_upper_bound",
        privileged_information=True,
        activation_s=activation_s,
        identical_local_prefix_before_activation=True,
        factors_changed_vs_C=[
            "guard_occupancy_full_current_bodies",
            "guard_occupancy_envelope_approximation",
        ],
        operational_perception_claimed=False,
        training_performed=False,
    )
    result = run(
        spec,
        output,
        method="causal_audit",
        measurement_contract="REC-METRO-1",
        policy_factory=factory,
        coordinator_factory=lambda: AuditPlanner(activation_s),
        observation_hook=observe,
        policy_version=record,
    )
    atomic_json(output / "audit-condition.json", record)
    manifest = json_file(output / "manifest.json")
    manifest["files"]["audit-condition.json"] = file_hash(
        output / "audit-condition.json"
    )
    atomic_json(output / "manifest.json", manifest)
    return result
