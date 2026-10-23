"""Comparaisons locales et appariées du passage et de la géométrie du garde."""

from functools import partial
from pathlib import Path

from ..laboratory.storage import atomic_json, file_hash, json_file
from .causal import AuditPolicy
from .environment import run
from .motion_audit import checked_model
from .motion_model import center_pose, center_sweep
from .safety import LocalGuard


def run_passage_condition(
    spec, output, *, passage=False, aligned_guard=False, fresh_partners=False,
    revalidate=False, verified=False, prioritized=False, continuous=False,
    envelope=False,
    label=None,
):
    """Un facteur à la fois ; le contrat physique REC-PHYS-3 reste constant."""
    output = Path(output)
    condition = dict(
        strategy=(
            "guard_envelope_clearance" if envelope else
            "continuous_clearance" if continuous else
            "prioritized_clearance" if prioritized else
            "verified_clearance" if verified else
            "revalidated_clearance" if revalidate else
            "fresh_partner_clearance" if fresh_partners else
            "coordinated_clearance" if passage else "clearance"
        ),
        trigger="local",
        information="local",
        not_before_s=69.0,
    )

    def factory(ego, endpoint, contract):
        actor = AuditPolicy(ego, endpoint, contract, condition, {})
        actor.policy.pose_predictor = partial(checked_model, center_pose)
        actor.policy.sweep_predictor = partial(checked_model, center_sweep)
        return actor

    def aligned(contract):
        return LocalGuard(
            contract,
            sweep_predictor=partial(checked_model, center_sweep),
        )

    record = dict(
        condition=(
            label or ("D" if passage and aligned_guard else "B" if passage else "C")
        ),
        factors={
            "resource_passage_commitment": any((
                passage, fresh_partners, revalidate, verified, prioritized,
                continuous, envelope,
            )),
            "center_aligned_guard": aligned_guard,
            "fresh_partners": fresh_partners,
            "option_revalidated": revalidate,
            "verified_passage_report": verified or prioritized or continuous or envelope,
            "upstream_passage_priority": prioritized or continuous or envelope,
            "in_flight_passage_priority": continuous or envelope,
            "guard_reserved_nominal_envelope": envelope,
        },
        domain="DefaultVehicle_only_local_development_scenes",
        motion_prediction="REC-MOTION-2",
        safety="REC-PHYS-3_limits_with_center_sweep_only_when_requested",
        privileged_information=False,
        trained_policy=False,
    )
    result = run(
        spec,
        output,
        method="causal_audit",
        measurement_contract="REC-METRO-1",
        policy_factory=factory,
        guard_factory=aligned if aligned_guard else None,
        policy_version=record,
    )
    atomic_json(output / "audit-condition.json", record)
    manifest = json_file(output / "manifest.json")
    manifest["files"]["audit-condition.json"] = file_hash(
        output / "audit-condition.json"
    )
    atomic_json(output / "manifest.json", manifest)
    return result
