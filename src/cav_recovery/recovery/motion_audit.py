"""Contrefactuels de prévision : même actionneur et même garde local REC-PHYS-3."""

from functools import partial
from pathlib import Path
import math

from ..laboratory.storage import atomic_json, json_file, file_hash
from .causal import AuditPolicy
from .feasibility import search
from .motion_model import center_pose, center_sweep
from .sequence_audit import SequencedCoordinator


def checked_model(function, own, action, distance):
    """Gabarit natif DefaultVehicle vérifié ; aucun paramètre appris sur l'issue."""
    if not (
        math.isclose(own["length_m"], 4.515, abs_tol=1e-6)
        and math.isclose(own["width_m"], 1.852, abs_tol=1e-6)
        and math.isclose(own["wheelbase_m"], 2.46894, abs_tol=1e-6)
    ):
        raise ValueError("REC-MOTION-2 n'est calibré ici que pour DefaultVehicle.")
    return function(own, action, distance, rear_axle_offset_m=1.4166)


class PhysicalSequenceCoordinator(SequencedCoordinator):
    def __init__(self):
        super().__init__()
        self.searcher = partial(
            search,
            pose_predictor=partial(checked_model, center_pose),
            sweep_predictor=partial(checked_model, center_sweep),
        )


def run_motion_condition(spec, output, *, local=False):
    """E modifie C2 ; F modifie D. Seule la prévision de manœuvre change."""
    from .environment import run

    output = Path(output)
    condition = dict(
        strategy="clearance" if local else "nominal",
        trigger="local",
        information="local",
        not_before_s=69.0,
    )

    def factory(ego, endpoint, contract):
        policy = AuditPolicy(ego, endpoint, contract, condition, {})
        if local:
            policy.policy.pose_predictor = partial(checked_model, center_pose)
            policy.policy.sweep_predictor = partial(checked_model, center_sweep)
        return policy

    record = dict(
        condition="F_local_center_model" if local else "E_global_center_model",
        motion_prediction="REC-MOTION-2_chassis_center_from_rear_axle",
        privileged_information=not local,
        safety_guard_changed=False,
        safety="same_local_REC_PHYS_3_guard_legacy_sweep_retained_for_controlled_comparison",
        training_performed=False,
        rear_axle_offset_m=1.4166,
        caution="La correction de prévision ne certifie pas le garde historique.",
    )
    result = run(
        spec,
        output,
        method="causal_audit",
        measurement_contract="REC-METRO-1",
        policy_factory=factory,
        coordinator_factory=None if local else PhysicalSequenceCoordinator,
        policy_version=record,
    )
    atomic_json(output / "audit-condition.json", record)
    manifest = json_file(output / "manifest.json")
    manifest["files"]["audit-condition.json"] = file_hash(
        output / "audit-condition.json"
    )
    atomic_json(output / "manifest.json", manifest)
    return result
