"""Contrefactuel fermé de l'entrée REC-PLAN-1, observateur K04 inchangé."""

from pathlib import Path
import gzip

from ..laboratory.storage import atomic_json, json_file, file_hash
from .causal import AuditPolicy
from .diagnostics import RecoveryDiagnostics
from .plan_input import OwnPlanProjector


def run_planned_trigger(spec, output):
    """Le facteur modifié est le plan propre fourni à D01, après REC-METRO-1."""
    from .environment import run

    output = Path(output)
    projector = OwnPlanProjector()
    service = RecoveryDiagnostics(measurement_contract="REC-METRO-1")
    condition = dict(
        strategy="current", information="local", trigger="d01_metro", not_before_s=0
    )

    def observe(instance):
        if "planned-diagnostics" not in instance.streams:
            for name in ["planned-inputs", "planned-diagnostics"]:
                instance.streams[name] = instance.stack.enter_context(
                    gzip.open(
                        output / (name + ".jsonl.gz"),
                        "wt",
                        encoding="utf-8",
                        compresslevel=3,
                    )
                )
        packets = {
            ego: projector.project(
                packet,
                wheelbase_m=instance.views[ego].own["wheelbase_m"],
                max_steering_rad=instance.views[ego].own["max_steering_rad"],
            )
            for ego, packet in instance.diagnostic_packets.items()
        }
        outputs = service.observe(packets, instance.at_s)
        for ego, view in instance.views.items():
            view.diagnostic = outputs[ego]["diagnostic"]
            view.graph = outputs[ego]["graph"]
            view.historical_diagnostic = outputs[ego]["historical_raw"]
            instance.write("planned-inputs", packets[ego])
            instance.write(
                "planned-diagnostics", dict(ego=ego, at_s=instance.at_s, **outputs[ego])
            )

    def factory(ego, endpoint, contract):
        return AuditPolicy(ego, endpoint, contract, condition, {})

    record = dict(
        condition=condition,
        diagnostic_input_contract="REC-PLAN-1",
        measurement_contract="REC-METRO-1",
        K04_changed=False,
        privileged_information=False,
        training_performed=False,
    )
    result = run(
        spec,
        output,
        method="causal_audit",
        measurement_contract="REC-METRO-1",
        policy_factory=factory,
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
