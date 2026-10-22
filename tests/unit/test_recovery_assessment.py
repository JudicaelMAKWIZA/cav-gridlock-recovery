"""Le comparateur refuse un faux appariement, même favorable à l'intervention."""

import gzip
import json
import pytest

from cav_recovery.laboratory.storage import atomic_json
from cav_recovery.recovery.assessment import paired_assessment


@pytest.mark.parametrize(
    "file,key",
    [
        ("spec.json", "seed"),
        ("spec.json", "horizon_s"),
        ("contract.json", "relapse_followup_s"),
    ],
)
def test_physical_conditions_cannot_be_relaxed(tmp_path, file, key):
    nominal, intervention = tmp_path / "n", tmp_path / "i"
    for folder in [nominal, intervention]:
        atomic_json(folder / "spec.json", dict(seed=21, horizon_s=180))
        atomic_json(folder / "contract.json", dict(relapse_followup_s=30))
    data = json.loads((intervention / file).read_text())
    data[key] += 1
    atomic_json(intervention / file, data)
    with pytest.raises(ValueError, match="Conditions physiques non appariées"):
        paired_assessment(nominal, intervention, tmp_path / "comparison.json")


def test_privileged_guard_divergence_is_not_a_paired_treatment_effect(tmp_path):
    nominal, intervention = tmp_path / "n", tmp_path / "i"
    for folder in [nominal, intervention]:
        atomic_json(folder / "spec.json", dict(seed=21))
        atomic_json(folder / "contract.json", dict(stall_s=35))
        atomic_json(folder / "physical-annotation.json", dict(events=[]))
        with gzip.open(folder / "reference.jsonl.gz", "wt", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(time_s=0, states={"V0": folder.name})) + "\n")
    with gzip.open(
        intervention / "commands.jsonl.gz", "wt", encoding="utf-8"
    ) as stream:
        stream.write(
            json.dumps(
                dict(at_s=0, guard=dict(accepted=True), applied=dict(reason="recovery"))
            )
            + "\n"
        )
    with pytest.raises(ValueError, match="divergent avant"):
        paired_assessment(nominal, intervention, tmp_path / "comparison.json")
