"""Rejeu causal des entrées enregistrées, sans moteur physique ni nouvelle campagne."""

import json
from pathlib import Path

from ..dataset.pipeline import read_rows
from .session import DetectionSession, graph_record


def replay(source, output, *, verify=True, temporal_model=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or output.is_relative_to(
        Path(r"D:\cav-recovery-data\metadrive-dense-20261018").resolve()
    ):
        raise ValueError(
            "replay output must be separate from immutable source evidence"
        )
    output.mkdir(parents=True, exist_ok=False)
    contexts = json.loads((source / "diagnostic/contexts.json").read_text("utf-8"))
    expected = (
        iter(read_rows(source / "diagnostic/decisions.jsonl.gz")) if verify else None
    )
    session = DetectionSession(
        output, compare_local=False, temporal_model=temporal_model
    )
    checked = 0
    at_s = 0.0
    try:
        for row in read_rows(source / "diagnostic/inputs.jsonl.gz"):
            at_s = row["at_s"]
            for key in row["measurements"]:
                if key not in session.contexts:
                    session.register(
                        key, contexts[key]["mission"], contexts[key]["map"]
                    )
            session.observe(at_s, row["measurements"])
            if expected is not None:
                old = next(expected)
                if old["at_s"] != at_s or set(old["vehicles"]) != set(
                    session.last["diagnostics"]
                ):
                    raise ValueError("replay frame alignment differs")
                for key, result in old["vehicles"].items():
                    if result["B"] != session.last["diagnostics"][key]:
                        raise ValueError(
                            "replayed local diagnostic differs from recorded source"
                        )
                    if result["C"] != graph_record(session.last["graphs"][key]):
                        raise ValueError(
                            "replayed partial graph differs from recorded source"
                        )
                checked += 1
    finally:
        session.finish(at_s)
    if expected is not None and next(expected, None) is not None:
        raise ValueError("replay ended before recorded decisions")
    result = dict(
        verified_frames=checked,
        verified_components=["D01", "local_C_RDG"],
        source=str(source),
        engine_initialized=False,
        physics_regenerated=False,
        comparison_A_disabled=True,
        timing_scope="local_diagnostics_graph_and_optional_statistical_monitor_without_native_acquisition",
        temporal_model=str(temporal_model) if temporal_model is not None else None,
        temporal_estimates_are_K04_confirmations=False,
    )
    (output / "replay.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
