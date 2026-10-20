"""Référence globale réservée à l'inspection, sans chemin vers les entrées CAV."""

from collections import deque

from ..dataset.contract import Contract
from ..dataset.qualify import qualify
from .evaluation import active_events, reference_groups


class PrefixReference:
    """Qualifie uniquement le passé déjà simulé pour la légende d'évaluation."""

    def __init__(self):
        self.frames = deque(maxlen=176)

    def update(self, frame, metadata, valid):
        self.frames.append(frame)
        reference = qualify(
            list(self.frames),
            metadata,
            dict(
                errors=[] if valid else ["invalid_physics"],
                complete_measurements=valid,
                all_arrived=False,
            ),
            Contract(),
        )
        at_s = frame["time_s"]
        affected = {event["requester"] for event in active_events(reference, at_s)}
        edges = [edge for edge in frame["witnesses"] if edge["requester"] in affected]
        return dict(
            at_s=at_s,
            affected_requesters=sorted(affected),
            groups=reference_groups(edges),
            source="independent_native_reference_for_display_only",
            valid=valid,
        )
