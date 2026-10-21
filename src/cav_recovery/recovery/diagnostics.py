"""Composition locale D01/C-RDG, avec domaine de mesure explicite et homogène."""

from collections import Counter

from ..communication.reports import DiagnosticExchange, make_report
from ..crdg.local import LocalCRDG
from ..d01.inputs import LocalInputFactory
from ..d01.observer import EgoObserver
from .metrology import StationaryMeasurement


class RecoveryDiagnostics:
    """Le domaine REC-METRO-1 n'est jamais substitué au flux brut historique.

    Le transport de rapports est interne à une exécution homogène. L'enveloppe
    de sortie précise la version de mesure ; mélanger des pairs de domaines
    différents requiert un autre protocole, et n'est pas pris en charge ici.
    """

    def __init__(self, *, measurement_contract="raw"):
        if measurement_contract not in {"raw", "REC-METRO-1"}:
            raise ValueError("Domaine de mesure inconnu.")
        self.measurement_contract = measurement_contract
        self.observers = {}
        self.raw_observers = {}
        self.factories = {}
        self.adapters = {}
        self.graphs = {}
        self.exchange = DiagnosticExchange()
        self.counts = Counter()

    def observe(self, packets, at_s):
        diagnostics, raw, receipts = {}, {}, {}
        for ego, packet in sorted(packets.items()):
            if ego not in self.factories:
                self.factories[ego] = LocalInputFactory(packet)
                self.observers[ego] = EgoObserver(detailed=False)
                self.raw_observers[ego] = EgoObserver(detailed=False)
                self.adapters[ego] = StationaryMeasurement()
                self.graphs[ego] = LocalCRDG(ego)
            local = {
                k: packet[k]
                for k in (
                    "ego_id",
                    "decision_at_s",
                    "ego",
                    "lidar",
                    "own_controls",
                    "v2v",
                )
            }
            raw[ego] = self.raw_observers[ego].update(self.factories[ego].build(local))
            if self.measurement_contract != "raw":
                local, receipts[ego] = self.adapters[ego].adapt(local)
            diagnostics[ego] = self.observers[ego].update(
                self.factories[ego].build(local)
            )
            self.counts[diagnostics[ego]["status"]] += 1
        reports = {
            ego: dict(
                position_xy_m=packets[ego]["ego"]["position_xy_m"],
                report=make_report(d),
            )
            for ego, d in diagnostics.items()
        }
        if at_s:
            self.exchange.observe(at_s, reports)
        result = {}
        for ego, diagnostic in diagnostics.items():
            graph = self.graphs[ego].update(
                diagnostic, self.exchange.knowledge(ego), at_s
            )
            result[ego] = dict(
                measurement_contract=self.measurement_contract,
                historical_raw=raw[ego],
                diagnostic=diagnostic,
                graph=graph,
                measurement_receipt=receipts.get(ego),
                scientific_scope="K04_thresholds_unchanged_measurement_variant_explicit",
            )
        return result
