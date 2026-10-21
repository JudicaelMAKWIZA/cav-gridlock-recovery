"""Composition des CAV locaux et du transport ; aucune action renvoyée au conducteur."""

from collections import Counter
import copy
import gzip
import json
from pathlib import Path
import time

from ..communication.reports import DiagnosticExchange, make_report
from ..crdg import LocalCRDG
from ..d01.inputs import LocalInputFactory
from ..d01.native import project_mailbox
from ..d01.observer import EgoObserver


def write_row(stream, value):
    stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def percentiles(values):
    values = sorted(values)
    if not values:
        return dict(p50=None, p95=None, p99=None, maximum=None)

    def at(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(len(values) - 1, lower + 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return dict(p50=at(0.5), p95=at(0.95), p99=at(0.99), maximum=values[-1])


def graph_record(view):
    """Conserve une représentation déterministe du graphe partiel réellement calculé."""
    current = [row for row in view["edges"] if row["state"] == "CURRENT"]
    return dict(
        groups=view["groups"],
        collective_status=view["collective_status"],
        simple_count_baseline=view["simple_count_baseline"],
        simple_count_requesters=view["simple_count_requesters"],
        edges=[
            {
                field: row[field]
                for field in (
                    "requester",
                    "blocker",
                    "movement_id",
                    "level",
                    "provenance",
                    "observed_at_s",
                )
            }
            for row in current
        ],
        historical_edge_count=len(view["edges"]) - len(current),
        indirect_dependencies=view["indirect_dependencies"],
        scope=view["scope"],
    )


class DetectionSession:
    """La racine de composition distribue une entrée propre à chaque instance.

    Les états des autres CAV n'entrent dans une instance que par ses messages
    réellement livrés. Le transport utilise les positions uniquement pour la
    portée. La visualisation et les traces globales restent des consommateurs.
    """

    project_mailbox = staticmethod(project_mailbox)

    def __init__(
        self,
        output,
        *,
        gui=False,
        compare_local=True,
        display_context=None,
        temporal_model=None,
    ):
        self.output = Path(output)
        self.contexts = {}
        self.factories = {}
        self.observers = {}
        self.local_observers = {}
        self.graphs = {}
        self.exchange = DiagnosticExchange()
        self.streams = {}
        self.costs = []
        self.statuses = Counter()
        self.reasons = Counter()
        self.last = {}
        self.gui = gui
        self.display_context = dict(display_context or {})
        self.compare_local = compare_local
        self.view = None
        self.finished = False
        self.reference_monitor = None
        self.reference_stats = None
        self.temporal_monitor = None
        if temporal_model is not None:
            from ..laboratory.inference import LocalTemporalMonitor

            self.temporal_monitor = LocalTemporalMonitor(temporal_model)
        import psutil

        self.process = psutil.Process()
        self.initial_rss = self.process.memory_info().rss
        self.peak_rss = self.initial_rss
        self.started = time.perf_counter()

    def register(self, key, mission, static_map):
        if key in self.contexts:
            raise ValueError("duplicate CAV identity")
        self.contexts[key] = dict(
            mission=copy.deepcopy(mission), map=copy.deepcopy(static_map)
        )
        self.observers[key] = EgoObserver(detailed=False)
        self.local_observers[key] = EgoObserver(detailed=False)
        self.graphs[key] = LocalCRDG(key)
        if not self.streams:
            directory = self.output / "diagnostic"
            directory.mkdir(exist_ok=False)
            for name in (
                "inputs",
                "decisions",
                "reports",
                "features",
                "communications",
            ):
                self.streams[name] = gzip.open(
                    directory / (name + ".jsonl.gz"),
                    "wt",
                    encoding="utf-8",
                    compresslevel=3,
                )
            if self.temporal_monitor is not None:
                self.streams["temporal_estimates"] = gzip.open(
                    directory / "temporal_estimates.jsonl.gz",
                    "wt",
                    encoding="utf-8",
                    compresslevel=3,
                )
        (self.output / "diagnostic" / "contexts.json").write_text(
            json.dumps(self.contexts, separators=(",", ":")), encoding="utf-8"
        )

    def _adapt(self, key, measurements):
        if key not in self.factories:
            context = self.contexts[key]
            initial = dict(
                schema_version="F01-input-1",
                contract_id="K04_APPROVED",
                **copy.deepcopy(context),
                **measurements,
            )
            factory = self.factories[key] = LocalInputFactory(initial)
            return factory.initial
        return self.factories[key].build(measurements)

    def inspect_reference(self, frame, metadata, valid):
        """Canal d'évaluation de la vue ; ces valeurs ne sont jamais envoyées aux modèles."""
        if self.reference_monitor is None:
            from .reference_view import PrefixReference

            self.reference_monitor = PrefixReference()
        if self.view is not None and self.view.show_reference:
            self.reference_stats = self.reference_monitor.update(frame, metadata, valid)
        else:
            # Conserver le préfixe est peu coûteux ; sa qualification est facultative.
            # Ce tampon d'inspection ne rejoint jamais les entrées des CAV.
            self.reference_monitor.frames.append(frame)
            self.reference_stats = None

    def observe(
        self,
        at_s,
        measurements,
        *,
        acquisition_ms=0.0,
        telemetry_ms=0.0,
        telemetry_events=(),
        reference_ms=0.0,
    ):
        costs = Counter()
        costs["independent_reference_view"] = reference_ms
        costs["native_sensor_plan_export"] = acquisition_ms
        costs["telemetry_communication"] = telemetry_ms
        diagnostics = {}
        local_diagnostics = {}
        for key, local in sorted(measurements.items()):
            begin = time.perf_counter()
            prepared = self._adapt(key, local)
            costs["adaptation"] += (time.perf_counter() - begin) * 1000
            begin = time.perf_counter()
            diagnostics[key] = self.observers[key].update(prepared)
            costs["detector"] += (time.perf_counter() - begin) * 1000
            self.statuses[diagnostics[key]["status"]] += 1
            self.reasons[diagnostics[key]["reason"]] += 1
            if self.compare_local:
                begin = time.perf_counter()
                own = dict(local, v2v=None)
                local_diagnostics[key] = self.local_observers[key].update(
                    self.factories[key].build(own)
                )
                costs["comparison_A"] += (time.perf_counter() - begin) * 1000
        begin = time.perf_counter()
        reports = {
            key: dict(
                position_xy_m=local["ego"]["position_xy_m"],
                report=make_report(diagnostics[key]),
            )
            for key, local in measurements.items()
        }
        deliveries = self.exchange.observe(at_s, reports) if at_s else []
        costs["diagnostic_communication"] += (time.perf_counter() - begin) * 1000
        graph_views = {}
        begin = time.perf_counter()
        for key in diagnostics:
            graph_views[key] = self.graphs[key].update(
                diagnostics[key], self.exchange.knowledge(key), at_s
            )
        costs["crdg"] += (time.perf_counter() - begin) * 1000
        begin = time.perf_counter()
        write_row(self.streams["inputs"], dict(at_s=at_s, measurements=measurements))
        for event in deliveries:
            if event["event"] == "emitted":
                write_row(self.streams["reports"], event)
        display_deliveries = [
            dict(channel="telemetry", **event) for event in telemetry_events
        ]
        display_deliveries += [
            dict(channel="diagnostic", **event) for event in deliveries
        ]
        for event in display_deliveries:
            if event["event"] == "emitted":
                write_row(self.streams["communications"], event)
        records = {}
        temporal_estimates = {}
        for key, diagnostic in diagnostics.items():
            view = graph_views[key]
            compact_graph = graph_record(view)
            own_a = local_diagnostics.get(key)
            records[key] = dict(
                B=diagnostic,
                C=compact_graph,
                A={
                    field: own_a[field]
                    for field in ("status", "reason", "current_CAV_occupancy_proven")
                }
                if own_a
                else None,
            )
            packet = measurements[key]
            inbox = packet["v2v"]
            ages = [
                at_s - row["message"]["observed_at_s"]
                for row in (inbox or {}).get("received", [])
            ]
            feature_record = dict(
                ego_id=key,
                movement_id=diagnostic["movement_id"],
                observed_at_s=at_s,
                available_at_s=at_s,
                own_state=packet["ego"],
                lidar=packet["lidar"]["normalized_ranges"],
                requested_speed_mps=packet["own_controls"]["requested_speed_mps"],
                accepted_speed_mps=packet["own_controls"]["accepted_speed_mps"],
                own_native_action=packet["own_controls"]["native_action"],
                received_message_ages_s=ages,
                v2v_box_available=inbox is not None,
                lidar_available=not packet["lidar"]["missing"],
                local_status=diagnostic["status"],
                current_received_relations=len(compact_graph["edges"]),
                observed_collective_groups=len(view["groups"]),
                scope="causal_local_measurements_only",
            )
            write_row(self.streams["features"], feature_record)
            if self.temporal_monitor is not None:
                temporal_begin = time.perf_counter()
                temporal_estimates[key] = self.temporal_monitor.observe(feature_record)
                costs["temporal_inference"] += (
                    time.perf_counter() - temporal_begin
                ) * 1000
        if self.temporal_monitor is not None:
            write_row(
                self.streams["temporal_estimates"],
                dict(at_s=at_s, vehicles=temporal_estimates),
            )
        write_row(self.streams["decisions"], dict(at_s=at_s, vehicles=records))
        costs["serialization"] += (time.perf_counter() - begin) * 1000 - costs[
            "temporal_inference"
        ]
        self.last = dict(
            at_s=at_s,
            diagnostics=diagnostics,
            graphs=graph_views,
            deliveries=display_deliveries,
            costs_ms=dict(costs),
            measurements=measurements,
            independent_reference=self.reference_stats,
            temporal_estimates=temporal_estimates,
        )
        if self.gui:
            begin = time.perf_counter()
            if self.view is None:
                from .view import ExperimentView

                self.view = ExperimentView(
                    output=self.output, context=self.display_context
                )
            if self.view.inspect(self.last) is False:
                self.gui = False
                print("Vue fermée proprement ; fin du calcul sans fenêtre.", flush=True)
            costs["visualization"] = (time.perf_counter() - begin) * 1000
        costs["detector_reports_crdg_total"] = sum(
            costs[key]
            for key in ("adaptation", "detector", "diagnostic_communication", "crdg")
        )
        costs["operational_with_acquisition"] = (
            costs["detector_reports_crdg_total"] + acquisition_ms + telemetry_ms
        )
        if self.temporal_monitor is not None:
            costs["operational_with_temporal"] = (
                costs["operational_with_acquisition"] + costs["temporal_inference"]
            )
        self.costs.append(dict(at_s=at_s, active=len(measurements), **costs))
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def finish(self, at_s):
        if self.finished:
            return
        self.finished = True
        for stream in self.streams.values():
            stream.close()
        if self.view is not None:
            self.view.close()
        if not self.streams:
            return
        terminal = {
            key: observer.finish(at_s)
            for key, observer in self.observers.items()
            if observer.last_output is not None
        }
        (self.output / "diagnostic" / "terminal.json").write_text(
            json.dumps(terminal), encoding="utf-8"
        )
        keys = sorted(
            {key for row in self.costs for key in row if key not in {"at_s", "active"}}
        )
        summary = dict(
            schema="integrated-local-detection-1",
            at_s=at_s,
            statuses=dict(self.statuses),
            reasons=dict(self.reasons),
            components_ms={
                key: percentiles([row.get(key, 0.0) for row in self.costs])
                for key in keys
            },
            busiest_batch_components_ms={
                key: percentiles(
                    [
                        row.get(key, 0.0)
                        for row in self.costs
                        if row["active"] >= max(r["active"] for r in self.costs)
                    ]
                )
                for key in keys
            },
            peak_active=max((row["active"] for row in self.costs), default=0),
            report_transport=self.exchange.snapshot()["counts"],
            wall_s=time.perf_counter() - self.started,
            compare_local=self.compare_local,
            memory=dict(
                initial_process_rss_bytes=self.initial_rss,
                sampled_peak_process_rss_bytes=self.peak_rss,
                scope="whole_process_including_simulator_when_present",
            ),
        )
        (self.output / "diagnostic" / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (self.output / "diagnostic" / "timings.json").write_text(
            json.dumps(self.costs), encoding="utf-8"
        )
        (self.output / "diagnostic" / "features-contract.json").write_text(
            json.dumps(
                dict(
                    sample_s=0.2,
                    causal=True,
                    future_features=False,
                    training_performed=False,
                    oracle_inputs=False,
                    targets_stored_separately=True,
                    group_by="episode_and_ego",
                    history_samples=176,
                    history_s=35.0,
                    sensor_scope="own_lidar_and_received_messages",
                    identifier_use="joins_only_not_predictors",
                    normalization="fit_on_train_only_in_future_work",
                    causal_window_interface="cav_recovery.experiments.temporal.causal_windows",
                    raw_local_inputs="inputs.jsonl.gz",
                    immutable_static_context="contexts.json",
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
