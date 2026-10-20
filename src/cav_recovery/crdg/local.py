"""Vue partielle, datée et causale des dépendances physiques au carrefour."""

from collections import OrderedDict, deque
import copy

import networkx as nx
from shapely import wkt
from shapely.errors import GEOSException

from ..communication.reports import make_report, validate_report, ReportError, finite


class LocalCRDG:
    """Aucune entrée globale : un diagnostic propre et une boîte réellement reçue."""

    def __init__(self, ego_id):
        self.ego_id = ego_id
        self.relations = OrderedDict()
        self.events = deque(maxlen=256)
        self.previous_current = set()
        self.last_time = None
        self.geometry_cache = OrderedDict()

    def _polygon(self, text):
        if text not in self.geometry_cache:
            try:
                polygon = wkt.loads(text)
            except GEOSException as error:
                raise ReportError("malformed occupied resource") from error
            if not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
                raise ReportError("invalid occupied resource")
            self.geometry_cache[text] = polygon
            if len(self.geometry_cache) > 256:
                self.geometry_cache.popitem(last=False)
        return self.geometry_cache[text]

    def update(self, diagnostic, knowledge, at_s):
        if diagnostic["ego_id"] != self.ego_id or (
            self.last_time is not None and at_s <= self.last_time
        ):
            raise ReportError("wrong ego or noncausal graph update")
        self.last_time = at_s
        reports = [(make_report(diagnostic), "own", at_s + 1.2, None)]
        for sender, entry in sorted((knowledge or {}).get("received", {}).items()):
            if not isinstance(entry, dict) or set(entry) != {
                "received_at_s",
                "message",
            }:
                raise ReportError("invalid report receipt fields")
            message = entry["message"]
            fields = {
                "protocol_version",
                "message_id",
                "sender_id",
                "sequence_number",
                "observed_at_s",
                "sent_at_s",
                "valid_until_s",
                "position_xy_m",
                "report",
            }
            if (
                not isinstance(message, dict)
                or set(message) != fields
                or message["protocol_version"] != 1
            ):
                raise ReportError("invalid diagnostic message fields")
            if not all(
                finite(message[key])
                for key in ("observed_at_s", "sent_at_s", "valid_until_s")
            ) or not finite(entry["received_at_s"]):
                raise ReportError("invalid receipt times")
            if (
                sender != message["sender_id"]
                or not message["sent_at_s"] <= entry["received_at_s"] <= at_s + 1e-7
            ):
                raise ReportError("invalid report receipt")
            if at_s <= message["valid_until_s"] + 1e-7:
                report = validate_report(message["report"], sender, at_s)
                reports.append(
                    (
                        report,
                        "received_v2v",
                        message["valid_until_s"],
                        message["message_id"],
                    )
                )
        for report, source, valid_until, envelope in reports:
            for relation in report["relations"]:
                identity = (
                    relation["requester"],
                    relation["movement_id"],
                    relation["blocker"],
                )
                self._polygon(relation["resource_wkt"])
                self.relations[identity] = dict(
                    **copy.deepcopy(relation),
                    reporter=report["ego_id"],
                    provenance=source,
                    report_message_id=envelope,
                    report_emitted_at_s=report["emitted_at_s"],
                    valid_until_s=valid_until,
                )
                self.relations.move_to_end(identity)
        for identity in list(self.relations):
            if self.relations[identity]["valid_until_s"] < at_s - 1e-7:
                del self.relations[identity]
        while len(self.relations) > 256:
            self.relations.popitem(last=False)
        edges = []
        for row in self.relations.values():
            current = abs(row["observed_at_s"] - at_s) <= 1e-7
            edges.append(dict(**row, state="CURRENT" if current else "HISTORICAL"))
        current_edges = [row for row in edges if row["state"] == "CURRENT"]
        identities = {
            (row["requester"], row["movement_id"], row["blocker"])
            for row in current_edges
        }
        for identity in sorted(identities - self.previous_current):
            self.events.append(
                dict(kind="RELATION_AVAILABLE", at_s=at_s, identity=list(identity))
            )
        for identity in sorted(self.previous_current - identities):
            self.events.append(
                dict(kind="CURRENT_PROOF_ENDED", at_s=at_s, identity=list(identity))
            )
        self.previous_current = identities
        groups = self._groups(current_edges)
        confirmed_requesters = sorted(
            {row["requester"] for row in current_edges if row["level"] == "CONFIRMED"}
        )
        return dict(
            ego_id=self.ego_id,
            at_s=at_s,
            scope="partial_one_hop_received_reports",
            edges=edges,
            groups=groups,
            collective_status="CONFIRMED_COLLECTIVE" if groups else "NOT_ESTABLISHED",
            simple_count_baseline=len(confirmed_requesters) >= 2,
            simple_count_requesters=confirmed_requesters,
            indirect_dependencies=self._indirect(current_edges),
            events=copy.deepcopy(list(self.events)[-12:]),
            anonymous_obstacles=copy.deepcopy(diagnostic["anonymous_obstacles"]),
        )

    def _groups(self, edges):
        confirmed = [row for row in edges if row["level"] == "CONFIRMED"]
        graph = nx.Graph()
        for row in confirmed:
            graph.add_edge(row["requester"], row["blocker"])
        for index, left in enumerate(confirmed):
            for right in confirmed[index + 1 :]:
                if (
                    self._polygon(left["resource_wkt"])
                    .intersection(self._polygon(right["resource_wkt"]))
                    .area
                    > 0
                ):
                    graph.add_edge(left["requester"], right["requester"])
        result = []
        for component in nx.connected_components(graph):
            affected = sorted(
                {row["requester"] for row in confirmed if row["requester"] in component}
            )
            if len(affected) < 2:
                continue
            members = [row for row in confirmed if row["requester"] in component]
            directed = nx.DiGraph((row["requester"], row["blocker"]) for row in members)
            result.append(
                dict(
                    affected_requesters=affected,
                    vehicles=sorted(component),
                    cycle_present=not nx.is_directed_acyclic_graph(directed),
                    observed_at_s=self.last_time,
                    relation_ids=[
                        [row["requester"], row["movement_id"], row["blocker"]]
                        for row in members
                    ],
                )
            )
        return sorted(result, key=lambda row: row["affected_requesters"])

    def _indirect(self, edges):
        graph = nx.DiGraph(
            (row["requester"], row["blocker"])
            for row in edges
            if row["level"] == "CONFIRMED"
        )
        result = []
        for source in sorted(graph):
            for target, path in sorted(
                nx.single_source_shortest_path(graph, source).items()
            ):
                if len(path) > 2 and not graph.has_edge(source, target):
                    result.append(
                        dict(
                            requester=source,
                            blocker=target,
                            path=path,
                            observed_at_s=self.last_time,
                        )
                    )
        return result
