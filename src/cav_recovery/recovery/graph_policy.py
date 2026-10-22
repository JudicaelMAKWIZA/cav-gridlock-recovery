"""Acteur-critique partagé sur graphe local, pilote hiérarchique sans oracle."""

import math
import numpy as np
import torch
from torch import nn
from shapely import wkt
from shapely.geometry import Point
from .cooperative import CooperativePolicy, option_bank
from .contracts import Action

FEATURES = 24


def local_graph(view, *, response=False):
    """Nœuds véhicules/ressources reçus ; jamais de groupe physique global.

    Les arêtes de communication et demandes déclarées restent distinctes des
    relations C-RDG étayées. Les identifiants servent au raccordement, pas aux
    caractéristiques numériques. Les dates futures ou expirées sont exclues.
    """
    own = view.own
    angle = own["heading_rad"]
    c, s = math.cos(angle), math.sin(angle)

    def relative(position):
        x, y = np.asarray(position) - own["position"]
        return [(c * x + s * y) / 30.0, (-s * x + c * y) / 30.0]

    nodes = []
    links = []
    identities = {view.ego: 0}
    ego = [0.0] * FEATURES
    ego[0] = 1.0
    ego[3] = own["speed_mps"] / 2.0
    ego[4] = own["progress_m"] / 200.0
    station = view.route.line.project(Point(own["position"]))
    ego[5:7] = relative(view.route.position(station + 3.0))
    ego[7] = view.reverse_used_m / 8.0
    ego[8] = max(0.0, view.own_commitment_until_s - view.at_s)
    controls = view.own_controls or {}
    ego[9] = (controls.get("applied") or {}).get("speed_mps", 0.0) / 2.0
    statuses = ["UNKNOWN", "NO_ALERT", "SUSPECTED", "CONFIRMED_OBSTRUCTION"]
    if view.diagnostic and view.diagnostic.get("status") in statuses:
        ego[10 + statuses.index(view.diagnostic["status"])] = 1.0
    ranges = np.asarray(view.scan["ranges"])
    for i, chunk in enumerate(np.array_split(ranges, 6)):
        ego[14 + i] = float(chunk.min()) if len(chunk) else 0.0
    nodes.append(ego)
    ego[20] = float(response)
    messages = [
        m
        for m in view.messages
        if m["sent_s"] <= view.at_s + 1e-7 and m["expires_s"] >= view.at_s - 1e-7
    ]
    for message in messages:
        if message["kind"] != "STATE":
            continue
        row = [0.0] * FEATURES
        row[1] = 1.0
        row[3:5] = relative(message["position"])
        payload = message["payload"]
        h = payload["heading_rad"] - angle
        row[5:9] = [
            math.sin(h),
            math.cos(h),
            payload["speed_mps"] / 2.0,
            max(0.0, view.at_s - message["sent_s"]),
        ]
        row[9:11] = [payload["length_m"] / 5.0, payload["width_m"] / 2.0]
        identities[message["sender"]] = len(nodes)
        nodes.append(row)
        links.extend([(0, len(nodes) - 1), (len(nodes) - 1, 0)])

    def resource(polygon, source, target, level, age):
        row = [0.0] * FEATURES
        row[2] = 1.0
        row[3:5] = relative([polygon.centroid.x, polygon.centroid.y])
        bounds = polygon.bounds
        row[5:9] = [
            (bounds[2] - bounds[0]) / 10.0,
            (bounds[3] - bounds[1]) / 10.0,
            level,
            age,
        ]
        index = len(nodes)
        nodes.append(row)
        for identity in [source, target]:
            if identity in identities:
                peer = identities[identity]
                links.extend([(peer, index), (index, peer)])

    for message in messages:
        if message["kind"] == "CLEAR_REQUEST":
            resource(
                wkt.loads(message["payload"]["space_wkt"]),
                message["sender"],
                message["target"],
                0.0,
                max(0.0, view.at_s - message["sent_s"]),
            )
        if message["kind"] == "INTENT":
            row = [0.0] * FEATURES
            row[2] = 1.0
            row[9] = 1.0
            row[3:5] = relative(message["position"])
            row[10:13] = [
                message["payload"]["speed_mps"] / 2.0,
                message["payload"]["steering"],
                message["payload"]["distance_m"] / 8.0,
            ]
            index = len(nodes)
            nodes.append(row)
            links.extend([(0, index), (index, 0)])
            if message["sender"] in identities:
                peer = identities[message["sender"]]
                links.extend([(peer, index), (index, peer)])
    if view.graph:
        levels = {"CANDIDATE": 0.25, "EFFECT_SUPPORTED": 0.5, "CONFIRMED": 1.0}
        for relation in view.graph["edges"]:
            if (
                relation["observed_at_s"] > view.at_s + 1e-7
                or relation["valid_until_s"] < view.at_s - 1e-7
            ):
                continue
            resource(
                wkt.loads(relation["resource_wkt"]),
                relation["requester"],
                relation["blocker"],
                levels.get(relation["level"], 0.0),
                max(0.0, view.at_s - relation["observed_at_s"]),
            )
    adjacency = np.eye(len(nodes), dtype=np.float32)
    for a, b in links:
        adjacency[a, b] = 1.0
    adjacency /= adjacency.sum(axis=1, keepdims=True)
    return np.asarray(nodes, dtype=np.float32), adjacency


class GraphActorCritic(nn.Module):
    """Deux échanges locaux ; critique local, aucun état central privilégié."""

    def __init__(self, hidden=32):
        super().__init__()
        self.input = nn.Linear(FEATURES, hidden)
        self.message = nn.Linear(hidden, hidden)
        self.actor = nn.Linear(hidden * 2, len(option_bank()))
        self.critic = nn.Linear(hidden * 2, 1)

    def forward(self, nodes, adjacency):
        hidden = torch.tanh(self.input(nodes))
        for _ in range(2):
            hidden = torch.tanh(hidden + self.message(adjacency @ hidden))
        representation = torch.cat([hidden[0], hidden.mean(dim=0)])
        return self.actor(representation), self.critic(representation).squeeze(-1)


class LearnedPolicy(CooperativePolicy):
    """Le réseau remplace entièrement le classement des options de la baseline.

    L'armement local, l'émission de demandes et la négociation sont une couche
    commune déterministe. Le choix de l'option est appris ; ni la recherche
    globale ni le classement géométrique de la baseline ne sont appelés.
    """

    def __init__(self, ego, endpoint, contract, model, record):
        super().__init__(ego, endpoint, contract)
        self.model = model
        self.record = record
        self.chosen_at = None
        self.hold_until = 0.0
        self.response_cache = {}

    def choose(self, view, mask, *, phase):
        features, adjacency = local_graph(
            view, response=phase == "cooperation_response"
        )
        with torch.no_grad():
            logits, value = self.model(
                torch.from_numpy(features), torch.from_numpy(adjacency)
            )
            distribution = torch.distributions.Categorical(
                logits=logits.masked_fill(~torch.tensor(mask), -1e9)
            )
            index = int(distribution.sample())
            logprob = float(distribution.log_prob(torch.tensor(index)))
        self.record(
            dict(
                ego=view.ego,
                at_s=view.at_s,
                features=features.tolist(),
                adjacency=adjacency.tolist(),
                mask=mask,
                action=index,
                old_logprob=logprob,
                value=float(value),
                phase=phase,
            )
        )
        self.chosen_at = view.at_s
        return index

    def candidates(self, view, guard):
        bank = option_bank()
        mask = [
            a.reason != "refuse_commitment"
            and guard.check(view, a, require_agreement=False)["accepted"]
            for a in bank
        ]
        index = self.choose(view, mask, phase="physical_option")
        if index == 0:
            self.hold_until = view.at_s + 1.0
            return []
        return [bank[index]]

    def decide(self, view, nominal, guard):
        intents = [
            m
            for m in view.messages
            if m["kind"] == "INTENT"
            and math.dist(m["position"], view.own["position"]) <= 10.0
        ]
        peers = {m["sender"] for m in intents}
        if self.attempt:
            peers.add(self.ego)
        winner = min(peers) if peers else None
        if winner is not None and winner != self.ego:
            invitation = next(m for m in intents if m["sender"] == winner)
            key = (winner, invitation["attempt"])
            cached = self.response_cache.get(key)
            if cached is None or view.at_s >= cached[0]:
                mask = [False] * len(option_bank())
                mask[0] = True
                mask[-1] = True
                index = self.choose(view, mask, phase="cooperation_response")
                self.response_cache[key] = (view.at_s + 0.6, index)
            else:
                index = cached[1]
            if index == len(option_bank()) - 1:
                self.outbox = []
                if self.attempt:
                    self.emit("CANCEL", view, {"reason": "neural_response_priority"})
                    self.attempt = None
                self.emit(
                    "REFUSE",
                    view,
                    {"reason": "neural_commitment_declined"},
                    target=winner,
                    attempt=invitation["attempt"],
                )
                return Action(0.0, reason="graph_policy_refuse_hold")
        action = super().decide(view, nominal, guard)
        if (
            self.attempt is None
            and view.at_s < self.hold_until
            and action.reason == "own_route"
        ):
            return Action(0.0, reason="graph_policy_hold")
        if action.reason == "cooperative_release":
            return Action(
                action.speed_mps, action.steering, "graph_policy_option", action.attempt
            )
        return action
