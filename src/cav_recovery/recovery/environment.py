"""Environnement d'intervention natif, sans écriture dans les campagnes M02."""

from contextlib import ExitStack
from dataclasses import asdict
import gzip
import math
from pathlib import Path
import time
import shutil

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from ..dataset.native import environment, introduce, vehicle_config, Route
from ..dataset.contract import Contract
from ..dataset.qualify import qualify
from ..laboratory.storage import (
    atomic_json,
    file_hash,
    protect_frozen_dataset,
    single_writer,
)
from .actuation import command as apply_recovery_command
from ..metadrive.native import configuration_record, missions_for
from ..metadrive.observation import body, rectangle, signed_speed
from ..metadrive.planning import requested_path
from .contracts import Action, RecoveryContract
from .safety import LocalGuard, LocalView
from .protocol import Endpoint, deliver
from .baseline import DeterministicPolicy


def nominal_action(view, contract):
    """Suivi de mission propre, sans liste globale des véhicules."""
    own = view.own
    station = view.route.line.project(Point(own["position"]))
    lookahead = 1.5 + max(0.0, own["speed_mps"]) * 0.35
    target = view.route.position(station + lookahead)
    alpha = (
        math.atan2(target[1] - own["position"][1], target[0] - own["position"][0])
        - own["heading_rad"]
    )
    steer = (
        math.atan2(2 * own["wheelbase_m"] * math.sin(alpha), lookahead)
        / own["max_steering_rad"]
    )
    curvature = (
        abs(
            math.remainder(
                view.route.heading(station + 4) - view.route.heading(station),
                2 * math.pi,
            )
        )
        / 4
    )
    speed = min(
        contract.maximum_forward_mps,
        own["desired_speed_mps"],
        math.sqrt(1.4 / max(0.001, curvature)),
    )
    return Action(speed, float(np.clip(steer, -1, 1)), "own_route")


class RecoveryEnv:
    """Les politiques ne reçoivent que LocalView, jamais l'audit physique."""

    def __init__(self, spec, output, contract=None, *, measurement_contract=None):
        self.spec, self.output = spec, Path(output)
        protect_frozen_dataset(self.output)
        self.contract = contract or RecoveryContract()
        if (
            not spec.get("agents")
            or spec["agents"][0]["id"] != "agent0"
            or len({a["id"] for a in spec["agents"]}) != len(spec["agents"])
            or spec.get("intersection_count") != 1
            or not math.isfinite(spec["horizon_s"])
            or spec["horizon_s"] <= 0
            or abs(spec["horizon_s"] / 0.2 - round(spec["horizon_s"] / 0.2)) > 1e-7
            or type(spec["active_limit"]) is not int
            or spec["active_limit"] < 1
        ):
            raise ValueError(
                "Un carrefour, des identités uniques et un horizon multiple de 0,2 s sont requis."
            )
        self.diagnostics = None
        if measurement_contract is not None:
            from .native_diagnostics import NativeRecoveryDiagnostics

            self.diagnostics = NativeRecoveryDiagnostics(measurement_contract)
        self.measurement_contract = measurement_contract
        self.guard = LocalGuard(self.contract)
        self.native = None
        self.stack = ExitStack()
        self.routes, self.missions, self.identities, self.endpoints = {}, {}, {}, {}
        self.initial, self.previous_positions, self.reverse_used = {}, {}, {}
        self.metadata = dict(vehicles={})
        self.pending = list(spec["agents"][1:])
        self.agent_specs = {a["id"]: a for a in spec["agents"]}
        self.at_s = 0.0
        self.physics_ticks = 0
        self.audit_frames, self.errors, self.arrivals = [], [], {}
        self.introduced = {}
        self.minimum_gap = None
        self.done = False
        self.outgoing = []
        self.last_commands = {}
        self.emission_positions = {}
        self.last_native_states = {}

    def write(self, name, row):
        import json

        self.streams[name].write(
            json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
        )

    def reset(self):
        self.output.mkdir(parents=True, exist_ok=False)
        snapshot = self.output / "source-snapshot"
        snapshot.mkdir()
        for source in Path(__file__).parent.glob("*.py"):
            shutil.copy2(source, snapshot / source.name)
        atomic_json(
            self.output / "source-hashes.json",
            {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        )
        self.streams = {
            name: self.stack.enter_context(
                gzip.open(
                    self.output / (name + ".jsonl.gz"),
                    "wt",
                    encoding="utf-8",
                    compresslevel=3,
                )
            )
            for name in [
                "local",
                "commands",
                "messages",
                "physics",
                "reference",
                "events",
                "diagnostics",
                "diagnostic-inputs",
            ]
        }
        atomic_json(self.output / "spec.json", self.spec)
        atomic_json(self.output / "contract.json", self.contract.record())
        self.native = environment(self.spec)
        self.native.config["vehicle_config"]["vehicle_model"] = "cgr_recovery_v2"
        self.native.config["agent_configs"]["agent0"]["vehicle_model"] = (
            "cgr_recovery_v2"
        )
        self.native.reset(seed=self.spec["seed"])
        atomic_json(
            self.output / "configuration.json",
            configuration_record(self.native, self.spec["seed"]),
        )
        self.surface = unary_union(
            [
                Polygon(lane.polygon).buffer(0)
                for links in self.native.current_map.road_network.graph.values()
                for lanes in links.values()
                for lane in lanes
            ]
        )
        self.metadata["surface_wkt"] = self.surface.wkt
        self.register("agent0")
        self.original_physics = self.native.engine.step_physics_world

        def measured_step():
            self.original_physics()
            self.physics_ticks += 1
            self.audit_physics(round(self.physics_ticks * 0.02, 8))

        self.native.engine.step_physics_world = measured_step
        self.audit_physics(0.0)
        self.views = self.observe()
        return self.views

    def register(self, key):
        vehicle = self.native.agents[key]
        if vehicle.WIDTH < self.contract.minimum_vehicle_width_m:
            raise ValueError("Gabarit hors du domaine de couverture LiDAR validé.")
        self.routes[key] = Route(self.native, self.agent_specs[key])
        self.missions[key] = missions_for(self.native)[key]
        self.identities[key] = vehicle.id
        self.endpoints[key] = Endpoint(key, self.contract.message_lifetime_s)
        self.initial[key] = self.routes[key].line.project(Point(vehicle.position))
        self.previous_positions[key] = list(map(float, vehicle.position))
        self.reverse_used[key] = 0.0
        self.introduced[key] = self.at_s
        self.metadata["vehicles"][key] = dict(
            length_m=float(vehicle.LENGTH),
            width_m=float(vehicle.WIDTH),
            mission=asdict(self.missions[key]),
            introduced_at_s=self.at_s,
        )
        self.write(
            "events",
            dict(event="introduced", ego=key, at_s=self.at_s, occupancy_checked=True),
        )

    def admit(self):
        if not self.pending or len(self.native.agents) >= self.spec["active_limit"]:
            return
        agent = self.pending[0]
        if agent["departure_s"] > self.at_s:
            return
        lane = self.native.current_map.road_network.get_lane(
            vehicle_config(agent)["spawn_lane_index"]
        )
        footprint = rectangle(
            lane.position(agent["longitude"], 0),
            lane.heading_theta_at(agent["longitude"]),
            4.515,
            1.852,
        ).buffer(1.5)
        # L'admission relève du simulateur ; elle n'est pas transmise comme une
        # perception privilégiée ni utilisée pour choisir une intervention.
        if any(footprint.intersects(body(v)) for v in self.native.agents.values()):
            return
        introduce(self.native, agent)
        self.pending.pop(0)
        self.register(agent["id"])

    def audit_physics(self, at):
        agents = self.native.agents
        ids = sorted(agents)
        bodies = {key: body(agents[key]) for key in ids}
        states = {
            key: list(
                map(
                    float,
                    [
                        *agents[key].position,
                        agents[key].heading_theta,
                        *agents[key].velocity,
                    ],
                )
            )
            for key in ids
        }
        self.last_native_states = states
        pairs = []
        xy = np.asarray([states[k][:2] for k in ids])
        for i, left in enumerate(ids):
            if not self.surface.buffer(1e-5).covers(bodies[left]):
                self.errors.append(dict(kind="outside_road", at_s=at, ego=left))
            nearby = np.flatnonzero(np.sum((xy[i + 1 :] - xy[i]) ** 2, axis=1) < 8**2)
            for offset in nearby:
                right = ids[i + 1 + offset]
                gap = bodies[left].distance(bodies[right])
                self.minimum_gap = (
                    gap if self.minimum_gap is None else min(gap, self.minimum_gap)
                )
                contact = (
                    self.native.engine.physics_world.dynamic_world.contactTestPair(
                        agents[left].chassis.node(), agents[right].chassis.node()
                    ).getNumContacts()
                    if gap < 0.1
                    else 0
                )
                overlap = (
                    bodies[left].intersection(bodies[right]).area if gap == 0 else 0.0
                )
                if contact or overlap > 1e-8:
                    self.errors.append(
                        dict(kind="contact_or_overlap", at_s=at, pair=[left, right])
                    )
                pairs.append([left, right, gap, contact, overlap])
        self.write(
            "physics",
            dict(at_s=at, states=states, pairs=pairs, pair_center_cutoff_m=8.0),
        )

    def observe(self):
        self.admit()
        views = {}
        positions = {}
        state_messages = []
        for key, v in sorted(self.native.agents.items()):
            if (
                v.id != self.identities[key]
                or v.config["destination"] != self.missions[key].destination
                or tuple(v.navigation.checkpoints) != self.missions[key].checkpoints
            ):
                raise RuntimeError("Identité ou mission modifiée.")
            position = list(map(float, v.position))
            positions[key] = position
            own = dict(
                position=position,
                heading_rad=float(v.heading_theta),
                speed_mps=signed_speed(v),
                length_m=float(v.LENGTH),
                width_m=float(v.WIDTH),
                wheelbase_m=float(v.FRONT_WHEELBASE + v.REAR_WHEELBASE),
                max_steering_rad=math.radians(v.config["max_steering"]),
                progress_m=self.routes[key].line.project(Point(position))
                - self.initial[key],
                desired_speed_mps=self.agent_specs[key]["target_speed_mps"],
            )
            scan = self.native.engine.get_sensor("lidar").perceive(
                v,
                self.native.engine.physics_world.dynamic_world,
                num_lasers=self.contract.rays,
                distance=self.contract.range_m,
                show=False,
            )[0]
            views[key] = LocalView(
                key,
                self.at_s,
                own,
                dict(
                    at_s=self.at_s,
                    ranges=list(map(float, scan)),
                    range_m=self.contract.range_m,
                    full_circle=True,
                    missing=False,
                ),
                [],
                self.surface,
                self.routes[key],
                self.reverse_used[key],
            )
            views[key].own_commitment_until_s = self.endpoints[key].committed_until_s
            views[key].own_controls = self.last_commands.get(key)
            state_messages.append(
                self.endpoints[key].send(
                    "STATE",
                    "state",
                    position,
                    self.at_s,
                    payload={
                        k: own[k]
                        for k in [
                            "speed_mps",
                            "heading_rad",
                            "length_m",
                            "width_m",
                            "progress_m",
                        ]
                    },
                )
            )
        actual_endpoints = {key: self.endpoints[key] for key in views}
        # Les intentions ont été émises au pas précédent : cette livraison
        # n'antidate aucun accord et ajoute un pas applicatif explicite.
        messages = [
            m
            for m in self.outgoing
            if m["sender"] in positions and m["expires_s"] >= self.at_s
        ]
        # Leur enveloppe position est celle de l'émission ; le transport utilise
        # la portée à l'émission, puis exige encore la présence du destinataire.
        for message in messages:
            emitted_positions = self.emission_positions
            for event in deliver(
                [message],
                actual_endpoints,
                emitted_positions,
                self.at_s,
                self.contract.range_m,
            ):
                self.write("messages", dict(event="delivered", **event))
        for event in deliver(
            state_messages,
            actual_endpoints,
            positions,
            self.at_s,
            self.contract.range_m,
        ):
            self.write("messages", dict(event="delivered", **event))
        self.outgoing = []
        for key, view in views.items():
            view.messages = self.endpoints[key].view(self.at_s)
            self.write(
                "local",
                dict(
                    ego=key,
                    at_s=self.at_s,
                    own=view.own,
                    scan=view.scan,
                    messages=view.messages,
                ),
            )
        self.audit_reference(views)
        self.views = views
        if self.diagnostics is not None and views:
            packets, outputs = self.diagnostics.observe(self)
            for key in views:
                self.write("diagnostic-inputs", packets[key])
                self.write("diagnostics", dict(ego=key, at_s=self.at_s, **outputs[key]))
        return views

    def audit_reference(self, views):
        """Vérité physique hors acteur, jamais passée à la baseline ou au garde."""
        bodies = {key: body(v) for key, v in self.native.agents.items()}
        witnesses = []
        states = {}
        for key, view in views.items():
            own = view.own
            station = self.routes[key].line.project(Point(own["position"]))
            states[key] = dict(
                progress_m=own["progress_m"],
                speed_mps=own["speed_mps"],
                movement_id=self.agent_specs[key]["movement_id"],
                resource_id=self.routes[key].resource(station),
            )
            poses = requested_path(
                self.routes[key],
                own["position"],
                own["heading_rad"],
                max(0.0, own["speed_mps"]),
                own["wheelbase_m"],
                own["max_steering_rad"],
                1.2 * own["length_m"],
            )
            sweep = unary_union(
                [rectangle(p[:2], p[2], own["length_m"], own["width_m"]) for p in poses]
            )
            required = sweep.difference(bodies[key])
            for other, polygon in bodies.items():
                if (
                    other == key
                    or math.dist(own["position"], views[other].own["position"]) > 12
                ):
                    continue
                intersection = required.intersection(polygon)
                if intersection.area > 1e-4:
                    witnesses.append(
                        dict(
                            requester=key,
                            blocker=other,
                            c1_expected=True,
                            road_admissible=self.surface.buffer(1e-5).covers(sweep),
                            body_intersection=True,
                            resource=dict(
                                id=states[key]["resource_id"],
                                polygon_wkt=intersection.wkt,
                            ),
                            movement=dict(id=self.agent_specs[key]["movement_id"]),
                            mechanism="independent_own_requested_path",
                        )
                    )
        frame = dict(time_s=self.at_s, states=states, witnesses=witnesses)
        self.audit_frames.append(frame)
        self.write("reference", frame)

    def step(self, actions, outgoing=()):
        if set(actions) != set(self.views):
            raise ValueError("Une commande propre est requise pour chaque CAV présent.")
        native_actions = {}
        for key, action in sorted(actions.items()):
            self.views[key].own_commitment_until_s = self.endpoints[
                key
            ].committed_until_s
            accepted, receipt = self.guard.limit(self.views[key], action)
            native_actions[key] = apply_recovery_command(
                self.native.agents[key],
                accepted,
                self.contract,
            )
            self.last_commands[key] = dict(
                proposed=asdict(action),
                applied=asdict(accepted),
                guard=receipt,
                native_action=native_actions[key],
                at_s=self.at_s,
            )
            self.write("commands", dict(ego=key, **self.last_commands[key]))
        self.outgoing = list(outgoing)
        self.emission_positions = {
            key: view.own["position"] for key, view in self.views.items()
        }
        for endpoint in self.endpoints.values():
            for event in endpoint.events:
                self.write("messages", event)
            endpoint.events.clear()
        _, _, terminated, truncated, infos = self.native.step(native_actions)
        self.at_s = round(self.at_s + self.contract.step_s, 8)
        for key, v in self.native.agents.items():
            previous = self.previous_positions[key]
            dx = np.asarray(v.position) - previous
            heading = self.views[key].own["heading_rad"]
            self.reverse_used[key] += max(
                0.0, -float(dx @ np.array([math.cos(heading), math.sin(heading)]))
            )
            self.previous_positions[key] = list(map(float, v.position))
        for key in actions:
            if not terminated.get(key) and not truncated.get(key):
                continue
            info = infos.get(key, {})
            pose = self.last_native_states.get(key)
            start, end = self.missions[key].final_lane[:2]
            destination_lanes = self.native.current_map.road_network.graph[start][end]
            corridor = False
            for lane in destination_lanes:
                longitudinal, lateral = (
                    lane.local_coordinates(pose[:2]) if pose else (-math.inf, math.inf)
                )
                corridor = corridor or (
                    abs(lateral) <= lane.width / 2 + 0.05
                    and longitudinal >= lane.length - 5.1
                )
            flags = {
                k: bool(info.get(k))
                for k in [
                    "arrive_dest",
                    "crash_vehicle",
                    "out_of_road",
                    "crash_sidewalk",
                    "crash_object",
                ]
            }
            if (
                flags["arrive_dest"]
                and corridor
                and not any(flags[k] for k in flags if k != "arrive_dest")
            ):
                self.arrivals[key] = self.at_s
                self.write(
                    "events",
                    dict(
                        event="native_arrival",
                        ego=key,
                        at_s=self.at_s,
                        flags=flags,
                        corridor_verified=corridor,
                    ),
                )
            else:
                self.errors.append(
                    dict(
                        kind="native_termination",
                        ego=key,
                        at_s=self.at_s,
                        flags=flags,
                        corridor_verified=corridor,
                    )
                )
        self.done = (
            bool(self.errors)
            or (not self.native.agents and not self.pending)
            or self.at_s >= self.spec["horizon_s"]
        )
        self.views = self.observe() if not self.errors else {}
        return self.views, self.done

    def close(self):
        if self.native is not None:
            if hasattr(self, "original_physics"):
                self.native.engine.step_physics_world = self.original_physics
            self.native.close()
        self.stack.close()


def run(
    spec,
    output,
    *,
    method="nominal",
    contract=None,
    measurement_contract=None,
    policy_factory=None,
    instrumentation_hook=None,
    policy_version=None,
):
    if method not in {
        "nominal",
        "deterministic",
        "cooperative",
        "global_search",
        "graph_policy",
    }:
        raise ValueError("Méthode inconnue.")
    if (method == "graph_policy") != (policy_factory is not None):
        raise ValueError(
            "La méthode apprise exige une fabrique d'acteurs locale explicite."
        )
    output = Path(output)
    protect_frozen_dataset(output)
    started, cpu = time.perf_counter(), time.process_time()
    instance = RecoveryEnv(
        spec, output, contract, measurement_contract=measurement_contract
    )
    policies = {}
    coordinator = None
    if method == "global_search":
        from .feasibility import GlobalCoordinator, GlobalBodyGuard

        coordinator = GlobalCoordinator()
        instance.guard = GlobalBodyGuard(instance)
    with single_writer(output.parent / (output.name + "-lock")):
        try:
            instance.views = instance.reset()
            while not instance.done:
                actions, outgoing = {}, []
                for ego, view in instance.views.items():
                    if ego not in policies:
                        if policy_factory is not None:
                            policies[ego] = policy_factory(
                                ego, instance.endpoints[ego], instance.contract
                            )
                        elif method == "cooperative":
                            from .cooperative import CooperativePolicy

                            policies[ego] = CooperativePolicy(
                                ego, instance.endpoints[ego], instance.contract
                            )
                        else:
                            policies[ego] = DeterministicPolicy(
                                ego,
                                instance.endpoints[ego],
                                instance.contract,
                                enabled=method == "deterministic",
                            )
                    actions[ego] = policies[ego].decide(
                        view, nominal_action(view, instance.contract), instance.guard
                    )
                    outgoing.extend(policies[ego].outbox)
                if coordinator is not None:
                    actions, outgoing = coordinator.decide(instance, actions)
                instance.step(actions, outgoing)
                if instrumentation_hook is not None:
                    # Ce consommateur produit les récompenses/traces après
                    # action ; son résultat n'entre jamais dans LocalView.
                    instrumentation_hook(instance)
            integrity = dict(
                errors=instance.errors,
                complete_measurements=True,
                arrivals=instance.arrivals,
                introduced=instance.introduced,
                not_introduced=[a["id"] for a in instance.pending],
                all_arrived=len(instance.arrivals) == len(spec["agents"]),
            )
            annotation = qualify(
                instance.audit_frames, instance.metadata, integrity, Contract()
            )
            atomic_json(output / "physical-annotation.json", annotation)
            atomic_json(output / "metadata.json", instance.metadata)
            summary = dict(
                method=method,
                seed=spec["seed"],
                scenario=spec.get("experiment_name", spec.get("family")),
                at_s=instance.at_s,
                errors=instance.errors,
                arrivals=instance.arrivals,
                active=sorted(instance.native.agents),
                pending=len(instance.pending),
                introduced=len(instance.introduced),
                minimum_gap_m=instance.minimum_gap,
                peak_active=max(len(f["states"]) for f in instance.audit_frames),
                reverse_m=instance.reverse_used,
                physical_label=annotation["label"],
                wall_s=time.perf_counter() - started,
                cpu_s=time.process_time() - cpu,
                decision_scope="local_state_rays_received_messages_only",
                global_audit_in_actor=False,
                activation="own_stall_plus_blocked_nominal_local_forward_or_agreed_reverse_not_K04_confirmation",
                trained_policy=bool(
                    policy_version and policy_version.get("updates", 0) > 0
                ),
            )
            summary["neural_policy_interface"] = policy_factory is not None
            summary["policy_version"] = policy_version
            summary["passive_diagnostic_measurement_contract"] = measurement_contract
            summary["global_audit_in_actor"] = coordinator is not None
            summary["decision_scope"] = (
                "privileged_global_feasibility_only"
                if coordinator
                else summary["decision_scope"]
            )
            if coordinator:
                atomic_json(output / "global-search.json", coordinator.log)
            atomic_json(output / "summary.json", summary)
        except Exception as error:
            if output.exists():
                atomic_json(
                    output / "failure.json",
                    dict(
                        error_type=type(error).__name__,
                        message=str(error),
                        at_s=instance.at_s,
                        errors=instance.errors,
                        results_complete=False,
                    ),
                )
            raise
        finally:
            instance.close()
    atomic_json(
        output / "manifest.json",
        dict(
            files={
                p.name: file_hash(p) for p in sorted(output.glob("*")) if p.is_file()
            }
        ),
    )
    return summary
