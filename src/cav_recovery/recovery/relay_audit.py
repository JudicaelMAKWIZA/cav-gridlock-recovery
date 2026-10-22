"""Contrefactuels de transport des rapports existants, sans preuve oracle ajoutée."""

from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
import math

from ..communication.reports import make_report
from ..crdg.local import LocalCRDG
from ..experiments.evaluation import active_events
from ..laboratory.storage import json_file
from .evaluation import rows


def replay_report_transport(folder):
    """Deux sauts à délai supplémentaire de 0,2 s et agrégation idéale séparée.

    L'origine, la date et l'expiration à 1,2 s sont conservées. Les coordonnées
    servent exclusivement au modèle de portée radio. Le schéma opérationnel
    ne prend pas encore en charge ces relais : c'est une expérience hors ligne.
    """
    folder = Path(folder)
    positions = {
        (r["ego"], r["at_s"]): r["own"]["position"]
        for r in rows(folder / "local.jsonl.gz")
    }
    eligible = {
        (r["ego"], r["at_s"]) for r in rows(folder / "causal-observations.jsonl.gz")
    }
    annotation = json_file(folder / "physical-annotation.json")
    modes = ("one_hop", "two_hop", "all_reports_ideal")
    mailboxes = {m: defaultdict(dict) for m in modes}
    graphs = {m: {} for m in modes}
    counts = {m: Counter() for m in modes}
    pending = defaultdict(list)
    trace = rows(folder / "diagnostics.jsonl.gz")
    for at, group in groupby(trace, key=lambda r: r["at_s"]):
        own = {r["ego"]: r["diagnostic"] for r in group if (r["ego"], at) in eligible}
        if not own:
            continue
        participants = sorted(own)

        def deliver(mode, recipient, message):
            sender = message["sender_id"]
            old = mailboxes[mode][recipient].get(sender)
            if old and old["message"]["sequence_number"] >= message["sequence_number"]:
                return
            if at <= message["valid_until_s"] + 1e-7:
                mailboxes[mode][recipient][sender] = dict(
                    received_at_s=at, message=message
                )
                counts[mode]["deliveries"] += 1

        for relay, message in pending.pop(round(at, 6), []):
            if relay not in own:
                continue
            counts["two_hop"]["relay_broadcasts"] += 1
            for recipient in participants:
                if (
                    recipient not in {relay, message["sender_id"]}
                    and math.dist(positions[relay, at], positions[recipient, at]) <= 30
                ):
                    deliver("two_hop", recipient, message)
        if at > 0 and abs(at / 0.4 - round(at / 0.4)) < 1e-7:
            for sender in participants:
                message = dict(
                    protocol_version=1,
                    message_id=f"{sender}:{round(at / 0.4)}",
                    sender_id=sender,
                    sequence_number=round(at / 0.4),
                    observed_at_s=at,
                    sent_at_s=at,
                    valid_until_s=at + 1.2,
                    position_xy_m=positions[sender, at],
                    report=make_report(own[sender]),
                )
                neighbors = [
                    p
                    for p in participants
                    if p != sender
                    and math.dist(positions[sender, at], positions[p, at]) <= 30
                ]
                for mode in modes:
                    counts[mode]["origin_broadcasts"] += 1
                    for recipient in (
                        participants if mode == "all_reports_ideal" else neighbors
                    ):
                        if recipient != sender:
                            deliver(mode, recipient, message)
                for relay in neighbors:
                    pending[round(at + 0.2, 6)].append((relay, message))
        truth = {
            (event["requester"], interval["blocker"])
            for event in active_events(annotation, at)
            for interval in event["occupant_intervals"]
            if interval["start_s"] - 1e-7 <= at <= interval["end_s"] + 1e-7
        }
        for mode in modes:
            for ego in participants:
                graph = graphs[mode].setdefault(ego, LocalCRDG(ego))
                mailbox = {
                    s: entry
                    for s, entry in mailboxes[mode][ego].items()
                    if at <= entry["message"]["valid_until_s"] + 1e-7
                }
                result = graph.update(own[ego], dict(received=mailbox), at)
                valid = {
                    (e["requester"], e["blocker"])
                    for e in result["edges"]
                    if e["level"] == "CONFIRMED"
                }
                current = {
                    (e["requester"], e["blocker"])
                    for e in result["edges"]
                    if e["level"] == "CONFIRMED" and e["state"] == "CURRENT"
                }
                counts[mode].update(
                    appointments=1,
                    valid_tp=len(valid & truth),
                    valid_fp=len(valid - truth),
                    valid_fn=len(truth - valid),
                    current_tp=len(current & truth),
                    current_fn=len(truth - current),
                    nonempty_truth_appointments=int(bool(truth)),
                    exact_valid_pair_set=int(bool(truth) and valid == truth),
                )
    return dict(
        variants={m: dict(c) for m, c in counts.items()},
        physics_rerun=False,
        diagnostic_outputs_changed=False,
        oracle_dependency_input=False,
        additional_hop_delay_s=0.2,
        original_ttl_s=1.2,
        origin_identity_and_timestamps_preserved=True,
        assumptions="Deux sauts supposent transport et authentification de l'origine étendus ; "
        "all_reports_ideal suppose une agrégation sans perte ni délai ajouté.",
        scope="reconstruction_only_no_closed_loop_recovery_gain_claimed",
    )
