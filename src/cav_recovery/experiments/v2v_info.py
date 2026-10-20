"""Lecture des échanges V2V ; aucune donnée affichée ne revient vers les CAV."""

import json
import math

from .presentation import FRENCH, french, label


def correspondents(snapshot, ego):
    """Regroupe les livraisons par correspondant, sans dérouler les messages reçus."""
    peers = {}
    for source, middle, target in snapshot["units"]:
        message = snapshot["nodes"][middle]
        sender = snapshot["nodes"][source]["vehicle_id"]
        recipient = snapshot["nodes"][target]["vehicle_id"]
        if ego not in (sender, recipient):
            continue
        outgoing = sender == ego
        peer = recipient if outgoing else sender
        row = peers.setdefault(
            peer,
            dict(
                vehicle_id=peer,
                sent=0,
                received=0,
                latest_out={},
                channels=set(),
                last_at_s=0,
            ),
        )
        row["sent" if outgoing else "received"] += 1
        row["channels"].add(message["channel"])
        row["last_at_s"] = max(row["last_at_s"], message["received_at_s"])
        if outgoing:
            previous = row["latest_out"].get(message["channel"])
            if (
                previous is None
                or snapshot["nodes"][previous]["sent_at_s"] <= message["sent_at_s"]
            ):
                row["latest_out"][message["channel"]] = middle
    return sorted(
        peers.values(),
        key=lambda row: (len(label(row["vehicle_id"])), label(row["vehicle_id"])),
    )


def own_lines(frame, ego):
    """Les mesures propres sont distinctes de ce que contient un message choisi."""
    local = frame.get("measurements", {}).get(ego, {})
    own = local.get("ego", {})
    diagnostic = frame.get("diagnostics", {}).get(ego, {})
    lines = [("ÉTAT PROPRE DE " + label(ego or "—"), True)]
    position = own.get("position_xy_m")
    if position is not None:
        lines.append(
            (
                f"Position : ({position[0]:.1f} ; {position[1]:.1f}) m · vitesse : {own['signed_speed_mps']:.2f} m/s",
                False,
            )
        )
    if diagnostic:
        destination = diagnostic.get("destination", "non renseignée")
        if destination.startswith("a") and destination.endswith("_sink"):
            destination = "sortie " + destination[1:-5]
        lines.append(
            ("Mission : " + destination + " · " + french(diagnostic["status"]), False)
        )
    return lines


FIELD_NAMES = {
    "protocol_version": "Version du protocole",
    "message_id": "Identifiant du message",
    "sender_id": "Émetteur",
    "sequence_number": "Numéro de séquence",
    "observed_at_s": "Mesure à (s)",
    "sent_at_s": "Émission à (s)",
    "valid_until_s": "Valide jusqu'à (s)",
    "position_xy_m": "Position (m)",
    "heading_rad": "Orientation (rad)",
    "signed_speed_mps": "Vitesse signée (m/s)",
    "length_m": "Longueur (m)",
    "width_m": "Largeur (m)",
    "intended_destination": "Destination annoncée",
    "progress_delta_m": "Distance parcourue (m)",
    "progress_kind": "Nature de cette distance",
    "progress_window_s": "Fenêtre de mesure (s)",
    "state_schema_version": "Format de l'état",
    "coordinate_frame": "Repère des coordonnées",
    "source": "Origine des données",
    "confidence": "Confiance déclarée",
    "report": "DIAGNOSTIC PARTAGÉ",
    "schema_version": "Format du diagnostic",
    "ego_id": "Véhicule déclarant",
    "movement_id": "Mouvement demandé",
    "destination": "Destination",
    "status": "Statut déclaré",
    "reason": "Motif déclaré",
    "evidence_through_s": "Preuves disponibles jusqu'à (s)",
    "emitted_at_s": "Rapport émis à (s)",
    "temporal_scope": "Cadence des preuves",
    "own_effect": "EFFET PROPRE RAPPORTÉ",
    "persistence": "PERSISTANCE RAPPORTÉE",
    "relations": "RELATIONS RAPPORTÉES",
    "anonymous_obstacles": "OBSTACLES SANS IDENTITÉ",
    "start_s": "Début (s)",
    "end_s": "Fin (s)",
    "samples": "Mesures",
    "progress_m": "Progression (m)",
    "low_speed_fraction": "Fraction à faible vitesse",
    "qualifies": "Effet qualifié",
    "proof_appointments": "Rendez-vous étayés",
    "own_effect_windows": "Fenêtres d'effet",
    "first_effect_end_s": "Fin du premier effet (s)",
    "last_effect_end_s": "Fin du dernier effet (s)",
    "all_effect_windows_qualify": "Toutes les fenêtres sont qualifiantes",
    "requester": "Demandeur",
    "blocker": "Occupant",
    "level": "Niveau déclaré",
    "resource_wkt": "Empreinte de l'espace requis",
    "error_bound_m": "Borne d'erreur (m)",
    "effect_qualifies": "Effet qualifié",
    "ray": "Rayon",
    "category": "Catégorie",
    "identity": "Identité",
}
VALUE_NAMES = {
    "metadrive_local_xy_m": "Repère local MetaDrive en mètres",
    "unsigned_distance_not_mission_progress": "Distance non signée ; ce n'est pas l'avancement de mission",
    "metadrive_ego_emulation_not_radio": "Mesures propres simulées ; pas de modèle radio",
    "simulated_exact_ego": "État propre simulé exact",
    "sampled_0.4s": "Échantillonnée toutes les 0,4 s",
    "CANDIDATE": "Candidate",
    "EFFECT_SUPPORTED": "Effet étayé",
    "CONFIRMED": "Confirmée",
}


def readable_fields(payload):
    """Présente le paquet reçu ; le bouton Brut conserve l'accès à tous les détails."""
    priority = (
        "position_xy_m",
        "signed_speed_mps",
        "heading_rad",
        "length_m",
        "width_m",
        "intended_destination",
        "progress_delta_m",
        "progress_kind",
        "progress_window_s",
        "report",
        "status",
        "reason",
        "destination",
        "movement_id",
        "own_effect",
        "persistence",
        "relations",
    )
    keys = [key for key in priority if key in payload] + [
        key for key in payload if key not in priority
    ]
    for key in keys:
        value = payload[key]
        title = FIELD_NAMES.get(key, key)
        if isinstance(value, dict):
            yield title, True
            yield from readable_fields(value)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            yield f"{title} : {len(value)}", True
            for index, item in enumerate(value, 1):
                yield f"Élément {index}", True
                yield from readable_fields(item)
        else:
            if value is None:
                text = "non renseigné dans le paquet"
            elif isinstance(value, bool):
                text = "oui" if value else "non"
            elif key == "resource_wkt":
                text = "géométrie transmise, consultable avec J · Brut"
            elif isinstance(value, list):
                text = (
                    "aucun"
                    if not value
                    else "; ".join(
                        format(v, ".6g") if isinstance(v, float) else str(v)
                        for v in value
                    )
                )
            elif isinstance(value, str):
                text = FRENCH.get(value, VALUE_NAMES.get(value, value))
                if key in ("sender_id", "ego_id", "requester", "blocker"):
                    text = label(value)
                if (
                    key in ("destination", "intended_destination")
                    and value.startswith("a")
                    and value.endswith("_sink")
                ):
                    text = "Sortie " + value[1:-5]
            elif key == "heading_rad":
                title = "Orientation (°)"
                text = f"{math.degrees(value):.1f}"
            elif isinstance(value, float):
                text = format(value, ".6g")
            else:
                text = str(value)
            yield title + " : " + text, False


def message_lines(message, ego, *, raw=False):
    """Aucune valeur n'est complétée à partir de l'état courant ou de l'oracle."""
    direction = "ÉMIS" if message["sender"] == ego else "REÇU"
    yield f"MESSAGE {direction} · {message['label']}", True
    yield f"{label(message['sender'])} · émission à {message['sent_at_s']:.1f} s", False
    yield (
        "Canal : "
        + ("diagnostic local" if message["channel"] == "diagnostic" else "télémétrie"),
        False,
    )
    yield (
        f"{len(message['recipients'])} destinataires livrés (instrumentation) ; liste en fin de détail.",
        False,
    )
    yield (
        "Paquet sélectionné conservé pour lecture. L · Retour ; molette pour défiler.",
        False,
    )
    payload = message.get("payload")
    if payload is None:
        yield "Contenu absent de cette trace ; aucune donnée n'est reconstruite.", False
    elif raw:
        yield "PAQUET BRUT EXACT", True
        for line in json.dumps(payload, ensure_ascii=False, indent=2).splitlines():
            yield line, False
    else:
        yield "CONTENU EFFECTIVEMENT PARTAGÉ", True
        yield from readable_fields(payload)
        if "report" in payload:
            yield (
                "Ce diagnostic est celui de l'émetteur ; ce n'est pas une annotation de l'oracle.",
                False,
            )
    yield "DESTINATAIRES LIVRÉS — INSTRUMENTATION", True
    yield (
        ", ".join(
            sorted(
                (label(key) for key in message["recipients"]),
                key=lambda text: (len(text), text),
            )
        ),
        False,
    )


def draw_correspondents(view):
    """La vue principale résume les correspondants et les derniers messages émis."""
    from .diagrams import heading, text_lines

    snapshot = view._v2v_snapshot()
    view.info_snapshot = snapshot
    rows = correspondents(snapshot, view.selected)
    scope = (
        "tous les échanges de l'ego"
        if view.v2v_mode == 2
        else "diagnostics reçus utiles"
    )
    plot = heading(
        view,
        "CORRESPONDANTS V2V · " + label(view.selected or "—"),
        "Livraisons des 0,6 dernières secondes · "
        + scope
        + ". Réceptions regroupées par CAV.",
    )
    y = plot.y
    for text, title in own_lines(view.frame, view.selected):
        y = text_lines(view, text, (plot.x, y), plot.width, small=not title)
    y = text_lines(
        view,
        "Mesures propres et paquet partagé sont distincts.",
        (plot.x, y),
        plot.width,
        small=True,
    )
    y = (
        text_lines(
            view,
            "Env. / reç. : livraisons par correspondant. Rectangles M : derniers paquets émis ; T = télémétrie, D = diagnostic. Clic pour lire.",
            (plot.x, y + 8),
            plot.width,
            small=True,
        )
        + 10
    )
    capacity = max(1, (plot.bottom - y - 34) // 40)
    pages = max(1, math.ceil(len(rows) / capacity))
    view.diagram_page %= pages
    view._button("diagram_prev", "<", (plot.right - 72, y, 30, 24))
    view._button("diagram_next", ">", (plot.right - 34, y, 30, 24))
    text_lines(
        view,
        f"{len(rows)} correspondants · page {view.diagram_page + 1}/{pages}",
        (plot.x, y),
        plot.width - 80,
        small=True,
    )
    y += 32
    view.graph_nodes = {}
    for row in rows[view.diagram_page * capacity : (view.diagram_page + 1) * capacity]:
        text_lines(
            view,
            f"{label(row['vehicle_id'])} · {row['sent']} env. / {row['received']} reç.",
            (plot.x, y + 6),
            max(80, plot.width - 146),
            small=True,
        )
        for index, channel in enumerate(("telemetry", "diagnostic")):
            middle = row["latest_out"].get(channel)
            if middle is None:
                continue
            node = snapshot["nodes"][middle]
            key = "packet:" + middle + ":" + row["vehicle_id"]
            view._button(
                key,
                node["label"] + (" T" if index == 0 else " D"),
                (plot.right - 140 + index * 72, y, 66, 28),
            )
            view.message_buttons[key] = middle
        y += 40
    if not rows:
        text_lines(
            view,
            "Aucune livraison récente liée à ce CAV.",
            (plot.x, y),
            plot.width,
            small=True,
        )
