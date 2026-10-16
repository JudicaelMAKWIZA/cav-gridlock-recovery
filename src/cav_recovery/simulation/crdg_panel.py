"""Fenêtre compagnon INFO C-RDG ; ne possède aucune connexion TraCI."""

import json
import math
from pathlib import Path
import sys
import time

import networkx as nx
from .crdg_live import atomic_json, edge_key, RELATIONS, SELECTED, CONSTRAINED, PARTICIPANT

FILTERS = {"Tous les groupes": "all", "Suivi longitudinal": "following",
           "Réception / espace aval": "receiving", "Passage au carrefour": "junction", "Cycles candidats": "cycle"}

DETECTION_STATES = {"normal_wait": "Attente au feu", "observed_constraint": "Contrainte locale observée",
                    "persistent_obstruction": "Contrainte locale prolongée", "persistent_immobility": "Faible progression prolongée",
                    "stopped": "Arrêt observé", "advancing": "Progression observée",
                    "partial_recovery": "Progrès antérieurs", "resolution_pending": "Dégagement à vérifier",
                    "resolved": "Passage dégagé", "resolved_by_complete_drain": "Toutes les missions arrivées",
                    "indeterminate": "Passage à vérifier", "interrupted": "Suivi interrompu"}
DETECTION_ACTORS = {"Gênés": "identified_members", "Arrêtés": "current_halted_members", "Ont avancé": "advanced_members",
                    "Encore ici": "current_present_members", "Tous": "members"}


def detection_place(junction):
    return {"magasin_nguma": "Magasin Nguma", "magasin_oua": "Magasin OUA",
            "magasin_west": "Magasin Ouest", "magasin_east": "Magasin Est"}.get(junction, f"Passage {junction}")


def detection_actors(row, choice):
    """Les auteurs d'un progrès passé ne sont pas forcément encore dans le passage."""
    if choice == "Gênés" and "identified_members" not in row:
        return sorted({edge["source_vehicle_id"] for edge in row.get("evidence", {}).get("qualified_dependencies", [])})
    return sorted(row.get(DETECTION_ACTORS[choice], []))


def detection_cause(row):
    if row.get("state") == "normal_wait":
        return "Feu interdit le passage"
    halted = row.get("current_halted_members", [])
    if (halted and len(row.get("signal_wait_members", [])) == len(halted)
            and row.get("cause_status") != "qualified"):
        return "Feu rouge maintenant ; longue attente à expliquer"
    if row.get("cause_status") == "occupation_only":
        return "Passage occupé ; gêne non confirmée"
    if row.get("cause_status") != "qualified":
        return "Cause inconnue"
    evidence = row.get("evidence", {})
    resources = {resource["id"]: resource["resource_type"]
                 for proof in evidence.get("vehicles", {}).values() for resource in proof.get("resources", [])}
    causes = set()
    for edge in evidence.get("qualified_dependencies", []):
        if edge["edge_type"] == "connection_obstacle":
            causes.add("Véhicule gênant le passage")
        elif resources.get(edge["target"]) == "receiving_space":
            causes.add("Voie suivante directe encombrée")
        elif resources.get(edge["target"]) == "junction_conflict":
            causes.add("Véhicule gênant le passage")
        else:
            causes.add("Contrainte observée ; cause à préciser")
    return " ; ".join(sorted(causes)) or "Cause inconnue"


def detection_ages(row):
    return [proof["low_progress_age_s"] for proof in row.get("evidence", {}).get("vehicles", {}).values()
            if proof.get("low_progress_age_s") is not None]


def detection_priority(row, active):
    """Montre d'abord les contraintes établies, puis les arrêts à vérifier."""
    state = row.get("state")
    level = (0 if row.get("obstruction_verdict") == "verified" else
             1 if state == "persistent_immobility" else
             2 if row.get("obstruction_verdict") == "candidate" else
             4 if state in ("normal_wait", "advancing", "resolution_pending", "resolved") else 3)
    ages = detection_ages(row)
    return (row["episode_id"] not in active, level, -max(ages, default=0), row["episode_id"])


def detection_text(row, aliases, names, *, terminal_status=None):
    """Garde l'état actuel court ; les événements passés sont dans les détails."""
    lines = [f"{detection_place(row.get('junction', '?'))} — {DETECTION_STATES.get(row.get('state'), 'État inconnu')}"]
    halted = len(row.get("current_halted_members", []))
    if halted:
        age = detection_ages(row)
        duration = f" · peu de progrès depuis {min(age):g}–{max(age):g} s" if age else " · durée non mesurée"
        lines.append(f"{halted} véhicule{'s' if halted > 1 else ''} arrêté{'s' if halted > 1 else ''}{duration}")
        prolonged = len(row.get("persistent_members", []))
        if prolonged:
            lines[-1] += f" · {prolonged} arrêt{'s' if prolonged > 1 else ''} prolongé{'s' if prolonged > 1 else ''}"
        cause = detection_cause(row)
        sources = detection_actors(row, "Gênés")
        if sources:
            cause += f" · établi pour {len(sources)} véhicule{'s' if len(sources) > 1 else ''}"
        lines.append(cause + ".")
        if "Voie suivante directe encombrée" in cause:
            lines.append("Évitement par une autre voie non évalué ici.")
        interval = row.get("progress_interval_s")
        if interval and interval[0] is not None:
            advancing = len(row.get("advancing_members", []))
            local = row.get("current_present_members", [])
            complete = bool(local) and all(row.get("member_observations", {}).get(item, {}).get("step_progress_m") is not None for item in local)
            if advancing:
                lines.append(f"{advancing} ont bougé entre {interval[0]:g} et {interval[1]:g} s.")
            elif complete:
                lines.append(f"Aucun déplacement entre {interval[0]:g} et {interval[1]:g} s.")
    elif row.get("state") == "resolution_pending":
        lines.append("Tous les membres ont quitté ce passage ; confirmation en attente.")
    elif row.get("physical_resolution") == "verified":
        lines.append("Résultat de ce passage, pas de tout le réseau.")
    elif row.get("advancing_members"):
        lines.append(f"{len(row['advancing_members'])} véhicule(s) ont avancé entre les deux dernières mesures.")
    else:
        lines.append("Le dégagement du passage reste à vérifier.")
    if row.get("right_censored") or (terminal_status == "horizon_reached"
            and row.get("end_s") is None and row.get("physical_resolution") != "verified"):
        lines.append("Simulation terminée ; issue inconnue.")
    elif terminal_status == "completed" and row.get("physical_resolution") != "verified":
        lines.append("Tous arrivés ; dégagement à confirmer après fermeture.")
    return "\n".join(lines)


def detection_details(row, aliases, names, *, choice="Tous", items=None):
    """Expose les dates individuelles sans leur prêter une reprise collective."""
    evidence = row.get("evidence", {})
    lines = []
    interval = row.get("progress_interval_s")
    items = detection_actors(row, choice) if items is None else items
    lines.append(f"{choice} : {len(items)} membres connus ; carte limitée aux présents.")
    for item in items:
        detail = row.get("member_observations", {}).get(item, {})
        parts = [aliases.get(item, item)]
        since = detail.get("measurement_since_s")
        if since is not None:
            parts.append(f"mesuré depuis {since:g} s")
        when, gain = detail.get("first_progress_s"), detail.get("first_progress_m")
        if when is not None and gain is not None:
            parts.append(f"a parcouru {gain:.3f} m à {when:g} s")
        elif item in row.get("advanced_members", []):
            parts.append("progrès ancien ; date non disponible dans ces données")
        total = row.get("progress_m", {}).get(item)
        if total is not None:
            parts.append(f"gain total {total:.3f} m")
        delta = detail.get("step_progress_m")
        if delta is not None and interval and interval[0] is not None:
            parts.append(f"{delta:g} m sur {interval[0]:g}–{interval[1]:g} s")
        arrived = row.get("arrivals_s", {}).get(item)
        if arrived is not None:
            parts.append(f"arrivé à destination à {arrived:g} s")
        if item in row.get("current_halted_members", []):
            parts.append("arrêté à la dernière observation")
        location = detail.get("vehicle") or evidence.get("vehicles", {}).get(item, {}).get("vehicle", {})
        if location.get("lane"):
            position = location.get("lane_position_m")
            when = detail.get("last_observed_s", row["last_seen_s"])
            parts.append("voie " + location["lane"] + (f", {position:.3f} m" if position is not None else "") + f" à {when:g} s")
        lines.append(" · ".join(parts))
    lines.extend((f"Suivi du passage : {row['first_seen_s']:g}–{row['last_seen_s']:g} s.",
                  "Chaque membre est mesuré depuis son entrée ou son retour.",
                  "Un progrès est enregistré quand une longueur de voiture a été parcourue."))
    if interval and interval[0] is not None:
        lines.append(f"Dernier intervalle de mesure : {interval[0]:g}–{interval[1]:g} s.")
    movements = [f"{names.get(a) or a} → {names.get(b) or b}" for a, b in evidence.get("known_movements", [])]
    if movements:
        lines.append("Mouvements : " + " ; ".join(dict.fromkeys(movements)))
    sources = sorted({edge["source_vehicle_id"] for edge in evidence.get("qualified_dependencies", [])})
    if sources:
        lines.append("Passage contraint pour : " + ", ".join(aliases.get(item, item) for item in sources))
        for edge in evidence.get("qualified_dependencies", []):
            if edge["edge_type"] == "connection_obstacle" and edge["target"].startswith("vehicle:"):
                target = edge["target"].removeprefix("vehicle:")
                lines.append(f"{aliases.get(edge['source_vehicle_id'], edge['source_vehicle_id'])} attend le passage gêné par {aliases.get(target, target)}.")
    occupants = sorted({foe["vehicle_id"] for foe in evidence.get("native_occupations", [])})
    if occupants:
        lines.append("Véhicules présents dans des passages : " + ", ".join(aliases.get(item, item) for item in occupants)
                     + " ; leur rôle de blocage n'est pas toujours confirmé.")
    if row.get("qualified_cause_age_s") is not None:
        lines.append(f"Attentes inchangées depuis {row['qualified_cause_age_s']:g} s.")
    if row.get("cycle_s") is None:
        lines.append("Pas de durée de référence connue pour confirmer un arrêt prolongé ici.")
    if not row.get("continuous_measurement", True):
        lines.append("Mesures manquantes ou invalides ; continuité non confirmée.")
    if evidence.get("footprint_uncertainty"):
        lines.append("Position de certains arrières inconnue ; passage non déclaré libre.")
    lines.extend(("Les autres voies possibles ne sont pas toutes vérifiées.",
                  "Les événements passés ne sont pas rejoués sur la carte.",
                  "Aucun gridlock confirmé. Aucune décision d'intervention."))
    return "\n".join(lines)


def graph_positions(graph):
    """Place une petite page sans ajouter de dépendance graphique."""
    nodes = sorted(graph)
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: (0.0, 0.0)}
    components = sorted((sorted(group) for group in nx.weakly_connected_components(graph)), key=lambda group: group[0])
    if len(components) > 1:
        columns = min(3, math.ceil(math.sqrt(len(components))))
        rows = math.ceil(len(components) / columns)
        result = {}
        # On sépare les groupes pour ne pas superposer leurs petites files.
        for index, component in enumerate(components):
            x = -1 + (2 * (index % columns) + 1) / columns
            y = -1 + (2 * (index // columns) + 1) / rows
            for node, (local_x, local_y) in graph_positions(graph.subgraph(component)).items():
                result[node] = (x + .8 * local_x / columns, y + .7 * local_y / rows)
        return result
    if (graph.number_of_edges() == len(nodes) - 1 and nx.is_directed_acyclic_graph(graph)
            and all(graph.in_degree(n) <= 1 and graph.out_degree(n) <= 1 for n in nodes)):
        order = list(nx.topological_sort(graph))
        columns, rows = min(4, len(nodes)), math.ceil(len(nodes) / 4)
        return {node: (-.9 + 1.8 * ((i % columns) if (i // columns) % 2 == 0 else columns - 1 - i % columns) / max(1, columns - 1),
                       0.0 if rows == 1 else -.7 + 1.4 * (i // columns) / (rows - 1)) for i, node in enumerate(order)}
    points = {node: [math.cos(2 * math.pi * i / len(nodes)), math.sin(2 * math.pi * i / len(nodes))]
              for i, node in enumerate(nodes)}
    spacing = math.sqrt(4 / len(nodes))
    for iteration in range(35):
        shifts = {node: [0.0, 0.0] for node in nodes}
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                dx, dy = points[a][0] - points[b][0], points[a][1] - points[b][1]
                distance = max(0.01, math.hypot(dx, dy))
                factor = spacing * spacing / distance ** 2
                for axis, delta in enumerate((dx, dy)):
                    shifts[a][axis] += delta * factor
                    shifts[b][axis] -= delta * factor
        for a, b in sorted(graph.edges):
            dx, dy = points[a][0] - points[b][0], points[a][1] - points[b][1]
            factor = math.hypot(dx, dy) / spacing
            for axis, delta in enumerate((dx, dy)):
                shifts[a][axis] -= delta * factor
                shifts[b][axis] += delta * factor
        temperature = 0.15 * (1 - iteration / 35)
        for node in nodes:
            length = max(0.01, math.hypot(*shifts[node]))
            for axis in (0, 1):
                points[node][axis] += shifts[node][axis] * min(length, temperature) / length
    centres = [sum(points[node][axis] for node in nodes) / len(nodes) for axis in (0, 1)]
    extent = max(abs(points[node][axis] - centres[axis]) for node in nodes for axis in (0, 1))
    return {node: tuple((points[node][axis] - centres[axis]) / max(extent, 0.01) for axis in (0, 1)) for node in nodes}


class InfoPanel:
    """Parcourt tous les groupes ; limite seulement le dessin à une page lisible."""

    def __init__(self, directory):
        import tkinter as tk
        from tkinter import ttk
        self.directory = Path(directory)
        self.aliases = json.loads((self.directory / "ids.json").read_text(encoding="utf-8"))
        self.reverse = {short: original for original, short in self.aliases.items()}
        self.frame, self.clock, self.frame_stamp = None, {}, None
        self.focus = json.loads((self.directory / "control.json").read_text())["focus"]
        self.group, self.page, self.inspected, self.map_vehicles = None, 0, None, None
        self.action_ns = None
        self.mode, self.detection_id, self.detection_nodes = "crdg", None, None
        self.v2v_data, self.v2v_stamp, self.v2v_visible = None, None, False
        self.detection_rows = {}
        self.actor_choice = "Arrêtés"
        self.graph_scope = "réseau"
        self.camera_feedback = None
        self.root = tk.Tk()
        self.root.title("INFO C-RDG")
        width = max(520, min(780, self.root.winfo_screenwidth() - 60))
        height = max(420, min(880, self.root.winfo_screenheight() - 120))
        self.root.geometry(f"{width}x{height}+20+70")
        self.root.minsize(520, 420)
        self.root.resizable(True, True)
        self.root.configure(background="#edf2f7")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Treeview", rowheight=23, font=("DejaVu Sans", 10))
        style.configure("Treeview.Heading", font=("DejaVu Sans", 10, "bold"))
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.viewport = tk.Canvas(self.root, highlightthickness=0, background="#edf2f7")
        self.viewport.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(self.root, orient="vertical", command=self.viewport.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(self.root, orient="horizontal", command=self.viewport.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.viewport.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.sizegrip = ttk.Sizegrip(self.root)
        self.sizegrip.grid(row=1, column=1, sticky="se")
        self.content = tk.Frame(self.viewport, background="#edf2f7")
        self.content_window = self.viewport.create_window(0, 0, window=self.content, anchor="nw")
        self.viewport.bind("<Configure>", self.fit_content)
        self.content.bind("<Configure>", self.fit_content)
        header = tk.Frame(self.content, background="#152b41")
        header.pack(fill="x")
        tk.Label(header, text="INFO C-RDG", background="#152b41", foreground="white", font=("DejaVu Sans", 17, "bold")).pack(anchor="w", padx=14, pady=(10, 0))
        self.time_label = tk.Label(header, text="En attente du premier graphe…", background="#152b41", foreground="#d5e8f6", font=("DejaVu Sans", 10))
        self.time_label.pack(anchor="w", padx=14, pady=(4, 10))
        self.count_label = tk.Label(self.content, background="#edf2f7", foreground="#223c55", anchor="w", font=("DejaVu Sans", 10))
        self.count_label.pack(fill="x", padx=12, pady=6)
        bar = ttk.Frame(self.content)
        bar.pack(fill="x", padx=10)
        ttk.Label(bar, text="Voiture").pack(side="left")
        self.choice = ttk.Combobox(bar, width=12, values=sorted(self.reverse))
        self.choice.pack(side="left", padx=5)
        if self.focus:
            self.choice.set(self.aliases[self.focus])
        self.choice.bind("<<ComboboxSelected>>", self.choose_vehicle)
        self.choice.bind("<Return>", self.choose_vehicle)
        ttk.Button(bar, text="Voir", command=self.choose_vehicle).pack(side="left")
        ttk.Button(bar, text="Groupes", command=self.overview).pack(side="left", padx=5)
        mode = ttk.Frame(self.content)
        mode.pack(fill="x", padx=10)
        self.mode_label = ttk.Label(mode, text="Mode : C-RDG")
        self.mode_label.pack(side="left")
        enabled = json.loads((self.directory / "control.json").read_text()).get("detection_enabled", False)
        ttk.Button(mode, text="Fermer la démo", command=self.request_close).pack(side="right", padx=4)
        self.pause_button = ttk.Button(mode, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side="right", padx=4)
        self.detection_button = ttk.Button(mode, text="Détecteur" if enabled else "Détecteur (inactif)",
                                          state="normal" if enabled else "disabled", command=self.toggle_detector)
        self.detection_button.pack(side="right")
        self.v2v_enabled = json.loads((self.directory / "control.json").read_text()).get("v2v_enabled", False)
        self.v2v_button = ttk.Button(mode, text="Communications V2V" if self.v2v_enabled else "V2V (inactif)",
                                     state="normal" if self.v2v_enabled else "disabled", command=self.toggle_v2v)
        self.v2v_button.pack(side="right", padx=4)
        camera = ttk.Frame(self.content)
        camera.pack(fill="x", padx=10, pady=(5, 0))
        for row, column, label, kind in ((0, 0, "Voir sur la carte", "fit"), (0, 1, "Vue du secteur", "sector"),
                                         (0, 2, "Vue du réseau", "network"), (1, 0, "Suivre la voiture", "follow"),
                                         (1, 1, "Caméra libre", "manual")):
            ttk.Button(camera, text=label, command=lambda kind=kind: self.request_camera(kind)).grid(row=row, column=column, padx=2, pady=2, sticky="w")
        labels = ttk.Frame(camera)
        labels.grid(row=1, column=2, sticky="w")
        filters = ttk.Frame(self.content)
        filters.pack(fill="x", padx=10, pady=5)
        ttk.Label(filters, text="Afficher").pack(side="left")
        self.filter = ttk.Combobox(filters, state="readonly", width=22, values=list(FILTERS))
        self.filter.set("Tous les groupes")
        self.filter.pack(side="left", padx=5)
        self.filter.bind("<<ComboboxSelected>>", self.change_filter)
        self.group_count = ttk.Label(filters)
        self.group_count.pack(side="left", padx=5)
        self.label_mode = ttk.Combobox(labels, state="readonly", width=9, values=("Sélection", "Toutes", "Aucune"))
        self.label_mode.set("Sélection")
        self.label_mode.pack(side="right")
        self.label_mode.bind("<<ComboboxSelected>>", self.change_labels)
        ttk.Label(labels, text="Noms").pack(side="right", padx=4)
        summary = ttk.Frame(self.content)
        summary.pack(fill="x", padx=12, pady=4)
        self.description = tk.StringVar(value="Aucun gridlock confirmé.")
        self.description_label = ttk.Label(summary, textvariable=self.description, wraplength=740)
        self.description_label.pack(fill="x")
        self.selection_notice = tk.StringVar()
        self.selection_label = ttk.Label(summary, textvariable=self.selection_notice, wraplength=740)
        self.selection_label.pack(fill="x", pady=2)
        self.v2v_box = ttk.LabelFrame(self.content, text="Communications V2V — transport simulé")
        self.v2v_status = ttk.Label(self.v2v_box, text="En attente du premier échange.", wraplength=720)
        self.v2v_status.pack(fill="x", padx=8, pady=4)
        v2v_controls = ttk.Frame(self.v2v_box)
        v2v_controls.pack(fill="x", padx=8)
        ttk.Label(v2v_controls, text="CAV").pack(side="left")
        self.v2v_choice = ttk.Combobox(v2v_controls, state="readonly", width=12,
                                       values=("Tous", *sorted(self.reverse)))
        self.v2v_choice.set("Tous")
        self.v2v_choice.pack(side="left", padx=5)
        self.v2v_choice.bind("<<ComboboxSelected>>", self.render_v2v)
        ttk.Button(v2v_controls, text="Voir dans le graphe", command=self.select_v2v_vehicle).pack(side="left")
        self.v2v_text = tk.Text(self.v2v_box, height=9, wrap="word", font=("DejaVu Sans", 9),
                                background="#f5f8fb", relief="flat")
        self.v2v_text.pack(fill="x", padx=8, pady=5)
        self.v2v_text.configure(state="disabled")
        panes = ttk.Panedwindow(self.content, orient="vertical")
        self.panes = panes
        # La légende garde sa place ; les détails restent dans une partie défilante.
        footer = ttk.Frame(self.content)
        footer.pack(side="bottom", fill="x", padx=10, pady=(0, 8))
        self.end_reason = tk.StringVar()
        self.end_label = ttk.Label(footer, textvariable=self.end_reason, wraplength=740)
        self.end_label.pack(fill="x")
        self.camera_label = ttk.Label(footer, text="Caméra déplacée seulement à votre demande.", wraplength=740)
        self.camera_label.pack(fill="x")
        self.labels_label = ttk.Label(footer, text="Étiquettes : sélection par défaut.", wraplength=740)
        self.labels_label.pack(fill="x")
        self.technical = tk.BooleanVar(value=False)
        ttk.Checkbutton(summary, text="Détails", variable=self.technical, command=self.details).pack(anchor="w")
        self.detail_box = ttk.Frame(panes)
        self.raw_details = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.detail_box, text="Données techniques", variable=self.raw_details, command=self.details).pack(anchor="w")
        detail_body = ttk.Frame(self.detail_box)
        detail_body.pack(fill="both", expand=True)
        self.detail_text = tk.Text(detail_body, height=6, font=("DejaVu Sans", 9), wrap="word", background="#f5f8fb", relief="flat")
        detail_scroll = ttk.Scrollbar(detail_body, orient="vertical", command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_scroll.set)
        self.detail_text.pack(side="left", fill="both", expand=True)
        detail_scroll.pack(side="right", fill="y")
        self.legend_box = self.make_legend(footer)
        panes.pack(fill="both", expand=True, padx=10, pady=7)
        table_frame = ttk.Frame(panes)
        self.groups = ttk.Treeview(table_frame, columns=("group", "label", "vehicles", "resources", "edges"), show="headings", height=4)
        for key, text, width in (("group", "Groupe", 65), ("label", "Relations observées", 245), ("vehicles", "Voitures", 70), ("resources", "Ressources", 80), ("edges", "Relations", 75)):
            self.groups.heading(key, text=text)
            self.groups.column(key, width=width, minwidth=width, anchor="center")
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.groups.yview)
        table_horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=self.groups.xview)
        table_horizontal.pack(side="bottom", fill="x")
        self.groups.configure(yscrollcommand=scroll.set, xscrollcommand=table_horizontal.set)
        self.groups.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.groups.bind("<<TreeviewSelect>>", self.choose_group)
        panes.add(table_frame, weight=1)
        graph_frame = ttk.Frame(panes)
        self.graph_frame = graph_frame
        self.canvas = tk.Canvas(graph_frame, background="white", highlightthickness=1, highlightbackground="#cad6df", height=220)
        self.canvas.bind("<Configure>", lambda event: self.draw())
        page_bar = ttk.Frame(graph_frame)
        page_bar.pack(side="bottom", fill="x")
        ttk.Button(page_bar, text="←", command=lambda: self.change_page(-1)).pack(side="left")
        ttk.Button(page_bar, text="→", command=lambda: self.change_page(1)).pack(side="left")
        self.page_label = ttk.Label(page_bar, text="Choisir un groupe ou une voiture")
        self.page_label.pack(side="left", padx=8)
        self.canvas.pack(fill="both", expand=True)
        panes.add(graph_frame, weight=3)
        edge_frame = ttk.Frame(panes)
        self.edge_frame = edge_frame
        self.edges = ttk.Treeview(edge_frame, columns=("from", "relation", "to"), show="headings", height=4)
        for key, text, width in (("from", "De", 95), ("relation", "Relation", 180), ("to", "Vers", 95)):
            self.edges.heading(key, text=text)
            self.edges.column(key, width=width)
        edge_scroll = ttk.Scrollbar(edge_frame, orient="vertical", command=self.edges.yview)
        self.edges.configure(yscrollcommand=edge_scroll.set)
        self.edges.pack(side="left", fill="both", expand=True)
        edge_scroll.pack(side="right", fill="y")
        self.edges.bind("<<TreeviewSelect>>", self.choose_edge)
        panes.add(edge_frame, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self.request_close)
        self.root.report_callback_exception = self.callback_error
        self.root.bind_all("<ButtonPress-1>", lambda event: setattr(self, "action_ns", time.monotonic_ns()), add="+")
        self.root.after_idle(lambda: atomic_json(self.directory / "ready.json", {"ready": True}))
        self.root.after(30, self.refresh)

    def fit_content(self, event=None):
        # Sur un petit écran, les commandes restent accessibles par défilement.
        width = max(480, self.viewport.winfo_width())
        for label in (self.description_label, self.selection_label, self.end_label,
                      self.camera_label, self.labels_label, self.count_label, self.time_label):
            label.configure(wraplength=max(240, width - 32))
        self.page_label.configure(wraplength=max(120, width - 280))
        columns = 4 if width >= 700 else 2
        for index, item in enumerate(self.legend_box.winfo_children()):
            item.grid_configure(row=index // columns, column=index % columns)
        height = max(self.content.winfo_reqheight(), self.viewport.winfo_height())
        self.viewport.itemconfigure(self.content_window, width=width, height=height)
        self.viewport.configure(scrollregion=(0, 0, width, height))

    def make_legend(self, parent):
        import tkinter as tk
        from tkinter import ttk
        box = ttk.Frame(parent)
        box.pack(fill="x", pady=4)
        for index, (colour, label) in enumerate(((SELECTED, "Sélection"), (CONSTRAINED, "Contrainte observée"),
                                                (PARTICIPANT, "Autre participant"), ((255, 255, 0), "Trafic ordinaire"))):
            item = ttk.Frame(box)
            item.grid(row=0, column=index, sticky="w", padx=4)
            dot = tk.Canvas(item, width=16, height=18, highlightthickness=0)
            dot.create_oval(3, 4, 13, 14, fill="#%02x%02x%02x" % colour[:3], outline="#697985")
            dot.pack(side="left")
            ttk.Label(item, text=label).pack(side="left")
        return box

    def callback_error(self, kind, error, traceback):
        import traceback as trace
        trace.print_exception(kind, error, traceback)
        self.end_reason.set(f"Erreur du panneau : {error}. Les preuves restent consultables.")
        self.command(ui_error=str(error))

    def command(self, **changes):
        started = time.monotonic_ns()
        path = self.directory / "control.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update(changes)
        value["sent_ns"] = time.monotonic_ns()
        atomic_json(path, value)
        click = self.action_ns or started
        self.action_ns = None
        def visible():
            with (self.directory / "ui_events.jsonl").open("a") as stream:
                stream.write(json.dumps({"action": sorted(changes), "click_ns": click,
                                        "processing_ns": started, "sent_ns": value["sent_ns"],
                                        "tk_visible_ns": time.monotonic_ns()}) + "\n")
        self.root.after_idle(visible)

    def request_close(self):
        self.command(close_requested=True, paused=False)
        self.description.set("Fermeture demandée ; arrêt contrôlé des ressources…")

    def change_labels(self, event=None):
        self.command(labels={"Sélection": "selected", "Toutes": "all", "Aucune": "none"}[self.label_mode.get()])

    def choose_vehicle(self, event=None):
        short = self.choice.get().strip()
        native = self.reverse.get(short, short if short in self.aliases else None)
        if native is None:
            self.description.set("Choisir un identifiant V… de cette simulation.")
            return
        self.focus, self.group, self.page = native, None, 0
        self.camera_feedback = None
        self.detection_nodes = None
        if self.mode != "detector":
            self.detection_id = None
        self.choice.set(self.aliases[native])
        self.map_vehicles = None
        self.inspected = None
        self.details()
        observed = self.frame and any(row["id"] == "vehicle:" + native for row in self.frame["snapshot"]["nodes"])
        self.description.set("Voiture sélectionnée : " + self.aliases[native] if observed else
                             "Aucun lien observé pour cette voiture ; graphe du réseau conservé.")
        self.command(focus=self.focus, highlighted=[self.focus])
        self.draw()
        self.show_detection()

    def choose_group(self, event=None):
        if self.mode == "detector":
            if event is not None and self.groups.selection() == (self.detection_id,):
                return
            self.choose_detection()
            return
        selected = self.groups.selection()
        if selected and selected[0] != self.group:
            self.group, self.focus, self.page = selected[0], None, 0
            self.map_vehicles = None
            self.inspected = None
            self.details()
            self.command(focus=None, highlighted=self.chosen_vehicles())
            self.description.set(next(g["label"] for g in self.frame["groups"] if g["id"] == self.group))
            self.draw()

    def overview(self):
        self.focus, self.group, self.page = None, None, 0
        self.inspected = self.map_vehicles = None
        self.detection_id = self.detection_nodes = None
        self.selection_notice.set("")
        self.command(focus=None, highlighted=[])
        self.draw()

    def filtered_groups(self):
        category = FILTERS[self.filter.get()]
        return [group for group in self.frame["groups"] if category == "all" or category in group["categories"]] if self.frame else []

    def fill_groups(self):
        if self.mode == "detector":
            self.fill_detections()
            return
        self.groups.delete(*self.groups.get_children())
        rows = self.filtered_groups()
        for group in rows:
            self.groups.insert("", "end", iid=group["id"], values=(group["id"], group["label"], group["vehicles"], group["resources"], group["edges"]))
        if self.group in {row["id"] for row in rows}:
            self.groups.selection_set(self.group)
        self.group_count.configure(text=f"{len(rows)} / {len(self.frame['groups']) if self.frame else 0} groupes")

    def toggle_detector(self):
        self.mode = "crdg" if self.mode == "detector" else "detector"
        self.mode_label.configure(text="Mode : Détecteur" if self.mode == "detector" else "Mode : C-RDG")
        self.detection_button.configure(text="C-RDG" if self.mode == "detector" else "Détecteur")
        columns = (("group", "Lieu", 220), ("label", "Situation", 210), ("vehicles", "Arrêtés", 65),
                   ("resources", "Peu de progrès", 110), ("edges", "Cause", 215)) if self.mode == "detector" else (
                   ("group", "Groupe", 65), ("label", "Relations observées", 245), ("vehicles", "Voitures", 70),
                   ("resources", "Ressources", 80), ("edges", "Relations", 75))
        for key, label, width in columns:
            self.groups.heading(key, text=label)
            self.groups.column(key, width=width, minwidth=width)
        self.filter.configure(state="disabled" if self.mode == "detector" else "readonly",
                              values=list(DETECTION_ACTORS) if self.mode == "detector" else list(FILTERS))
        self.filter.set(self.actor_choice if self.mode == "detector" else "Tous les groupes")
        if self.mode == "detector":
            self.panes.forget(self.edge_frame)
        else:
            self.panes.add(self.edge_frame, weight=1)
        self.overview()
        self.fill_groups()
        self.details()
        if self.mode == "detector" and self.groups.get_children():
            first = self.groups.get_children()[0]
            self.groups.selection_set(first)
            self.choose_detection()
        else:
            self.description.set("Aucune situation observée pour l'instant."
                                 if self.mode == "detector" else "Choisir un groupe C-RDG.")

    def toggle_v2v(self):
        self.v2v_visible = not self.v2v_visible
        if self.v2v_visible:
            self.v2v_box.pack(fill="x", padx=10, pady=5, before=self.panes)
            self.v2v_button.configure(text="Masquer V2V")
            self.render_v2v()
        else:
            self.v2v_box.pack_forget()
            self.v2v_button.configure(text="Communications V2V")

    def select_v2v_vehicle(self):
        alias = self.v2v_choice.get()
        if alias in self.reverse:
            self.choice.set(alias)
            self.choose_vehicle()

    def render_v2v(self, event=None):
        if not self.v2v_visible or self.v2v_data is None:
            return
        state = self.v2v_data
        counts = state["counts"]
        self.v2v_status.configure(text=(
            f"SUMO {state['time_s']:g} s · {state['active_cav']} CAV actifs · "
            f"{state['communicating_pairs']} paires joignables · "
            f"{counts['emitted']} messages émis / {counts['delivered']} réceptions · "
            f"{counts['expired']} informations périmées · portée {state['range_m']:g} m"))
        alias = self.v2v_choice.get()
        item = self.reverse.get(alias)
        lines = []
        if item is not None:
            local = state["participants"].get(item)
            if local is None:
                lines.append(f"{alias} n'est pas encore présent ou est arrivé : aucun état actif.")
            else:
                own = local["own"]
                lines.append(f"{alias} — observation propre émulée par TraCI à {own['observed_at_s']:g} s : "
                             f"voie {own['lane_id']}, vitesse {own['speed_mps']:.2f} m/s, "
                             f"position ({own['position_xy_m'][0]:.1f}, {own['position_xy_m'][1]:.1f}) m.")
                progress = own["progress_delta_m"]
                lines.append("Progrès récent : inconnu au premier pas." if progress is None else
                             f"Progrès récent : {progress:.2f} m en {own['progress_window_s']:g} s.")
                neighbors = state["reachable_neighbors"].get(item, [])
                lines.append("Voisins joignables : " +
                             (", ".join(self.aliases[name] for name in neighbors) if neighbors else "aucun"))
                received = local["received"]
                lines.append(f"Messages valides reçus : {len(received)} émetteur(s).")
                for sender, entry in sorted(received.items()):
                    message = entry["message"]
                    lines.append(f"  {self.aliases[sender]} : reçu à {entry['received_at_s']:g} s, "
                                 f"mesuré à {message['observed_at_s']:g} s, "
                                 f"valide jusqu'à {message['valid_until_s']:g} s ; "
                                 f"voie {message['lane_id']}, vitesse {message['speed_mps']:.2f} m/s "
                                 f"({message['source']}).")
                if local["expired"]:
                    lines.append("Informations périmées : " +
                                 ", ".join(self.aliases[name] for name in sorted(local["expired"])))
                sent = local["last_sent"]
                lines.append("Dernière émission : " +
                             (f"{sent['message_id']} à {sent['sent_at_s']:g} s ; "
                              "les destinataires sont connus du transport, pas de l'émetteur." if sent else "aucune"))
                others = [name for name in sorted(state["participants"]) if name != item and name not in received]
                lines.append(f"Diagnostic du transport : {len(others)} états non reçus ; "
                             f"leur contenu demeure inconnu pour {alias}.")
                for name in others[:6]:
                    key = "|".join(sorted((item, name)))
                    distance = state["transport_distances_m"][key]
                    lines.append(f"  {self.aliases[name]} : {distance:.1f} m, "
                                 f"{'hors de portée' if distance > state['range_m'] else 'joignable, pas encore reçu'} "
                                 "(diagnostic du transport, non connaissance du CAV).")
        else:
            lines.append("Choisir un CAV pour distinguer son état propre de ses messages reçus.")
        lines.append("Échanges récents :")
        recent = [row for row in state["recent_emissions"] if
                  (item is None or row["message"]["sender_id"] == item or item in row["delivered_to"])]
        for row in recent[-10:]:
            sender = self.aliases[row["message"]["sender_id"]]
            recipients = ", ".join(self.aliases[name] for name in row["delivered_to"]) or "personne"
            lines.append(f"  {row['time_s']:g} s · {sender} → {recipients} · "
                         f"{row['out_of_range_count']} hors portée.")
        if not recent:
            lines.append("  Aucun message pertinent à cet instant.")
        self.v2v_text.configure(state="normal")
        self.v2v_text.delete("1.0", "end")
        self.v2v_text.insert("end", "\n".join(lines))
        self.v2v_text.configure(state="disabled")

    def change_filter(self, event=None):
        if self.mode == "detector":
            if self.filter.get() in DETECTION_ACTORS:
                self.actor_choice = self.filter.get()
                self.choose_detection()
        else:
            self.fill_groups()

    def fill_detections(self):
        data = self.frame.get("detection") if self.frame else None
        inspected = self.inspected
        rows = [*(data or {}).get("completed_episodes", []), *(data or {}).get("episodes", [])]
        self.detection_rows = {row["episode_id"]: row for row in rows}
        active = {row["episode_id"] for row in (data or {}).get("episodes", [])}
        self.groups.delete(*self.groups.get_children())
        for row in sorted(rows, key=lambda row: detection_priority(row, active)):
            state = DETECTION_STATES.get(row["state"], "État inconnu")
            if row.get("right_censored") or (data.get("run_status") in ("completed", "horizon_reached")
                                             and row["episode_id"] in active):
                state = "Fin · " + state
            elif row["episode_id"] not in active:
                state = "Passé · " + state
            ages = detection_ages(row)
            age = f"{min(ages):g}–{max(ages):g} s" if ages else "—"
            cause = detection_cause(row)
            label = "O" + row["episode_id"].split(":")[-1].lstrip("0")
            self.groups.insert("", "end", iid=row["episode_id"], values=(label + " · " + detection_place(row["junction"]),
                state, len(row.get("current_halted_members", [])), age, cause))
        if data.get("run_status") == "completed":
            count_text = f"{len(active)} à clôturer / {len(rows)} suivies"
        else:
            count_text = f"{len(active)} en cours / {len(rows)} suivies"
        self.group_count.configure(text=count_text)
        if self.detection_id in self.detection_rows:
            self.groups.selection_set(self.detection_id)
            if self.focus is None:
                self.choose_detection(send_command=False)
                if inspected and inspected.get("edge_type"):
                    item = next((edge for edge in self.frame["snapshot"]["edges"] if edge_key(edge) == edge_key(inspected)), None)
                    if item:
                        self.inspect(item)
                elif inspected and inspected.get("node_type") == "resource":
                    item = next((node for node in self.frame["snapshot"]["nodes"] if node["id"] == inspected["id"]), None)
                    if item:
                        self.inspect(item)
            else:
                self.show_detection()
                self.details()
        elif self.detection_id is not None:
            self.detection_id = self.detection_nodes = self.map_vehicles = self.inspected = None
            self.command(highlighted=[])
            self.description.set("Cette situation n'est plus disponible.")
        if data is None:
            self.description.set("En attente des premières observations du détecteur.")

    def choose_detection(self, *, send_command=True):
        selected = self.groups.selection()
        if not selected or selected[0] not in self.detection_rows:
            return
        changed = self.detection_id != selected[0]
        self.detection_id = selected[0]
        self.camera_feedback = None
        self.groups.see(self.detection_id)
        row = self.detection_rows[self.detection_id]
        self.focus, self.group, self.page = None, None, 0
        self.choice.set("")
        present = set(self.clock.get("present", []))
        choices = [label for label in DETECTION_ACTORS if detection_actors(row, label)]
        if changed:
            self.actor_choice = "Gênés" if "Gênés" in choices else "Arrêtés"
        if self.actor_choice not in choices:
            self.actor_choice = "Tous" if "Tous" in choices else choices[0] if choices else "Arrêtés"
        self.filter.configure(values=choices, state="readonly" if choices else "disabled")
        self.filter.set(self.actor_choice if choices else "Aucun membre connu")
        candidates = detection_actors(row, self.actor_choice)
        self.map_vehicles = set(candidates) & present
        self.detection_nodes = {"vehicle:" + item for item in self.map_vehicles}
        self.inspected = row
        if send_command:
            self.command(focus=None, highlighted=sorted(self.map_vehicles))
            if self.map_vehicles:
                self.request_camera("fit")
        self.details()
        self.draw()
        self.show_detection()

    def show_detection(self):
        if self.mode != "detector" or self.detection_id not in self.detection_rows:
            return
        row = self.detection_rows[self.detection_id]
        data = self.frame.get("detection") or {}
        self.description.set(detection_text(row, self.aliases, data.get("street_names", {}), terminal_status=data.get("run_status")))
        event = row.get("end_s") if row.get("end_s") is not None else row["last_seen_s"]
        historic = row.get("end_s") is not None or row.get("state") in ("resolved", "resolved_by_complete_drain", "interrupted")
        caption = "Historique" if historic else "Observé"
        now = self.clock.get("time_s", self.frame["snapshot"]["time_s"])
        selected = self.chosen_vehicles()
        selection = self.aliases.get(self.focus, self.focus) if self.focus else self.actor_choice
        if self.inspected and self.inspected.get("edge_type"):
            selection = "Acteurs du lien"
        elif self.inspected and self.inspected.get("node_type") == "resource":
            selection = "Acteurs du passage"
        represented = {"vehicle:" + item for item in selected} & {n["id"] for n in self.frame["snapshot"]["nodes"]}
        self.selection_notice.set(f"{caption} à {event:g} s · carte actuelle à {now:g} s\n"
                                  f"{selection} : {len(set(selected) & set(self.clock.get('present', [])))} présents · "
                                  f"{len(represented)} dans le graphe · vue {self.graph_scope}.")
        if not self.focus and self.actor_choice == "Ont avancé":
            times = [detail["first_progress_s"] for item, detail in row.get("member_observations", {}).items()
                     if item in row.get("advanced_members", []) and detail.get("first_progress_s") is not None]
            if times:
                self.selection_notice.set(self.selection_notice.get() + f"\nProgrès enregistrés à {min(times):g}–{max(times):g} s ; distances dans Détails.")

    def chosen_vehicles(self):
        if self.map_vehicles is not None:
            return sorted(self.map_vehicles)
        if self.focus:
            return [self.focus]
        return sorted(node.removeprefix("vehicle:") for node in self.selected_nodes() if node.startswith("vehicle:"))

    def request_camera(self, kind):
        vehicles = self.chosen_vehicles()
        if kind in ("fit", "follow") and not vehicles:
            if self.mode == "detector" and self.detection_id:
                self.camera_feedback = "Aucun membre encore présent ; aucun ancien cadrage utilisé."
                self.camera_label.configure(text=self.camera_feedback)
            else:
                self.description.set("Choisir un groupe, une voiture ou une relation.")
            return
        if kind == "follow" and len(vehicles) != 1:
            self.description.set("Choisir une seule voiture pour la suivre.")
            return
        command = json.loads((self.directory / "control.json").read_text())
        previous = command.get("camera", {})
        self.camera_feedback = None
        self.command(camera={"sequence": previous.get("sequence", 0) + 1, "kind": kind, "vehicles": vehicles},
                     highlighted=vehicles if kind in ("fit", "follow") else command.get("highlighted", []),
                     focus=vehicles[0] if kind == "follow" else self.focus)

    def toggle_pause(self):
        path = self.directory / "control.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        paused = not value.get("paused", False)
        self.command(paused=paused)
        self.pause_button.configure(text="Reprendre" if paused else "Pause")

    def change_page(self, increment):
        self.page = max(0, self.page + increment)
        self.draw()

    def label(self, node):
        if node.startswith("vehicle:"):
            return self.aliases[node.removeprefix("vehicle:")]
        return self.frame["resources"].get(node, "R?")

    def relation_label(self, edge):
        if edge["edge_type"] == "waits_for":
            resource = next((n for n in self.frame["snapshot"]["nodes"] if n["id"] == edge["target"]), {})
            return {"receiving_space": "Espace insuffisant", "junction_conflict": "Conflit de passage"}.get(resource.get("resource_type"), "Cause inconnue")
        return RELATIONS.get(edge["edge_type"], "Contrainte")

    def selected_nodes(self):
        if not self.frame:
            return set()
        snapshot = self.frame["snapshot"]
        if self.detection_nodes is not None:
            found = self.detection_nodes & {row["id"] for row in snapshot["nodes"]}
            if not found:
                self.graph_scope = "réseau (aucun lien pour la sélection)"
                return {row["id"] for row in snapshot["nodes"]}
            self.graph_scope = "voisinage"
            for _ in range(2):
                previous = set(found)
                for edge in snapshot["edges"]:
                    if edge["source"] in previous or edge["target"] in previous:
                        found.update((edge["source"], edge["target"]))
            return found
        if self.focus:
            root = "vehicle:" + self.focus
            if not any(row["id"] == root for row in snapshot["nodes"]):
                self.graph_scope = "réseau (aucun lien pour la sélection)"
                return {row["id"] for row in snapshot["nodes"]}
            self.graph_scope = "voisinage"
            found = {root}
            for _ in range(2):
                previous = set(found)
                for edge in snapshot["edges"]:
                    if edge["source"] in previous:
                        found.add(edge["target"])
                    if edge["target"] in previous:
                        found.add(edge["source"])
            return found
        self.graph_scope = "groupe" if self.group else "réseau"
        return set(next((row["nodes"] for row in self.frame["groups"] if row["id"] == self.group),
                        [row["id"] for row in snapshot["nodes"]]))

    def draw(self):
        if not self.frame:
            return
        self.canvas.delete("all")
        chosen = self.selected_nodes()
        snapshot = self.frame["snapshot"]
        nodes = {row["id"]: row for row in snapshot["nodes"] if row["id"] in chosen}
        edges = [edge for edge in snapshot["edges"] if edge["source"] in chosen and edge["target"] in chosen]
        self.edge_rows = {edge_key(edge): edge for edge in edges}
        selection = self.edges.selection()
        self.edges.delete(*self.edges.get_children())
        inactive = set(self.clock.get("inactive_edges", []))
        for edge in edges:
            label = self.relation_label(edge)
            if edge_key(edge) in inactive:
                label += " (terminée)"
            self.edges.insert("", "end", iid=edge_key(edge), values=(self.label(edge["source"]), label, self.label(edge["target"])))
        if selection and selection[0] in self.edge_rows:
            self.edges.selection_set(selection[0])
        if not nodes:
            self.canvas.create_text(max(100, self.canvas.winfo_width() / 2), 90, text="Pas de liens observés à cet instant.", fill="#567086", font=("DejaVu Sans", 11))
            self.page_label.configure(text=f"Vue générale : {len(self.frame['groups'])} groupes accessibles")
            if self.focus and self.mode != "detector":
                reason = snapshot.get("waiting_states", {}).get(self.focus)
                self.description.set({"signal": "Attente au feu — passage non autorisé.", "unknown": "Cause inconnue — aucune relation inventée."}.get(reason, "Pas de dépendance représentée à cet instant."))
            return
        width = self.canvas.winfo_width()
        limit = (12 if width >= 900 else 8 if width >= 650 else 4) if self.mode == "detector" else (
                 24 if width >= 1100 else 12 if width >= 800 else 8 if width >= 650 else 4)
        if self.canvas.winfo_height() < 180:
            limit = min(limit, 2)
        pages = max(1, math.ceil(len(nodes) / limit))
        self.page = min(self.page, pages - 1)
        priority = set()
        if self.mode == "detector" and self.detection_id in self.detection_rows:
            # On montre d'abord les relations qualifiées de la situation sélectionnée.
            for edge in self.detection_rows[self.detection_id].get("evidence", {}).get("qualified_dependencies", []):
                priority.update(("vehicle:" + edge["source_vehicle_id"], edge["target"]))
        ordered = sorted(nodes, key=lambda node: (node not in priority, node))
        visible = ordered[self.page * limit:(self.page + 1) * limit]
        links = [edge for edge in edges if edge["source"] in visible and edge["target"] in visible]
        self.page_label.configure(text=f"Page {self.page + 1}/{pages} · {len(visible)}/{len(nodes)} éléments · {len(links)}/{len(edges)} liens dessinés")
        graph = nx.DiGraph()
        graph.add_nodes_from(visible)
        graph.add_edges_from((edge["source"], edge["target"]) for edge in links)
        key = (tuple(visible), tuple(sorted(graph.edges)))
        if key != getattr(self, "layout_key", None):
            self.layout_key, self.layout = key, graph_positions(graph)
        layout = self.layout
        width, height = max(200, self.canvas.winfo_width()), max(80, self.canvas.winfo_height())
        if self.canvas.winfo_height() < 70:
            self.canvas.create_text(width / 2, max(12, self.canvas.winfo_height() / 2),
                                    text="Agrandir la zone du graphe.", fill="#567086", font=("DejaVu Sans", 10))
            return
        points = {node: (width / 2 + float(point[0]) * (width / 2 - 55), height / 2 + float(point[1]) * (height / 2 - 40)) for node, point in layout.items()}
        for edge in links:
            a, b = points[edge["source"]], points[edge["target"]]
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = max(1, math.hypot(dx, dy))
            ux, uy = dx / length, dy / length
            colour = "#b5c0ca" if edge_key(edge) in inactive else "#526c85"
            item = self.canvas.create_line(a[0] + ux * 22, a[1] + uy * 22, b[0] - ux * 24, b[1] - uy * 24, arrow="last", width=2, fill=colour)
            self.canvas.tag_bind(item, "<Button-1>", lambda event, edge=edge: self.inspect(edge))
        constrained = set(self.clock.get("constrained", []))
        participants = set(self.clock.get("participants", []))
        for node in visible:
            x, y = points[node]
            resource = nodes[node]["node_type"] == "resource"
            selected = node == "vehicle:" + str(self.focus) or nodes[node].get("vehicle_id") in self.clock.get("highlighted", [])
            vehicle = nodes[node].get("vehicle_id")
            colour = ("#1991dc" if selected and vehicle in self.clock.get("present", []) else
                      "#f3bc68" if vehicle in constrained else "#d9c5f1" if vehicle in participants else
                      "#d8e6e8" if resource else "#e5e9ed")
            item = self.canvas.create_rectangle(x - 24, y - 18, x + 24, y + 18, fill=colour, outline="#658092") if resource else self.canvas.create_oval(x - 23, y - 23, x + 23, y + 23, fill=colour, outline="#658092")
            text = self.canvas.create_text(x, y, text=self.label(node), font=("DejaVu Sans", 10, "bold"), fill="#18334b")
            for glyph in (item, text):
                self.canvas.tag_bind(glyph, "<Button-1>", lambda event, value=nodes[node]: self.inspect(value))

    def choose_edge(self, event=None):
        selected = self.edges.selection()
        if selected and selected[0] in self.edge_rows:
            self.inspect(self.edge_rows[selected[0]])

    def inspect(self, item):
        self.inspected = item
        if "edge_type" in item:
            related = {item["source"], item["target"]}
            for edge in self.frame["snapshot"]["edges"]:
                if edge["source"] in related and edge["source"].startswith("resource:"):
                    related.add(edge["target"])
            self.map_vehicles = {node.removeprefix("vehicle:") for node in related if node.startswith("vehicle:")}
            if self.focus not in self.map_vehicles:
                self.focus = None
                self.choice.set("")
            self.command(focus=self.focus if self.focus in self.map_vehicles else None,
                         highlighted=sorted(self.map_vehicles))
            label = self.relation_label(item)
            description = f"{self.label(item['source'])} → {self.label(item['target'])} · {label}"
            if item["edge_type"] == "waits_for":
                resource = next(row for row in self.frame["snapshot"]["nodes"] if row["id"] == item["target"])
                if resource["resource_type"] == "receiving_space":
                    value = resource.get("max_free_space_m")
                    description += f"\nLibre : {'inconnu' if value is None else f'{value:g} m'} · Nécessaire : {resource['required_space_m']:g} m"
            if edge_key(item) in set(self.clock.get("inactive_edges", [])):
                description += "\nRelation disparue depuis cette observation."
            description += f"\nMesures observées à {self.frame['snapshot']['time_s']:g} s."
        else:
            description = self.label(item["id"])
            if item["node_type"] == "vehicle":
                self.focus = item["vehicle_id"]
                self.choice.set(self.aliases[self.focus])
                self.map_vehicles = {self.focus}
                self.command(focus=self.focus, highlighted=[self.focus])
                reason = {"signal": "Attente au feu", "unknown": "Cause inconnue", "moving": "Roule",
                          "leader": "Suivi limité", "connection_obstacle": "Passage contraint",
                          "receiving_space": "Espace insuffisant", "junction_conflict": "Conflit de passage",
                          "multiple": "Contraintes multiples"}.get(item.get("waiting_reason"), "Voiture sélectionnée")
                description += " · " + reason
            else:
                description += " · Espace de réception" if item["resource_type"] == "receiving_space" else " · Passage conflictuel"
                related = {node for edge in self.frame["snapshot"]["edges"] if item["id"] in (edge["source"], edge["target"])
                           for node in (edge["source"], edge["target"]) if node.startswith("vehicle:")}
                self.map_vehicles, self.focus = {node.removeprefix("vehicle:") for node in related}, None
                self.choice.set("")
                self.command(focus=None, highlighted=sorted(self.map_vehicles))
        if self.mode == "detector":
            self.show_detection()
        else:
            self.description.set(description)
        self.details()

    def details(self):
        self.detail_text.delete("1.0", "end")
        shown = str(self.detail_box) in self.panes.panes()
        if not self.technical.get():
            if shown:
                self.panes.forget(self.detail_box)
        elif self.mode == "detector" and self.detection_id in self.detection_rows and not self.raw_details.get():
            data = self.frame.get("detection") or {}
            self.detail_text.configure(height=6)
            if not shown:
                self.panes.add(self.detail_box, weight=1)
            row = self.detection_rows[self.detection_id]
            if self.inspected and (self.inspected.get("edge_type") or self.inspected.get("node_type") == "resource"):
                item = self.inspected
                if item.get("edge_type"):
                    self.detail_text.insert("end", f"{self.label(item['source'])} → {self.label(item['target'])} : {self.relation_label(item)}.\n"
                        f"Lien observé à {self.frame['snapshot']['time_s']:g} s.\n"
                        "Ces acteurs sont ceux du lien, pas forcément tous les véhicules du suivi.")
                else:
                    self.detail_text.insert("end", f"{self.label(item['id'])} : passage observé à {self.frame['snapshot']['time_s']:g} s.")
                    if item["resource_type"] == "receiving_space":
                        free = item.get("max_free_space_m")
                        self.detail_text.insert("end", f"\nPlace disponible : {'inconnue' if free is None else f'{free:g} m'}."
                                                     f"\nPlace nécessaire : {item['required_space_m']:g} m.")
            elif self.focus and self.focus not in row["members"]:
                self.detail_text.insert("end", f"{self.aliases[self.focus]} n'est pas membre de ce suivi.\nConsulter ses données techniques dans le graphe.")
            else:
                self.detail_text.insert("end", detection_details(row, self.aliases, data.get("street_names", {}),
                                             choice=self.actor_choice, items=[self.focus] if self.focus else None))
        elif self.inspected:
            self.detail_text.configure(height=6)
            if not shown:
                self.panes.add(self.detail_box, weight=1)
            self.detail_text.insert("end", json.dumps(self.inspected, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            if shown:
                self.panes.forget(self.detail_box)
        self.root.after_idle(self.fit_content)

    def refresh(self):
        try:
            clock_path = self.directory / "clock.json"
            self.clock = json.loads(clock_path.read_text(encoding="utf-8"))
            v2v_path = self.directory / "v2v.json"
            if self.v2v_enabled and v2v_path.exists():
                stamp = v2v_path.stat().st_mtime_ns
                if stamp != self.v2v_stamp:
                    self.v2v_stamp = stamp
                    self.v2v_data = json.loads(v2v_path.read_text(encoding="utf-8"))
                    self.render_v2v()
            status = self.clock.get("status")
            if status == "closing":
                self.root.destroy()
                return
            frame_path = self.directory / "frame.json"
            if frame_path.exists():
                stamp = frame_path.stat().st_mtime_ns
                if stamp != self.frame_stamp:
                    incoming = json.loads(frame_path.read_text(encoding="utf-8"))
                    if incoming["snapshot"]["time_s"] > self.clock["time_s"]:
                        self.time_label.configure(text="Mise à jour en cours…")
                        self.root.after(50, self.refresh)
                        return
                    self.frame_stamp, self.frame = stamp, incoming
                    if self.frame.get("detection") is not None:
                        self.detection_button.configure(state="normal")
                        if self.mode != "detector":
                            self.detection_button.configure(text="Détecteur")
                    if self.group and not any(g["id"] == self.group for g in self.frame["groups"]):
                        self.group = self.map_vehicles = self.inspected = None
                        self.command(highlighted=[])
                        self.description.set("Composition modifiée : choisir un groupe actuel.")
                    self.fill_groups()
                    self.draw()
                    if self.inspected and "edge_type" in self.inspected:
                        key = edge_key(self.inspected)
                        item = next((edge for edge in self.frame["snapshot"]["edges"] if edge_key(edge) == key), None)
                        if item:
                            self.inspect(item)
                        else:
                            self.inspected = None
                            self.map_vehicles = None
                            self.description.set("Cette relation n'est plus représentée au nouvel instant.")
                            self.details()
            observed = self.frame["snapshot"]["time_s"] if self.frame else None
            now = self.clock["time_s"]
            age = None if observed is None else now - observed
            freshness = "aucun graphe encore observé" if age is None else f"observé {observed:g} s · âge {age:g} s"
            no_step = time.time() - clock_path.stat().st_mtime > 3
            status_text = {"starting": "préparation", "running": "en cours", "paused": "pause",
                           "completed": "vidé", "horizon_reached": "horizon atteint", "failed": "échec"}.get(status, status)
            self.time_label.configure(text=f"SUMO {now:g} s · {freshness} · {status_text}" + (" · aucun nouveau pas reçu" if no_step else ""))
            self.pause_button.configure(text="Reprendre" if status == "paused" else "Pause")
            if status in ("completed", "horizon_reached", "failed", "user_closed"):
                self.pause_button.configure(state="disabled")
                self.end_reason.set(self.clock.get("reason") or {"completed": "Toutes les missions sont arrivées.",
                    "horizon_reached": "Horizon atteint : des missions restent présentes. Dernier état inspectable.",
                    "failed": "Exécution en erreur : consulter les diagnostics.", "user_closed": "Fermeture demandée."}[status])
            camera = self.clock.get("camera")
            if camera:
                mode = {"fit": "cadrée", "follow": "suivi", "sector": "secteur", "network": "réseau", "manual": "libre"}[camera["kind"]]
                message = camera["message"] or f"Carte {mode} · {len(camera['vehicles'])} voitures présentes à {camera['time_s']:g} s"
                if status == "paused":
                    message += " · Temps physique inchangé."
                if camera.get("native_redraw") not in (None, "not_initialized", "x11_ready"):
                    message += " · Redessin non confirmé : utiliser les contrôles natifs SUMO."
                self.camera_label.configure(text=self.camera_feedback or message)
            labels = self.clock.get("labels")
            if labels:
                mode = {"selected": "Sélection", "all": "Toutes", "none": "Aucune"}[labels["mode"]]
                self.labels_label.configure(text=f"Étiquettes {mode} : {labels['enabled']} activées / {labels['present']} voitures."
                                           + (" Zoomer ou choisir un alias pour les autres." if labels["mode"] == "all" and labels["suppressed"] else ""))
            if self.frame:
                source = self.frame["snapshot"]
                self.count_label.configure(text=(f"Graphe : {source['vehicle_count']} véhicules · {source['edge_count']} liens (à {observed:g} s)"
                    if self.mode == "detector" else
                    f"{len(self.frame['groups'])} groupes · {source['vehicle_count']} voitures · {source['resource_count']} ressources · {source['edge_count']} relations (à {observed:g} s)"))
                self.show_detection()
            # Une disparition connue est indiquée avant le prochain snapshot à 5 s.
            signature = (tuple(self.clock.get("inactive_edges", [])), tuple(self.clock.get("constrained", [])),
                         tuple(self.clock.get("participants", [])), self.clock.get("focus"))
            signature += (tuple(self.clock.get("highlighted", [])),)
            if signature != getattr(self, "inactive_signature", ()):
                self.inactive_signature = signature
                self.draw()
                if self.inspected and "edge_type" in self.inspected:
                    self.inspect(self.inspected)
        except (OSError, ValueError, KeyError) as error:
            self.time_label.configure(text=f"Données indisponibles : {error}")
        self.root.after(50, self.refresh)


def main():
    panel = InfoPanel(Path(sys.argv[1]))
    panel.root.mainloop()


if __name__ == "__main__":
    main()
