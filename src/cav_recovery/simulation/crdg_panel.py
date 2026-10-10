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
        self.pause_button = ttk.Button(bar, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side="right")
        ttk.Button(bar, text="Fermer la démo", command=self.request_close).pack(side="right", padx=4)
        mode = ttk.Frame(self.content)
        mode.pack(fill="x", padx=10)
        ttk.Label(mode, text="Mode : C-RDG").pack(side="left")
        ttk.Button(mode, text="Détecteur — indisponible", state="disabled").pack(side="right")
        camera = ttk.Frame(self.content)
        camera.pack(fill="x", padx=10, pady=(5, 0))
        ttk.Button(camera, text="Voir sur la carte", command=lambda: self.request_camera("fit")).pack(side="left")
        ttk.Button(camera, text="Vue du secteur", command=lambda: self.request_camera("sector")).pack(side="left", padx=4)
        ttk.Button(camera, text="Vue du réseau", command=lambda: self.request_camera("network")).pack(side="left")
        ttk.Button(camera, text="Suivre la voiture", command=lambda: self.request_camera("follow")).pack(side="left", padx=4)
        ttk.Button(camera, text="Caméra libre", command=lambda: self.request_camera("manual")).pack(side="left")
        filters = ttk.Frame(self.content)
        filters.pack(fill="x", padx=10, pady=5)
        ttk.Label(filters, text="Afficher").pack(side="left")
        self.filter = ttk.Combobox(filters, state="readonly", width=26, values=list(FILTERS))
        self.filter.set("Tous les groupes")
        self.filter.pack(side="left", padx=5)
        self.filter.bind("<<ComboboxSelected>>", lambda event: self.fill_groups())
        self.group_count = ttk.Label(filters)
        self.group_count.pack(side="left", padx=5)
        self.label_mode = ttk.Combobox(filters, state="readonly", width=10, values=("Sélection", "Toutes", "Aucune"))
        self.label_mode.set("Sélection")
        self.label_mode.pack(side="right")
        self.label_mode.bind("<<ComboboxSelected>>", self.change_labels)
        ttk.Label(filters, text="Étiquettes").pack(side="right", padx=4)
        # La légende et l'explication gardent leur place quand le dessin rétrécit.
        footer = ttk.Frame(self.content)
        footer.pack(side="bottom", fill="x", padx=10, pady=(0, 8))
        self.description = tk.StringVar(value="Un groupe n'est pas un deadlock confirmé.")
        self.end_reason = tk.StringVar()
        ttk.Label(footer, textvariable=self.end_reason, wraplength=740).pack(fill="x")
        ttk.Label(footer, textvariable=self.description, wraplength=740).pack(fill="x", pady=5)
        self.camera_label = ttk.Label(footer, text="Caméra déplacée seulement à votre demande.", wraplength=740)
        self.camera_label.pack(fill="x")
        self.labels_label = ttk.Label(footer, text="Étiquettes : sélection par défaut.", wraplength=740)
        self.labels_label.pack(fill="x")
        self.technical = tk.BooleanVar(value=False)
        ttk.Checkbutton(footer, text="Détails techniques", variable=self.technical, command=self.details).pack(anchor="w")
        self.detail_text = tk.Text(footer, height=4, font=("DejaVu Sans", 9), wrap="word", background="#f5f8fb", relief="flat")
        self.legend_box = self.make_legend(footer)
        panes = ttk.Panedwindow(self.content, orient="vertical")
        panes.pack(fill="both", expand=True, padx=10, pady=7)
        table_frame = ttk.Frame(panes)
        self.groups = ttk.Treeview(table_frame, columns=("group", "label", "vehicles", "resources", "edges"), show="headings", height=4)
        for key, text, width in (("group", "Groupe", 65), ("label", "Relations observées", 245), ("vehicles", "Voitures", 70), ("resources", "Ressources", 80), ("edges", "Relations", 75)):
            self.groups.heading(key, text=text)
            self.groups.column(key, width=width, anchor="center")
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.groups.yview)
        self.groups.configure(yscrollcommand=scroll.set)
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
        width = max(760, self.viewport.winfo_width(), self.content.winfo_reqwidth())
        height = max(760, self.viewport.winfo_height())
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
        self.choice.set(self.aliases[native])
        self.map_vehicles = None
        self.inspected = None
        self.details()
        observed = self.frame and any(row["id"] == "vehicle:" + native for row in self.frame["snapshot"]["nodes"])
        self.description.set("Voiture sélectionnée : " + self.aliases[native] if observed else
                             "Aucune dépendance qualifiée pour cette voiture à l'instant affiché ; graphe réseau conservé.")
        self.command(focus=self.focus, highlighted=[self.focus])
        self.draw()

    def choose_group(self, event=None):
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
        self.command(focus=None, highlighted=[])
        self.draw()

    def filtered_groups(self):
        category = FILTERS[self.filter.get()]
        return [group for group in self.frame["groups"] if category == "all" or category in group["categories"]] if self.frame else []

    def fill_groups(self):
        self.groups.delete(*self.groups.get_children())
        rows = self.filtered_groups()
        for group in rows:
            self.groups.insert("", "end", iid=group["id"], values=(group["id"], group["label"], group["vehicles"], group["resources"], group["edges"]))
        if self.group in {row["id"] for row in rows}:
            self.groups.selection_set(self.group)
        self.group_count.configure(text=f"{len(rows)} / {len(self.frame['groups']) if self.frame else 0} groupes")

    def chosen_vehicles(self):
        if self.map_vehicles is not None:
            return sorted(self.map_vehicles)
        if self.focus:
            return [self.focus]
        return sorted(node.removeprefix("vehicle:") for node in self.selected_nodes() if node.startswith("vehicle:"))

    def request_camera(self, kind):
        vehicles = self.chosen_vehicles()
        if kind in ("fit", "follow") and not vehicles:
            self.description.set("Choisir un groupe, une voiture ou une relation.")
            return
        if kind == "follow" and len(vehicles) != 1:
            self.description.set("Choisir une seule voiture pour la suivre.")
            return
        command = json.loads((self.directory / "control.json").read_text())
        previous = command.get("camera", {})
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
        if self.focus:
            root = "vehicle:" + self.focus
            if not any(row["id"] == root for row in snapshot["nodes"]):
                return {row["id"] for row in snapshot["nodes"]}
            found = {root}
            for _ in range(2):
                previous = set(found)
                for edge in snapshot["edges"]:
                    if edge["source"] in previous:
                        found.add(edge["target"])
                    if edge["target"] in previous:
                        found.add(edge["source"])
            return found
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
            self.canvas.create_text(max(100, self.canvas.winfo_width() / 2), 90, text="Aucune dépendance représentée à cet instant.", fill="#567086", font=("DejaVu Sans", 11))
            self.page_label.configure(text=f"Vue générale : {len(self.frame['groups'])} groupes accessibles")
            if self.focus:
                reason = snapshot.get("waiting_states", {}).get(self.focus)
                self.description.set({"signal": "Attente au feu — passage non autorisé.", "unknown": "Cause inconnue — aucune relation inventée."}.get(reason, "Pas de dépendance représentée à cet instant."))
            return
        pages = max(1, math.ceil(len(nodes) / 24))
        self.page = min(self.page, pages - 1)
        visible = sorted(nodes)[self.page * 24:(self.page + 1) * 24]
        links = [edge for edge in edges if edge["source"] in visible and edge["target"] in visible]
        self.page_label.configure(text=f"Page {self.page + 1}/{pages} · {len(visible)}/{len(nodes)} nœuds · {len(links)}/{len(edges)} arcs dessinés")
        graph = nx.DiGraph()
        graph.add_nodes_from(visible)
        graph.add_edges_from((edge["source"], edge["target"]) for edge in links)
        key = (tuple(visible), tuple(sorted(graph.edges)))
        if key != getattr(self, "layout_key", None):
            self.layout_key, self.layout = key, graph_positions(graph)
        layout = self.layout
        width, height = max(200, self.canvas.winfo_width()), max(80, self.canvas.winfo_height())
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
        self.description.set(description)
        self.details()

    def details(self):
        self.detail_text.delete("1.0", "end")
        if self.technical.get() and self.inspected:
            self.detail_text.pack(fill="x", pady=4)
            self.detail_text.insert("end", json.dumps(self.inspected, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            self.detail_text.pack_forget()

    def refresh(self):
        try:
            clock_path = self.directory / "clock.json"
            self.clock = json.loads(clock_path.read_text(encoding="utf-8"))
            status = self.clock.get("status")
            if status == "closing":
                self.root.destroy()
                return
            frame_path = self.directory / "frame.json"
            if frame_path.exists():
                stamp = frame_path.stat().st_mtime_ns
                if stamp != self.frame_stamp:
                    self.frame_stamp = stamp
                    self.frame = json.loads(frame_path.read_text(encoding="utf-8"))
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
                self.camera_label.configure(text=message)
            labels = self.clock.get("labels")
            if labels:
                mode = {"selected": "Sélection", "all": "Toutes", "none": "Aucune"}[labels["mode"]]
                self.labels_label.configure(text=f"Étiquettes {mode} : {labels['enabled']} activées / {labels['present']} voitures."
                                           + (" Zoomer ou choisir un alias pour les autres." if labels["mode"] == "all" and labels["suppressed"] else ""))
            if self.frame:
                source = self.frame["snapshot"]
                self.count_label.configure(text=f"{len(self.frame['groups'])} groupes · {source['vehicle_count']} voitures · {source['resource_count']} ressources · {source['edge_count']} relations (à {observed:g} s)")
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
