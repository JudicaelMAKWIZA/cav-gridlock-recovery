"""Demande le redessin natif SUMO sous X11 et limite les étiquettes affichées."""

import ctypes as C
import ctypes.util
import os
from pathlib import Path


def visible_labels(aliases, readings, selected, mode, boundary=None, size=None):
    """Les sélectionnés passent avant les autres ; masquer un label ne retire pas sa voiture."""
    if mode not in ("selected", "all", "none"):
        raise ValueError("Mode d'étiquettes inconnu.")
    chosen = set(selected).intersection(readings)
    if mode == "none":
        return set()
    if boundary is None or size is None:
        return chosen
    (left, bottom), (right, top) = boundary
    width, height = size
    if right <= left or top <= bottom or width <= 0 or height <= 0:
        return chosen
    scale = min(width / (right - left), height / (top - bottom))
    cx, cy = (left + right) / 2, (bottom + top) / 2
    result, boxes = set(), []
    items = chosen if mode == "selected" else readings
    for item in sorted(items, key=lambda v: (v not in chosen, aliases[v])):
        x, y = readings[item]["position"]
        x, y = width / 2 + (x - cx) * scale, height / 2 - (y - cy) * scale
        if not (0 <= x <= width and 0 <= y <= height):
            continue
        # Marge visuelle conservatrice pour le texte natif ; pas une distance de sécurité.
        half = (len(aliases[item]) * 9 + 6) / 2
        box = (x - half, y - 12, x + half, y + 12)
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            continue
        if item in chosen or not any(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1] for b in boxes):
            result.add(item)
            boxes.append(box)
    return result


class ExposeEvent(C.Structure):
    _fields_ = [("type", C.c_int), ("serial", C.c_ulong), ("send_event", C.c_int),
                ("display", C.c_void_p), ("window", C.c_ulong), ("x", C.c_int), ("y", C.c_int),
                ("width", C.c_int), ("height", C.c_int), ("count", C.c_int)]


class Event(C.Union):
    _fields_ = [("expose", ExposeEvent), ("pad", C.c_long * 24)]


class Attributes(C.Structure):
    _fields_ = [(name, C.c_int) for name in ("x", "y", "width", "height", "border_width", "depth")]
    _fields_ += [("visual", C.c_void_p), ("root", C.c_ulong), ("class_", C.c_int),
                 ("bit_gravity", C.c_int), ("win_gravity", C.c_int), ("backing_store", C.c_int),
                 ("backing_planes", C.c_ulong), ("backing_pixel", C.c_ulong), ("save_under", C.c_int),
                 ("colormap", C.c_ulong), ("map_installed", C.c_int), ("map_state", C.c_int),
                 ("all_event_masks", C.c_long), ("your_event_mask", C.c_long),
                 ("do_not_propagate_mask", C.c_long), ("override_redirect", C.c_int), ("screen", C.c_void_p)]


class NativeView:
    """Cible uniquement la fenêtre SUMO appartenant au processus de ce run."""

    def __init__(self, pid):
        if os.name != "posix" or not os.environ.get("DISPLAY"):
            raise RuntimeError("Redessin X11 indisponible ; utiliser les contrôles natifs SUMO.")
        library = C.util.find_library("X11")
        if not library:
            raise RuntimeError("Bibliothèque X11 indisponible.")
        self.x = C.CDLL(library)
        x = self.x
        x.XOpenDisplay.argtypes, x.XOpenDisplay.restype = [C.c_char_p], C.c_void_p
        x.XDefaultRootWindow.argtypes, x.XDefaultRootWindow.restype = [C.c_void_p], C.c_ulong
        x.XQueryTree.argtypes = [C.c_void_p, C.c_ulong, C.POINTER(C.c_ulong), C.POINTER(C.c_ulong), C.POINTER(C.POINTER(C.c_ulong)), C.POINTER(C.c_uint)]
        x.XFetchName.argtypes = [C.c_void_p, C.c_ulong, C.POINTER(C.c_void_p)]
        x.XInternAtom.argtypes, x.XInternAtom.restype = [C.c_void_p, C.c_char_p, C.c_int], C.c_ulong
        x.XGetWindowProperty.argtypes = [C.c_void_p, C.c_ulong, C.c_ulong, C.c_long, C.c_long, C.c_int, C.c_ulong,
                                        C.POINTER(C.c_ulong), C.POINTER(C.c_int), C.POINTER(C.c_ulong), C.POINTER(C.c_ulong), C.POINTER(C.c_void_p)]
        x.XGetWindowAttributes.argtypes = [C.c_void_p, C.c_ulong, C.POINTER(Attributes)]
        x.XSendEvent.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_long, C.POINTER(Event)]
        x.XFree.argtypes = [C.c_void_p]
        x.XFlush.argtypes = x.XCloseDisplay.argtypes = [C.c_void_p]
        x.XSync.argtypes = [C.c_void_p, C.c_int]
        x.XSetErrorHandler.argtypes, x.XSetErrorHandler.restype = [C.c_void_p], C.c_void_p
        self.display = x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError("Affichage X11 absent.")
        self.x_error = False
        def on_error(display, error):
            self.x_error = True
            return 0
        self._error_handler = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_void_p)(on_error)
        self._old_handler = x.XSetErrorHandler(C.cast(self._error_handler, C.c_void_p))
        try:
            owners, commands, pending = set(), set(), [pid]
            while pending:
                parent = pending.pop()
                if parent in owners:
                    continue
                owners.add(parent)
                command = Path(f"/proc/{parent}/cmdline")
                if command.exists():
                    commands.add(tuple(command.read_bytes().rstrip(b"\0").split(b"\0")))
                children = Path(f"/proc/{parent}/task/{parent}/children")
                if children.exists():
                    pending.extend(map(int, children.read_text().split()))
            root = x.XDefaultRootWindow(self.display)
            windows = []
            for outer in self.children(root):
                for window in [outer, *self.children(outer)]:
                    if self.title(window).endswith("SUMO 1.27.1") and self.command(window) in commands:
                        windows.append(window)
            if len(windows) != 1:
                raise RuntimeError("Fenêtre SUMO de ce run non identifiée sans ambiguïté.")
            self.window = windows[0]
            self.viewport = self.find_viewport(self.window)
        except Exception:
            self.close()
            raise

    def children(self, window):
        root, parent, values, count = C.c_ulong(), C.c_ulong(), C.POINTER(C.c_ulong)(), C.c_uint()
        if not self.x.XQueryTree(self.display, window, C.byref(root), C.byref(parent), C.byref(values), C.byref(count)):
            return []
        try:
            return list(values[:count.value])
        finally:
            if values:
                self.x.XFree(values)

    def title(self, window):
        value = C.c_void_p()
        if not self.x.XFetchName(self.display, window, C.byref(value)) or not value:
            return ""
        try:
            return C.string_at(value).decode("utf-8", errors="replace")
        finally:
            self.x.XFree(value)

    def command(self, window):
        atom = self.x.XInternAtom(self.display, b"WM_COMMAND", 1)
        kind, form, count, remaining, value = C.c_ulong(), C.c_int(), C.c_ulong(), C.c_ulong(), C.c_void_p()
        self.x.XGetWindowProperty(self.display, window, atom, 0, 65536, 0, 0,
                                 C.byref(kind), C.byref(form), C.byref(count), C.byref(remaining), C.byref(value))
        try:
            return tuple(C.string_at(value, count.value).rstrip(b"\0").split(b"\0")) if value and form.value == 8 else None
        finally:
            if value:
                self.x.XFree(value)

    def dimensions(self, window):
        attributes = Attributes()
        if not self.x.XGetWindowAttributes(self.display, window, C.byref(attributes)):
            raise RuntimeError("La vue SUMO n'est plus disponible.")
        return attributes.width, attributes.height

    def find_viewport(self, window):
        pending, leaves = [window], []
        while pending:
            item = pending.pop()
            children = self.children(item)
            if children:
                pending.extend(children)
            else:
                leaves.append(item)
        chosen = max(leaves, key=lambda item: self.dimensions(item)[0] * self.dimensions(item)[1])
        width, height = self.dimensions(chosen)
        if width < 300 or height < 200:
            raise RuntimeError("Vue routière SUMO non identifiée.")
        return chosen

    @property
    def size(self):
        return self.dimensions(self.viewport)

    def redraw(self):
        width, height = self.size
        event = Event()
        event.expose = ExposeEvent(12, 0, 1, self.display, self.viewport, 0, 0, width, height, 0)
        if not self.x.XSendEvent(self.display, self.viewport, 0, 1 << 15, C.byref(event)):
            raise RuntimeError("Redessin SUMO non transmis.")
        self.x.XFlush(self.display)
        self.x.XSync(self.display, 0)
        if self.x_error:
            raise RuntimeError("La fenêtre SUMO a refusé le redessin X11.")

    def close(self):
        if self.display:
            self.x.XCloseDisplay(self.display)
            self.display = None
            self.x.XSetErrorHandler(self._old_handler)
