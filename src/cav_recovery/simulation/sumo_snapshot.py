"""Enregistre un état SUMO et ses observations, sans faire avancer le trafic."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .road_network import write_xml


def save_scene(connection, current, readings, directory, focus, depth, provenance):
    """Garde l'état natif au même instant pour vérifier ou reproduire une preuve."""
    if connection.simulation.getTime() != current["time_s"]:
        raise ValueError("Le graphe et la scène doivent avoir le même instant.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    connection.simulation.saveState(str(directory / "state.xml.gz"))
    if connection.simulation.getTime() != current["time_s"]:
        raise ValueError("Le temps a changé pendant la sauvegarde.")
    data = {"snapshot": current, "readings": readings, "provenance": provenance,
            "state_sha256": hashlib.sha256((directory / "state.xml.gz").read_bytes()).hexdigest(),
            "focus": focus, "depth": depth}
    (directory / "scene.json").write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding="utf-8")
    root = ET.Element("configuration")
    for group, values in {"input": {"net-file": "../../network.net.xml", "route-files": "../../traffic.rou.xml",
                                  "load-state": "state.xml.gz"},
                          "time": {"begin": str(current["time_s"]), "end": str(current["time_s"]), "step-length": "0.5"},
                          "processing": {"time-to-teleport": "-1", "collision.check-junctions": "true"},
                          "gui_only": {"gui-settings-file": "../../view.xml"}}.items():
        parent = ET.SubElement(root, group)
        for key, value in values.items():
            ET.SubElement(parent, key, value=value)
    write_xml(directory / "scene.sumocfg", root)
    return {"time_s": current["time_s"], "focus": focus,
            "focus_present": focus in readings if focus else None}
