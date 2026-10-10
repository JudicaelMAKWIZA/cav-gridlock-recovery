import cav_recovery
from pathlib import Path
import tomllib


def test_package_importable():
    assert cav_recovery is not None


def test_install_pins_the_sumo_client_and_declares_one_entrypoint():
    config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert {"eclipse-sumo==1.27.1", "traci==1.27.1", "sumolib==1.27.1",
            "sumo-data==1.27.1", "pyproj==3.7.2"}.issubset(config["project"]["dependencies"])
    assert config["project"]["scripts"] == {"cgr": "cav_recovery.cli:main"}
