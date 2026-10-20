import cav_recovery
from pathlib import Path
import tomllib


def test_package_importable():
    assert cav_recovery is not None


def test_install_pins_the_engine_without_legacy_runtime():
    config = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert "sumo" not in config["project"]["optional-dependencies"]
    assert config["project"]["dependencies"] == ["networkx==3.7"]
    assert config["project"]["scripts"] == {"cgr": "cav_recovery.cli:main"}
