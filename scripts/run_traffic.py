"""Garde les anciennes commandes du dépôt sur le même pipeline que cgr."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cav_recovery.cli import main as cli_main
from cav_recovery.simulation.traffic_run import run_traffic


def main():
    return cli_main(legacy=True, runner=run_traffic)


if __name__ == "__main__":
    raise SystemExit(main())
