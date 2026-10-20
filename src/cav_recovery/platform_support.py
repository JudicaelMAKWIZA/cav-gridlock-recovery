"""Mesures de ressources du système, sans décision de simulation ou de qualification."""

import os


def peak_rss_kib():
    if os.name == "nt":
        import psutil

        return psutil.Process().memory_info().peak_wset / 1024
    import resource
    import sys

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024 if sys.platform == "darwin" else value


def process_alive(pid):
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is not a harmless existence probe on Windows.
        import psutil

        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def available_memory_kib():
    if os.name == "nt":
        import psutil

        return psutil.virtual_memory().available // 1024
    from pathlib import Path

    memory = Path("/proc/meminfo")
    if memory.exists():
        return int(
            next(
                line.split()[1]
                for line in memory.read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
    return None
