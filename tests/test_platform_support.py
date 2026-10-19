"""Native resource accounting and non-destructive process probes."""
import os
import subprocess
import sys
from cav_recovery.platform_support import available_memory_kib, peak_rss_kib, process_alive


def test_live_probe_does_not_terminate_its_target():
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.5)'])
    try:
        assert process_alive(child.pid)
        assert child.wait(timeout=10) == 0
        assert not process_alive(child.pid)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)


def test_accounting_reports_native_units_and_keeps_current_process_alive():
    assert process_alive(os.getpid()) and not process_alive(-1)
    assert peak_rss_kib() > 0
    available = available_memory_kib()
    assert available is None or available > 0
