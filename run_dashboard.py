"""
run_dashboard.py

Launcher for the local SAA dashboard.

    python run_dashboard.py

Starts Streamlit on http://localhost:8501 and opens a browser. This
wrapper exists so the app can be started without remembering the
streamlit invocation, and so the repo root is on sys.path before the
`core` and `analysis` packages are imported.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "dashboard" / "app.py"
URL = "http://localhost:8501"


def main() -> int:
    if not APP.exists():
        print(f"Cannot find {APP}", file=sys.stderr)
        return 1

    # headless=true skips Streamlit's first-run email prompt, which otherwise
    # blocks startup on a fresh machine. The browser is opened here instead,
    # so the behaviour the user sees is unchanged.
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP),
           "--server.port", "8501", "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]

    print(f"Starting the SAA dashboard on {URL} ...")
    print("First run fetches price history and takes a minute or two.")
    print("Press Ctrl+C to stop.\n")

    threading.Timer(4.0, lambda: webbrowser.open(URL)).start()
    try:
        return subprocess.call(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
