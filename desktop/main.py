"""
desktop/main.py

Desktop application shell for the SAA dashboard.

Runs the Streamlit server as a private child process bound to loopback on a
port nobody else is using, then renders it in a native OS window via WebView2
(Chromium) on Windows, WebKit on macOS/Linux. No browser, no URL, no visible
localhost — it opens and behaves like an application.

Why this shape rather than a rewrite in a desktop toolkit: the analysis layer
is the product, and it is thousands of lines of pandas and scipy that must run
in CPython. Any genuinely native UI would either reimplement the whole
dashboard or shell out to Python anyway. Embedding a real browser engine keeps
one implementation of every view while still giving a real window, a taskbar
entry, and a clean shutdown.

Design decisions that matter for reliability:

  * The port is chosen by binding to port 0 and asking the OS what it got, so
    two copies can run at once and a stale process never blocks startup.
  * The server is started with `--server.headless true` so Streamlit never
    tries to open a browser behind the window, and with its usage-stats and
    file-watcher features off, which are pointless in a packaged app.
  * The child is launched in its own process group and terminated on close.
    Streamlit spawns a server thread that ignores a plain terminate on
    Windows, so the shutdown path escalates to the process tree.
  * Startup polls the health endpoint rather than sleeping a fixed interval,
    because a cold run has to import pandas, sklearn and statsmodels before it
    can answer, and that takes anywhere from two to twenty seconds depending
    on the machine and whether the import cache is warm.
"""

from __future__ import annotations

import argparse
import atexit
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "dashboard" / "app.py"

WINDOW_TITLE = "Strategic Asset Allocation"
MIN_SIZE = (1100, 720)
DEFAULT_SIZE = (1480, 940)
STARTUP_TIMEOUT = 180  # seconds; a cold first run also fetches price history


def _free_port() -> int:
    """Ask the OS for an unused loopback port instead of guessing one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# Orphan prevention
# ---------------------------------------------------------------------------

def _attach_to_job(proc: subprocess.Popen) -> object | None:
    """
    Put the server in a Windows job object that kills it when this process
    exits, however this process exits.

    The graceful paths (window closed, exception, atexit) already call
    stop_server. This covers the ungraceful ones -- Task Manager "End task",
    a crash, a hard power-state change -- where no Python cleanup runs at all.
    Without it an orphaned Streamlit server keeps running, holds its port, and
    silently consumes memory with no window attached to it.

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE ties the job's lifetime to this
    process's last handle, so the OS does the cleanup rather than trusting the
    app to. Returns the handle, which the caller must keep alive -- letting it
    be garbage-collected would close the job and kill the server immediately.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_uint64),
                        ("WriteOperationCount", ctypes.c_uint64),
                        ("OtherOperationCount", ctypes.c_uint64),
                        ("ReadTransferCount", ctypes.c_uint64),
                        ("WriteTransferCount", ctypes.c_uint64),
                        ("OtherTransferCount", ctypes.c_uint64)]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                        ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9
        PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
                job, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info)):
            kernel32.CloseHandle(job)
            return None

        handle = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, proc.pid)
        if not handle:
            kernel32.CloseHandle(job)
            return None
        ok = kernel32.AssignProcessToJobObject(job, handle)
        kernel32.CloseHandle(handle)
        if not ok:
            kernel32.CloseHandle(job)
            return None
        return job
    except Exception:
        # Orphan prevention is a safety net, never a startup blocker.
        return None


def _server_command(port: int) -> list[str]:
    return [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false",
        "--global.developmentMode", "false",
    ]


def start_server(port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    # Streamlit's first-run prompt writes to a credentials file and blocks on
    # stdin; setting an empty email up front skips it in a packaged app.
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    creationflags = 0
    if sys.platform == "win32":
        # Own process group so the whole tree can be signalled, and no console
        # window flashing up behind the app.
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

    return subprocess.Popen(
        _server_command(port),
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )


def wait_for_server(port: int, proc: subprocess.Popen, timeout: int = STARTUP_TIMEOUT) -> bool:
    """
    Poll the health endpoint until the server answers. Returns False if the
    child dies first, so the caller can surface the real error rather than a
    blank window.
    """
    url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.4)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    """
    Terminate the server and everything it spawned.

    A plain terminate() is not enough on Windows: Streamlit's tornado loop runs
    in a thread that does not always honour it, which would leave an orphaned
    process holding the port. taskkill /T walks the tree.
    """
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=15)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _drain(proc: subprocess.Popen, sink: list[str]) -> None:
    """Keep the child's stdout pipe empty so it cannot deadlock on a full buffer."""
    try:
        for line in iter(proc.stdout.readline, b""):
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                sink.append(text)
                if len(sink) > 400:
                    del sink[:200]
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAA desktop application")
    parser.add_argument("--debug", action="store_true",
                        help="Show the server log and open developer tools.")
    parser.add_argument("--browser", action="store_true",
                        help="Serve to the default browser instead of a native window.")
    args = parser.parse_args(argv)

    if not APP.exists():
        print(f"Cannot find the dashboard at {APP}", file=sys.stderr)
        return 1

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"Starting the analysis engine on {url} ...")

    proc = start_server(port)
    atexit.register(stop_server, proc)
    # Kept in a local for the lifetime of main(): if this handle is collected,
    # the job closes and takes the server down with it.
    _job = _attach_to_job(proc)  # noqa: F841

    log: list[str] = []
    threading.Thread(target=_drain, args=(proc, log), daemon=True).start()

    if not wait_for_server(port, proc):
        stop_server(proc)
        print("The analysis engine failed to start.", file=sys.stderr)
        if log:
            print("\n--- server output ---", file=sys.stderr)
            print("\n".join(log[-40:]), file=sys.stderr)
        return 1

    print("Ready.")

    if args.browser:
        import webbrowser
        webbrowser.open(url)
        print("Serving in your browser. Press Ctrl+C to stop.")
        try:
            proc.wait()
        except KeyboardInterrupt:
            pass
        finally:
            stop_server(proc)
        return 0

    try:
        import webview
    except ImportError:
        stop_server(proc)
        print("pywebview is not installed. Install it with:\n"
              "    pip install pywebview\n"
              "or run with --browser to use your default browser instead.",
              file=sys.stderr)
        return 1

    window = webview.create_window(
        WINDOW_TITLE, url,
        width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1],
        min_size=MIN_SIZE,
        confirm_close=False,
        text_select=True,          # analysts copy numbers out of tables
        background_color="#FCFCFB",  # matches the dashboard surface, so no white flash
    )

    def _on_closed() -> None:
        stop_server(proc)

    window.events.closed += _on_closed

    try:
        webview.start(debug=args.debug)
    finally:
        stop_server(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
