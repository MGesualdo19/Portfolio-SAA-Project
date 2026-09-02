"""
tests/test_desktop_shell.py

Tests for the desktop shell's process management, which is where a packaged
app actually breaks: a port collision, a server that never comes up, or an
orphaned child still holding the port after the window closes.

The window itself is not tested here -- creating one needs a display and would
block. Everything underneath it is.
"""

from __future__ import annotations

import socket
import sys
import time
import urllib.request

import pytest

from desktop.main import _free_port, _server_command, start_server, stop_server, wait_for_server


def test_free_port_is_actually_free():
    port = _free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))  # would raise if the port were taken


def test_free_port_varies():
    """Two launches must not collide, so a second copy can run alongside the first."""
    ports = {_free_port() for _ in range(5)}
    assert len(ports) > 1


def test_server_command_is_headless_and_loopback_only():
    """
    A desktop app must not open a browser behind its own window, and must not
    expose the analysis server on the network -- this dashboard renders a real
    portfolio.
    """
    cmd = _server_command(12345)
    joined = " ".join(cmd)
    assert "--server.headless true" in joined
    assert "--server.address 127.0.0.1" in joined
    assert "--server.port 12345" in joined
    assert "gatherUsageStats false" in joined
    assert cmd[0] == sys.executable


@pytest.mark.network
@pytest.mark.slow
def test_server_starts_answers_and_stops_cleanly():
    """
    End-to-end on the process layer: the server comes up, answers health, and
    releases the port when stopped. A leaked child here means the app cannot
    be reopened after being closed.
    """
    port = _free_port()
    proc = start_server(port)
    try:
        assert wait_for_server(port, proc, timeout=180), "server did not become healthy"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=10) as r:
            assert r.status == 200
            assert b"ok" in r.read()
    finally:
        stop_server(proc)

    # The port must be reusable immediately after shutdown.
    for _ in range(30):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
            break
        except OSError:
            time.sleep(1)
    else:
        pytest.fail("port was still held after stop_server -- the child leaked")


def test_wait_for_server_reports_failure_when_the_child_dies():
    """
    If the engine crashes on startup the shell must return rather than hang,
    so the user sees the real error instead of an empty window.
    """
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        assert wait_for_server(_free_port(), proc, timeout=30) is False
    finally:
        stop_server(proc)


def test_stop_server_is_safe_on_an_already_dead_process():
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "pass"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    proc.wait()
    stop_server(proc)  # must not raise
