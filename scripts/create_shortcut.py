"""
scripts/create_shortcut.py

Put the SAA app on the Desktop and in the Start menu.

    python scripts/create_shortcut.py            # Desktop + Start menu
    python scripts/create_shortcut.py --remove   # take them away again

The shortcut points at pythonw.exe in this repo's virtual environment, so
launching it never shows a console window and never depends on which Python
happens to be on PATH. A generated .ico gives it a real icon rather than the
default Python one.

Windows only. On macOS or Linux, run `python -m desktop.main` (or add
`desktop/main.py` to your launcher of choice) -- pywebview uses WebKit there
and the app behaves the same.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "Strategic Asset Allocation"
ICON = ROOT / "desktop" / "saa.ico"


# ---------------------------------------------------------------------------
# Icon
# ---------------------------------------------------------------------------

def _png(size: int) -> bytes:
    """
    A small PNG drawn in pure Python: a rounded square in the dashboard's
    primary blue with a rising three-bar chart in white.

    Generated rather than committed as a binary blob so the icon is
    inspectable, matches the app palette by construction, and can be
    regenerated at any size without a design tool in the loop.
    """
    bg = (0x2A, 0x78, 0xD6)
    fg = (0xFC, 0xFC, 0xFB)
    s = size
    px = [[(0, 0, 0, 0) for _ in range(s)] for _ in range(s)]

    r = max(2, s // 6)  # corner radius
    for y in range(s):
        for x in range(s):
            # Rounded-rectangle mask.
            cx = min(max(x, r), s - 1 - r)
            cy = min(max(y, r), s - 1 - r)
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                px[y][x] = (*bg, 255)

    # Three ascending bars, inset from the edges.
    pad = max(2, s // 5)
    inner = s - 2 * pad
    bar_w = max(1, inner // 4)
    gap = max(1, (inner - 3 * bar_w) // 2)
    heights = [0.42, 0.68, 1.0]
    for i, h in enumerate(heights):
        x0 = pad + i * (bar_w + gap)
        bh = int(inner * h)
        y0 = s - pad - bh
        for y in range(y0, s - pad):
            for x in range(x0, min(x0 + bar_w, s - pad)):
                if 0 <= x < s and 0 <= y < s and px[y][x][3]:
                    px[y][x] = (*fg, 255)

    raw = b"".join(b"\x00" + b"".join(bytes(px[y][x]) for x in range(s)) for y in range(s))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", s, s, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def build_icon(path: Path = ICON) -> Path:
    """Write a multi-resolution .ico containing PNG-compressed frames."""
    sizes = [16, 32, 48, 64, 128, 256]
    images = [_png(sz) for sz in sizes]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for sz, img in zip(sizes, images):
        entries += struct.pack("<BBBBHHII", sz % 256, sz % 256, 0, 0, 1, 32, len(img), offset)
        blobs += img
        offset += len(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + entries + blobs)
    return path


# ---------------------------------------------------------------------------
# Shortcuts
# ---------------------------------------------------------------------------

def _target() -> tuple[str, str]:
    """(executable, arguments) for the shortcut."""
    pythonw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    if not pythonw.exists():
        pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        pythonw = Path(sys.executable)
    return str(pythonw), "-m desktop.main"


def _locations() -> list[Path]:
    import os

    desktop = Path(os.path.expanduser("~")) / "Desktop"
    # OneDrive-redirected Desktop is the common case on this machine.
    onedrive = os.environ.get("OneDrive")
    if onedrive and (Path(onedrive) / "Desktop").exists():
        desktop = Path(onedrive) / "Desktop"
    start = (Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows"
             / "Start Menu" / "Programs")
    out = [p / f"{NAME}.lnk" for p in (desktop, start) if p.exists()]
    return out


def create() -> list[Path]:
    if sys.platform != "win32":
        raise SystemExit("Shortcut creation is Windows-only. "
                         "On macOS/Linux run: python -m desktop.main")
    try:
        import win32com.client  # type: ignore
    except ImportError:
        raise SystemExit(
            "pywin32 is required to create shortcuts:\n"
            "    pip install pywin32\n"
            "Alternatively just double-click 'SAA Dashboard.bat' in the repo root.")

    import time

    icon = build_icon()
    exe, args = _target()
    shell = win32com.client.Dispatch("WScript.Shell")

    def _write(path: Path) -> bool:
        lnk = shell.CreateShortCut(str(path))
        lnk.TargetPath = exe
        lnk.Arguments = args
        lnk.WorkingDirectory = str(ROOT)
        lnk.IconLocation = str(icon)
        lnk.Description = "Strategic asset allocation model and dashboard"
        lnk.WindowStyle = 7  # minimised: nothing to show, the app opens its own window
        lnk.save()
        return path.exists()

    made, failed = [], []
    for path in _locations():
        # WScript.Shell's save() reports nothing and occasionally does nothing --
        # an antivirus or search-indexer lock on the Start-menu folder makes the
        # first write vanish silently. Verify the file landed and retry once,
        # rather than printing a success the user will not find.
        ok = False
        for attempt in range(2):
            try:
                ok = _write(path)
            except Exception:
                ok = False
            if ok:
                break
            time.sleep(0.6)
        (made if ok else failed).append(path)

    if failed:
        print("Could not create (try again, or use 'SAA Dashboard.bat'):", file=sys.stderr)
        for p in failed:
            print(f"  {p}", file=sys.stderr)
    return made


def remove() -> list[Path]:
    gone = []
    for path in _locations():
        if path.exists():
            path.unlink()
            gone.append(path)
    return gone


def main() -> int:
    ap = argparse.ArgumentParser(description="Create or remove the SAA app shortcuts.")
    ap.add_argument("--remove", action="store_true", help="Remove the shortcuts instead.")
    ap.add_argument("--icon-only", action="store_true", help="Just regenerate desktop/saa.ico.")
    a = ap.parse_args()

    if a.icon_only:
        print(f"Wrote {build_icon()}")
        return 0
    if a.remove:
        gone = remove()
        print("Removed:" if gone else "Nothing to remove.")
        for p in gone:
            print(f"  {p}")
        return 0

    made = create()
    if not made:
        print("Could not find a Desktop or Start-menu folder to write to.")
        return 1
    print("Created:")
    for p in made:
        print(f"  {p}")
    print("\nLaunch it from the Desktop or the Start menu. No console window, no localhost URL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
