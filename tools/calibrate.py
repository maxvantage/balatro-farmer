"""Screenshot the Balatro window for coordinate calibration.

``config.json`` stores click points as normalized (0..1) positions inside the
window's client rect, so they survive a window resize. This tool captures the
window and can overlay either a labelled grid (to read coordinates off) or the
currently configured points (to verify them).

Usage:
    python tools/calibrate.py --grid --name blind_select
    python tools/calibrate.py --overlay --name verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from farmer.vision import grab  # noqa: E402
from farmer.window import BalatroWindow  # noqa: E402

OUT_DIR = ROOT / "logs" / "calibrate"

GRID_COLOR = (0, 220, 255)
POINT_COLOR = (0, 255, 0)


def draw_grid(img, step: float = 0.05, label_every: int = 2):
    h, w = img.shape[:2]
    n = int(round(1 / step))
    for i in range(n + 1):
        frac = i * step
        x, y = int(frac * w), int(frac * h)
        heavy = i % label_every == 0
        thickness = 1
        shade = GRID_COLOR if heavy else (90, 90, 90)
        cv2.line(img, (x, 0), (x, h), shade, thickness)
        cv2.line(img, (0, y), (w, y), shade, thickness)
        if heavy:
            cv2.putText(img, f"{frac:.2f}", (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, GRID_COLOR, 1, cv2.LINE_AA)
            cv2.putText(img, f"{frac:.2f}", (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, GRID_COLOR, 1, cv2.LINE_AA)
    return img


def draw_points(img, coords: dict):
    h, w = img.shape[:2]
    for name, value in coords.items():
        if not isinstance(value, list) or len(value) != 2:
            continue
        x, y = int(value[0] * w), int(value[1] * h)
        cv2.drawMarker(img, (x, y), POINT_COLOR, cv2.MARKER_CROSS, 26, 2)
        cv2.circle(img, (x, y), 16, POINT_COLOR, 2)
        cv2.putText(img, name, (x + 20, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, POINT_COLOR, 2, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="shot")
    ap.add_argument("--grid", action="store_true", help="overlay a labelled 0.05 grid")
    ap.add_argument("--overlay", action="store_true", help="overlay configured coords")
    ap.add_argument("--config", type=Path, default=ROOT / "config.json")
    ap.add_argument("--settle", type=float, default=0.6,
                    help="seconds to wait after focusing before capturing")
    args = ap.parse_args()

    window = BalatroWindow.find()
    # Balatro runs borderless full-screen here, so anything else on top would be
    # captured instead of the game. Focus it and let the compositor settle.
    if not window.focus():
        print("warning: could not focus Balatro; the capture may be occluded")
    time.sleep(args.settle)

    rect = window.client_rect()
    print(f"window: {window!r}  client={rect}")

    frame = grab(rect)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = OUT_DIR / f"{args.name}_raw.png"
    cv2.imwrite(str(raw), frame)
    print(f"wrote {raw}")

    if args.grid or args.overlay:
        marked = frame.copy()
        if args.grid:
            marked = draw_grid(marked)
        if args.overlay:
            coords = json.loads(args.config.read_text(encoding="utf-8"))["coords"]
            marked = draw_points(marked, coords)
        out = OUT_DIR / f"{args.name}_marked.png"
        cv2.imwrite(str(out), marked)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
