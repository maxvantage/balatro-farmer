"""Resolution-independence check for the Soul detector.

The live-frame test (``test_detector_live.py``) is the stronger evidence, but it
only covers the one window size we happened to capture. This one renders synthetic
Mega Arcana packs at several window sizes, laid out according to ``PackGeometry``,
and confirms the derived-scale search finds The Soul at each.

That matters because scale is *computed* from window height rather than swept:
the real bug this replaced was a fixed sweep topping out at 1.05x when the actual
card needed 1.58x. If the derivation is wrong for some resolution, this catches it.

Run:  .venv/Scripts/python.exe tests/test_detector.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from farmer.vision import PackGeometry, SoulFinder, count_cards  # noqa: E402
from tools.extract_soul_template import BALATRO_EXE, GRID_COLS, GRID_ROWS  # noqa: E402

SOUL_CELL = (2, 2)
DECOY_CELLS = [(0, 0), (1, 0), (3, 0), (4, 0), (5, 0), (6, 0), (0, 1), (3, 1)]

# Window sizes to exercise: 720p-ish, Matt's old reported size, 1200p, his actual.
WINDOWS = [(1280, 800), (1463, 914), (1920, 1200), (2560, 1599)]

# Vertical centre of the card row, measured live (1128/1599).
ROW_CENTER_Y = 0.7055


def load_atlas() -> np.ndarray:
    with zipfile.ZipFile(BALATRO_EXE) as z:
        data = z.read("resources/textures/2x/Tarots.png")
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def cell(atlas: np.ndarray, col: int, row: int) -> np.ndarray:
    ch = atlas.shape[0] // GRID_ROWS
    cw = atlas.shape[1] // GRID_COLS
    crop = atlas[row * ch : (row + 1) * ch, col * cw : (col + 1) * cw]
    if crop.shape[2] == 4:
        alpha = crop[:, :, 3]
        ys, xs = np.where(alpha > 10)
        if len(xs):
            crop = crop[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
    return crop


def build_pack(atlas, geom: PackGeometry, size, soul_slot, seed=0):
    """Render a 5-card pack laid out the way the game does."""
    rng = np.random.default_rng(seed)
    w, h = size
    # Balatro dims the board behind an open pack rather than blanking it.
    img = np.full((h, w, 3), 46, dtype=np.uint8)
    img[:, :, 0] = 84  # purplish backdrop
    img[:, :, 2] = 70
    img = np.clip(img.astype(int) + rng.integers(-14, 15, img.shape), 0, 255).astype(np.uint8)

    card_w, card_h = geom.card_size(h)
    gap = int(card_w * 0.19)
    total = 5 * card_w + 4 * gap
    x0 = (w - total) // 2
    y0 = int(ROW_CENTER_Y * h - card_h / 2)

    decoys = [DECOY_CELLS[i % len(DECOY_CELLS)] for i in rng.permutation(5)]
    soul_center = None
    for i in range(5):
        art = cell(atlas, *SOUL_CELL) if i == soul_slot else cell(atlas, *decoys[i])
        card = cv2.resize(art, (card_w, card_h), interpolation=cv2.INTER_AREA)
        x = x0 + i * (card_w + gap)
        img[y0 : y0 + card_h, x : x + card_w] = card
        if i == soul_slot:
            soul_center = (x + card_w // 2, y0 + card_h // 2)
    return img, soul_center


def main() -> int:
    atlas = load_atlas()
    geom = PackGeometry()
    finder = SoulFinder(geom)
    failures: list[str] = []

    print(f"threshold {finder.threshold}\n")
    print(f"{'window':>12} {'card':>10} {'cards':>6} {'soul':>7} {'none':>7} "
          f"{'margin':>7} {'located':>8}")
    print("-" * 66)

    for size in WINDOWS:
        w, h = size
        with_img, center = build_pack(atlas, geom, size, soul_slot=2, seed=h)
        without_img, _ = build_pack(atlas, geom, size, soul_slot=None, seed=h)

        hit = finder.search(with_img)
        miss = finder.search(without_img)
        hit_score = hit.score if hit else 0.0
        miss_score = miss.score if miss else 0.0
        n_cards = count_cards(with_img, geom)

        located = False
        if hit is not None and center is not None:
            tol = geom.card_size(h)[1] * 0.3
            located = (abs(hit.center[0] - center[0]) < tol
                       and abs(hit.center[1] - center[1]) < tol)

        card_w, card_h = geom.card_size(h)
        print(f"{w}x{h:<6} {card_w}x{card_h:<6} {n_cards:>6} {hit_score:>7.3f} "
              f"{miss_score:>7.3f} {hit_score - miss_score:>7.3f} {str(located):>8}")

        if hit_score < finder.threshold:
            failures.append(f"{w}x{h}: Soul scored {hit_score:.3f}, below threshold")
        if miss_score >= finder.threshold:
            failures.append(f"{w}x{h}: Soul-free pack scored {miss_score:.3f}, too high")
        if not located:
            failures.append(f"{w}x{h}: best match was not on the Soul card")
        if n_cards != 5:
            failures.append(f"{w}x{h}: count_cards saw {n_cards}, expected 5")

    print("-" * 66)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("detector is resolution-independent across all tested window sizes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
