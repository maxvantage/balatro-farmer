"""Detector checks against real captured frames.

The synthetic test (``test_detector.py``) proved the matcher works on clean
composites. This one uses actual screenshots from a live Charm Tag pack, which is
what caught the two real bugs:

* the on-screen cards render 1.58x the template's native height, outside any
  fixed scale sweep -- so scale is now derived from window height
* searching the whole window scored 0.46 on the animated purple backdrop, close
  enough to a real threshold to matter -- so the search is confined to the card row

The known-positive is produced by compositing the game's own Soul sprite into a
real Soul-free pack at the real measured card size. Waiting for a genuine
1-in-500 Soul to validate this would be absurd.

Run:  .venv/Scripts/python.exe tests/test_detector_live.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from farmer.vision import (  # noqa: E402
    ASSETS,
    PackGeometry,
    SoulFinder,
    count_cards,
    find_use_button,
    identify_pack,
)

FIXTURES = ROOT / "tests" / "fixtures"

# Real captures, kept as fixtures so these tests do not depend on run output.
SETTLED = FIXTURES / "pack_settled.png"        # 5 tarots dealt, no Soul
MID_TEAR = FIXTURES / "pack_unsettled.png"     # dealt but still materializing
CLICKED = FIXTURES / "pack_card_selected.png"  # a card selected, USE showing

# Card boxes measured live at 2560x1599 (x, y, w, h).
CARD_BOXES = [
    (811, 969, 201, 303),
    (1055, 974, 201, 301),
    (1293, 979, 220, 309),
    (1542, 972, 206, 296),
    (1783, 970, 205, 298),
]


def composite_soul(frame: np.ndarray, slot: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Paste the real Soul sprite over one card slot at true card size.

    The sprite is cropped to its opaque art first: the atlas cell carries
    transparent padding, and pasting the padded cell would render a Soul narrower
    than a real card, making this an easier target than the live case.
    """
    sprite = cv2.imread(str(ASSETS / "soul_2x.png"), cv2.IMREAD_UNCHANGED)
    if sprite.shape[2] == 4:
        alpha = sprite[:, :, 3]
        ys, xs = np.where(alpha > 10)
        sprite = sprite[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        sprite = cv2.cvtColor(sprite, cv2.COLOR_BGRA2BGR)
    x, y, w, h = CARD_BOXES[slot]
    out = frame.copy()
    out[y : y + h, x : x + w] = cv2.resize(sprite, (w, h), interpolation=cv2.INTER_AREA)
    return out, (x + w // 2, y + h // 2)


def main() -> int:
    for path in (SETTLED, MID_TEAR, CLICKED):
        if not path.exists():
            print(f"missing capture {path}; run tools/calibrate.py during a pack first")
            return 2

    settled = cv2.imread(str(SETTLED))
    mid_tear = cv2.imread(str(MID_TEAR))
    clicked = cv2.imread(str(CLICKED))

    # Use the thresholds the bot will actually run with, not library defaults.
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    det, pack_cfg = cfg["detector"], cfg["pack"]
    geom = PackGeometry(
        region=tuple(det["region"]),
        card_height_frac=float(det["card_height_frac"]),
        scale_tolerance=float(det["scale_tolerance"]),
        card_brightness=int(det["card_brightness"]),
    )
    finder = SoulFinder(geom, threshold=float(det["threshold"]))
    failures: list[str] = []

    h = settled.shape[0]
    print(f"window height {h}; expected card {geom.card_size(h)}; "
          f"scales {finder.scales_for(h)[0]:.2f}..{finder.scales_for(h)[-1]:.2f}")
    print(f"threshold {finder.threshold}\n")

    # -- readiness signal ------------------------------------------------
    # The unsettled frame is *expected* to pass the blob count -- that is exactly
    # why counting blobs was not a sufficient readiness test. The identification
    # gate further down is what actually rejects it.
    n_settled = count_cards(settled, geom)
    n_tear = count_cards(mid_tear, geom)
    print(f"count_cards: settled={n_settled} (want 5)   unsettled={n_tear} "
          f"(also 5 -- why the count alone is not enough)")
    if n_settled != 5:
        failures.append(f"settled pack counted {n_settled} cards, expected 5")

    # -- negative: a real Soul-free pack --------------------------------
    neg = finder.search(settled)
    neg_score = neg.score if neg else 0.0
    print(f"\nreal Soul-free pack: {neg_score:.3f}  (must stay below threshold)")
    if neg_score >= finder.threshold:
        failures.append(f"Soul-free pack scored {neg_score:.3f}, at/above threshold")

    # -- positive: real frame + real sprite, every slot ------------------
    print("\nknown-positive, Soul composited into each slot:")
    print(f"  {'slot':>4} {'score':>7} {'located':>8}")
    for slot in range(5):
        img, center = composite_soul(settled, slot)
        m = finder.search(img)
        score = m.score if m else 0.0
        located = False
        if m is not None:
            located = (abs(m.center[0] - center[0]) < 60
                       and abs(m.center[1] - center[1]) < 60)
        print(f"  {slot:>4} {score:>7.3f} {str(located):>8}")
        if score < finder.threshold:
            failures.append(f"slot {slot}: Soul scored {score:.3f}, below threshold")
        if not located:
            failures.append(f"slot {slot}: best match was not on the Soul card")

    margin = min(
        finder.search(composite_soul(settled, s)[0]).score for s in range(5)
    ) - neg_score
    print(f"\nworst-case separation: {margin:.3f}")
    if margin < 0.15:
        failures.append(f"separation {margin:.3f} is too thin to trust")

    # -- naming every card in the pack -----------------------------------
    slots, boxes = identify_pack(settled, geom)
    got = [s.name if s else "?" for s in slots]
    expected = ["The Chariot", "The Moon", "The Emperor", "The Hierophant", "Death"]
    print(f"\nidentified: {', '.join(got)}")
    if got != expected:
        failures.append(f"pack identified as {got}, expected {expected}")

    # The Soul must win its slot wherever it sits.
    print("Soul identification by slot:")
    for slot in range(5):
        img, _ = composite_soul(settled, slot)
        ids, _ = identify_pack(img, geom)
        s = ids[slot] if slot < len(ids) else None
        ok = s is not None and s.is_soul
        print(f"  slot {slot}: {s.name if s else '?':<22} "
              f"score={s.score if s else 0:.3f} margin={s.margin if s else 0:.3f}")
        if not ok:
            failures.append(f"slot {slot}: Soul identified as {s.name if s else None}")
        elif s.margin < 0.15:
            failures.append(f"slot {slot}: Soul margin only {s.margin:.3f}")

    # -- readiness gate ---------------------------------------------------
    # Two animations to survive. Cards dissolve in when dealt (a frame then has 5
    # card-shaped blobs whose faces are not drawn), and every card carries a
    # permanent ambient_tilt so even settled cards never hold still -- one slot
    # measured 0.285-0.748 across 40 live frames. A per-slot gate would therefore
    # reject good packs on an unlucky frame, so the gate leans on the mean.
    MIN_SLOT = float(pack_cfg["min_slot_score"])
    MIN_MEAN = float(pack_cfg["min_mean_slot_score"])

    def gate(scans) -> tuple[bool, float, float]:
        vals = [s.score for s in scans if s]
        if len(vals) < 5:
            return False, 0.0, 0.0
        avg, low = sum(vals) / len(vals), min(vals)
        return (avg >= MIN_MEAN and low >= MIN_SLOT), avg, low

    tear_slots, _ = identify_pack(mid_tear, geom)
    ok_settled, avg_s, low_s = gate(slots)
    ok_tear, avg_t, low_t = gate(tear_slots)
    print(f"\nreadiness gate (mean>={MIN_MEAN}, slot>={MIN_SLOT}):")
    print(f"  settled    mean={avg_s:.3f} min={low_s:.3f} -> {'PASS' if ok_settled else 'REJECT'}")
    print(f"  unsettled  mean={avg_t:.3f} min={low_t:.3f} -> {'PASS' if ok_tear else 'REJECT'}")
    if not ok_settled:
        failures.append(f"settled pack (mean {avg_s:.3f}) would be rejected by the gate")
    if ok_tear:
        failures.append(f"unsettled frame (mean {avg_t:.3f}) would pass the gate")
    if avg_s - avg_t < 0.25:
        failures.append(f"gate separation only {avg_s-avg_t:.3f}; too thin")

    # A tilted Soul scores ~0.10-0.14 below a flat composite (measured live). Even
    # with the worst observed penalty applied, it must still win its slot clearly.
    TILT_PENALTY = 0.135
    worst_margin = min(
        identify_pack(composite_soul(settled, s)[0], geom)[0][s].margin for s in range(5)
    )
    print(f"\nSoul margin worst-case {worst_margin:.3f}; "
          f"minus measured tilt penalty {TILT_PENALTY} -> {worst_margin-TILT_PENALTY:.3f}")
    if worst_margin - TILT_PENALTY < 0.20:
        failures.append(
            f"Soul margin {worst_margin:.3f} leaves only "
            f"{worst_margin-TILT_PENALTY:.3f} after tilt"
        )

    # -- USE button ------------------------------------------------------
    use = find_use_button(clicked, geom)
    none_use = find_use_button(settled, geom)
    print(f"\nfind_use_button: clicked={use} (want ~(1398,1251))  "
          f"unclicked={none_use} (want None)")
    if use is None or abs(use[0] - 1398) > 40 or abs(use[1] - 1251) > 40:
        failures.append(f"USE button located at {use}, expected near (1398,1251)")
    if none_use is not None:
        failures.append(f"found a USE button at {none_use} on a frame with no selection")

    print("\n" + "-" * 52)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all live-frame detector checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
