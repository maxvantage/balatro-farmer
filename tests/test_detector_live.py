"""Detector checks against real captured frames.

Every threshold here came from measurement, and most of them came from a bug that
only showed up on real data:

* Cards render 1.58x the template's native height at 2560x1599 -- outside any fixed
  scale sweep. Scale is now derived from window height.
* The atlas cell is padded (142x190 cell, 126x186 art), which distorted the aspect
  ratio and capped scores near 0.70.
* Dealt cards materialize through a dissolve shader; such a frame still has five
  card-sized blobs, so blob count is not readiness.
* Every card carries a permanent ambient_tilt, so even settled cards never hold
  still -- boxes wander up to 111px and one slot ranged 0.285-0.748 over 40 frames.
* **The Soul is not one sprite.** It is Tarots(2,2) plus an Enhancers(0,1) overlay
  drawn with animated scale and +/-0.17 rad rotation. The first live run found two
  real Souls that scored only 0.448/0.520 against a flat template -- below the then
  0.55 threshold, so that signal missed BOTH, and only the argmax signal saved it.

Hence the positive cases below composite the *animated* Soul, not a flat sprite.
A flat composite is an unrealistically easy target and is what hid the bug.

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

from farmer.cards import ARCANA_POOL, _inset, identify_slot  # noqa: E402
from farmer.vision import (  # noqa: E402
    PackGeometry,
    SoulFinder,
    count_cards,
    find_card_slots,
    identify_pack,
)

FIXTURES = ROOT / "tests" / "fixtures"
SETTLED = FIXTURES / "pack_settled.png"        # 5 tarots dealt, no Soul
UNSETTLED = FIXTURES / "pack_unsettled.png"    # dealt but still materializing

EXPECTED_NAMES = ["The Chariot", "The Moon", "The Emperor", "The Hierophant", "Death"]


def load_cfg():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    det, pack = cfg["detector"], cfg["pack"]
    geom = PackGeometry(
        region=tuple(det["region"]),
        card_height_frac=float(det["card_height_frac"]),
        scale_tolerance=float(det["scale_tolerance"]),
        card_brightness=int(det["card_brightness"]),
        match_scale=float(det["match_scale"]),
    )
    return cfg, geom, det, pack


def paste(frame: np.ndarray, box, art: np.ndarray) -> np.ndarray:
    x, y, w, h = box
    out = frame.copy()
    bgr = art[:, :, :3] if art.shape[2] == 4 else art
    out[y : y + h, x : x + w] = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return out


def main() -> int:
    for path in (SETTLED, UNSETTLED):
        if not path.exists():
            print(f"missing fixture {path}")
            return 2

    cfg, geom, det, pack_cfg = load_cfg()
    settled = cv2.imread(str(SETTLED))
    unsettled = cv2.imread(str(UNSETTLED))

    pool = ARCANA_POOL()
    soul = next(c for c in pool if c.is_soul)
    finder = SoulFinder(geom, threshold=float(det["threshold"]))
    soul_floor = float(det["soul_score_threshold"])
    failures: list[str] = []

    h = settled.shape[0]
    print(f"window height {h}; expected card {geom.card_size(h)}")
    print(f"soul variants {len(soul.all_art)}; soul_floor {soul_floor}; "
          f"template threshold {finder.threshold}\n")

    # -- naming ----------------------------------------------------------
    slots, boxes = identify_pack(settled, geom)
    names = [s.name if s else "?" for s in slots]
    print(f"identified: {', '.join(names)}")
    if names != EXPECTED_NAMES:
        failures.append(f"named {names}, expected {EXPECTED_NAMES}")

    # -- negative --------------------------------------------------------
    neg_soul = max((s.soul_score for s in slots if s), default=0.0)
    neg_tmpl = (finder.search(settled) or type("x", (), {"score": 0.0})).score
    print(f"\nreal Soul-free pack: max soul score {neg_soul:.3f} (floor {soul_floor}), "
          f"template {neg_tmpl:.3f} (threshold {finder.threshold})")
    if neg_soul >= soul_floor:
        failures.append(f"Soul-free pack scored {neg_soul:.3f}, at/above the floor")
    if neg_tmpl >= finder.threshold:
        failures.append(f"Soul-free template score {neg_tmpl:.3f}, at/above threshold")
    if any(s and s.is_soul for s in slots):
        failures.append("a Soul was named in a pack that has none")

    # -- positive: ANIMATED soul, every slot, several animation phases ----
    print("\nanimated Soul composited into each slot (all variants):")
    print(f"  {'slot':>4} {'variant':>8} {'named':>7} {'score':>7} {'margin':>7} {'soul':>7}")
    worst_margin, worst_soul = 1.0, 1.0
    for slot in range(5):
        for vi in (0, 4, 9):  # extremes and middle of the animation bank
            img = paste(settled, boxes[slot], soul.all_art[vi])
            sid = identify_slot(img, boxes[slot], pool, match_scale=geom.match_scale)
            named = sid is not None and sid.is_soul
            score = sid.score if sid else 0.0
            margin = sid.margin if sid else 0.0
            sscore = sid.soul_score if sid else 0.0
            worst_margin = min(worst_margin, margin)
            worst_soul = min(worst_soul, sscore)
            if slot in (0, 4) and vi == 4:
                print(f"  {slot:>4} {vi:>8} {str(named):>7} {score:>7.3f} "
                      f"{margin:>7.3f} {sscore:>7.3f}")
            if not named:
                failures.append(f"slot {slot} variant {vi}: named {sid.name if sid else None}")
            if sscore < soul_floor:
                failures.append(
                    f"slot {slot} variant {vi}: soul score {sscore:.3f} below floor")
    print(f"  worst over all 15 cases: margin {worst_margin:.3f}, soul score {worst_soul:.3f}")
    if worst_margin < 0.20:
        failures.append(f"worst Soul margin only {worst_margin:.3f}")

    # -- regression: a flat template would have missed these -------------
    # This is the bug the first live run exposed; keep it measured, not assumed.
    flat_scores, bank_scores = [], []
    for slot in (0, 2, 4):
        for vi in (0, 4, 9):
            img = paste(settled, boxes[slot], soul.all_art[vi])
            x, y, w, h_ = boxes[slot]
            region = img[y : y + h_, x : x + w]
            tw, th = int(w * 0.8), int(h_ * 0.8)

            def sc(art):
                # Inset exactly as production does; without it the template's border
                # misaligns and both numbers collapse meaninglessly.
                t = cv2.resize(_inset(art), (tw, th), interpolation=cv2.INTER_AREA)
                return float(cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED).max())

            flat_scores.append(sc(soul.art))                       # plain atlas cell
            bank_scores.append(max(sc(a) for a in soul.all_art))   # composite bank
    print(f"\nflat atlas template on animated Souls: "
          f"min {min(flat_scores):.3f} mean {sum(flat_scores)/len(flat_scores):.3f}")
    print(f"composite bank on the same:            "
          f"min {min(bank_scores):.3f} mean {sum(bank_scores)/len(bank_scores):.3f}")
    if min(bank_scores) - min(flat_scores) < 0.15:
        failures.append("composite bank is not clearly better than the flat template")

    # -- readiness gate ---------------------------------------------------
    MIN_SLOT = float(pack_cfg["min_slot_score"])
    MIN_MEAN = float(pack_cfg["min_mean_slot_score"])

    def gate(scans):
        vals = [s.score for s in scans if s]
        if len(vals) < 5:
            return False, 0.0, 0.0
        avg, low = sum(vals) / len(vals), min(vals)
        return (avg >= MIN_MEAN and low >= MIN_SLOT), avg, low

    un_slots, _ = identify_pack(unsettled, geom)
    ok_s, avg_s, low_s = gate(slots)
    ok_u, avg_u, low_u = gate(un_slots)
    print(f"\nreadiness gate (mean>={MIN_MEAN}, slot>={MIN_SLOT}):")
    print(f"  settled    mean={avg_s:.3f} min={low_s:.3f} -> {'PASS' if ok_s else 'REJECT'}")
    print(f"  unsettled  mean={avg_u:.3f} min={low_u:.3f} -> {'PASS' if ok_u else 'REJECT'}")
    print(f"  count_cards: settled={count_cards(settled, geom)} "
          f"unsettled={count_cards(unsettled, geom)} (both 5 -- count alone is not enough)")
    if not ok_s:
        failures.append(f"settled pack (mean {avg_s:.3f}) would be rejected")
    if ok_u:
        failures.append(f"unsettled frame (mean {avg_u:.3f}) would pass the gate")
    if avg_s - avg_u < 0.25:
        failures.append(f"gate separation only {avg_s-avg_u:.3f}")


    print("\n" + "-" * 56)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all live-frame detector checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
