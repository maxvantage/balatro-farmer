"""Finding The Soul on screen.

This is the *only* part of the bot that looks at pixels. Tag detection comes from
the save file; pack contents cannot, because ``save_run()`` runs before a skip tag
is applied -- verified live: with a pack open on screen, ``save.jkr`` still read
``STATE=7`` and contained no ``pack_cards`` area at all.

The template is Balatro's own art, sliced out of the texture atlas inside
Balatro.exe (see ``tools/extract_soul_template.py``), so it is pixel-exact.

Two things make the match reliable rather than a guess:

* **Scale is derived, not swept blindly.** On a 2560x1599 window the pack cards
  render 205x301 px, i.e. 1.58x the template's native height -- outside any
  plausible fixed sweep. Expressing card height as a fraction of window height
  and computing the scale from the live frame makes this resolution-independent.
* **The search is confined to the pack card row.** Searching the whole window
  invited spurious small-scale matches on the animated purple backdrop (those
  scored ~0.46, uncomfortably close to a real threshold).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .window import Rect

__all__ = [
    "Match",
    "PackGeometry",
    "SoulFinder",
    "grab",
    "annotate",
    "count_cards",
    "find_card_slots",
    "identify_pack",
]

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# The sprite has transparent rounded corners and a border the game redraws;
# matching only the inner face avoids both.
_INSET = 0.10

# Aspect ratio of a Balatro card face (width / height), measured live.
CARD_ASPECT = 205 / 301


@dataclass(frozen=True)
class PackGeometry:
    """Where pack cards sit, as fractions of the window client area.

    Defaults measured from a real Mega Arcana Pack at 2560x1599. Because every
    value is a fraction, they hold at other window sizes too.
    """

    # The top edge deliberately excludes the played-hand row above the pack: its
    # white playing cards are bright enough to bridge the gaps between pack cards
    # and merge all five into a single blob.
    region: tuple[float, float, float, float] = (0.27, 0.57, 0.83, 0.86)
    card_height_frac: float = 0.1882
    scale_tolerance: float = 0.18
    card_brightness: int = 175
    # Identification matches at reduced resolution: 4x faster, no accuracy loss.
    match_scale: float = 0.5

    def pixel_region(self, width: int, height: int) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.region
        return (
            max(0, int(x0 * width)),
            max(0, int(y0 * height)),
            min(width, int(x1 * width)),
            min(height, int(y1 * height)),
        )

    def card_size(self, height: int) -> tuple[int, int]:
        card_h = self.card_height_frac * height
        return int(round(card_h * CARD_ASPECT)), int(round(card_h))


@dataclass(frozen=True)
class Match:
    score: float
    center: tuple[int, int]  # full-frame pixel coords
    scale: float
    size: tuple[int, int]

    def screen_center(self, rect: Rect) -> tuple[int, int]:
        return (rect.left + self.center[0], rect.top + self.center[1])


def grab(rect: Rect) -> np.ndarray:
    """Screenshot a screen rect as a BGR array."""
    import mss  # imported lazily; only needed when we actually look at the screen

    with mss.mss() as sct:
        shot = sct.grab(rect.as_mss())
    return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)


def _inset_art(img: np.ndarray) -> np.ndarray:
    """Drop alpha and inset past the drawn border and rounded corners."""
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[:2]
    dx, dy = int(w * _INSET), int(h * _INSET)
    return img[dy : h - dy, dx : w - dx]


class SoulFinder:
    """Sliding-window search for The Soul across the pack region.

    Matches against the whole bank of animated composites (see ``cards``), because
    the plain atlas cell misses the overlay that dominates the rendered card. This
    is the backstop signal; per-slot identification is the primary one.
    """

    def __init__(
        self,
        geometry: PackGeometry | None = None,
        threshold: float = 0.50,
        templates: tuple | None = None,
        scale_steps: int = 3,
    ) -> None:
        from .cards import soul_variants  # local: avoids an import cycle

        self.geometry = geometry or PackGeometry()
        self.templates = tuple(
            _inset_art(t) for t in (templates or soul_variants())
        )
        self.threshold = threshold
        self.scale_steps = scale_steps

    def scales_for(self, frame_height: int) -> np.ndarray:
        """Scale factors to try, centred on the geometry's implied card size."""
        _, card_h = self.geometry.card_size(frame_height)
        implied = (card_h * (1 - 2 * _INSET)) / self.templates[0].shape[0]
        tol = self.geometry.scale_tolerance
        return np.linspace(implied * (1 - tol), implied * (1 + tol), self.scale_steps)

    def search(self, image: np.ndarray) -> Match | None:
        """Best match inside the pack region; centre is in full-frame coords."""
        ih, iw = image.shape[:2]
        x0, y0, x1, y1 = self.geometry.pixel_region(iw, ih)
        roi = image[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        best: Match | None = None
        rh, rw = roi.shape[:2]
        for scale in self.scales_for(ih):
            for template in self.templates:
                th, tw = template.shape[:2]
                w, h = int(tw * scale), int(th * scale)
                if w < 12 or h < 12 or w > rw or h > rh:
                    continue
                resized = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
                result = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if best is None or max_val > best.score:
                    best = Match(
                        score=float(max_val),
                        center=(x0 + max_loc[0] + w // 2, y0 + max_loc[1] + h // 2),
                        scale=float(scale),
                        size=(w, h),
                    )
        return best

    def find(self, image: np.ndarray) -> Match | None:
        best = self.search(image)
        return best if best is not None and best.score >= self.threshold else None


def find_card_slots(
    image: np.ndarray, geometry: PackGeometry | None = None
) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of the dealt pack cards, left to right.

    Cards are much brighter than the dimmed purple backdrop. The brightness cutoff
    matters: too low and the white played-hand cards above bridge the gaps and
    merge all five into one blob (which is also why the region excludes that row).
    """
    geom = geometry or PackGeometry()
    ih, iw = image.shape[:2]
    x0, y0, x1, y1 = geom.pixel_region(iw, ih)
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return []

    card_w, card_h = geom.card_size(ih)
    expected = card_w * card_h

    mask = (roi.max(axis=2) > geom.card_brightness).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # Constrain the box *shape*, not just its area. Filtering on area and height
        # alone let a merged blob through as a card -- one measured 347x372 where a
        # card is 205x301. The crop was then misaligned, every candidate scored ~0.25,
        # and c_soul won that slot on noise by a margin of 0.033: a false Soul that
        # stopped a five-hour run. A card's silhouette is a known shape; use it.
        if not 0.55 * card_w <= w <= 1.45 * card_w:
            continue
        if not 0.6 * card_h <= h <= 1.4 * card_h:
            continue
        if not 0.35 * expected <= area <= 1.6 * expected:
            continue
        boxes.append((x0 + int(x), y0 + int(y), int(w), int(h)))
    return sorted(boxes, key=lambda b: b[0])


def count_cards(image: np.ndarray, geometry: PackGeometry | None = None) -> int:
    """How many card-sized bright blobs are sitting in the pack row.

    Used as a readiness signal: a Charm Tag's pack plays a tear-open animation
    before dealing, and an early screenshot catches the wrapper with no cards at
    all. Waiting on this instead of a fixed sleep removes that race -- and it
    counts The Soul too, whose face is bright blue/gold rather than tarot tan.
    """
    return len(find_card_slots(image, geometry))


def identify_pack(image: np.ndarray, geometry: PackGeometry | None = None):
    """Name every card in the open pack.

    Returns ``(slots, boxes)`` where ``slots[i]`` identifies ``boxes[i]``. A slot
    that could not be identified is ``None``. This is the bot's second, independent
    Soul signal -- picking the best of 23 candidates per slot does not depend on a
    threshold being tuned right, and it doubles as the logged record of what each
    pack actually contained.
    """
    from .cards import ARCANA_POOL, identify_slot  # local: avoids an import cycle

    geom = geometry or PackGeometry()
    boxes = find_card_slots(image, geom)
    pool = ARCANA_POOL()
    slots = [
        identify_slot(image, box, pool, match_scale=geom.match_scale) for box in boxes
    ]
    return slots, boxes


def annotate(
    image: np.ndarray,
    match: Match | None,
    label: str = "",
    geometry: PackGeometry | None = None,
) -> np.ndarray:
    """Draw the search region and match box, for audit screenshots."""
    out = image.copy()
    ih, iw = out.shape[:2]
    x0, y0, x1, y1 = (geometry or PackGeometry()).pixel_region(iw, ih)
    cv2.rectangle(out, (x0, y0), (x1, y1), (120, 120, 120), 1)

    if match is not None:
        w, h = match.size
        cx, cy = match.center
        tl = (cx - w // 2, cy - h // 2)
        br = (cx + w // 2, cy + h // 2)
        cv2.rectangle(out, tl, br, (0, 255, 0), 3)
        text = f"{label} score={match.score:.3f} scale={match.scale:.2f}"
        cv2.putText(out, text, (tl[0], max(20, tl[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
    elif label:
        cv2.putText(out, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return out
