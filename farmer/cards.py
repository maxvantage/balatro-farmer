"""Identifying every card in a booster pack by name.

Rather than only asking "is The Soul here?" against one threshold, this identifies
all five cards against the pool of what an Arcana Pack can actually contain (the
22 Tarots, plus The Soul). That buys two things:

* **A much stronger Soul signal.** Picking the best of 23 candidates per slot is a
  relative decision, not an absolute cutoff, so it does not hinge on a threshold
  being tuned correctly. A missed Soul costs ~500 resets, so having two
  independent signals that must agree is worth the small cost.
* **A real audit trail.** Every pack's contents get logged by name, so after the
  run you can confirm no Soul was ever passed over -- without re-examining
  screenshots by eye.

Sprites and positions both come from the game's own files, so the catalogue cannot
drift out of sync with the installed version.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

__all__ = ["Card", "SlotID", "load_catalog", "ARCANA_POOL", "identify_slot"]

BALATRO_EXE = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Balatro\Balatro.exe")

GRID_COLS, GRID_ROWS = 10, 6
ATLAS = "resources/textures/{scale}x/Tarots.png"

# Matching ignores the outer border/rounded corners, which the game redraws.
_INSET = 0.10

# A card definition line in game.lua, e.g.
#   c_soul= {order = 17, ... name = "The Soul", pos = {x=2,y=2}, set = "Spectral", ...}
_DEF = re.compile(r"^\s*(c_[a-z0-9_]+)\s*=\s*\{([^\n]*)$", re.M)


@dataclass(frozen=True)
class Card:
    key: str
    name: str
    col: int
    row: int
    card_set: str
    art: np.ndarray  # BGR, cropped to the opaque art bounds

    @property
    def is_soul(self) -> bool:
        return self.key == "c_soul"


@dataclass(frozen=True)
class SlotID:
    """Identification of a single pack slot."""

    key: str
    name: str
    score: float
    runner_up: str
    runner_up_score: float

    @property
    def margin(self) -> float:
        return self.score - self.runner_up_score

    @property
    def is_soul(self) -> bool:
        return self.key == "c_soul"

    @property
    def confident(self) -> bool:
        """Whether this identification is clear-cut enough to trust silently."""
        return self.score >= 0.45 and self.margin >= 0.04

    def __str__(self) -> str:
        flag = "" if self.confident else "?"
        return f"{self.name}{flag}"


def _crop_to_art(cell: np.ndarray) -> np.ndarray:
    """Trim an atlas cell to its opaque art and drop alpha.

    The cell is 142x190 but the art is only ~126x186; keeping the transparent
    padding would distort the aspect ratio when scaled to an on-screen card.
    """
    if cell.shape[2] == 4:
        ys, xs = np.where(cell[:, :, 3] > 10)
        if len(xs):
            cell = cell[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        cell = cv2.cvtColor(cell, cv2.COLOR_BGRA2BGR)
    return cell


@lru_cache(maxsize=4)
def load_catalog(scale: int = 2, exe: Path = BALATRO_EXE) -> tuple[Card, ...]:
    """Every consumable sprite in Tarots.png, keyed and named from game.lua."""
    with zipfile.ZipFile(exe) as z:
        lua = z.read("game.lua").decode("utf-8", "replace")
        atlas = cv2.imdecode(
            np.frombuffer(z.read(ATLAS.format(scale=scale)), np.uint8),
            cv2.IMREAD_UNCHANGED,
        )

    ch, cw = atlas.shape[0] // GRID_ROWS, atlas.shape[1] // GRID_COLS
    cards: list[Card] = []
    for match in _DEF.finditer(lua):
        key, body = match.group(1), match.group(2)
        name = re.search(r'name\s*=\s*"([^"]+)"', body)
        pos = re.search(r"pos\s*=\s*\{x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\}", body)
        card_set = re.search(r'set\s*=\s*"(\w+)"', body)
        if not (name and pos and card_set):
            continue
        col, row = int(pos.group(1)), int(pos.group(2))
        cell = atlas[row * ch : (row + 1) * ch, col * cw : (col + 1) * cw]
        cards.append(
            Card(key, name.group(1), col, row, card_set.group(1), _crop_to_art(cell))
        )
    return tuple(cards)


def ARCANA_POOL(scale: int = 2) -> tuple[Card, ...]:
    """What a Mega Arcana Pack can contain: the Tarots, plus The Soul."""
    return tuple(
        c for c in load_catalog(scale) if c.card_set == "Tarot" or c.is_soul
    )


def _inset(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    dx, dy = int(w * _INSET), int(h * _INSET)
    return img[dy : h - dy, dx : w - dx]


def identify_slot(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    pool: tuple[Card, ...],
    pad_frac: float = 0.09,
) -> SlotID | None:
    """Name the card in one slot by best correlation across the pool.

    The candidate is matched *within a padded region* around the slot rather than
    compared to an exactly-cropped one. Slot boxes come from a brightness blob, and
    a card's bright area does not perfectly coincide with its rectangle -- two
    different masks disagreed by 13 px on the same frame, and with no translation
    tolerance that alone dropped a correct score from 0.74 to 0.30. Since The Soul's
    art is much brighter than a tarot's, its blob is exactly the case most likely to
    sit a few pixels off.
    """
    ih, iw = image.shape[:2]
    x, y, w, h = box
    pad_x, pad_y = int(w * pad_frac), int(h * pad_frac)
    region = image[
        max(0, y - pad_y) : min(ih, y + h + pad_y),
        max(0, x - pad_x) : min(iw, x + w + pad_x),
    ]
    tw, th = int(w * (1 - 2 * _INSET)), int(h * (1 - 2 * _INSET))
    if tw < 8 or th < 8 or region.shape[0] < th or region.shape[1] < tw:
        return None

    scored: list[tuple[float, Card]] = []
    for card in pool:
        art = cv2.resize(_inset(card.art), (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(region, art, cv2.TM_CCOEFF_NORMED)
        scored.append((float(result.max()), card))

    scored.sort(key=lambda s: -s[0])
    (best_score, best), (second_score, second) = scored[0], scored[1]
    return SlotID(best.key, best.name, best_score, second.name, second_score)
