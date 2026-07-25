"""Typed, cached views over Balatro's save files.

The bot makes every one of its decisions from these reads rather than from the
screen. Two facts from the game's Lua make that possible:

* ``save_run()`` fires on entry to BLIND_SELECT, and it serializes
  ``GAME.round_resets.blind_tags`` -- which holds *both* the Small and Big blind
  skip tags. So "is there a Charm Tag?" is a file read, not a screenshot.
* ``GAME.pseudorandom.seed`` changes on every restart, giving an exact signal for
  "the new run has begun" instead of a guessed sleep.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .jkr import JkrError, read_jkr

__all__ = [
    "STATES",
    "SAVE_DIR",
    "RunState",
    "SaveWatcher",
    "MetaWatcher",
    "TARGET_JOKER",
    "CHARM_TAG",
]

# From globals.lua: G.STATES
STATES = {
    "SELECTING_HAND": 1,
    "HAND_PLAYED": 2,
    "DRAW_TO_HAND": 3,
    "GAME_OVER": 4,
    "SHOP": 5,
    "PLAY_TAROT": 6,
    "BLIND_SELECT": 7,
    "ROUND_EVAL": 8,
    "TAROT_PACK": 9,
    "PLANET_PACK": 10,
    "MENU": 11,
    "TUTORIAL": 12,
    "SPLASH": 13,
    "SANDBOX": 14,
    "SPECTRAL_PACK": 15,
    "DEMO_CTA": 16,
    "STANDARD_PACK": 17,
    "BUFFOON_PACK": 18,
    "NEW_ROUND": 19,
}

# States in which a booster pack is on screen and awaiting a pick.
PACK_STATES = frozenset(
    {
        STATES["TAROT_PACK"],
        STATES["PLANET_PACK"],
        STATES["SPECTRAL_PACK"],
        STATES["STANDARD_PACK"],
        STATES["BUFFOON_PACK"],
    }
)

CHARM_TAG = "tag_charm"
TARGET_JOKER = "j_yorick"

SAVE_DIR = Path(os.environ["APPDATA"]) / "Balatro"


def profile_dir(profile: int = 1) -> Path:
    return SAVE_DIR / str(profile)


@dataclass(frozen=True)
class RunState:
    """A snapshot of the parts of ``save.jkr`` the bot cares about."""

    seed: str
    state: int
    ante: int
    blind_tags: dict[str, str]
    blind_states: dict[str, str]
    seeded: bool
    challenge: bool
    joker_keys: tuple[str, ...]

    @property
    def at_blind_select(self) -> bool:
        return self.state == STATES["BLIND_SELECT"]

    @property
    def in_pack(self) -> bool:
        return self.state in PACK_STATES

    # A blind can still be skipped only while it is upcoming or on deck.
    SKIPPABLE = frozenset({"Select", "Upcoming"})

    @property
    def charm_slots(self) -> list[str]:
        """Blinds ('Small'/'Big') offering a Charm Tag we can still claim.

        ``blind_tags`` keeps reporting a tag after it has been consumed, so the
        blind's state has to be checked too -- otherwise, resuming on a run whose
        blinds are already skipped would try to click a dead button.
        """
        return [
            b
            for b in ("Small", "Big")
            if self.blind_tags.get(b) == CHARM_TAG
            and self.blind_states.get(b) in self.SKIPPABLE
        ]

    def describe_tags(self) -> str:
        small = self.blind_tags.get("Small", "?").removeprefix("tag_")
        big = self.blind_tags.get("Big", "?").removeprefix("tag_")
        return f"Small={small:<12} Big={big}"


class _FileWatcher:
    """Re-parses a save file only when its mtime/size changes.

    Balatro writes saves from a background thread, so a read can land mid-write
    and fail to inflate. Callers get a retry loop rather than an exception.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stamp: tuple[float, int] | None = None
        self._cached: dict[str, Any] | None = None

    def _stat(self) -> tuple[float, int] | None:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return None
        return (st.st_mtime, st.st_size)

    def raw(self, *, force: bool = False, retries: int = 6) -> dict[str, Any] | None:
        """Return the parsed table, or None if the file is missing."""
        stamp = self._stat()
        if stamp is None:
            return None
        if not force and stamp == self._stamp and self._cached is not None:
            return self._cached
        for attempt in range(retries):
            try:
                data = read_jkr(self.path)
            except (JkrError, OSError):
                # Mid-write; back off briefly and try again.
                time.sleep(0.03 * (attempt + 1))
                continue
            self._stamp = self._stat()
            self._cached = data
            return data
        return self._cached


def _joker_keys(save: dict[str, Any]) -> tuple[str, ...]:
    """Centre keys of the Jokers currently held, e.g. ``('j_yorick',)``.

    Used to confirm a Soul actually resolved into a Legendary: ``meta.jkr`` only
    changes for a *newly* discovered Joker, so it cannot tell us anything when the
    Soul rolls one we already own.
    """
    cards = (((save.get("cardAreas") or {}).get("jokers") or {}).get("cards")) or {}
    keys: list[str] = []
    for _, card in sorted(cards.items(), key=lambda kv: kv[0]):
        if not isinstance(card, dict):
            continue
        center = (card.get("save_fields") or {}).get("center")
        if isinstance(center, str):
            keys.append(center)
    return tuple(keys)


class SaveWatcher(_FileWatcher):
    """Reads ``save.jkr`` -- the live run."""

    def __init__(self, profile: int = 1) -> None:
        super().__init__(profile_dir(profile) / "save.jkr")

    def read(self, *, force: bool = False) -> RunState | None:
        data = self.raw(force=force)
        if not data:
            return None
        game = data.get("GAME") or {}
        resets = game.get("round_resets") or {}
        pseudo = game.get("pseudorandom") or {}
        seed = pseudo.get("seed")
        if not isinstance(seed, str):
            return None
        return RunState(
            joker_keys=_joker_keys(data),
            seed=seed,
            state=int(data.get("STATE") or 0),
            ante=int(resets.get("ante") or 0),
            blind_tags={
                k: v for k, v in (resets.get("blind_tags") or {}).items() if isinstance(v, str)
            },
            blind_states={
                k: v for k, v in (resets.get("blind_states") or {}).items() if isinstance(v, str)
            },
            seeded=bool(game.get("seeded")),
            challenge=bool(game.get("challenge")),
        )


class MetaWatcher(_FileWatcher):
    """Reads ``meta.jkr`` -- the permanent collection, i.e. our win condition."""

    def __init__(self, profile: int = 1) -> None:
        super().__init__(profile_dir(profile) / "meta.jkr")

    def discovered_jokers(self) -> set[str]:
        data = self.raw(force=True) or {}
        discovered = data.get("discovered") or {}
        return {k for k, v in discovered.items() if k.startswith("j_") and v}

    def has_target(self, key: str = TARGET_JOKER) -> bool:
        return key in self.discovered_jokers()
