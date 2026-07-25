"""The fast-reset farming loop.

Strategy (the standard community technique, automated):

    1. At Ante 1 blind select, read both skip tags straight out of ``save.jkr``.
    2. If neither is a Charm Tag, hold ``R`` to restart instantly and go again.
    3. If one is, skip into it. A Charm Tag grants a free Mega Arcana Pack.
    4. Look for The Soul among the 5 cards. If it's there, take it -- it creates
       a random Legendary Joker.
    5. Stop when ``meta.jkr`` reports the target Joker as discovered.

Seeded runs would make this trivial, but Balatro disables all unlocks and
discoveries on a seeded run, so brute force on random seeds is the only route.

Modes:
    observe    read-only; sends no input at all. Used to verify that parsed tags
               match what is actually on screen before trusting the automation.
    skip-only  drives the reset loop, but stops at the first Charm pack and just
               reports the detector's score. Calibration for the live run.
    live       the full unattended loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .gamestate import TARGET_JOKER, MetaWatcher, RunState, SaveWatcher
from .input import PanicAbort, click_screen, hold_key, move_screen, panic_pressed
from .vision import (
    Match,
    PackGeometry,
    SoulFinder,
    annotate,
    grab,
    identify_pack,
)
from .window import BalatroWindow, Rect, WindowNotFound

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
PACK_DIR = LOG_DIR / "packs"
# Soul evidence lives OUTSIDE logs/ on purpose. Routine pack shots are bulk and get
# cleared between runs; frames of an actual Soul are rare, hard-won, and were the
# only reason the two-sprite render bug was ever diagnosed. Clearing logs must not
# be able to destroy them.
SOUL_DIR = ROOT / "souls"

LEGENDARIES = ("j_caino", "j_triboulet", "j_yorick", "j_chicot", "j_perkeo")
# Canio's internal key really is "j_caino" -- a typo in the game's own source.
PRETTY = {
    "j_caino": "Canio",
    "j_triboulet": "Triboulet",
    "j_yorick": "Yorick",
    "j_chicot": "Chicot",
    "j_perkeo": "Perkeo",
}


class Stop(RuntimeError):
    """Raised to unwind the loop for any non-success reason."""


@dataclass(frozen=True)
class SoulOutcome:
    """What happened after The Soul was used."""

    target_unlocked: bool
    joker_keys: tuple[str, ...]
    resolved: bool  # False means we could not confirm anything either way


@dataclass
class PackScan:
    """Everything we concluded about one opened pack, from one frame."""

    frame: np.ndarray
    slots: list  # list[SlotID | None], left to right
    boxes: list  # list[(x, y, w, h)]
    match: Match | None  # best Soul template match
    ready: bool  # cards finished materializing before we read them

    @property
    def names(self) -> list[str]:
        return [str(s) if s else "?" for s in self.slots]

    @property
    def min_score(self) -> float:
        scores = [s.score for s in self.slots if s]
        return min(scores) if scores else 0.0

    @property
    def mean_score(self) -> float:
        scores = [s.score for s in self.slots if s]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def soul_slot(self) -> int | None:
        for i, s in enumerate(self.slots):
            if s and s.is_soul:
                return i
        return None

    @property
    def max_soul_score(self) -> float:
        """Best Soul correlation over all slots, whoever won each slot."""
        return max((s.soul_score for s in self.slots if s), default=0.0)

    @property
    def best_soul_slot(self) -> int | None:
        """Slot the Soul matched best, used when only the score signal fires."""
        scored = [(s.soul_score, i) for i, s in enumerate(self.slots) if s]
        return max(scored)[1] if scored else None

    @property
    def template_score(self) -> float:
        return self.match.score if self.match else 0.0

    def soul_point(self, rect: Rect) -> tuple[int, int]:
        """Where to click for The Soul, preferring the identified slot box."""
        idx = self.soul_slot
        if idx is None:
            idx = self.best_soul_slot
        if idx is not None and idx < len(self.boxes):
            x, y, w, h = self.boxes[idx]
            return (rect.left + x + w // 2, rect.top + y + h // 2)
        assert self.match is not None
        return self.match.screen_center(rect)


@dataclass
class Config:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "Config":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def coord(self, name: str) -> tuple[float, float] | None:
        value = self.raw["coords"].get(name)
        if value is None:
            return None
        return (float(value[0]), float(value[1]))

    def require_coord(self, name: str) -> tuple[float, float]:
        value = self.coord(name)
        if value is None:
            raise Stop(
                f"coords.{name} is not calibrated yet. Run tools/calibrate.py with "
                f"Balatro at the blind-select screen, then fill it into config.json."
            )
        return value


@dataclass
class Stats:
    resets: int = 0
    charm_packs: int = 0
    charm_lost: int = 0
    souls_found: int = 0
    suspicious: int = 0
    focus_recoveries: int = 0
    legendaries: list[str] = field(default_factory=list)
    cards_seen: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def summary(self) -> str:
        mins = (time.time() - self.started) / 60
        rate = self.resets / mins if mins else 0
        got = ", ".join(PRETTY.get(k, k) for k in self.legendaries) or "none"
        lines = [
            f"{self.resets} resets in {mins:.1f} min ({rate:.1f}/min)",
            f"charm packs: {self.charm_packs} | tarots seen: {len(self.cards_seen)}"
            f" | souls: {self.souls_found}",
            f"legendaries rolled: {got}",
        ]
        if self.suspicious or self.charm_lost:
            lines.append(
                f"flagged for audit: {self.suspicious} | charm packs lost: "
                f"{self.charm_lost}   (run tools/report.py --suspicious)"
            )
        if self.focus_recoveries:
            lines.append(f"focus recovered {self.focus_recoveries}x")
        return "\n".join(lines)


class Farmer:
    def __init__(self, cfg: Config, mode: str, target: str = TARGET_JOKER) -> None:
        self.cfg = cfg
        self.mode = mode
        self.target = target
        self.profile = int(cfg["profile"])
        self.save = SaveWatcher(self.profile)
        self.meta = MetaWatcher(self.profile)
        self.stats = Stats()
        self.window: BalatroWindow | None = None
        det = cfg["detector"]
        self.geometry = PackGeometry(
            region=tuple(det["region"]),
            card_height_frac=float(det["card_height_frac"]),
            scale_tolerance=float(det["scale_tolerance"]),
            card_brightness=int(det["card_brightness"]),
            match_scale=float(det["match_scale"]),
        )
        self.finder = SoulFinder(self.geometry, threshold=float(det["threshold"]))
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        PACK_DIR.mkdir(parents=True, exist_ok=True)
        SOUL_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / "run.jsonl"

    # -- infrastructure ---------------------------------------------------

    def log(self, **record: Any) -> None:
        record["t"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    @property
    def sends_input(self) -> bool:
        return self.mode != "observe"

    def rect(self) -> Rect:
        assert self.window is not None
        return self.window.client_rect()

    def guard(self) -> None:
        """Abort conditions checked before every action and every loop turn."""
        if panic_pressed():
            raise PanicAbort("panic key (F12) pressed")
        if self.sends_input:
            if self.window is None or not self.window.exists():
                raise Stop("Balatro window disappeared")
            if not self.window.is_foreground():
                self._recover_focus()
        limits = self.cfg["limits"]
        if self.stats.resets >= int(limits["max_resets"]):
            raise Stop(f"hit max_resets ({limits['max_resets']})")
        if (time.time() - self.stats.started) / 3600 >= float(limits["max_hours"]):
            raise Stop(f"hit max_hours ({limits['max_hours']})")

    def _recover_focus(self) -> None:
        """Regain foreground before continuing, or stop.

        Never clicks while unfocused either way -- but over a multi-hour unattended
        run a transient focus steal should not end the hunt, so try to take it back
        first and only give up if that fails.
        """
        assert self.window is not None
        timeout = float(self.cfg["timeouts"]["refocus"])
        self.log(event="focus_lost")
        print("        !! Balatro lost focus -- pausing and trying to reclaim it")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if panic_pressed():
                raise PanicAbort("panic key (F12) pressed")
            if self.window.focus(timeout=1.0):
                self.stats.focus_recoveries += 1
                self.log(event="focus_recovered")
                print("        -- focus reclaimed, continuing")
                time.sleep(float(self.cfg["timeouts"]["ui_settle"]))
                return
            time.sleep(1.0)
        raise Stop(
            f"Balatro stayed unfocused for {timeout:.0f}s -- stopping rather than "
            "clicking blind"
        )

    def click_norm(self, nx: float, ny: float) -> None:
        rect = self.rect()
        x, y = rect.norm_to_screen(nx, ny)
        click_screen(x, y, rect)

    def click_until(
        self,
        coord: tuple[float, float],
        predicate: Callable[[RunState], bool],
        what: str,
        attempts: int = 3,
        timeout: float | None = None,
    ) -> RunState:
        """Click, confirm the game state actually changed, and retry if not.

        The first live run needed a retry on *every single* skip click -- all 298,
        always succeeding on the second attempt. That looked like dropped clicks, but
        measuring it showed the truth: a skip takes **3.7-4.2s** to reach save.jkr
        (the tag animation blocks the event queue before ``save_run`` fires), and the
        wait was simply too short. So the timeout is generous and the retry is what
        it should have been all along -- a genuine safety net, not the normal path.
        """
        if timeout is None:
            timeout = float(self.cfg["timeouts"]["skip_confirm"])
        last: Stop | None = None
        for attempt in range(attempts):
            self.click_norm(*coord)
            try:
                return self.wait_until(predicate, timeout, what)
            except Stop as exc:
                last = exc
                self.log(event="click_retry", what=what, attempt=attempt + 1)
                time.sleep(0.3)
        raise last or Stop(f"could not confirm {what}")

    def wait_until(
        self,
        predicate: Callable[[RunState], bool],
        timeout: float,
        what: str,
    ) -> RunState:
        """Poll the save file until ``predicate`` holds. Never a bare sleep."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if panic_pressed():
                raise PanicAbort("panic key (F12) pressed")
            state = self.save.read()
            if state is not None and predicate(state):
                return state
            time.sleep(float(self.cfg["poll_interval"]))
        raise Stop(f"timed out after {timeout:.0f}s waiting for {what}")

    # -- run control ------------------------------------------------------

    def wait_for_new_run(self, previous_seed: str | None, timeout: float) -> RunState:
        def ready(s: RunState) -> bool:
            return (
                s.at_blind_select
                and s.seed != previous_seed
                and "Small" in s.blind_tags
                and "Big" in s.blind_tags
            )

        return self.wait_until(ready, timeout, "a new run at blind select")

    def restart(self, current_seed: str) -> RunState:
        """Hold R for a new run, with one recovery attempt.

        ``key_hold_update`` ignores R while the controller is locked (which it can
        be mid-pack), so a failed restart is expected occasionally rather than
        exceptional -- close the pack and try once more before giving up.
        """
        hold = float(self.cfg["restart_hold_seconds"])
        timeout = float(self.cfg["timeouts"]["new_run"])
        hold_key("r", hold)
        try:
            state = self.wait_for_new_run(current_seed, timeout)
        except Stop:
            pack_skip = self.cfg.coord("pack_skip")
            if pack_skip is None:
                raise
            self.log(event="restart_recovery", seed=current_seed)
            self.click_norm(*pack_skip)
            time.sleep(0.6)
            hold_key("r", hold)
            state = self.wait_for_new_run(current_seed, timeout)
        # save.jkr is written on entry to BLIND_SELECT, but the panels are still
        # animating in for a moment after that; clicking too early gets swallowed.
        time.sleep(float(self.cfg["timeouts"]["ui_settle"]))
        self.stats.resets += 1
        return state

    # -- charm pack handling ----------------------------------------------

    def skip_into_charm(self, state: RunState) -> None:
        """Skip blinds until the Charm Tag is claimed and its pack opens.

        Every click is verified against ``blind_states`` rather than assumed.
        """
        if "Small" not in state.charm_slots:
            # Charm is on the Big blind; burn the Small tag first. No Ante-1 tag
            # other than Charm opens a pack, so this is always safe and instant.
            self.click_until(
                self.cfg.require_coord("skip_small"),
                lambda s: s.blind_states.get("Small") == "Skipped"
                and s.blind_states.get("Big") == "Select",
                "Small blind to be skipped",
            )
            self.click_until(
                self.cfg.require_coord("skip_big"),
                lambda s: s.blind_states.get("Big") == "Skipped",
                "Big blind to be skipped (claiming Charm)",
            )
        else:
            self.click_until(
                self.cfg.require_coord("skip_small"),
                lambda s: s.blind_states.get("Small") == "Skipped",
                "Small blind to be skipped (claiming Charm)",
            )

    def _read_pack(self, rect: Rect) -> PackScan:
        """One frame's read. The sliding template search is deliberately skipped.

        Identification already yields a per-slot Soul correlation for free, which is
        the cheap second signal; the sliding search costs ~1s a frame and is run only
        to confirm an actual candidate (see ``_confirm_soul``).
        """
        frame = grab(rect)
        slots, boxes = identify_pack(frame, self.geometry)
        return PackScan(frame, slots, boxes, None, ready=False)

    def _confirm_soul(self, scan: PackScan) -> PackScan:
        """Run the independent sliding search on a Soul candidate."""
        scan.match = self.finder.search(scan.frame)
        return scan

    def _legible(self, scan: PackScan) -> bool:
        """Whether a frame is a trustworthy read of a settled pack.

        Counting card-shaped blobs is not sufficient: Balatro materializes dealt
        cards with a dissolve shader, and a mid-animation frame has five card-sized
        blobs whose faces are not drawn (measured: every slot 0.08-0.22).

        But every card also carries a permanent ``ambient_tilt`` that Card:draw
        recomputes each frame, so a *settled* card's score still swings a lot -- one
        slot measured 0.285 to 0.748 across 40 frames. Gating on "every slot above
        0.30" would therefore reject good packs on an unlucky frame. Hence a mean
        gate (settled ~0.60-0.74 vs unsettled ~0.13) plus a low per-slot floor that
        still sits above the unsettled ceiling.
        """
        cfg = self.cfg["pack"]
        if len(scan.boxes) < int(cfg["min_cards"]):
            return False
        if not all(s for s in scan.slots):
            return False
        return (
            scan.mean_score >= float(cfg["min_mean_slot_score"])
            and scan.min_score >= float(cfg["min_slot_score"])
        )

    def scan_pack(self) -> tuple[PackScan, PackScan | None, int]:
        """Read the pack over several frames.

        Returns ``(cleanest, soul_scan, frames_read)``. Because the cards are always
        in motion, the Soul verdict is taken across *every* frame sampled rather
        than from one chosen frame -- if any frame names a Soul, that frame is
        returned and used for the click. That is strictly more sensitive than
        picking one frame and hoping it was not the skewed one.
        """
        pack_cfg = self.cfg["pack"]
        timeout = float(self.cfg["timeouts"]["pack_ready"])
        rect = self.rect()

        cleanest: PackScan | None = None
        soul_scan: PackScan | None = None
        frames = 0
        ready = False

        soul_floor = float(self.cfg["detector"]["soul_score_threshold"])

        def consider(scan: PackScan) -> None:
            nonlocal cleanest, soul_scan
            if cleanest is None or scan.mean_score > cleanest.mean_score:
                cleanest = scan
            # Only a legible frame may nominate a Soul. Without this, a
            # mid-materialize frame where just one box was found and c_soul won on
            # noise (score 0.333, below the floor) reported a Soul -- two false
            # positives in 18 resets. A frame we would not trust to read the pack is
            # not a frame we should trust to spot the Soul.
            if not self._legible(scan):
                return
            candidate = scan.soul_slot is not None or scan.max_soul_score >= soul_floor
            if candidate and (
                soul_scan is None or scan.max_soul_score > soul_scan.max_soul_score
            ):
                soul_scan = scan

        deadline = time.time() + timeout
        while time.time() < deadline:
            if panic_pressed():
                raise PanicAbort("panic key (F12) pressed")
            scan = self._read_pack(rect)
            frames += 1
            consider(scan)
            if self._legible(scan):
                ready = True
                break
            time.sleep(0.15)

        # Keep sampling once settled: more frames, more chances to catch a Soul.
        for _ in range(max(0, int(pack_cfg["scan_frames"]) - 1)):
            time.sleep(float(pack_cfg["scan_interval"]))
            if panic_pressed():
                raise PanicAbort("panic key (F12) pressed")
            scan = self._read_pack(rect)
            frames += 1
            consider(scan)
            ready = ready or self._legible(scan)

        assert cleanest is not None
        cleanest.ready = ready
        if soul_scan is not None:
            soul_scan.ready = ready
            self._confirm_soul(soul_scan)
        return cleanest, soul_scan, frames

    def save_pack_shot(self, seed: str, scan: PackScan, *, full_res: bool) -> Path:
        """Save an audit shot of every Charm pack.

        A full hunt opens roughly 330 packs; full-frame 2560x1599 PNGs run about
        2.4 MB each, i.e. most of a gigabyte. So the routine case is cropped to the
        card row and halved (~100 KB), which is still legible enough to check by
        eye. Anything interesting -- a Soul, or a pack the reader was unsure about
        -- is kept at full resolution instead, and those are rare.
        """
        if full_res:
            # Keep an unannotated copy too: the overlays draw over the cards, which
            # would spoil the frame as evidence (and as a future template source).
            cv2.imwrite(str(SOUL_DIR / f"{seed}_clean.png"), scan.frame)

        marked = annotate(
            scan.frame, scan.match, label=seed, geometry=self.geometry
        )
        for i, (box, name) in enumerate(zip(scan.boxes, scan.names)):
            x, y, w, h = box
            cv2.rectangle(marked, (x, y), (x + w, y + h), (200, 200, 0), 1)
            cv2.putText(marked, name, (x, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 255, 0), 1, cv2.LINE_AA)

        ih, iw = marked.shape[:2]
        x0, y0, x1, y1 = self.geometry.pixel_region(iw, ih)
        pad_x, pad_y = int(0.03 * iw), int(0.07 * ih)
        crop = marked[
            max(0, y0 - pad_y) : min(ih, y1 + pad_y),
            max(0, x0 - pad_x) : min(iw, x1 + pad_x),
        ]
        if not full_res:
            crop = cv2.resize(crop, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        path = PACK_DIR / f"{seed}.png"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        return path

    def take_soul(self, point: tuple[int, int], seed: str) -> "SoulOutcome":
        """Click The Soul, Use it, and confirm what came out.

        Selecting a pack card raises it and puts a small red USE button at its
        lower edge. That button is located by colour rather than clicked at a fixed
        offset -- this is the one moment in the whole run that has to work, since a
        fumbled Soul costs roughly 500 resets to see another.

        Confirmation leans on ``meta.jkr`` rather than ``save.jkr``. A Mega Arcana
        Pack is "Choose 2", so the pack stays open after the Soul is used and
        ``save_run()`` does not fire -- but ``discover_card`` calls
        ``save_progress()`` immediately, so a newly discovered Joker lands in
        ``meta.jkr`` right away. The pack is closed partway through the wait to
        force a run save too, which is what tells us *which* Legendary we got when
        it is one already discovered.
        """
        rect = self.rect()
        cx, cy = point
        click_screen(cx, cy, rect)
        time.sleep(0.45)
        cv2.imwrite(str(SOUL_DIR / f"{seed}_selected.png"), grab(rect))

        self._click_use(rect, cx, cy)
        time.sleep(0.6)
        cv2.imwrite(str(SOUL_DIR / f"{seed}_used.png"), grab(rect))

        timeout = float(self.cfg["timeouts"]["soul_resolved"])
        pack_skip = self.cfg.coord("pack_skip")
        started = time.time()
        retried_use = False
        closed_pack = False
        jokers: tuple[str, ...] = ()

        while time.time() - started < timeout:
            if panic_pressed():
                raise PanicAbort("panic key (F12) pressed")
            if self.meta.has_target(self.target):
                state = self.save.read()
                return SoulOutcome(True, state.joker_keys if state else (), True)
            state = self.save.read()
            if state is not None and state.joker_keys:
                return SoulOutcome(False, state.joker_keys, True)

            elapsed = time.time() - started
            # Retry only after the save could plausibly have landed. Firing at 2s
            # meant every single Soul logged a spurious retry, because a run save
            # takes ~4s to appear. The card is re-selected first in case the original
            # selection was what failed.
            if not retried_use and elapsed > float(self.cfg["timeouts"]["use_retry"]):
                self.log(event="use_retry", seed=seed)
                click_screen(cx, cy, rect)
                time.sleep(0.4)
                self._click_use(rect, cx, cy)
                retried_use = True
            # Closing the pack forces save_run(), which reveals the new Joker even
            # when it is one already discovered.
            elif not closed_pack and pack_skip is not None and elapsed > 8.0:
                self.click_norm(*pack_skip)
                closed_pack = True
            time.sleep(float(self.cfg["poll_interval"]))

        return SoulOutcome(self.meta.has_target(self.target), jokers, False)

    def _click_use(self, rect: Rect, cx: int, cy: int) -> None:
        """Click the USE button, at a fixed offset below the selected card.

        Deliberately geometric rather than vision-based. Detecting the button by
        colour bought nothing: the Soul is *detected* entirely from the card row, so
        the button only matters for the click that follows -- and the calibrated
        offset has landed within 7-10px of the button centre on every real Soul so
        far, against a button roughly 106x70px.

        Selecting a card raises it, and this offset is calibrated against that raised
        position, which is why it is measured from the card rather than the button.
        """
        dx, dy = self.cfg.coord("use_button_offset") or (0.0, 0.082)
        click_screen(cx + int(dx * rect.width), cy + int(dy * rect.height), rect)

    # -- modes -------------------------------------------------------------

    def run(self) -> int:
        if self.meta.has_target(self.target):
            print(f"{PRETTY.get(self.target, self.target)} is already discovered. Nothing to do.")
            return 0

        if self.sends_input:
            self.window = BalatroWindow.find()
            if not self.window.focus_or_prompt():
                raise Stop("could not bring Balatro to the foreground")
            print(f"attached to {self.window!r}")
        else:
            try:
                self.window = BalatroWindow.find()
            except WindowNotFound:
                self.window = None

        state = self.save.read(force=True)
        if state is None:
            raise Stop(
                "No readable run in save.jkr. Start an unseeded run and get to the "
                "blind-select screen first."
            )
        self._preflight(state)

        print(f"mode={self.mode} target={PRETTY.get(self.target, self.target)}")
        print("panic key: hold F12 to abort\n")

        return self._loop(state)

    def _preflight(self, state: RunState) -> None:
        if state.seeded:
            raise Stop(
                "This run is SEEDED. Balatro disables unlocks and discoveries on "
                "seeded runs, so farming it would achieve nothing. Start an "
                "unseeded run."
            )
        if state.challenge:
            raise Stop("This is a challenge run; start a normal unseeded run instead.")

    def _loop(self, state: RunState) -> int:
        while True:
            self.guard()

            if state.at_blind_select and state.ante == 1:
                charm = bool(state.charm_slots)
                self.log(
                    event="run",
                    seed=state.seed,
                    small=state.blind_tags.get("Small"),
                    big=state.blind_tags.get("Big"),
                    charm=charm,
                )
                marker = "  <<< CHARM" if charm else ""
                print(
                    f"[{self.stats.resets:>5}] {state.seed}  {state.describe_tags()}{marker}"
                )

                if charm and self.mode != "observe":
                    result = self._handle_charm(state)
                    if result is not None:
                        return result

            if self.mode == "observe":
                # No input: wait for the player to restart by hand.
                state = self.wait_for_new_run(state.seed, timeout=3600.0)
            else:
                state = self.restart(state.seed)

    def _handle_charm(self, state: RunState) -> int | None:
        """Returns an exit code to stop, or None to keep farming."""
        self.stats.charm_packs += 1
        try:
            self.skip_into_charm(state)
        except Stop as exc:
            # One dropped click should not end an overnight run, but a Charm pack we
            # never opened is a real lost chance at a Soul, so it is recorded loudly
            # and surfaced in the report rather than swallowed.
            self.stats.charm_lost += 1
            self.log(event="charm_lost", seed=state.seed, error=str(exc))
            print(f"        !! LOST this charm pack: {exc}")
            return None

        # Park the cursor away from the cards: hovering one raises it and pops a
        # tooltip, either of which can corrupt a slot read.
        neutral = self.cfg.coord("neutral")
        if neutral is not None:
            rect = self.rect()
            move_screen(*rect.norm_to_screen(*neutral), rect)

        cleanest, soul_scan, frames = self.scan_pack()
        scan = soul_scan or cleanest

        # Three signals, any of which is enough (a false positive merely halts the
        # bot; a miss silently costs ~500 resets):
        #   name     - the Soul won its slot outright against the other 22 candidates
        #   score    - Soul correlation cleared an absolute floor (tarots <=0.35,
        #              a real Soul >=0.87)
        #   template - independent sliding search, run only on candidates
        soul_floor = float(self.cfg["detector"]["soul_score_threshold"])
        by_name = scan.soul_slot is not None
        by_score = scan.max_soul_score >= soul_floor
        by_template = scan.template_score >= self.finder.threshold
        soul = by_name or by_score or by_template
        # Flag when the cards were unsettled, or when the two cheap signals split.
        suspicious = (not cleanest.ready) or (by_name != by_score)

        self.stats.cards_seen.extend(
            s.key for s in scan.slots if s and not s.is_soul
        )
        shot = self.save_pack_shot(state.seed, scan, full_res=soul or suspicious)
        print(f"        pack: {', '.join(scan.names)}")
        print(f"        soul: name={by_name} score={scan.max_soul_score:.3f}"
              f"/{soul_floor:.2f}"
              + (f" template={scan.template_score:.3f}" if scan.match else "")
              + f" | slots mean {cleanest.mean_score:.2f}"
              f" worst {cleanest.min_score:.2f} over {frames} frames"
              f"{'' if cleanest.ready else '  [NOT SETTLED]'}")
        self.log(
            event="charm_pack",
            seed=state.seed,
            cards=[s.key if s else None for s in scan.slots],
            scores=[round(s.score, 3) if s else None for s in scan.slots],
            soul_scores=[round(s.soul_score, 3) if s else None for s in scan.slots],
            mean_score=round(cleanest.mean_score, 3),
            min_score=round(cleanest.min_score, 3),
            max_soul_score=round(scan.max_soul_score, 3),
            frames=frames,
            template_score=round(scan.template_score, 3) if scan.match else None,
            soul_by_name=by_name,
            soul_by_score=by_score,
            soul_by_template=by_template,
            ready=cleanest.ready,
            suspicious=suspicious,
            shot=shot.name,
        )
        if suspicious:
            self.stats.suspicious += 1
            print(f"        !! flagged for audit -> {shot.name}")

        if self.mode == "skip-only":
            print(f"\nskip-only mode: stopping at the first Charm pack.\n"
                  f"Inspect {shot} to confirm the read.")
            return 0

        if not soul:
            return None

        self.stats.souls_found += 1
        print("        *** SOUL DETECTED -- taking it ***")
        outcome = self.take_soul(scan.soul_point(self.rect()), state.seed)

        rolled = [k for k in outcome.joker_keys if k in LEGENDARIES]
        self.stats.legendaries.extend(rolled)
        self.log(
            event="soul_used",
            seed=state.seed,
            jokers=list(outcome.joker_keys),
            target_unlocked=outcome.target_unlocked,
            resolved=outcome.resolved,
        )

        if outcome.target_unlocked:
            print(f"\n*** {PRETTY.get(self.target, self.target)} UNLOCKED ***")
            self.log(event="success", seed=state.seed, target=self.target)
            return 0

        if not outcome.resolved:
            # The Soul was almost certainly consumed but we could not confirm the
            # result. Stopping beats silently farming on with an unknown state.
            print(f"        !! used the Soul but could not confirm the result.")
            print(f"        !! see {shot} and {SOUL_DIR / (state.seed + '_used.png')}")
            return 2

        pretty = ", ".join(PRETTY.get(k, k) for k in rolled) or "unknown"
        print(f"        rolled: {pretty} (not the target) -- continuing")
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("observe", "skip-only", "live"), default="observe")
    ap.add_argument("--config", type=Path, default=ROOT / "config.json")
    ap.add_argument("--target", default=TARGET_JOKER)
    ap.add_argument("--max-resets", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    if args.max_resets is not None:
        cfg.raw["limits"]["max_resets"] = args.max_resets

    farmer = Farmer(cfg, args.mode, args.target)
    try:
        return farmer.run()
    except PanicAbort as exc:
        print(f"\naborted: {exc}")
        return 130
    except (Stop, WindowNotFound) as exc:
        print(f"\nstopped: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    finally:
        print(farmer.stats.summary())


if __name__ == "__main__":
    sys.exit(main())
