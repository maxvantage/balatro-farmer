# balatro-farmer

Automates Balatro's "fast reset farming" technique to hunt **The Soul**, the only
source of Legendary Jokers.

Target: **Yorick** (`j_yorick`) — the last of 150 Jokers missing from this profile.

## How it works

The interesting part is how little of this is screen-scraping.

Balatro's `.jkr` save files are raw-DEFLATE serialized Lua tables, and the game's
own Lua is readable straight out of `Balatro.exe` (it's a LÖVE archive, i.e. a zip).
Reading both turned up three facts that shape the whole design:

1. **Both skip tags are in the save file.** At blind select,
   `GAME.round_resets.blind_tags` holds the Small *and* Big blind tags. The
   central decision — "is there a Charm Tag?" — is a file read, not a screenshot.
2. **`GAME.pseudorandom.seed` changes every restart**, giving an exact "new run
   has started" signal. Every wait in the bot is a state assertion on the save
   file; there are no guessed sleeps in the main loop.
3. **Holding `R` restarts instantly** (`Controller:key_hold_update`, >0.7s),
   keeping the same deck and stake and staying unseeded.

Vision is used for exactly one job: spotting The Soul among the 5 cards of a Mega
Arcana Pack. Pack contents never reach the save file — `save_run()` is called
*before* a skip tag is applied. This was confirmed live, not just read: with a pack
open on screen, `save.jkr` still reported `STATE=7` and contained no `pack_cards`
area at all. The template is Balatro's own art, sliced out of its texture atlas.

Three things about that step only became clear from real screenshots:

- **Scale is derived, not swept.** At 2560×1599 the pack cards render 205×301 px —
  1.58× the template's native height, outside any plausible fixed sweep. Card
  height is expressed as a fraction of window height and the scale computed per
  frame, which `tests/test_detector.py` checks across four resolutions.
- **The search is confined to the card row.** Searching the whole window scored
  0.46 on the animated purple backdrop, uncomfortably close to a real threshold.
- **The atlas cell is padded.** The cell is 142×190 but the art is only 126×186;
  scaling the padding in distorted the aspect ratio (0.747 vs 0.681 as rendered)
  and capped match scores near 0.70. Cropping to the opaque bounds fixed it.

### The Soul is not one sprite

This is the finding that mattered most, and only a real Soul exposed it. The Soul
renders as **two** layers: its atlas cell `Tarots(2,2)`, plus `G.shared_soul` =
`Enhancers(0,1)` (a white pearl) drawn through a dissolve shader with continuously
animated scale *and rotation*:

```lua
scale_mod  = 0.05 + 0.05*sin(1.8T) + 0.07*sin(frac(T)*pi*14)*(1-frac(T))^3
rotate_mod = 0.1*sin(1.219T) + 0.07*sin(T*pi*5)*(1-frac(T))^2
```

Scale spans ~0–0.17, rotation ±0.17 rad (≈±10°). A flat single template misses the
overlay that visually dominates the card. On the first live run, two real Souls
scored **0.448 and 0.520** on the template matcher — below its 0.55 threshold, so
that signal **missed both**, and only the argmax signal caught them.

The fix is a bank of 10 composites (5 rotations × 2 scales) covering the animation,
matched with max-over-bank. Measured on a real Soul afterwards: **0.857**
identification, **0.789** template — both signals now fire, and agree.

| | flat template | composite bank |
|---|---|---|
| Real Souls (live) | 0.448 / 0.520 | **0.857** |
| Animated Souls (measured) | 0.537–0.593 | **0.908–0.920** |
| Soul-free slots | — | ≤0.347 |

### The cards never hold still

Every card sets `ambient_tilt = 0.2`, and `Card:draw` recomputes a shader tilt each
frame from real time with a per-card phase:

```lua
local tilt_angle = G.TIMERS.REAL*(1.56 + (self.ID/1.14212)%1) + self.ID/1.35122
self.tilt_var.amt = self.ambient_tilt*(0.5+math.cos(tilt_angle))*tilt_factor
```

Measured across 40 consecutive frames of one live pack:

| | |
|---|---|
| Detected box wander | up to **111 px** in x, 50 px in y |
| One slot's score range | **0.285 → 0.748** (std 0.148) |
| Name stability | **200/200 slot reads correct, 0 misidentifications** |
| Tilt penalty vs a flat composite | **0.102 mean, 0.135 worst** |

So flat-composite tests overstate scores by ~0.1, and a single frame can dip below
any per-slot gate. Three things make it hold up anyway:

- Matching is **translation-tolerant** (searched within a padded region), which is
  what absorbs the box wander.
- The readiness gate uses the **mean** slot score (settled 0.68–0.79 vs unsettled
  ~0.13 across 24 live packs) rather than requiring every slot to clear a bar on one
  unlucky frame.
- The Soul verdict is taken across **every frame sampled** (8 per pack), not from a
  single chosen frame. Applying the worst measured tilt penalty, a real Soul still
  wins its slot by **~0.37–0.53**.

### Why not just use a seed?

Balatro disables all unlocks and discoveries on a seeded run. Brute force on
random seeds is the only route that actually unlocks anything.

## Odds

| | |
|---|---|
| Ante-1 tag pool | 15 of 24 tags → P(Charm) ≈ 1/15 per tag |
| Tags visible per reset | 2 (Small + Big) → P(Charm) ≈ 12.9% |
| Soul rate | 0.3%/card × 5 cards → ≈1.49% per Mega Arcana Pack |
| **Souls per reset** | **≈1 in 520** |
| Legendary is Yorick | 1 in 5 |
| **Expected resets** | **≈2,600** |

Measured over 200+ real resets: charm rate **12.6%** against a 12.9% prediction, and
throughput **420–580 resets/hour** depending on how many Charm packs come up (each
costs a couple of seconds to read). At the true charm rate that's ~520/hour, putting
the expected hunt at **~5 hours** — with high variance. It could finish inside an
hour or run past twelve.

## Setup

```bash
py -m venv .venv
.venv/Scripts/python.exe -m pip install pillow numpy opencv-python mss
.venv/Scripts/python.exe tools/extract_soul_template.py
```

## Usage

Start Balatro, begin an **unseeded** run on any deck, and get to the blind-select
screen. Then:

```bash
.venv/Scripts/python.exe -m farmer.bot --mode observe
```

Modes, in the order you should use them:

- **`observe`** — sends no input at all. Prints the seed and both blind tags each
  time you restart by hand, so you can confirm the parsed tags match what's on
  screen before trusting any automation.
- **`skip-only`** — drives the reset loop, but stops at the first Charm pack and
  reports the detector's score with a screenshot. This is how the detector
  threshold gets set from real data rather than a guess.
- **`live`** — the full unattended loop. Stops when `meta.jkr` reports Yorick
  discovered.

**Panic key: hold `F12` to abort.** The bot also stops on its own if Balatro
loses foreground focus, rather than clicking blindly at whatever is underneath.

## Logging and the report

Every reset appends one line to `logs/run.jsonl`, and every Charm pack records the
**names of all five cards** it contained. To summarise a run:

```bash
.venv/Scripts/python.exe tools/report.py
.venv/Scripts/python.exe tools/report.py --cards        # full tarot tally
.venv/Scripts/python.exe tools/report.py --suspicious   # only packs needing a look
```

It reports resets, packs, charm rate against theory, the tag distribution, what
tarots turned up, soul outcomes, and an **integrity** section.

### Not missing a Soul

Three signals decide, and any one firing is treated as a Soul (a false positive just
halts the bot; a missed Soul silently costs ~500 resets). All three read the card
row — **none of them involve the USE button**, which only matters for the click
*after* detection:

1. **Identification (primary)** — each card is named against the 23 things a Mega
   Arcana Pack can contain. The Soul wins its slot by 0.50–0.66, a relative decision
   that does not depend on a tuned cutoff.
2. **Absolute Soul score** — free from the same pass. Tarot slots score ≤0.347, a
   real Soul ≥0.874, so the 0.60 floor sits in a 0.53-wide gap.
3. **Sliding template search** — fully independent, and expensive, so it runs only
   to confirm a candidate rather than on every frame.

Only a frame that is itself legible may nominate a Soul. Without that rule, a
mid-materialize frame where one box was found and `c_soul` won on noise produced two
false positives in 18 resets.

A pack is **flagged for audit** if the cards had not finished materializing or if
the two signals disagreed. Flagged packs and Souls are kept at full resolution;
routine ones are cropped and halved. `--suspicious` lists exactly what to re-check,
so a bug can never quietly swallow a Soul without leaving a record.

Disk: measured **~202 KB per pack**, so a full hunt lands near **80 MB**.

### Clearing logs between runs

`logs/` is bulk output and safe to delete — the bot recreates it on startup:

```bash
rmdir /s /q logs
```

Frames of an **actual Soul** are written to `souls/` instead, deliberately outside
`logs/`. They are the rarest artefact this project produces (two in three hours) and
were the only reason the two-sprite render bug was ever diagnosed, so a routine log
clear must not be able to destroy them. To archive a run instead of discarding it:

```bash
move logs logs_run1
```

## Calibration

`config.json` stores click points as normalized (0..1) positions inside the
window client rect, so they survive a resize. To (re)derive them:

```bash
.venv/Scripts/python.exe tools/calibrate.py --grid --name blind_select
.venv/Scripts/python.exe tools/calibrate.py --overlay --name verify
```

## Tests

```bash
.venv/Scripts/python.exe tests/test_detector.py
.venv/Scripts/python.exe tests/test_detector_live.py
```

- `test_detector.py` — renders synthetic packs from the game's atlas at four
  window sizes and checks the derived-scale search finds The Soul at each
  (resolution independence).
- `test_detector_live.py` — runs against real captured frames from a live Charm
  pack. The known-positive composites the game's own Soul sprite into a real
  Soul-free pack at true card size, in all five slots. Also checks the
  card-dealt readiness signal and the USE-button locator.

### Measured detector separation

| | score |
|---|---|
| Real Soul-free packs (3 observed) | 0.235 – 0.339 |
| Soul composited into real frames, all 5 slots | 0.722 – 0.844 |
| **Threshold** | **0.55** |

The threshold is biased toward recall on purpose: a missed Soul silently costs
~500 resets, while a false positive just halts the bot with screenshots. Every
Charm pack is saved to `logs/packs/` so the whole run can be audited by eye
afterwards.

## Results: done — 150/150

**Yorick landed on the 33rd Soul**, seed `16ERTR3N`, 2026-07-26 04:26 local.

📊 **[Read the illustrated report →](https://maxvantage.github.io/balatro-farmer/)**
(source in [`docs/index.html`](docs/index.html))

| | resets | Charm packs | Souls | rolled |
|---|---|---|---|---|
| Run 1 (~3 h) | 1,561 | 199 | 2 | Perkeo, Triboulet |
| Run 2 (5.9 h) | 3,269 | 389 | 4 | Canio, Perkeo, Chicot, Perkeo |
| Run 3 (13.9 h) | 7,742 | 946 | 18 | Canio ×5, Triboulet ×5, Chicot ×5, Perkeo ×3 |
| Run 4 (4.7 h) | 2,547 | 332 | 9 | Canio ×3, Chicot ×3, Perkeo ×2, **Yorick** |
| **total (27.5 h)** | **15,119** | **1,866** | **33** | Canio 9, Chicot 9, Perkeo 8, Triboulet 6, Yorick 1 |

The runs are one hunt; they were only split to check on progress. 550 resets/hour,
one every 6.5 s. Charm rate 12.34% against a 12.9% prediction. A Soul in 1.77% of
packs (1 in 57), i.e. 1 per 458 resets — 9,330 tarots read to find 33.

Each Soul is an even 1-in-5, so 32 straight misses before the hit is a
0.8³² = **0.079%** outcome, about 1 in 1,262. Five Souls is the fair price for one
named Legendary; this cost 33.

**In its final form the detector went 27 for 27 with no false positives** — every
Soul in runs 3 and 4. Verified independently rather than on its own say-so: a
reimplementation of Balatro's RNG (below) names exactly which packs hold a Soul, and
it agrees with the vision pipeline on all **1,278** of those packs, no misses and no
false alarms. Runs 1–2 predate the shape fix (run 2 ended on the false positive
described below) and their logs have since been rotated away, so those 6 Souls are
not cross-checkable.

### Why it is only ever 1-in-5

Worth recording, because the profile makes it look otherwise. Yorick is the single
joker of 150 missing from `unlocked` in `meta.jkr`, which reads like the smoking gun
for "it can never spawn". It is not: `get_current_pool` culls on
`(v.unlocked ~= false or v.rarity == 4)`, and Legendaries are rarity 4, so the lock
is bypassed. All five are structurally identical in `game.lua` (orders 146–150).

The roll is `pseudorandom_element(pool, pseudoseed('Joker4'))` — deterministic in the
run seed. Reimplementing `pseudohash`, `pseudoseed` and LuaJIT's Tausworthe
`math.random` reproduces **all 19 outcomes that have ground truth** exactly (run 3's
18, plus calling the winning seed `16ERTR3N` as Yorick before `meta.jkr` was read),
and a 1.5M-seed sweep puts Yorick at 19.9%. Both draws share `hashed_seed`, so the conditional case was checked
too: given a Soul in the pack, Yorick is 19.7% (χ²=3.22, 4 df, p≈0.52). No bias.
There is nothing to fix; it is variance.

Run 1 produced the two-layer Soul sprite discovery, and the finding that **every
skip click was retrying** — all 298 of them, always succeeding on the second
attempt. A skip takes a measured **3.7–4.2 s** to reach `save.jkr` (the tag
animation blocks the event queue before `save_run` fires); the timeout was simply
shorter than that.

Run 2 ended on a **false positive**, and the mechanism is worth recording. A merged
brightness blob — 347×372 where a card is 205×301 — passed `find_card_slots`, which
filtered on area and height but never width. The misaligned crop made every
candidate in that slot score ~0.25, and `c_soul` won it on noise by a margin of
0.033. The bot claimed a Soul, failed to use it, and stopped rather than continue in
an unknown state. Two fixes: `find_card_slots` now constrains box *shape*, and
winning a slot outright no longer triggers on its own.

That last point is the useful lesson. Over 389 real packs:

| signal | fired on 385 Soul-free packs | fired on 4 real Souls |
|---|---|---|
| absolute Soul score | **0** (ceiling 0.375, floor 0.60) | 4/4 (0.831–0.878) |
| sliding template | **0** | 4/4 (0.767–0.813) |
| slot argmax | **1** ← the false positive | 4/4 |

argmax caught nothing the other two missed and was the only source of false
positives, so it is now recorded for audit but does not trigger. A redundant signal
that only adds false positives is not a safety net.

Run 3 found that **the Soul confirmation had been working by accident.** `use_retry`
fired on 18 Souls out of 18 — the same "the retry is the normal path" smell as run 1,
but with a worse cause. Using the Soul ends the *whole* pack despite the "Choose 2"
label: 0.6s after the USE click the card row is empty, the Joker is in its slot and
the screen is back at blind select (`count_cards` reads 5 on every pre-use frame and
0 on every post-use frame, across all 18). Closing the pack does not fire
`save_run()`, so `save.jkr` never revealed the Joker on its own — and the 5.5s retry
was therefore guaranteed to fire, re-clicking the card position *on the live
blind-select screen*, roughly where the blind panels sit. One of those clicks started
a blind, which fired `save_run()`, which produced the answer 4–5s later. Every
resolution in the run came out of a stray click into a running game.

The trap: raising `use_retry` above the 8s pack-close — the obvious reading of "the
retry timer is too short" — would have removed the only thing producing a save, and
every future Soul would have timed out unresolved and stopped the bot. So the timer
is gone instead. The USE is confirmed by watching the pack disappear, the retry
re-clicks only the button (never the card, which would burn a different one), and
nothing is clicked after the Soul is spent: the target arrives in `meta.jkr` within a
frame via `discover_card` → `save_progress()`, and naming a non-target Legendary is
telemetry not worth clicking blind for.

Run 4 confirmed both halves of that. `use_retry` fired **0 times in 9 Souls** (against
18 of 18 before), so no stray click ever entered a live run. And every `soul_used`
logged `jokers: []`, which is the predicted consequence: with nothing clicked after
the Soul is spent, `save_run()` never fires and the non-target names are genuinely
unavailable from the save. Those 8 names were recovered afterwards from the RNG model
instead — which is what the `jokers: []` entries in `logs/run.jsonl` mean, rather than
a failure.

### Not moving to a seed filter

The roll being deterministic in the run seed means the bot *could* read `save.jkr` at
blind select, predict both the Soul and the Legendary, and only ever open a pack that
is a guaranteed Yorick — about 3.8 h expected instead of unbounded variance. That is
deliberately not implemented. It is lookahead into the RNG rather than automation of
play, which is the line this project does not cross.

## Safety notes

- **Read-only on save files.** Nothing here ever writes into `%APPDATA%/Balatro`.
- Every click is bounds-checked against Balatro's client rect.
- The bot refuses to run on a seeded or challenge run (they can't unlock anything).
- **Holding `R` zeroes your current win streak** — that's in the game's own
  restart handler and is unavoidable with this technique.
