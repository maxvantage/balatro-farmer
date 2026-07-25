"""Summarise a farming run from logs/run.jsonl.

Answers: how many resets, how many packs, what was in them, what the tag
distribution looked like versus theory, and -- most importantly -- whether any pack
was flagged as unreliable so a real Soul could have slipped past.

Usage:
    python tools/report.py
    python tools/report.py --cards        # full tally of every tarot seen
    python tools/report.py --suspicious   # only the packs needing a human look
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG = ROOT / "logs" / "run.jsonl"
PACK_DIR = ROOT / "logs" / "packs"

PRETTY_LEGENDARY = {
    "j_caino": "Canio",
    "j_triboulet": "Triboulet",
    "j_yorick": "Yorick",
    "j_chicot": "Chicot",
    "j_perkeo": "Perkeo",
}


def load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"no log at {path} yet")
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def pretty_card(key: str | None) -> str:
    if not key:
        return "?"
    return key.removeprefix("c_").replace("_", " ").title()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=LOG)
    ap.add_argument("--cards", action="store_true", help="tally every tarot seen")
    ap.add_argument("--suspicious", action="store_true", help="only flagged packs")
    args = ap.parse_args()

    events = load(args.log)
    if not events:
        return 0

    runs = [e for e in events if e.get("event") == "run"]
    packs = [e for e in events if e.get("event") == "charm_pack"]
    souls = [e for e in events if e.get("event") == "soul_used"]
    flagged = [e for e in packs if e.get("suspicious")]
    lost = [e for e in events if e.get("event") == "charm_lost"]
    retries = [e for e in events if e.get("event") == "click_retry"]

    if args.suspicious:
        if not flagged:
            print("No packs were flagged. Every pack was read cleanly.")
            return 0
        print(f"{len(flagged)} pack(s) flagged for a human look:\n")
        for e in flagged:
            why = []
            if not e.get("ready"):
                why.append("cards not settled")
            if e.get("soul_by_name") != e.get("soul_by_score", e.get("soul_by_template")):
                why.append("soul signals disagreed")
            print(f"  {e['seed']}  {', '.join(why) or 'flagged'}")
            print(f"    cards:  {', '.join(pretty_card(c) for c in e.get('cards') or [])}")
            print(f"    scores: {e.get('scores')}")
            if e.get("soul_scores") is not None:
                print(f"    soul:   {e.get('soul_scores')}  "
                      f"max={e.get('max_soul_score')}  template={e.get('template_score')}")
            else:
                print(f"    template={e.get('template_score')}")
            print(f"    shot:   {PACK_DIR / (e['seed'] + '.png')}\n")
        return 0

    # -- timing ----------------------------------------------------------
    stamps = [datetime.fromisoformat(e["t"]) for e in events if "t" in e]
    span_h = 0.0
    if len(stamps) > 1:
        span_h = (max(stamps) - min(stamps)).total_seconds() / 3600

    print("=" * 58)
    print("BALATRO FARMER REPORT")
    print("=" * 58)
    print(f"  resets            {len(runs)}")
    print(f"  charm packs       {len(packs)}")
    print(f"  souls found       {len([s for s in souls])}")
    print(f"  elapsed           {span_h:.2f} h")
    if span_h > 0:
        print(f"  rate              {len(runs)/span_h:.0f} resets/hour")

    # -- charm rate vs theory -------------------------------------------
    if runs:
        rate = len(packs) / len(runs) * 100
        print(f"\n  charm rate        {rate:.1f}%   (theory ~12.9%: two tags at 1/15)")

    # -- tag distribution ------------------------------------------------
    tags = Counter()
    for e in runs:
        for slot in ("small", "big"):
            if e.get(slot):
                tags[e[slot].removeprefix("tag_")] += 1
    if tags:
        total = sum(tags.values())
        print(f"\n  tag distribution ({total} tags seen, ~1/15 = 6.7% each)")
        for name, n in tags.most_common():
            bar = "#" * max(1, round(n / total * 120))
            print(f"    {name:<12} {n:>5} {n/total*100:>5.1f}%  {bar}")

    # -- pack contents ---------------------------------------------------
    cards = Counter()
    for e in packs:
        for key in e.get("cards") or []:
            if key:
                cards[key] += 1
    if cards:
        print(f"\n  tarots seen       {sum(cards.values())} across {len(packs)} packs")
        shown = cards.most_common() if args.cards else cards.most_common(8)
        for key, n in shown:
            print(f"    {pretty_card(key):<22} {n:>4}")
        if not args.cards and len(cards) > 8:
            print(f"    ... {len(cards)-8} more (use --cards)")
        if cards.get("c_soul"):
            print(f"\n  *** The Soul appeared {cards['c_soul']} time(s) ***")

    # -- souls -----------------------------------------------------------
    if souls:
        print("\n  soul outcomes")
        for e in souls:
            rolled = [PRETTY_LEGENDARY.get(k, k) for k in e.get("jokers") or []]
            status = "TARGET!" if e.get("target_unlocked") else ", ".join(rolled) or "unknown"
            confirm = "" if e.get("resolved") else "  (UNCONFIRMED)"
            print(f"    {e['seed']}  -> {status}{confirm}")

    # -- integrity -------------------------------------------------------
    print("\n  integrity")
    if flagged:
        print(f"    {len(flagged)} pack(s) FLAGGED -- run: tools/report.py --suspicious")
    else:
        print("    all packs read cleanly (cards settled, both soul signals agreed)")
    if lost:
        print(f"    {len(lost)} charm pack(s) LOST to failed skips -- chances at a Soul"
              " that were never seen:")
        for e in lost:
            print(f"      {e['seed']}: {e.get('error')}")
    else:
        print("    no charm packs lost")
    if retries:
        print(f"    {len(retries)} click retry/retries (self-healed)")
    refocus = [e for e in events if e.get("event") == "focus_recovered"]
    if refocus:
        print(f"    {len(refocus)} focus loss(es) recovered")
    unresolved = [e for e in souls if not e.get("resolved")]
    if unresolved:
        print(f"    {len(unresolved)} soul(s) used without confirmation -- check screenshots")

    # -- disk ------------------------------------------------------------
    if PACK_DIR.exists():
        files = list(PACK_DIR.glob("*.png"))
        size = sum(f.stat().st_size for f in files)
        print(f"\n  disk              {len(files)} screenshots, {human_bytes(size)}")
        if files:
            avg = size / len(files)
            print(f"                    avg {human_bytes(avg)}; "
                  f"~330 packs projects to {human_bytes(avg*330*1.2)}")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
