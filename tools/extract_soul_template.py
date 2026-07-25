"""Extract The Soul's sprite from Balatro's own texture atlas.

Balatro.exe is a LÖVE archive (a zip), so the game's art is readable directly --
which gives us a pixel-exact template instead of a screenshot crop.

From game.lua:
    c_soul = {... name = "The Soul", pos = {x=2, y=2}, set = "Spectral", hidden = true}

Consumable sprites live in ``Tarots.png``, a 10x6 grid of 71x95 cells (142x190 in
the 2x atlas). The player's ``texture_scaling`` setting picks which atlas the game
actually renders from.

Usage:
    python tools/extract_soul_template.py [--scale 2] [--out assets/soul_2x.png]
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

from PIL import Image

BALATRO_EXE = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Balatro\Balatro.exe")

# Grid geometry of Tarots.png, and The Soul's cell within it.
GRID_COLS, GRID_ROWS = 10, 6
SOUL_COL, SOUL_ROW = 2, 2


def extract(exe: Path, scale: int) -> Image.Image:
    with zipfile.ZipFile(exe) as z:
        data = z.read(f"resources/textures/{scale}x/Tarots.png")
    atlas = Image.open(io.BytesIO(data)).convert("RGBA")
    cw, ch = atlas.width // GRID_COLS, atlas.height // GRID_ROWS
    box = (SOUL_COL * cw, SOUL_ROW * ch, (SOUL_COL + 1) * cw, (SOUL_ROW + 1) * ch)
    return atlas.crop(box)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=BALATRO_EXE)
    ap.add_argument("--scale", type=int, default=2, choices=(1, 2))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sprite = extract(args.exe, args.scale)
    out = args.out or Path(__file__).resolve().parent.parent / "assets" / f"soul_{args.scale}x.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sprite.save(out)
    print(f"wrote {out}  ({sprite.width}x{sprite.height})")

    # A contact sheet of the whole atlas makes it easy to eyeball that the cell
    # indices line up with the card we expect.
    with zipfile.ZipFile(args.exe) as z:
        atlas = Image.open(io.BytesIO(z.read(f"resources/textures/{args.scale}x/Tarots.png")))
    sheet = out.parent / f"tarot_atlas_{args.scale}x.png"
    atlas.save(sheet)
    print(f"wrote {sheet}  ({atlas.width}x{atlas.height})")


if __name__ == "__main__":
    main()
