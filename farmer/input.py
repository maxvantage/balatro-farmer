"""Synthetic keyboard/mouse input via SendInput.

Scancode-level keyboard events are used because SDL2 (which LÖVE, and therefore
Balatro, sits on) keys off scancodes. The ``R`` restart is a genuine press/hold/
release: Balatro's ``Controller:key_press_update`` sets ``held_key_times[key]=0``
on press and ``key_hold_update`` accumulates dt per frame while the key stays in
``held_keys``, firing the restart past 0.7s. One keydown plus a wait plus one
keyup is therefore equivalent to a hardware hold.

Every click is bounds-checked against Balatro's client rect. The bot will not
click at a coordinate outside the game window.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import time

from .window import Rect, user32

__all__ = [
    "hold_key",
    "tap_key",
    "click_screen",
    "move_screen",
    "panic_pressed",
    "PanicAbort",
    "OutOfBounds",
]

# --- SendInput plumbing --------------------------------------------------

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79

VK_F12 = 0x7B

# Scancodes (set 1) for the few keys we need.
SCAN = {"r": 0x13, "escape": 0x01, "space": 0x39}


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wt.LONG),
        ("dy", wt.LONG),
        ("mouseData", wt.DWORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wt.WORD),
        ("wScan", wt.WORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _INPUTUNION)]


def _send(*inputs: _INPUT) -> None:
    n = len(inputs)
    arr = (_INPUT * n)(*inputs)
    sent = user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
    if sent != n:
        raise OSError(f"SendInput sent {sent}/{n} events (err {ctypes.get_last_error()})")


class OutOfBounds(RuntimeError):
    """Raised when a click would land outside the Balatro window."""


class PanicAbort(RuntimeError):
    """Raised when the user hits the panic key."""


# --- keyboard ------------------------------------------------------------


def _key_event(scan: int, up: bool) -> _INPUT:
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    return _INPUT(
        type=INPUT_KEYBOARD,
        u=_INPUTUNION(ki=_KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0)),
    )


def hold_key(key: str, seconds: float, *, poll_panic: bool = True) -> None:
    """Press ``key``, hold it for ``seconds``, then release.

    The release is in a ``finally`` so a panic abort can never leave a key stuck
    down.
    """
    scan = SCAN[key]
    _send(_key_event(scan, up=False))
    try:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            time.sleep(0.01)
            if poll_panic and panic_pressed():
                raise PanicAbort("panic key pressed during key hold")
    finally:
        _send(_key_event(scan, up=True))


def tap_key(key: str) -> None:
    scan = SCAN[key]
    _send(_key_event(scan, up=False))
    time.sleep(0.03)
    _send(_key_event(scan, up=True))


# --- mouse ---------------------------------------------------------------


def _to_absolute(x: int, y: int) -> tuple[int, int]:
    """Map screen pixels to SendInput's 0..65535 virtual-desktop space."""
    vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    ax = int(round((x - vx) * 65535 / max(1, vw - 1)))
    ay = int(round((y - vy) * 65535 / max(1, vh - 1)))
    return ax, ay


def _mouse_event(flags: int, ax: int = 0, ay: int = 0) -> _INPUT:
    return _INPUT(
        type=INPUT_MOUSE,
        u=_INPUTUNION(
            mi=_MOUSEINPUT(dx=ax, dy=ay, mouseData=0, dwFlags=flags, time=0, dwExtraInfo=0)
        ),
    )


def move_screen(x: int, y: int, bounds: Rect | None = None) -> None:
    if bounds is not None and not bounds.contains(x, y):
        raise OutOfBounds(f"({x},{y}) is outside the Balatro window {bounds}")
    ax, ay = _to_absolute(x, y)
    _send(_mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, ax, ay))


def click_screen(
    x: int,
    y: int,
    bounds: Rect | None = None,
    *,
    hover_delay: float = 0.10,
    press_delay: float = 0.05,
) -> None:
    """Move to (x, y), let the game register the hover, then left-click.

    Balatro resolves what you clicked from the cursor's collision node, which is
    updated on its own frame tick -- so the hover delay is required, not padding.
    """
    move_screen(x, y, bounds)
    time.sleep(hover_delay)
    _send(_mouse_event(MOUSEEVENTF_LEFTDOWN))
    time.sleep(press_delay)
    _send(_mouse_event(MOUSEEVENTF_LEFTUP))


# --- panic ---------------------------------------------------------------


def panic_pressed() -> bool:
    """True while the panic key (F12) is held."""
    return bool(user32.GetAsyncKeyState(VK_F12) & 0x8000)
