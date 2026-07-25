"""Locating and inspecting the Balatro window, via ctypes/user32.

Deliberately avoids pywin32: everything needed here is a handful of user32 calls,
and the system Python could not install pywin32's post-install step anyway.

The process is made per-monitor DPI aware at import time so that client rects,
cursor positions and screenshots all agree on one coordinate space. Without this,
a scaled display silently shifts every click.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import time
from dataclasses import dataclass

__all__ = ["Rect", "BalatroWindow", "WindowNotFound", "set_dpi_aware"]

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_VK_MENU = 0x12
_KEYEVENTF_KEYUP = 0x0002


def _tap_alt() -> None:
    """Press and release ALT via keybd_event, to nudge the foreground lock."""
    user32.keybd_event(_VK_MENU, 0, 0, 0)
    time.sleep(0.02)
    user32.keybd_event(_VK_MENU, 0, _KEYEVENTF_KEYUP, 0)
    time.sleep(0.02)

_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)


def set_dpi_aware() -> None:
    """Opt into per-monitor DPI awareness; harmless if already set."""
    try:
        user32.SetProcessDpiAwarenessContext(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
    except Exception:  # pragma: no cover - older Windows
        try:
            ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
        except Exception:
            user32.SetProcessDPIAware()


set_dpi_aware()


class WindowNotFound(RuntimeError):
    """Raised when the Balatro window cannot be located."""


@dataclass(frozen=True)
class Rect:
    """A screen-space rectangle (left/top inclusive, right/bottom exclusive)."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def norm_to_screen(self, nx: float, ny: float) -> tuple[int, int]:
        """Map normalized (0..1) window coords to absolute screen pixels."""
        return (
            self.left + int(round(nx * self.width)),
            self.top + int(round(ny * self.height)),
        )

    def as_mss(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


_ENUM_PROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)


def _window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _find_windows(title_contains: str) -> list[int]:
    found: list[int] = []
    needle = title_contains.lower()

    def callback(hwnd: wt.HWND, _param: wt.LPARAM) -> bool:
        if user32.IsWindowVisible(hwnd) and needle in _window_title(hwnd).lower():
            found.append(int(hwnd))
        return True

    user32.EnumWindows(_ENUM_PROC(callback), 0)
    return found


class BalatroWindow:
    """Handle to the running Balatro window."""

    TITLE = "Balatro"

    def __init__(self, hwnd: int) -> None:
        self.hwnd = hwnd

    @classmethod
    def find(cls, title: str = TITLE) -> "BalatroWindow":
        matches = _find_windows(title)
        if not matches:
            raise WindowNotFound(
                f"No visible window with {title!r} in the title. Is Balatro running?"
            )
        if len(matches) > 1:
            # Prefer an exact title match over a substring hit.
            exact = [h for h in matches if _window_title(h).strip() == title]
            matches = exact or matches
        return cls(matches[0])

    # -- geometry ---------------------------------------------------------

    @property
    def title(self) -> str:
        return _window_title(self.hwnd)

    def exists(self) -> bool:
        return bool(user32.IsWindow(self.hwnd))

    def client_rect(self) -> Rect:
        """Screen-space rect of the client (drawable) area."""
        rc = wt.RECT()
        if not user32.GetClientRect(self.hwnd, ctypes.byref(rc)):
            raise WindowNotFound("GetClientRect failed; window may have closed")
        origin = wt.POINT(0, 0)
        user32.ClientToScreen(self.hwnd, ctypes.byref(origin))
        return Rect(
            origin.x,
            origin.y,
            origin.x + rc.right,
            origin.y + rc.bottom,
        )

    # -- focus ------------------------------------------------------------

    def is_foreground(self) -> bool:
        return int(user32.GetForegroundWindow()) == self.hwnd

    def focus(self, *, timeout: float = 1.5) -> bool:
        """Bring Balatro to the foreground. Returns True if it took.

        Windows refuses SetForegroundWindow from a process that does not own the
        current foreground window, so a plain call silently fails. The reliable
        workaround is to attach to the foreground window's input thread first,
        which makes the two threads share a foreground state. A synthetic ALT tap
        is also tried, since Windows relaxes the restriction while a key is down.

        Mouse clicks reach whatever window is under the cursor regardless, but the
        R restart is a *keyboard* event and only goes to the focused window -- so
        this genuinely has to work before the bot can drive anything.
        """
        if self.is_foreground():
            return True

        user32.ShowWindow(self.hwnd, 9)  # SW_RESTORE

        fg = user32.GetForegroundWindow()
        fg_thread = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        my_thread = kernel32.GetCurrentThreadId()
        attached = False
        if fg_thread and fg_thread != my_thread:
            attached = bool(user32.AttachThreadInput(my_thread, fg_thread, True))
        try:
            user32.BringWindowToTop(self.hwnd)
            user32.SetForegroundWindow(self.hwnd)
            user32.SetActiveWindow(self.hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(my_thread, fg_thread, False)

        if self._wait_foreground(0.4):
            return True

        # Second attempt: Windows lifts the foreground lock briefly around a
        # keypress, so tap ALT and immediately retry.
        _tap_alt()
        user32.SetForegroundWindow(self.hwnd)
        return self._wait_foreground(timeout)

    def _wait_foreground(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_foreground():
                return True
            time.sleep(0.05)
        return self.is_foreground()

    def focus_or_prompt(self, timeout: float = 30.0) -> bool:
        """Focus Balatro, falling back to asking the user to click on it.

        Some foreground-lock situations cannot be beaten programmatically. Rather
        than fail the run, wait for the player to click the game window.
        """
        if self.focus():
            return True
        print(
            f"Could not focus Balatro programmatically.\n"
            f"  -> Click on the Balatro window now (waiting up to {timeout:.0f}s)..."
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_foreground():
                print("  -> Balatro is focused; continuing.")
                return True
            time.sleep(0.2)
        return False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        try:
            rect = self.client_rect()
            return f"<BalatroWindow hwnd={self.hwnd} client={rect.width}x{rect.height}>"
        except Exception:
            return f"<BalatroWindow hwnd={self.hwnd} (gone)>"
