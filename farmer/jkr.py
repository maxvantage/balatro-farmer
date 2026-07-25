"""Reader for Balatro's ``.jkr`` save files.

A ``.jkr`` file is a raw-DEFLATE stream (no zlib/gzip header, hence ``wbits=-15``)
containing a serialized Lua table literal of the form::

    return {["STATE"]=7,["BLIND"]={["chips"]=0,},}

This module inflates the file and parses that literal into plain Python objects.
Tables become dicts; Lua's 1-based integer keys are preserved as ints, so an
"array" table looks like ``{1: ..., 2: ...}``.

Everything here is read-only. Nothing in this package ever writes into the
Balatro save directory.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

__all__ = ["read_jkr", "loads", "JkrError"]


class JkrError(Exception):
    """Raised when a save file cannot be inflated or parsed."""


def read_jkr(path: str | Path) -> dict[str, Any]:
    """Inflate and parse a ``.jkr`` file into a dict."""
    raw = Path(path).read_bytes()
    try:
        text = zlib.decompress(raw, -15).decode("utf-8", "replace")
    except zlib.error as exc:  # partially-written file, or not actually a save
        raise JkrError(f"could not inflate {path}: {exc}") from exc
    return loads(text)


def loads(text: str) -> dict[str, Any]:
    """Parse a serialized Lua table literal (with optional ``return`` prefix)."""
    return _Parser(text).parse()


# Lua literals we translate directly.
_KEYWORDS = {"true": True, "false": False, "nil": None}
_WS = " \t\r\n"


class _Parser:
    """Recursive-descent parser for the subset of Lua that Balatro serializes.

    Balatro's serializer only ever emits nested tables, quoted strings, numbers
    and booleans, so this deliberately does not implement the rest of Lua.
    """

    def __init__(self, text: str) -> None:
        self.s = text
        self.i = 0
        self.n = len(text)

    # -- plumbing ---------------------------------------------------------

    def parse(self) -> dict[str, Any]:
        self._skip_ws()
        if self.s.startswith("return", self.i):
            self.i += len("return")
            self._skip_ws()
        value = self._value()
        if not isinstance(value, dict):
            raise JkrError("top-level value is not a table")
        return value

    def _skip_ws(self) -> None:
        s, n = self.s, self.n
        i = self.i
        while i < n and s[i] in _WS:
            i += 1
        self.i = i

    def _fail(self, what: str) -> JkrError:
        near = self.s[max(0, self.i - 40) : self.i + 40].replace("\n", " ")
        return JkrError(f"{what} at offset {self.i}; near: ...{near}...")

    # -- grammar ----------------------------------------------------------

    def _value(self) -> Any:
        self._skip_ws()
        if self.i >= self.n:
            raise self._fail("unexpected end of input")
        c = self.s[self.i]
        if c == "{":
            return self._table()
        if c == '"':
            return self._string()
        if c == "'":
            return self._string(quote="'")
        for kw, val in _KEYWORDS.items():
            if self.s.startswith(kw, self.i):
                # Guard against an identifier that merely starts with a keyword.
                end = self.i + len(kw)
                if end >= self.n or not (self.s[end].isalnum() or self.s[end] == "_"):
                    self.i = end
                    return val
        return self._number()

    def _table(self) -> dict[Any, Any]:
        self.i += 1  # consume '{'
        out: dict[Any, Any] = {}
        positional = 0  # Lua array-style entries with no explicit key
        while True:
            self._skip_ws()
            if self.i >= self.n:
                raise self._fail("unterminated table")
            if self.s[self.i] == "}":
                self.i += 1
                return out
            key = self._key()
            if key is _NO_KEY:
                positional += 1
                out[positional] = self._value()
            else:
                out[key] = self._value()
            self._skip_ws()
            if self.i < self.n and self.s[self.i] == ",":
                self.i += 1

    def _key(self) -> Any:
        """Parse ``["k"]=``, ``[1]=`` or bare ``k=``. Returns _NO_KEY if absent."""
        self._skip_ws()
        start = self.i
        if self.i < self.n and self.s[self.i] == "[":
            self.i += 1
            key = self._value()
            self._skip_ws()
            if self.i >= self.n or self.s[self.i] != "]":
                raise self._fail("expected ']' closing table key")
            self.i += 1
            self._skip_ws()
            if self.i >= self.n or self.s[self.i] != "=":
                raise self._fail("expected '=' after table key")
            self.i += 1
            return key

        # Bare identifier key, e.g. ``name="Charm Tag"``. Balatro's own saves use
        # the bracket form, but the Lua in the game archive uses this one and the
        # same parser is handy for both.
        j = self.i
        while j < self.n and (self.s[j].isalnum() or self.s[j] == "_"):
            j += 1
        if j > self.i:
            ident = self.s[self.i : j]
            k = j
            while k < self.n and self.s[k] in _WS:
                k += 1
            if k < self.n and self.s[k] == "=" and self.s[k + 1 : k + 2] != "=":
                self.i = k + 1
                return ident
        self.i = start
        return _NO_KEY

    def _string(self, quote: str = '"') -> str:
        self.i += 1  # consume opening quote
        chunks: list[str] = []
        s, n = self.s, self.n
        while True:
            if self.i >= n:
                raise self._fail("unterminated string")
            c = s[self.i]
            if c == "\\":
                nxt = s[self.i + 1 : self.i + 2]
                chunks.append(_ESCAPES.get(nxt, nxt))
                self.i += 2
                continue
            if c == quote:
                self.i += 1
                return "".join(chunks)
            chunks.append(c)
            self.i += 1

    def _number(self) -> float | int:
        start = self.i
        s, n = self.s, self.n
        if self.i < n and s[self.i] in "+-":
            self.i += 1
        while self.i < n and (s[self.i].isdigit() or s[self.i] in ".eE+-"):
            # '+'/'-' are only part of the number right after an exponent marker.
            if s[self.i] in "+-" and s[self.i - 1] not in "eE":
                break
            self.i += 1
        text = s[start : self.i]
        if not text:
            raise self._fail("expected a value")
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError as exc:
            raise self._fail(f"bad number {text!r}") from exc


_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "'": "'", "\\": "\\"}


class _NoKey:
    """Sentinel for a table entry written without an explicit key."""

    __slots__ = ()


_NO_KEY = _NoKey()
