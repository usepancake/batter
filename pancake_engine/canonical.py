"""Canonical serialization for Pancake Engine 0.3.

Engine 0.3 is correctness-first, not TS parity. Known TS divergences are
documented in docs/math-audit-0.4.md.

This module implements byte-identical canonical serialization compatible with
ECMA-262 §6.1.6.1.13 NumberToString and V8's ``JSON.stringify`` number output.

The canonical bytes are the substrate for every hash in Engine 0.3:
``schema_sha256``, ``rows_sha256``, ``compiled_spec_hash``, ``config_hash``,
``result_hash``. Cross-runtime byte-equality is the determinism gate.

Rules:

- ``null``, ``true``, ``false`` — literal.
- ``int`` — decimal repr; reject ``|x| > 2**53`` (silent precision loss on JS round-trip).
- ``float`` — ECMA-262 NumberToString. Reject NaN, +Inf, -Inf. Normalize -0 to 0.
- ``str`` — NFC-normalize then JSON-escape per RFC 8259. Reject lone surrogates.
- ``list`` / ``tuple`` — order preserved; never sorted.
- ``dict`` — keys sorted by Unicode codepoint, recursively. Duplicate-key
  rejection happens **at JSON parse time** in :mod:`pancake_engine.io.load`,
  not here — raw Python dicts cannot detect duplicates after parse.
- ``datetime`` and other types — rejected. Callers serialize times to unix-int seconds first.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any

__all__ = ["canonicalize", "canonical_string", "MAX_SAFE_INTEGER"]

# JavaScript Number.MAX_SAFE_INTEGER = 2**53 - 1; we allow up to 2**53 inclusive
# because 2**53 is exactly representable as a float. Anything strictly larger
# loses precision on the V8 side.
MAX_SAFE_INTEGER = 2**53


def canonicalize(obj: Any) -> bytes:
    """Return the canonical UTF-8 byte representation of ``obj``."""
    return canonical_string(obj).encode("utf-8")


def canonical_string(obj: Any) -> str:
    """Return the canonical string representation (UTF-8 encoding deferred to caller)."""
    return _canon(obj)


def _canon(obj: Any) -> str:
    # bool is a subclass of int in Python; match it before int.
    if obj is None:
        return "null"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, bool):
        # Defensive — `obj is True/False` above already covers all bool instances,
        # but a custom subclass of bool would slip through without this branch.
        return "true" if obj else "false"
    if isinstance(obj, int):
        if abs(obj) > MAX_SAFE_INTEGER:
            raise ValueError(
                f"E_INTEGER_TOO_LARGE: {obj} exceeds 2**53; "
                "use a string for large integers to avoid silent precision loss on JS round-trip"
            )
        return str(obj)
    if isinstance(obj, float):
        return _number_to_string(obj)
    if isinstance(obj, str):
        return _escape_string(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_canon(x) for x in obj) + "]"
    if isinstance(obj, dict):
        # Sort keys by Unicode codepoint order. Python's default `<` on str
        # is codepoint-ordered.
        keys = sorted(obj.keys())
        parts = []
        for k in keys:
            if not isinstance(k, str):
                raise ValueError(f"E_NON_STRING_KEY: {k!r}")
            parts.append(_escape_string(k) + ":" + _canon(obj[k]))
        return "{" + ",".join(parts) + "}"
    raise ValueError(f"E_UNSUPPORTED_TYPE: {type(obj).__name__}")


def _number_to_string(x: float) -> str:
    """ECMA-262 §6.1.6.1.13 NumberToString for finite floats.

    Matches V8 ``JSON.stringify(<number>)`` byte-for-byte on finite values.
    Rejects NaN and ±Infinity. Normalizes -0 to 0.

    The shortest round-trip digit sequence is sourced from CPython's
    ``repr(float)``, which uses Grisu/Ryu — same algorithm V8 uses. CPython
    and V8 differ only in exponential-notation thresholds; this function
    re-applies the ECMA thresholds.
    """
    if math.isnan(x):
        raise ValueError("E_NONFINITE: NaN is not representable in canonical form")
    if math.isinf(x):
        raise ValueError("E_NONFINITE: Infinity is not representable in canonical form")
    if x == 0:
        # -0.0 == 0.0 in Python; both normalize to "0", matching V8.
        return "0"

    negative = x < 0
    if negative:
        x = -x

    r = repr(x)

    # Split into mantissa and explicit exponent
    if "e" in r:
        mantissa_str, exp_str = r.split("e")
        exp = int(exp_str)
    else:
        mantissa_str = r
        exp = 0

    # Split mantissa into integer and fractional parts
    if "." in mantissa_str:
        int_part, frac_part = mantissa_str.split(".")
        if frac_part == "0":
            frac_part = ""
    else:
        int_part = mantissa_str
        frac_part = ""

    # Combine digits; adjust exp for the decimal point position
    all_digits = int_part + frac_part
    exp -= len(frac_part)

    # Strip leading zeros (e.g., 0.001 → int_part="0", frac="001", all="0001")
    stripped = all_digits.lstrip("0")
    if stripped == "":
        # Unreachable under x != 0; defensive.
        return "0"
    all_digits = stripped

    # Strip trailing zeros, rolling them into the exponent
    rstripped = all_digits.rstrip("0")
    if rstripped == "":
        rstripped = "0"
    exp += len(all_digits) - len(rstripped)
    all_digits = rstripped

    # The canonical digit string has no leading/trailing zeros (except "0" itself).
    # Numeric value = int(all_digits) × 10**exp.
    # k = number of significant digits
    # n = decimal point position from the left, where n == k means integer ending,
    #     n > k means trailing zeros, n < k means digits after decimal point.
    k = len(all_digits)
    n = k + exp

    if k <= n <= 21:
        # Integer in regular notation: digits then trailing zeros
        result = all_digits + "0" * (n - k)
    elif 0 < n <= 21:
        # Fractional in regular notation: split digits at position n
        result = all_digits[:n] + "." + all_digits[n:]
    elif -6 < n <= 0:
        # Small number: 0. then leading zeros then digits
        result = "0." + "0" * (-n) + all_digits
    elif k == 1:
        # Single-digit scientific
        e = n - 1
        sign = "+" if e >= 0 else "-"
        result = all_digits + "e" + sign + str(abs(e))
    else:
        # Multi-digit scientific
        e = n - 1
        sign = "+" if e >= 0 else "-"
        result = all_digits[0] + "." + all_digits[1:] + "e" + sign + str(abs(e))

    return ("-" if negative else "") + result


def _escape_string(s: str) -> str:
    """JSON-escape a string with NFC normalization and lone-surrogate rejection."""
    s = unicodedata.normalize("NFC", s)

    # Reject lone surrogates by attempting UTF-8 encode. Python strings can
    # contain unpaired surrogates if produced from certain decode operations;
    # we refuse to canonicalize them rather than emit garbled UTF-8.
    try:
        s.encode("utf-8")
    except UnicodeEncodeError as e:
        raise ValueError(f"E_LONE_SURROGATE: {e}") from e

    out: list[str] = ['"']
    for ch in s:
        code = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)
