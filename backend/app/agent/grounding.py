"""Best-effort guard against invented numbers. A number in the answer that sits next to a unit
(degrees, mm, %, rupees, quintal, kg, pH...) or is >= 100 must match a numeric value that appeared
in tool output / alerts / the user's question. Not a proof of truth: it catches made-up figures."""

import re
from typing import Any

_MARK = re.compile(r"\[S\d+(?:\s*,\s*S\d+)*\]")
CITE_IDS = re.compile(r"S\d+")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}:\d{2}")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_UNIT = re.compile(
    r"(?<![a-z])(?:mm|km|kg|qtl|ph|rs\.?|inr|quintals?)(?![a-z])"
    r"|[°℃%₹]|मिमी|क्विंटल|रुपये|रु|डिग्री|प्रतिशत|ಮಿ\.?ಮೀ|ಕ್ವಿಂಟಾಲ್|ರೂ|ಡಿಗ್ರಿ|ಶೇ"
)


def cited_ids(text: str) -> set[str]:
    return {i for m in _MARK.findall(text) for i in CITE_IDS.findall(m)}


def collect_numbers(obj: Any) -> list[float]:
    """Numeric leaves only (strings such as timestamps are deliberately ignored)."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, int | float):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in collect_numbers(v)]
    if isinstance(obj, list | tuple):
        return [n for v in obj for n in collect_numbers(v)]
    return []


def numbers_in_text(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUM.findall(_DATE.sub(" ", _MARK.sub(" ", text))):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def _close(v: float, known: list[float]) -> bool:
    tol = 1.0 if v < 100 else max(0.5, 0.005 * v)
    return any(abs(v - k) <= tol for k in known)


def find_ungrounded(text: str, known: list[float]) -> list[str]:
    clean = _DATE.sub(" ", _MARK.sub(" ", text))
    bad: list[str] = []
    for m in _NUM.finditer(clean):
        try:
            v = float(m.group().replace(",", ""))
        except ValueError:
            continue
        near = clean[max(0, m.start() - 4) : m.start()] + " " + clean[m.end() : m.end() + 10]
        if (v >= 100 or _UNIT.search(near.lower())) and not _close(v, known):
            bad.append(m.group())
    return bad
