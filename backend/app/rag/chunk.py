"""Sentence-aware chunking with overlap. Chunks never cross page boundaries (exact citations)."""

import re

_SENT = re.compile(r"(?<=[.!?\u0964])\s+")  # includes the Devanagari danda


def _len(parts: list[str]) -> int:
    return sum(map(len, parts)) + max(len(parts) - 1, 0)


def _units(text: str, max_chars: int, overlap: int) -> list[str]:
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        for sent in _SENT.split(" ".join(para.split())):
            if not sent:
                continue
            if len(sent) <= max_chars:
                out.append(sent)
            else:  # no sentence breaks (tables, lists): hard-slice with overlap
                step = max_chars - overlap
                out.extend(sent[i : i + max_chars] for i in range(0, len(sent), step))
    return out


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 150) -> list[str]:
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")
    chunks: list[str] = []
    cur: list[str] = []
    for unit in _units(text, max_chars, overlap):
        if cur and _len(cur) + 1 + len(unit) > max_chars:
            chunks.append(" ".join(cur))
            carry: list[str] = []
            for prev in reversed(cur):  # trailing sentences become the next chunk's lead-in
                if _len([prev, *carry]) > overlap:
                    break
                carry.insert(0, prev)
            cur = carry if _len([*carry, unit]) <= max_chars else []
        cur.append(unit)
    if cur:
        chunks.append(" ".join(cur))
    return chunks
