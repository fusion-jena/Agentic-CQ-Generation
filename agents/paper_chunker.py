"""
Recursive character splitter for research papers.

Style follows LangChain's RecursiveCharacterTextSplitter conventions, but kept
dependency-free. Tries to split on the largest natural separator that still
yields chunks within budget: section headings -> blank lines -> sentences -> spaces.
"""
from __future__ import annotations

import re
from typing import List


# Order matters: largest natural separator first.
_SEPARATORS = [
    "\n\n\n",   # major break (often before section heading)
    "\n\n",     # paragraph
    "\n",       # line
    ". ",       # sentence
    " ",        # word
    "",         # char (last resort)
]


def _split_with(text: str, sep: str) -> List[str]:
    if sep == "":
        return list(text)
    if sep == ". ":
        # Keep the period attached to the preceding sentence.
        parts = text.split(sep)
        return [p + ("." if i < len(parts) - 1 else "") for i, p in enumerate(parts)]
    return text.split(sep)


def _recursive_split(text: str, target_size: int, sep_idx: int = 0) -> List[str]:
    if len(text) <= target_size:
        return [text]
    if sep_idx >= len(_SEPARATORS):
        return [text[i:i + target_size] for i in range(0, len(text), target_size)]

    sep = _SEPARATORS[sep_idx]
    parts = _split_with(text, sep)
    if len(parts) == 1:
        return _recursive_split(text, target_size, sep_idx + 1)

    chunks: List[str] = []
    buf = ""
    glue = sep if sep != "" else ""
    for part in parts:
        candidate = buf + (glue if buf else "") + part
        if len(candidate) <= target_size:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        if len(part) <= target_size:
            buf = part
        else:
            # The part itself is too big - recurse with a smaller separator.
            sub = _recursive_split(part, target_size, sep_idx + 1)
            if sub:
                chunks.extend(sub[:-1])
                buf = sub[-1]
            else:
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out: List[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        out.append(prev_tail + chunks[i])
    return out


def _strip_boilerplate(text: str) -> str:
    """Light cleanup: collapse runaway whitespace, drop pure-numeric page-number lines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    lines = [ln for ln in text.split("\n") if not re.fullmatch(r"\s*\d{1,4}\s*", ln)]
    return "\n".join(lines).strip()


def chunk_paper(text: str, chunk_chars: int = 3500, chunk_overlap: int = 200) -> List[str]:
    """Split a paper into character-bounded chunks with overlap."""
    cleaned = _strip_boilerplate(text)
    if not cleaned:
        return []
    base = _recursive_split(cleaned, chunk_chars)
    base = [c.strip() for c in base if c and c.strip()]
    return _add_overlap(base, chunk_overlap)
