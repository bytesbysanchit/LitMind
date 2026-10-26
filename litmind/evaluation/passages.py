"""Local passage retrieval, so Gemini only ever sees a handful of relevant excerpts, never the whole book."""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

_SENT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = 900) -> list[str]:
    chunks: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) < 40:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        cur = ""
        for s in _SENT.split(para):
            if cur and len(cur) + len(s) > max_chars:
                chunks.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if len(cur) >= 40:
            chunks.append(cur)
    return chunks


class PassageIndex:
    def __init__(self, text: str, max_chars: int = 900):
        self.chunks = chunk_text(text, max_chars) if text else []
        self._vec = self._mat = None
        if self.chunks:
            self._vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=1)
            self._mat = self._vec.fit_transform(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 3, max_len: int = 420) -> list[tuple[int, str]]:
        """Top-k (passage id, passage text) for a query. Ids are stable for this index."""
        if not self.chunks or not query.strip():
            return []
        q = self._vec.transform([query])
        sims = linear_kernel(q, self._mat).ravel()
        order = sims.argsort()[::-1][:k]
        return [(int(i), self.chunks[i][:max_len]) for i in order if sims[i] > 0]

    def get(self, pid: int) -> str:
        return self.chunks[pid] if 0 <= pid < len(self.chunks) else ""


def sample_text(text: str, budget: int = 140_000, windows: int = 6) -> str:
    """Head + evenly spaced windows + tail: a bounded, representative sample for long books."""
    if len(text) <= budget:
        return text
    head, tail = int(budget * 0.2), int(budget * 0.14)
    w = (budget - head - tail) // windows
    parts = [text[:head]]
    span = len(text) - head - tail
    for i in range(windows):
        start = head + int(span * (i + 0.5) / windows) - w // 2
        parts.append(text[start:start + w])
    parts.append(text[-tail:])
    return "\n\n[...]\n\n".join(parts)
