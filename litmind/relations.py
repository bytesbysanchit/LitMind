"""Character relationship scoring.

Two separate questions, answered by two different tools:

1. HOW STRONGLY are two characters connected?   -> statistics (this module, offline)
2. WHAT is the relationship and WHY?              -> Gemini, using evidence sentences from (1)

The old scorer tried to answer both with keyword rules, which produced confident
but unreliable labels. Here the strength is a transparent number and the type/reason
is only produced from quoted evidence (and may be "unclear").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

from .characters import CharacterResult


@dataclass
class Relation:
    a: str
    b: str
    together: int                 # windows in which both appear
    strength: float               # 0..1 (cosine of the two presence vectors)
    pmi: float                    # >0 means they meet more often than chance
    evidence: list[str] = field(default_factory=list)
    # Filled by Gemini (optional):
    relation_type: str = ""
    reason: str = ""
    tone: str = ""


def compute_relations(
    result: CharacterResult,
    *,
    window: int = 3,
    min_together: int = 3,
    top_n: int = 20,
    evidence_per_pair: int = 3,
    evidence_chars: int = 320,
) -> list[Relation]:
    """Score every pair of characters by how often they appear in the same
    `window`-sentence block. Returns the strongest `top_n` pairs."""
    n_sent = len(result.sentences)
    if n_sent == 0 or len(result.characters) < 2:
        return []
    n_blocks = math.ceil(n_sent / window)
    presence = {c.name: {sid // window for sid in c.sentence_ids} for c in result.characters}

    rels: list[Relation] = []
    for ca, cb in combinations(result.characters, 2):
        A, B = presence[ca.name], presence[cb.name]
        both = A & B
        if len(both) < min_together:
            continue
        strength = len(both) / math.sqrt(len(A) * len(B))
        pmi = math.log((len(both) * n_blocks) / (len(A) * len(B)))
        rels.append(Relation(ca.name, cb.name, len(both), round(strength, 4), round(pmi, 3)))

    # Rank by strength, but ignore pairs that meet less often than chance would predict.
    rels = [r for r in rels if r.pmi > 0]
    rels.sort(key=lambda r: (-r.strength, -r.together))
    rels = rels[:top_n]

    # Evidence: blocks where both appear, spread evenly across the book.
    for r in rels:
        both = sorted(presence[r.a] & presence[r.b])
        if len(both) > evidence_per_pair:
            step = len(both) / evidence_per_pair
            both = [both[int(i * step)] for i in range(evidence_per_pair)]
        for blk in both:
            text = " ".join(result.sentences[blk * window:(blk + 1) * window])
            r.evidence.append(text[:evidence_chars].strip())
    return rels
