"""Reference-annotation schema (what an uploaded corpus is converted into) and the prediction bundle.

JSON files may use this schema directly:

{
  "book_title": "Dracula",
  "characters": [{"name": "Mina Harker", "aliases": ["Mina Murray", "Mina"], "description": "...", "traits": ["brave"]}],
  "relationships": [{"a": "Lucy Westenra", "b": "Arthur Holmwood", "type": "romance", "directed": false, "evidence": "..."}],
  "protagonists": [{"name": "Jonathan Harker", "narrative_role": "...", "traits": [], "motivations": "...",
                    "conflicts": "...", "arc": "...", "key_relationships": [], "passages": []}],
  "genres": ["Gothic", "Horror"],
  "multi_label_genre": true,
  "characters_exhaustive": false,
  "text": "optional raw book text used to find supporting passages"
}
Every field is optional; a category with no annotations is reported as "Not Evaluated".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator

from ..gemini_client import ProtagonistAnalysis


def split_list(value) -> list[str]:
    """'a; b | c' / ['a','b'] / None  ->  clean list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        parts = [str(v) for v in value]
    else:
        s = str(value).strip()
        if not s or s.casefold() in ("nan", "none", "null", "n/a", "-"):
            return []
        parts = re.split(r"[;|\n]+", s) if re.search(r"[;|\n]", s) else s.split(",")
    return [p.strip() for p in parts if p and p.strip()]


def _text(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.casefold() in ("nan", "none", "null") else s


class RefCharacter(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    traits: list[str] = Field(default_factory=list)
    declared: bool = True              # False = only seen as a relationship endpoint

    _l = field_validator("aliases", "traits", mode="before")(lambda v: split_list(v))
    _t = field_validator("name", "description", mode="before")(lambda v: _text(v))


class RefRelation(BaseModel):
    a: str
    b: str
    type: str = ""
    directed: bool = False
    evidence: str = ""

    _t = field_validator("a", "b", "type", "evidence", mode="before")(lambda v: _text(v))


class RefProtagonist(BaseModel):
    name: str = ""
    aliases: list[str] = Field(default_factory=list)
    narrative_role: str = ""
    traits: list[str] = Field(default_factory=list)
    motivations: str = ""
    conflicts: str = ""
    arc: str = ""
    key_relationships: list[str] = Field(default_factory=list)
    passages: list[str] = Field(default_factory=list)

    _l = field_validator("aliases", "traits", "key_relationships", "passages", mode="before")(lambda v: split_list(v))
    _t = field_validator("name", "narrative_role", "motivations", "conflicts", "arc", mode="before")(lambda v: _text(v))


class ReferenceAnnotations(BaseModel):
    book_title: str = ""
    characters: list[RefCharacter] = Field(default_factory=list)
    relationships: list[RefRelation] = Field(default_factory=list)
    protagonists: list[RefProtagonist] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    multi_label_genre: bool | None = None
    characters_exhaustive: bool = False
    relationships_exhaustive: bool = False
    source_text: str = ""
    origin: str = "uploaded"           # "uploaded" | "ai_proposed_user_confirmed"
    confirmed: bool = True

    _l = field_validator("genres", mode="before")(lambda v: split_list(v))

    def available(self) -> dict[str, bool]:
        return {"characters": any(c.declared and c.name for c in self.characters),
                "relationships": any(r.a and r.b for r in self.relationships),
                "protagonist": any(p.name for p in self.protagonists),
                "genre": bool(self.genres)}

    @property
    def is_multi_label(self) -> bool:
        return self.multi_label_genre if self.multi_label_genre is not None else len(self.genres) > 1


# --------------------------------------------------------------------------- predictions
@dataclass
class PredCharacter:
    name: str
    aliases: list[str] = field(default_factory=list)
    mentions: int = 0
    description: str = ""


@dataclass
class PredRelation:
    a: str
    b: str
    type: str = ""                     # empty until Gemini explains it; we never guess
    reason: str = ""
    strength: float = 0.0
    together: int = 0


@dataclass
class Prediction:
    book_title: str = ""
    characters: list[PredCharacter] = field(default_factory=list)
    relationships: list[PredRelation] = field(default_factory=list)
    protagonist: ProtagonistAnalysis | None = None
    genres: list[str] = field(default_factory=list)      # primary genre first
    summary: str = ""

    def available(self) -> dict[str, bool]:
        return {"characters": bool(self.characters), "relationships": True,       # empty list is a valid result
                "protagonist": self.protagonist is not None, "genre": bool(self.genres)}


def build_prediction(book_title, chars_result, relations, roles: dict[str, str] | None = None,
                     protagonist: ProtagonistAnalysis | None = None, analysis=None) -> Prediction:
    """Assemble LitMind's current results into a Prediction (all inputs optional)."""
    roles = {k.casefold(): v for k, v in (roles or {}).items()}
    chars = []
    if chars_result is not None:
        for c in chars_result.characters:
            chars.append(PredCharacter(c.name, c.alias_list, c.mentions, roles.get(c.name.casefold(), "")))
    rels = [PredRelation(r.a, r.b, (r.relation_type or "").strip(), r.reason or "", r.strength, r.together)
            for r in (relations or [])]
    genres, summary = [], ""
    if analysis is not None:
        genres = [analysis.primary_genre] + list(analysis.sub_genres)
        summary = analysis.summary
    return Prediction(book_title, chars, rels, protagonist, genres, summary)
