"""Parse an uploaded reference corpus (.xlsx, .csv, .json, .txt, .pdf) into ReferenceAnnotations.

Steps:  parse_corpus()  ->  suggest_mappings()  ->  (user reviews / edits)  ->  build_annotations()
Nothing is guessed silently: every automatic choice is exposed so the UI can show and override it.
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .schema import (RefCharacter, RefProtagonist, RefRelation, ReferenceAnnotations, split_list)

SUPPORTED = (".xlsx", ".csv", ".json", ".txt", ".pdf")
MAX_BYTES = 25 * 1024 * 1024
KINDS = ("characters", "relationships", "protagonist", "genre", "ignore")


class CorpusError(ValueError):
    """Raised with a message that is safe and useful to show to the user."""


@dataclass
class ParsedCorpus:
    filename: str
    kind: str                                   # xlsx | csv | json | txt | pdf
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    json_data: dict | list | None = None
    text: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def has_annotations_possible(self) -> bool:
        return bool(self.tables) or isinstance(self.json_data, dict)


@dataclass
class TableMapping:
    sheet: str
    kind: str                                   # one of KINDS
    columns: dict[str, str] = field(default_factory=dict)       # schema field -> column header
    confidence: float = 0.0                     # 0..1, how sure the auto-detector was


# --------------------------------------------------------------------------- parsing
def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise CorpusError("The file's text encoding could not be read.")


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")] if (~df.columns.str.startswith("Unnamed")).any() else df
    return df.reset_index(drop=True)


def parse_corpus(filename: str, data: bytes) -> ParsedCorpus:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise CorpusError(f"Unsupported file type '{ext or filename}'. Use one of: {', '.join(SUPPORTED)}.")
    if not data:
        raise CorpusError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise CorpusError(f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    out = ParsedCorpus(filename, ext.lstrip("."))

    if ext == ".xlsx":
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str, engine="openpyxl")
        except Exception as exc:                                    # corrupt, password protected, wrong format
            raise CorpusError("This is not a readable Excel (.xlsx) workbook. It may be corrupt or "
                              "password-protected.") from exc
        for name, df in sheets.items():
            df = _clean_df(df)
            if df.empty:
                out.warnings.append(f"Sheet '{name}' is empty and was skipped.")
            else:
                out.tables[str(name)] = df
        if not out.tables:
            raise CorpusError("The workbook has no sheets with data.")

    elif ext == ".csv":
        text = _decode(data)
        if not text.strip():
            raise CorpusError("The CSV file is empty.")
        try:
            sep = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
        except csv.Error:
            sep = ","
        try:
            df = _clean_df(pd.read_csv(io.StringIO(text), sep=sep, dtype=str))
        except Exception as exc:
            raise CorpusError("The CSV file could not be parsed. Check that it is a table with a header row.") from exc
        if df.empty:
            raise CorpusError("The CSV file has a header but no rows.")
        out.tables[Path(filename).stem] = df

    elif ext == ".json":
        try:
            obj = json.loads(_decode(data))
        except ValueError as exc:
            raise CorpusError(f"The JSON file is malformed ({exc}).") from exc
        if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
            out.tables[Path(filename).stem] = _clean_df(pd.DataFrame(obj).astype(str))
        elif isinstance(obj, dict) and obj:
            out.json_data = obj
            if isinstance(obj.get("text"), str):
                out.text = obj["text"]
        else:
            raise CorpusError("The JSON file must contain an object (see the annotation schema) or a list of records.")

    elif ext == ".txt":
        out.text = _decode(data).strip()
        if len(out.text) < 200:
            raise CorpusError("The text file is too short to be a book or a corpus (under 200 characters).")

    else:  # .pdf
        from ..text_io import read_upload
        try:
            out.text = read_upload(filename, data)
        except ValueError as exc:
            raise CorpusError(str(exc)) from exc
        except Exception as exc:
            raise CorpusError("The PDF could not be read.") from exc
    return out


# --------------------------------------------------------------------------- table detection
def _n(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).casefold())


SYNONYMS = {
    "characters": {
        "name": ["name", "character", "charactername", "characters", "person", "mainname"],
        "aliases": ["aliases", "alias", "alsoknownas", "aka", "othernames", "nicknames", "alternatenames"],
        "description": ["description", "details", "bio", "about", "summary", "role"],
        "traits": ["traits", "personality", "personalitytraits", "characteristics"]},
    "relationships": {
        "a": ["character1", "charactera", "source", "from", "person1", "char1", "entity1", "first", "a", "character"],
        "b": ["character2", "characterb", "target", "to", "person2", "char2", "entity2", "second", "b", "relatedto"],
        "type": ["relationship", "relation", "relationshiptype", "type", "relationtype", "category"],
        "direction": ["direction", "directed"],
        "evidence": ["evidence", "passage", "quote", "text", "excerpt"]},
    "protagonist": {
        "name": ["name", "protagonist", "protagonistname", "character"],
        "narrative_role": ["narrativerole", "role"],
        "traits": ["traits", "personality", "personalitytraits"],
        "motivations": ["motivations", "motivation", "goals", "goal", "motivationsgoals"],
        "conflicts": ["conflicts", "conflict", "majorconflicts"],
        "arc": ["arc", "characterarc", "development", "characterdevelopment"],
        "key_relationships": ["relationships", "keyrelationships", "importantrelationships"],
        "passages": ["passages", "evidence", "supportingpassages", "quotes"]},
    "genre": {"genre": ["genre", "genres", "label", "labels", "category", "bookgenre"]},
}
SHEET_HINTS = {"characters": ["character", "cast", "people"], "relationships": ["relation", "link", "network", "edges"],
               "protagonist": ["protagonist", "hero", "mainchar", "lead"], "genre": ["genre", "category", "label"]}
REQUIRED = {"characters": ["name"], "relationships": ["a", "b"], "protagonist": [], "genre": ["genre"]}


def _match_columns(kind: str, columns: list[str]) -> dict[str, str]:
    norm = {_n(c): c for c in columns}
    found, used = {}, set()
    for fld, syns in SYNONYMS[kind].items():
        for syn in syns:
            if syn in norm and norm[syn] not in used:
                found[fld] = norm[syn]
                used.add(norm[syn])
                break
    return found


def suggest_mapping(sheet: str, df: pd.DataFrame) -> TableMapping:
    cols = list(df.columns)
    # key/value layout for a protagonist sheet: two columns 'field' and 'value'
    if len(cols) == 2 and _n(cols[0]) in ("field", "attribute", "property", "key") and _n(cols[1]) in ("value", "details", "description"):
        return TableMapping(sheet, "protagonist", {"field": cols[0], "value": cols[1]}, 0.9)
    best = ("ignore", {}, 0.0)
    sname = _n(sheet)
    for kind in ("characters", "relationships", "protagonist", "genre"):
        cm = _match_columns(kind, cols)
        if any(r not in cm for r in REQUIRED[kind]):
            continue
        score = len(cm) / max(1, len(SYNONYMS[kind]))
        if any(h in sname for h in SHEET_HINTS[kind]):
            score += 0.5
        if kind == "relationships" and len(cm) < 2:
            continue
        if score > best[2]:
            best = (kind, cm, score)
    # a relationships table also matches 'characters' by its first column: prefer the more specific kind
    return TableMapping(sheet, best[0], best[1], min(1.0, best[2]))


def suggest_mappings(parsed: ParsedCorpus) -> list[TableMapping]:
    return [suggest_mapping(s, df) for s, df in parsed.tables.items()]


# --------------------------------------------------------------------------- build annotations
_PROTAG_KEYS = {"name": "name", "protagonist": "name", "aliases": "aliases", "alias": "aliases",
                "narrativerole": "narrative_role", "role": "narrative_role", "traits": "traits", "personality": "traits",
                "personalitytraits": "traits", "motivations": "motivations", "motivation": "motivations", "goals": "motivations",
                "conflicts": "conflicts", "conflict": "conflicts", "majorconflicts": "conflicts", "arc": "arc",
                "characterarc": "arc", "development": "arc", "characterdevelopment": "arc",
                "keyrelationships": "key_relationships", "relationships": "key_relationships",
                "importantrelationships": "key_relationships", "passages": "passages", "evidence": "passages"}
_DIRECTED = {"directed", "directional", "one-way", "oneway", "->", "→", "yes", "true", "1", "y"}


def _cell(row, col) -> str:
    if not col or col not in row.index:
        return ""
    v = row[col]
    return "" if pd.isna(v) else str(v).strip()


def build_annotations(parsed: ParsedCorpus, mappings: list[TableMapping] | None = None, *,
                      title: str = "", characters_exhaustive: bool = False,
                      relationships_exhaustive: bool = False, multi_label: bool | None = None) -> ReferenceAnnotations:
    ref = ReferenceAnnotations(book_title=title or Path(parsed.filename).stem, source_text=parsed.text,
                               characters_exhaustive=characters_exhaustive,
                               relationships_exhaustive=relationships_exhaustive, multi_label_genre=multi_label)
    if isinstance(parsed.json_data, dict):
        d = dict(parsed.json_data)
        alt = {"text": "source_text", "genre": "genres", "protagonist": "protagonists"}
        for k, v in alt.items():
            if k in d and v not in d:
                d[v] = d.pop(k)
        if isinstance(d.get("protagonists"), dict):
            d["protagonists"] = [d["protagonists"]]
        if isinstance(d.get("characters"), list):
            d["characters"] = [{"name": c} if isinstance(c, str) else c for c in d["characters"]]
        try:
            loaded = ReferenceAnnotations.model_validate({**d, "source_text": d.get("source_text", parsed.text)})
        except Exception as exc:
            raise CorpusError(f"The JSON annotations do not match the schema: {str(exc).splitlines()[0]}") from exc
        loaded.book_title = loaded.book_title or ref.book_title
        loaded.characters_exhaustive = loaded.characters_exhaustive or characters_exhaustive
        loaded.relationships_exhaustive = loaded.relationships_exhaustive or relationships_exhaustive
        if multi_label is not None:
            loaded.multi_label_genre = multi_label
        ref = loaded

    for m in mappings or []:
        df = parsed.tables.get(m.sheet)
        if df is None or m.kind in ("ignore", ""):
            continue
        c = m.columns
        if m.kind == "characters":
            for _, row in df.iterrows():
                name = _cell(row, c.get("name"))
                if name:
                    ref.characters.append(RefCharacter(name=name, aliases=_cell(row, c.get("aliases")),
                                                       description=_cell(row, c.get("description")), traits=_cell(row, c.get("traits"))))
        elif m.kind == "relationships":
            for _, row in df.iterrows():
                a, b = _cell(row, c.get("a")), _cell(row, c.get("b"))
                if a and b:
                    ref.relationships.append(RefRelation(a=a, b=b, type=_cell(row, c.get("type")),
                                                         directed=_cell(row, c.get("direction")).casefold() in _DIRECTED,
                                                         evidence=_cell(row, c.get("evidence"))))
        elif m.kind == "protagonist":
            if "field" in c and "value" in c:                         # key/value layout
                kv: dict[str, str] = {}
                for _, row in df.iterrows():
                    key = _PROTAG_KEYS.get(_n(_cell(row, c["field"])))
                    if key:
                        kv[key] = (kv[key] + "; " if key in kv else "") + _cell(row, c["value"])
                if kv.get("name"):
                    ref.protagonists.append(RefProtagonist(**kv))
            else:                                                     # one protagonist per row
                for _, row in df.iterrows():
                    vals = {f: _cell(row, col) for f, col in c.items()}
                    if vals.get("name"):
                        ref.protagonists.append(RefProtagonist(**vals))
        elif m.kind == "genre":
            for _, row in df.iterrows():
                ref.genres += split_list(_cell(row, c.get("genre")))
    ref.genres = list(dict.fromkeys(g for g in ref.genres if g))
    # relationship endpoints that are not in the characters list still need to be matchable
    known = {x.name.casefold() for x in ref.characters} | {a.casefold() for x in ref.characters for a in x.aliases}
    for r in ref.relationships:
        for end in (r.a, r.b):
            if end.casefold() not in known:
                ref.characters.append(RefCharacter(name=end, declared=False))
                known.add(end.casefold())
    return ref


def validate_reference(ref: ReferenceAnnotations) -> list[str]:
    """Human-readable problems and cautions about the reference (never raises)."""
    notes = []
    av = ref.available()
    if not any(av.values()):
        notes.append("No usable annotations were found (characters, relationships, protagonist or genre).")
    for cat, ok in av.items():
        if not ok:
            notes.append(f"No {cat} annotations: this category will be reported as Not Evaluated.")
    names = [c.name.casefold() for c in ref.characters if c.declared]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        notes.append("Duplicate reference characters: " + ", ".join(sorted(dup)))
    if ref.origin != "uploaded" and not ref.confirmed:
        notes.append("These annotations were proposed by an AI model and are UNVERIFIED until you confirm them.")
    return notes
