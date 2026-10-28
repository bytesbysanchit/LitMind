"""Gemini integration: summary + genre, character review, relationship explanations.

Setup
-----
    pip install google-genai python-dotenv
    # put your key in .env (never in code, never in git):
    GEMINI_API_KEY=your-key-here
    # optional:
    GEMINI_MODEL=gemini-3.8-flash

The key is read from the environment (or a `.env` file). Model names change over time,
so the model is configurable and a fallback list is tried if the first one is not found.

Long books are summarised with map-reduce (chunk notes -> final analysis) so that a
whole novel never has to fit into one request.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from pydantic import BaseModel, Field

DEFAULT_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
SINGLE_CALL_CHARS = int(os.getenv("GEMINI_SINGLE_CALL_CHARS", "350000"))
CHUNK_CHARS = int(os.getenv("GEMINI_CHUNK_CHARS", "60000"))

GENRES = [
    "Literary Fiction", "Mystery", "Crime", "Thriller", "Horror", "Gothic", "Fantasy",
    "Science Fiction", "Romance", "Historical Fiction", "Adventure", "Drama / Play",
    "Humour / Satire", "Young Adult", "Children's", "Biography / Memoir", "Non-fiction", "Other",
]


class GeminiError(RuntimeError):
    """Raised with a message that is safe to show to the user."""


# --------------------------------------------------------------------------- #
# Response schemas
# --------------------------------------------------------------------------- #
class BookAnalysis(BaseModel):
    one_line_pitch: str = Field(description="One sentence that sells the book.")
    summary: str = Field(description="Plot summary, 200-300 words, plain prose.")
    primary_genre: str
    sub_genres: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    setting: str = ""
    tone: str = ""


class ChunkNotes(BaseModel):
    notes: str = Field(description="150-200 words: what happens in this part, who is involved.")


class CharacterRole(BaseModel):
    name: str
    role: str = Field(description="Max 12 words: who this character is in the story.")


class CharacterReview(BaseModel):
    merge_groups: list[list[str]] = Field(default_factory=list)
    drop: list[str] = Field(default_factory=list)
    roles: list[CharacterRole] = Field(default_factory=list)

class ProposedCharacter(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""


class ProposedRelation(BaseModel):
    a: str
    b: str
    type: str = ""
    evidence: str = ""


class ReferenceProposal(BaseModel):
    characters: list[ProposedCharacter] = Field(default_factory=list)
    relationships: list[ProposedRelation] = Field(default_factory=list)


class RelationExplanation(BaseModel):
    a: str
    b: str
    relation_type: str = Field(description="Short label, e.g. 'husband and wife', 'rivals', 'unclear'.")
    reason: str = Field(description="One or two sentences grounded in the evidence.")
    tone: str = Field(description="positive, negative, mixed or neutral")


class RelationExplanations(BaseModel):
    relations: list[RelationExplanation] = Field(default_factory=list)


@dataclass
class BookReport:
    analysis: BookAnalysis
    model: str
    calls: int
    mode: str                      # "single" or "map-reduce"


# --------------------------------------------------------------------------- #
# Backend (real or fake)
# --------------------------------------------------------------------------- #
class Backend(Protocol):
    def generate(self, *, model: str, prompt: str, schema: type[BaseModel], system: str | None) -> str: ...


class _GenAIBackend:
    """Thin wrapper over the official `google-genai` SDK."""

    def __init__(self, api_key: str):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise GeminiError("Install the SDK first:  pip install google-genai") from exc
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def generate(self, *, model, prompt, schema, system):
        from google.genai import types
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            system_instruction=system,
            temperature=0.3,
        )
        resp = self._client.models.generate_content(model=model, contents=prompt, config=cfg)
        return resp.text or ""


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    env = Path(".env")
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_api_key() -> str | None:
    _load_dotenv()
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
SYSTEM = (
    "You are a careful literary analyst. Base every statement on the supplied text. "
    "If the text does not show something, say so instead of guessing. Return only JSON."
)


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 backend: Backend | None = None, max_retries: int = 2,
                 sleep: Callable[[float], None] = time.sleep):
        self.models = [model] if model else []
        env_model = os.getenv("GEMINI_MODEL")
        for m in ([env_model] if env_model else []) + DEFAULT_MODELS:
            if m and m not in self.models:
                self.models.append(m)
        if backend is None:
            key = api_key or get_api_key()
            if not key:
                raise GeminiError("No Gemini API key found. Add GEMINI_API_KEY to your .env file.")
            backend = _GenAIBackend(key)
        self.backend = backend
        self.max_retries = max_retries
        self._sleep = sleep
        self.calls = 0
        self.active_model = self.models[0]

    # ----------------------------------------------------------------- core
    def _call(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        last: Exception | None = None
        for model in list(self.models):
            for attempt in range(self.max_retries + 1):
                try:
                    self.calls += 1
                    raw = self.backend.generate(model=model, prompt=prompt, schema=schema, system=SYSTEM)
                    self.active_model = model
                    return schema.model_validate(_parse_json(raw))
                except Exception as exc:  # noqa: BLE001 - classify below
                    last = exc
                    kind = _classify(exc)
                    if kind == "auth":
                        raise GeminiError("Gemini rejected the API key (invalid or lacks access).") from exc
                    if kind == "model":
                        break                                   # try the next model name
                    if kind == "retry":
                        if attempt < self.max_retries:
                            self._sleep(min(60, 2 ** attempt * 3))
                            continue
                        break                                   # still overloaded: try the next model
                    if kind == "bad_output" and attempt < 1:
                        continue                                # one more try for malformed JSON
                    raise GeminiError(f"Gemini request failed: {_short(exc)}") from exc
        if _is_network(last):
            raise GeminiError(
                "Could not keep a connection to Gemini (network dropped: " + _short(last) + "). "
                "Check your internet, turn off any VPN/proxy/firewall, then try again. "
                "Run `python3 scripts/check_gemini.py` to test the connection on its own.") from last
        raise GeminiError(f"None of the models worked ({', '.join(self.models)}): {_short(last)}")

    # --------------------------------------------------- summary and genre
    def analyze_book(self, text: str, title: str = "", author: str = "",
                     progress: Callable[[str], None] | None = None) -> BookReport:
        say = progress or (lambda _m: None)
        start_calls = self.calls
        hint = f'Title: "{title}"' + (f" by {author}" if author else "") + "\n" if title else ""
        final_prompt = (
            f"{hint}Analyse this book and return JSON with: one_line_pitch, summary (200-300 words, "
            f"cover the whole story arc; you may name the ending), primary_genre (choose ONE of: "
            f"{', '.join(GENRES)}), sub_genres (0-3), themes (3-6 short phrases), setting, tone.\n\n"
        )
        if len(text) <= SINGLE_CALL_CHARS:
            say("Reading the whole book in one request…")
            analysis = self._call(final_prompt + "BOOK TEXT:\n" + text, BookAnalysis)
            mode = "single"
        else:
            parts = _split_chunks(text, CHUNK_CHARS)
            notes = []
            for i, part in enumerate(parts, 1):
                say(f"Summarising part {i} of {len(parts)}…")
                n = self._call(
                    f"{hint}This is part {i} of {len(parts)} of a novel. Write notes on what happens "
                    f"in THIS part only (150-200 words), naming the main characters involved.\n\n{part}",
                    ChunkNotes)
                notes.append(f"Part {i}: {n.notes}")
            say("Combining notes into the final analysis…")
            opening = text[:3000]
            analysis = self._call(
                final_prompt + "You are given chronological notes for each part of the book, plus its opening.\n\n"
                "OPENING:\n" + opening + "\n\nNOTES:\n" + "\n\n".join(notes), BookAnalysis)
            mode = "map-reduce"
        analysis.primary_genre = normalise_genre(analysis.primary_genre)
        return BookReport(analysis, self.active_model, self.calls - start_calls, mode)

    # ------------------------------------------------ character review pass
    def review_characters(self, result, contexts: dict[str, list[str]] | None = None) -> CharacterReview:
        """Ask Gemini to merge aliases of the same person and flag non-characters.
        Only names that were already detected can be affected (see characters.apply_review)."""
        lines = []
        for c in result.characters:
            alias = ", ".join(c.alias_list[:6]) or "-"
            ctx = " | ".join((contexts or {}).get(c.name, [])[:2])
            lines.append(f"- {c.name} (mentions: {c.mentions}; also called: {alias})" + (f"\n    e.g. {ctx}" if ctx else ""))
        prompt = (
            "Below are names detected automatically in a novel. Decide:\n"
            "1. merge_groups: groups of listed names that are definitely the SAME person (e.g. maiden and "
            "married name, a person and their noble title). Use the listed names exactly; first name in a "
            "group is kept. Only merge when you are confident.\n"
            "2. drop: listed names that are NOT characters (places, rivers, nationalities, objects, real "
            "historical figures only mentioned in passing).\n"
            "3. roles: for each remaining character, a role of at most 12 words.\n\n"
            "DETECTED NAMES:\n" + "\n".join(lines))
        return self._call(prompt, CharacterReview)

    def propose_reference(
        self,
        sampled_text: str,
        title: str = ""
    ) -> ReferenceProposal:
        prompt = (
            f"Title: {title}\n"
            "Extract characters with canonical names and aliases, and relationships "
            "with types and short evidence quotes. Use only the supplied text. "
            "Do not invent facts.\n\nTEXT:\n"
            + sampled_text[:30000]
        )
        return self._call(prompt, ReferenceProposal)

    # ------------------------------------------- relationship explanations
    def explain_relations(self, relations: list, max_pairs: int = 12) -> list:
        """Fill relation_type / reason / tone on `relations` using only their evidence."""
        pairs = relations[:max_pairs]
        if not pairs:
            return relations
        blocks = []
        for r in pairs:
            ev = "\n".join(f"  [{i + 1}] {e}" for i, e in enumerate(r.evidence)) or "  (no evidence)"
            blocks.append(f"PAIR: {r.a} & {r.b}\n{ev}")
        prompt = (
            "For each pair, state the relationship between the two characters using ONLY the evidence "
            "passages. If the passages do not show it, use relation_type 'unclear' and say what the "
            "passages do show. Keep `a` and `b` exactly as given.\n\n" + "\n\n".join(blocks))
        out = self._call(prompt, RelationExplanations)
        lookup = {(x.a.casefold(), x.b.casefold()): x for x in out.relations}
        for r in pairs:
            hit = lookup.get((r.a.casefold(), r.b.casefold())) or lookup.get((r.b.casefold(), r.a.casefold()))
            if hit:
                r.relation_type, r.reason, r.tone = hit.relation_type, hit.reason, hit.tone
        return relations


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _split_chunks(text: str, size: int) -> list[str]:
    parts, pos = [], 0
    while pos < len(text):
        end = min(pos + size, len(text))
        if end < len(text):
            cut = text.rfind("\n\n", pos + size // 2, end)
            if cut != -1:
                end = cut
        parts.append(text[pos:end])
        pos = end
    return parts


def _parse_json(raw: str):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    if not raw:
        raise ValueError("empty response")
    return json.loads(raw)


def normalise_genre(g: str) -> str:
    low = (g or "").strip().casefold()
    for known in GENRES:
        if low == known.casefold():
            return known
    for known in GENRES:
        if known.casefold() in low or low in known.casefold() and low:
            return known
    return (g or "Other").strip() or "Other"


def _status(exc: Exception) -> int | None:
    for attr in ("code", "status_code", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    m = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(m.group(1)) if m else None


def _is_network(exc: BaseException | None) -> bool:
    """Connection reset / dropped / timed out / DNS / TLS problems, however the SDK wraps them."""
    seen = 0
    while exc is not None and seen < 6:
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)) and not isinstance(exc, FileNotFoundError):
            return True
        name = type(exc).__name__
        if any(k in name for k in ("ReadError", "ConnectError", "RemoteProtocol", "Timeout",
                                   "ServerDisconnected", "ProxyError", "SSLError", "WriteError")):
            return True
        msg = str(exc).lower()
        if any(k in msg for k in ("connection reset", "connection aborted", "connection refused",
                                  "timed out", "name resolution", "ssl", "broken pipe", "eof occurred")):
            return True
        exc = exc.__cause__ or exc.__context__
        seen += 1
    return False


def _classify(exc: Exception) -> str:
    if _is_network(exc):
        return "retry"
    code = _status(exc)
    msg = str(exc).lower()
    if isinstance(exc, (json.JSONDecodeError, ValueError)) or "validation error" in msg:
        return "bad_output"
    if code in (401, 403) or "api key not valid" in msg or "permission_denied" in msg:
        return "auth"
    if code == 404 or "not found" in msg and "model" in msg:
        return "model"
    if code in (429, 500, 502, 503, 504) or "resource_exhausted" in msg or "unavailable" in msg:
        return "retry"
    return "other"


def _short(exc: Exception | None) -> str:
    s = str(exc or "unknown error")
    return s if len(s) < 300 else s[:300] + "…"
