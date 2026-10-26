"""Turn raw model output into text that is safe and pleasant to display.

Gemini is asked for JSON, but fields can still arrive with artefacts: the whole JSON object stuffed
into the summary field, literal "\\n" sequences, a repeated paragraph, or a genre given as a stringified
list. These helpers repair that without rewriting valid content.
"""
from __future__ import annotations

import ast
import json
import re

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)
_META_LINE = re.compile(
    r'^\s*[{,]?\s*"?(primary_genre|sub_genres|themes|setting|tone|one_line_pitch|overview|summary|'
    r'genre_rationale)"?\s*[:=]', re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“‘'(])")


def _maybe_json(text: str):
    t = _FENCE.sub("", text.strip())
    if t[:1] in "{[":
        try:
            return json.loads(t)
        except ValueError:
            return None
    return None


def _unescape(s: str) -> str:
    return (s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ")
             .replace('\\"', '"').replace("\r\n", "\n"))


def clean_summary(raw) -> str:
    """Readable prose with paragraph breaks; never raw JSON. Returns '' if nothing usable."""
    if raw is None:
        return ""
    if isinstance(raw, dict):
        raw = raw.get("summary") or raw.get("detailed_summary") or ""
    text = str(raw).strip()
    obj = _maybe_json(text)
    if isinstance(obj, dict):
        text = str(obj.get("summary") or obj.get("detailed_summary") or "")
    elif isinstance(obj, list):
        text = " ".join(str(x) for x in obj)
    else:
        # a broken/partial JSON object: pull the summary field out with a regex
        m = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m and text.lstrip()[:1] in "{\"":
            try:
                text = json.loads('"' + m.group(1) + '"')
            except ValueError:
                text = m.group(1)
    text = _unescape(text)
    lines = [ln for ln in text.split("\n") if not _META_LINE.match(ln)]
    text = "\n".join(lines).strip().strip('"').strip()
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) == 1 and len(paras[0]) > 700:                # one wall of text: group sentences
        sents = _SENT_SPLIT.split(paras[0])
        paras = [" ".join(sents[i:i + 4]) for i in range(0, len(sents), 4)]
    out, seen = [], set()
    for p in paras:                                            # drop repeated paragraphs and sentences
        key = re.sub(r"\W+", "", p.casefold())
        if not key or key in seen:
            continue
        seen.add(key)
        sents, kept, sseen = _SENT_SPLIT.split(p), [], set()
        for s in sents:
            k = re.sub(r"\W+", "", s.casefold())
            if k and k not in sseen:
                sseen.add(k)
                kept.append(s)
        out.append(" ".join(kept))
    return "\n\n".join(out)


def summary_paragraphs(raw) -> list[str]:
    return [p for p in clean_summary(raw).split("\n\n") if p]


def clean_label_list(value) -> list[str]:
    """Genre-like input (list, JSON text, "a, b", python repr) -> clean unique labels."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, dict):
        items = list(value.values())
    else:
        s = str(value).strip()
        obj = _maybe_json(s)
        if obj is None and s[:1] in "[(":
            try:
                obj = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                obj = None
        if isinstance(obj, (list, tuple)):
            items = list(obj)
        elif isinstance(obj, dict):
            items = list(obj.values())
        else:
            items = re.split(r"[;,|/]\s*|\s+and\s+", s) if s else []
    out, seen = [], set()
    for it in items:
        label = re.sub(r"\s+", " ", str(it)).strip(" \"'[]{}")
        if label and label.casefold() not in seen:
            seen.add(label.casefold())
            out.append(label)
    return out


def format_genres(primary, secondary=None) -> tuple[str, list[str]]:
    """(primary label, secondary labels) as readable strings; primary never repeated in secondary."""
    prim_list = clean_label_list(primary)
    main = prim_list[0] if prim_list else ""
    extra = prim_list[1:] + clean_label_list(secondary)
    seen, sec = {main.casefold()}, []
    for g in extra:
        if g.casefold() not in seen:
            seen.add(g.casefold())
            sec.append(g)
    return main, sec
