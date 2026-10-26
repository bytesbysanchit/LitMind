"""Deterministic matching and metrics. No LLM is involved here, so these numbers are reproducible.

Conventions
-----------
* A reference item that has no counterpart in the prediction is a MISS (false negative).
* A prediction that has no counterpart is "Not in reference". It only counts as a false positive
  (lowering precision) when the reference is declared exhaustive; otherwise precision is reported as
  unavailable, because a partial reference cannot prove a prediction wrong.
* Missing reference *fields* are "Not Evaluated", never an incorrect prediction.
* A short name that could belong to several reference characters is NEVER guessed: it is listed under
  "Needs confirmation" and excluded from the metrics until the user resolves it.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ..characters import parse_surface
from .schema import (PredCharacter, PredRelation, Prediction, RefCharacter, RefProtagonist, RefRelation,
                     ReferenceAnnotations)

CORRECT, PARTIAL, INCORRECT, NOT_EVAL = "Correct", "Partial", "Incorrect", "Not Evaluated"


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


# --------------------------------------------------------------------------- names
def norm_name(s: str) -> str:
    """Comparable form: accents/case/titles/possessives removed. 'Dr. Seward's' -> 'seward'."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    _, toks = parse_surface(s if any(c.isupper() for c in s) else s.title())
    base = " ".join(t.lower() for t in toks) if toks else re.sub(r"[^a-z0-9 ]", "", s.casefold()).strip()
    return re.sub(r"^the ", "", base).strip()


def forms(name: str, aliases=()) -> set[str]:
    return {f for f in (norm_name(x) for x in [name, *aliases]) if f}


def _tokens(f: str) -> set[str]:
    return set(f.split())


# --------------------------------------------------------------------------- characters
@dataclass
class CharRow:
    ref: RefCharacter | None
    pred: PredCharacter | None
    status: str                     # Matched | Missing | Not in reference | Duplicate | Needs confirmation
    how: str = ""                   # exact | alias | name part | confirmed by user
    candidates: list[str] = field(default_factory=list)


@dataclass
class CharMatch:
    rows: list[CharRow]
    pred_to_ref: dict[int, int]     # pred index -> index into ref.characters (declared or not)
    counts: dict


def match_characters(pred: list[PredCharacter], ref: list[RefCharacter],
                     overrides: dict[str, str | None] | None = None) -> CharMatch:
    """overrides: {predicted name: reference name | None}. None = user says 'not in the reference'."""
    overrides = {norm_name(k): (norm_name(v) if v else None) for k, v in (overrides or {}).items()}
    rform = [forms(r.name, r.aliases) for r in ref]
    by_canon = {norm_name(r.name): i for i, r in enumerate(ref)}
    assign: dict[int, tuple[int | None, str, list[int]]] = {}       # pred idx -> (ref idx, how, candidates)
    for pi, p in enumerate(pred):
        pf = forms(p.name, p.aliases)
        key = norm_name(p.name)
        if key in overrides:
            tgt = overrides[key]
            assign[pi] = (by_canon.get(tgt) if tgt else None, "confirmed by user", [])
            continue
        exact = [i for i, rf in enumerate(rform) if pf & rf]
        if len(exact) > 1:                                          # prefer the one matching the main name
            main = [i for i in exact if key in rform[i]]
            exact = main if len(main) == 1 else exact
        if len(exact) == 1:
            assign[pi] = (exact[0], "exact" if key in rform[exact[0]] else "alias", [])
            continue
        if len(exact) > 1:
            assign[pi] = (None, "ambiguous", exact)
            continue
        part = []
        for i, rf in enumerate(rform):                              # name part: 'Mina' ~ 'Mina Harker'
            for a in {key}:
                ta = _tokens(a)
                if any(ta and (ta < _tokens(f) or _tokens(f) < ta) for f in rf):
                    part.append(i)
                    break
        if len(part) == 1:
            assign[pi] = (part[0], "name part", [])
        elif len(part) > 1:
            assign[pi] = (None, "ambiguous", part)
        else:
            assign[pi] = (None, "none", [])

    rows, p2r, taken = [], {}, {}
    order = sorted(range(len(pred)), key=lambda i: -pred[i].mentions)
    for pi in order:
        ri, how, cands = assign[pi]
        if how == "ambiguous":
            rows.append(CharRow(None, pred[pi], "Needs confirmation", how, [ref[c].name for c in cands]))
        elif ri is None:
            rows.append(CharRow(None, pred[pi], "Not in reference", how))
        elif ri in taken:
            rows.append(CharRow(ref[ri], pred[pi], "Duplicate", how))
            p2r[pi] = ri
        else:
            taken[ri] = pi
            p2r[pi] = ri
            rows.append(CharRow(ref[ri], pred[pi], "Matched", how))
    declared = [i for i, r in enumerate(ref) if r.declared]
    for ri in declared:
        if ri not in taken:
            rows.append(CharRow(ref[ri], None, "Missing"))
    tp = sum(1 for ri in declared if ri in taken)
    counts = {"tp": tp, "fn": len(declared) - tp, "fp": sum(r.status == "Not in reference" for r in rows),
              "dup": sum(r.status == "Duplicate" for r in rows), "unresolved": sum(r.status == "Needs confirmation" for r in rows),
              "ref_total": len(declared), "pred_total": len(pred)}
    return CharMatch(rows, p2r, counts)


def character_metrics(m: CharMatch, exhaustive: bool) -> dict:
    c = m.counts
    out = {"recall": c["tp"] / c["ref_total"] if c["ref_total"] else None, **{k: c[k] for k in ("tp", "fn", "dup", "unresolved")},
           "not_in_reference": c["fp"], "ref_total": c["ref_total"], "pred_total": c["pred_total"]}
    if exhaustive:
        fp = c["fp"] + c["dup"]                                     # duplicates lower precision too
        d = prf(c["tp"], fp, c["fn"])
        out.update(precision=d["precision"], f1=d["f1"], fp=fp)
    else:
        out.update(precision=None, f1=None, fp=None,
                   precision_note="Not computed: the reference is not declared exhaustive, so extra predictions "
                                  "cannot be called wrong.")
    return out


# --------------------------------------------------------------------------- relationships
CANON = {
    "friendship": ["friend", "friends", "friendship", "companion", "companions", "ally", "allies", "comrade", "buddy"],
    "family": ["family", "sibling", "siblings", "brother", "sister", "brothers", "sisters", "parent", "mother", "father",
               "child", "son", "daughter", "cousin", "uncle", "aunt", "relative", "kin", "stepfather", "stepdaughter"],
    "romance": ["romance", "romantic", "lover", "lovers", "love", "engaged", "fiance", "fiancee", "betrothed", "married",
                "marriage", "spouse", "husband", "wife", "couple", "suitor", "courtship"],
    "conflict": ["conflict", "enemy", "enemies", "rival", "rivals", "rivalry", "antagonist", "antagonism", "foe", "hostile",
                 "adversary", "opponent", "opponents", "nemesis"],
    "mentorship": ["mentor", "mentorship", "teacher", "student", "pupil", "master", "apprentice", "guide", "protege"],
    "professional": ["professional", "employer", "employee", "colleague", "colleagues", "coworker", "partner", "partners",
                     "business", "client", "doctor", "patient", "detective", "assistant", "servant", "work"],
}
_CANON_LOOKUP = {w: k for k, ws in CANON.items() for w in ws}
UNCLEAR = {"", "unclear", "unknown", "none", "n/a", "not determined", "insufficient evidence"}


def canon_type(t: str) -> str | None:
    """Relationship label -> canonical class, or None if it cannot be classified confidently."""
    s = re.sub(r"[^a-z ]", " ", t.casefold()).split()
    hits = {_CANON_LOOKUP[w] for w in s if w in _CANON_LOOKUP}
    return next(iter(hits)) if len(hits) == 1 else None


@dataclass
class RelRow:
    ref: RefRelation | None
    pred: PredRelation | None
    status: str             # Correct | Partial | Incorrect | Missing | Not in reference | Duplicate | Pending semantic check
    pair_match: bool = False
    note: str = ""
    key: str = ""


def _ref_entity(name: str, ref: list[RefCharacter]) -> int | None:
    f = norm_name(name)
    for i, r in enumerate(ref):
        if f in forms(r.name, r.aliases):
            return i
    return None


def match_relationships(pred: list[PredRelation], pred_chars: list[PredCharacter], ref_rels: list[RefRelation],
                        ref_chars: list[RefCharacter], cm: CharMatch) -> list[RelRow]:
    pidx = {norm_name(c.name): i for i, c in enumerate(pred_chars)}
    ends = lambda r: tuple(_ref_entity(x, ref_chars) for x in (r.a, r.b))
    ref_key = {}
    for j, r in enumerate(ref_rels):
        a, b = ends(r)
        ref_key[j] = (a, b) if r.directed else tuple(sorted((a, b), key=lambda x: -1 if x is None else x))
    rows, used_ref, seen_pred = [], set(), {}
    for p in pred:
        a = cm.pred_to_ref.get(pidx.get(norm_name(p.a), -1))
        b = cm.pred_to_ref.get(pidx.get(norm_name(p.b), -1))
        if a is None or b is None:
            rows.append(RelRow(None, p, "Not in reference", note="One or both characters are not matched to the reference."))
            continue
        k = tuple(sorted((a, b)))
        hit = next((j for j, rk in ref_key.items() if (rk == k or (ref_rels[j].directed and rk == (a, b)))), None)
        if hit is None:
            rows.append(RelRow(None, p, "Not in reference", note="This pair has no relationship in the reference."))
        elif k in seen_pred:
            rows.append(RelRow(ref_rels[hit], p, "Duplicate", True, "Same pair already predicted."))
        else:
            seen_pred[k] = hit
            used_ref.add(hit)
            rows.append(_compare_type(ref_rels[hit], p))
    for j, r in enumerate(ref_rels):
        if j not in used_ref:
            rows.append(RelRow(r, None, "Missing", note="Reference relationship not predicted."))
    return rows


def _compare_type(ref: RelRow | RefRelation, p: PredRelation) -> RelRow:
    rt, pt = (ref.type or "").strip(), (p.type or "").strip()
    if not rt:
        return RelRow(ref, p, "Correct", True, "Pair matches; the reference gives no type, so type was not evaluated.")
    if pt.casefold() in UNCLEAR:
        return RelRow(ref, p, PARTIAL, True, "Pair found, but LitMind has not determined the relationship type.")
    cr, cp = canon_type(rt), canon_type(pt)
    if cr and cp:
        if cr == cp:
            return RelRow(ref, p, CORRECT, True, f"Both are '{cr}'.")
        return RelRow(ref, p, INCORRECT, True, f"Reference '{rt}' ({cr}) vs predicted '{pt}' ({cp}).")
    if re.sub(r"\W+", "", rt.casefold()) == re.sub(r"\W+", "", pt.casefold()):
        return RelRow(ref, p, CORRECT, True, "Same label.")
    return RelRow(ref, p, "Pending semantic check", True, "Labels cannot be compared mechanically.")


def relationship_metrics(rows: list[RelRow], exhaustive: bool) -> dict:
    ref_total = sum(1 for r in rows if r.ref is not None and r.status != "Duplicate")
    pair_tp = sum(1 for r in rows if r.pair_match and r.status != "Duplicate")
    strict_tp = sum(1 for r in rows if r.status == CORRECT and r.pair_match)
    extra = sum(1 for r in rows if r.status == "Not in reference")
    dup = sum(1 for r in rows if r.status == "Duplicate")
    with_type = [r for r in rows if r.pair_match and r.ref and r.ref.type and r.status in (CORRECT, INCORRECT, PARTIAL)]
    out = {"ref_total": ref_total, "pairs_found": pair_tp, "pair_recall": pair_tp / ref_total if ref_total else None,
           "strict_recall": strict_tp / ref_total if ref_total else None, "not_in_reference": extra, "duplicates": dup,
           "type_accuracy": (sum(1 for r in with_type if r.status == CORRECT) / len(with_type)) if with_type else None,
           "pending": sum(1 for r in rows if r.status == "Pending semantic check")}
    if exhaustive:
        pp = prf(pair_tp, extra + dup, ref_total - pair_tp)
        sp = prf(strict_tp, extra + dup + (pair_tp - strict_tp), ref_total - strict_tp)
        out.update(pair_precision=pp["precision"], pair_f1=pp["f1"], strict_precision=sp["precision"], strict_f1=sp["f1"])
    else:
        out.update(pair_precision=None, pair_f1=None, strict_precision=None, strict_f1=None,
                   precision_note="Precision/F1 not computed: the reference is not declared exhaustive.")
    return out


# --------------------------------------------------------------------------- protagonist
PROTAG_FIELDS = [("name", "Name and aliases"), ("narrative_role", "Narrative role"), ("traits", "Personality and traits"),
                 ("motivations", "Motivations and goals"), ("conflicts", "Major conflicts"),
                 ("arc", "Character development / arc"), ("key_relationships", "Important relationships"),
                 ("passages", "Supporting passages")]
SCORE = {CORRECT: 1.0, PARTIAL: 0.5, INCORRECT: 0.0}


@dataclass
class ProtagRow:
    field: str
    label: str
    reference: str
    predicted: str
    status: str
    explanation: str = ""
    evidence: list[str] = field(default_factory=list)
    semantic: bool = False          # needs a Gemini judgement


def _pred_field(pred, fld: str) -> str:
    m = {"name": pred.name, "narrative_role": pred.narrative_role, "traits": "; ".join(pred.personality_traits),
         "motivations": pred.motivations, "conflicts": pred.major_conflicts, "arc": pred.character_arc,
         "key_relationships": "; ".join(pred.key_relationships), "passages": ""}
    return (m.get(fld) or "").strip()


def _ref_field(ref: RefProtagonist, fld: str) -> str:
    v = getattr(ref, fld)
    return "; ".join(v) if isinstance(v, list) else (v or "").strip()


def evaluate_protagonist_names(pred, refs: list[RefProtagonist]) -> tuple[RefProtagonist | None, list[ProtagRow]]:
    """Name check (exact, via aliases) and the list of fields still to judge. pred: ProtagonistAnalysis | None."""
    refs = [r for r in refs if r.name]
    if not refs:
        return None, []
    target = refs[0]
    pf = forms(pred.name, pred.aliases) if pred and pred.name else set()
    hit = next((r for r in refs if pf & forms(r.name, r.aliases)), None)
    rows: list[ProtagRow] = []
    if pred is None or not pred.determinable or not pred.name:
        why = "LitMind did not identify a protagonist."
        rows.append(ProtagRow("name", "Name and aliases", _ref_field(target, "name"), "", INCORRECT, why))
        hit = None
    elif hit is None:
        rows.append(ProtagRow("name", "Name and aliases", ", ".join(r.name for r in refs), pred.name, INCORRECT,
                              f"Predicted '{pred.name}', which is not a reference protagonist."))
    else:
        rows.append(ProtagRow("name", "Name and aliases", hit.name, pred.name, CORRECT, "Exact match (via names and aliases)."))
        target = hit
    for fld, label in PROTAG_FIELDS[1:]:
        refv = _ref_field(target, fld)
        if not refv:
            rows.append(ProtagRow(fld, label, "", _pred_field(pred, fld) if pred else "", NOT_EVAL, "The reference has no value for this field."))
        elif fld == "passages":
            rows.append(ProtagRow(fld, label, refv, "", NOT_EVAL, "LitMind does not output supporting passages."))
        elif hit is None:
            rows.append(ProtagRow(fld, label, refv, _pred_field(pred, fld) if pred else "", INCORRECT,
                                  "The predicted protagonist is a different character (or none)."))
        elif not _pred_field(pred, fld):
            rows.append(ProtagRow(fld, label, refv, "", INCORRECT, "LitMind produced no value for this field."))
        else:
            rows.append(ProtagRow(fld, label, refv, _pred_field(pred, fld), "Pending semantic check", semantic=True))
    return target, rows


def protagonist_score(rows: list[ProtagRow]) -> dict:
    scored = [r for r in rows if r.status in SCORE]
    ev = [r for r in rows if r.status != NOT_EVAL]
    return {"score": (sum(SCORE[r.status] for r in scored) / len(scored)) if scored else None,
            "fields_evaluated": len(scored), "fields_not_evaluated": sum(1 for r in rows if r.status == NOT_EVAL),
            "fields_pending": sum(1 for r in rows if r.status == "Pending semantic check"),
            "name_correct": any(r.field == "name" and r.status == CORRECT for r in rows), "n_rows": len(ev)}


# --------------------------------------------------------------------------- genre
GENRE_SYN = {"sci fi": "science fiction", "scifi": "science fiction", "sf": "science fiction", "gothic fiction": "gothic",
             "gothic horror": "gothic horror", "detective": "mystery", "detective fiction": "mystery", "crime fiction": "crime",
             "historical": "historical fiction", "humor": "humour / satire", "humour": "humour / satire", "satire": "humour / satire",
             "ya": "young adult", "non fiction": "non-fiction", "nonfiction": "non-fiction", "drama": "drama / play",
             "play": "drama / play", "biography": "biography / memoir", "memoir": "biography / memoir",
             "children s": "children's", "childrens": "children's", "fantasy fiction": "fantasy", "horror fiction": "horror"}


def norm_genre(g: str) -> str:
    s = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9/' -]", " ", str(g).casefold().replace("-", " ").replace("_", " "))).strip()
    return GENRE_SYN.get(s, s)


def genre_evaluation(ref_genres: list[str], pred_genres: list[str], multi: bool) -> dict:
    ref = list(dict.fromkeys(norm_genre(g) for g in ref_genres if g.strip()))
    pred = list(dict.fromkeys(norm_genre(g) for g in pred_genres if g.strip()))
    if not ref:
        return {"evaluated": False, "reason": "The reference has no genre label."}
    if not pred:
        return {"evaluated": False, "reason": "LitMind has no genre prediction yet."}
    out = {"evaluated": True, "multi_label": multi, "reference": ref, "predicted": pred}
    if not multi:
        out.update(exact_match=(pred[0] == ref[0]), accuracy=1.0 if pred[0] == ref[0] else 0.0,
                   in_secondary=(pred[0] != ref[0] and ref[0] in pred[1:]),
                   missing=[] if pred[0] == ref[0] else [ref[0]], extra=[] if pred[0] == ref[0] else [pred[0]])
    else:
        tp = len(set(ref) & set(pred))
        d = prf(tp, len(set(pred) - set(ref)), len(set(ref) - set(pred)))
        out.update(precision=d["precision"], recall=d["recall"], f1=d["f1"], exact_match=set(ref) == set(pred),
                   jaccard=tp / len(set(ref) | set(pred)), missing=sorted(set(ref) - set(pred)), extra=sorted(set(pred) - set(ref)))
    return out
