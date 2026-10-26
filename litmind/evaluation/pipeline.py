"""Staged evaluation pipeline.

  1. prepare   - choose the passage source (reference text first, else the analysed book) and index it locally
  2. match     - deterministic matching and metrics (no LLM)
  3. judge     - category-specific Gemini judgements on the few items that need semantic comparison
  4. aggregate - category scores and the documented overall score
  5. report    - EvaluationReport (UI tables, CSV / Excel export)

Gemini is optional: without it every deterministic metric is still produced and the semantic fields are
reported as "Not Evaluated (semantic check unavailable)".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..gemini_client import GeminiClient, GeminiError
from . import matching as M
from .gemini_eval import explain_genre, judge
from .passages import PassageIndex
from .report import CategoryResult, EvaluationReport
from .schema import Prediction, ReferenceAnnotations

UNAVAILABLE = "Not Evaluated (semantic check unavailable)"


@dataclass
class EvalOptions:
    use_gemini: bool = True
    overrides: dict[str, str | None] = field(default_factory=dict)     # user-confirmed name matches


def run_evaluation(prediction: Prediction, reference: ReferenceAnnotations, *, client: GeminiClient | None = None,
                   book_text: str = "", options: EvalOptions | None = None,
                   progress: Callable[[str, float], None] | None = None) -> EvaluationReport:
    opt = options or EvalOptions()
    say = progress or (lambda *_: None)
    gem = client if (opt.use_gemini and client is not None) else None
    start_calls = client.calls if client is not None else 0
    warnings: list[str] = []
    rep = EvaluationReport(reference.book_title or prediction.book_title, reference.origin,
                           reference.confirmed, {n: CategoryResult(n) for n in ("Characters", "Relationships", "Protagonist", "Genre")},
                           warnings, gemini_used=gem is not None)
    if not reference.confirmed:
        warnings.append("The reference annotations are AI-proposed and were NOT confirmed by a person. "
                        "Treat every result as unverified.")

    say("Preparing passages", 0.05)
    text = reference.source_text or book_text
    index = PassageIndex(text) if text else PassageIndex("")
    av_ref, av_pred = reference.available(), prediction.available()

    def gem_call(label, fn, *a, **k):
        """Run one Gemini step; on failure keep deterministic results and record the problem."""
        try:
            return fn(*a, **k)
        except GeminiError as exc:
            warnings.append(f"{label}: Gemini step failed ({exc}). Deterministic results are still shown.")
            return None

    # ------------------------------------------------------------------ characters
    say("Matching characters", 0.2)
    cat = rep.categories["Characters"]
    cm = M.match_characters(prediction.characters, reference.characters, opt.overrides)   # also needed for relationships
    if not av_ref["characters"]:
        cat.reason = "The reference has no character annotations."
    elif not av_pred["characters"]:
        cat.reason = "LitMind has not detected any characters yet. Run character detection first."
    else:
        exhaustive = reference.characters_exhaustive
        cat.metrics = M.character_metrics(cm, exhaustive)
        items, item_rows = [], {}
        for r in cm.rows:
            if r.status == "Matched" and r.ref and r.pred and (r.ref.description or r.ref.traits) and r.pred.description:
                iid = len(items)
                ref_txt = "; ".join(x for x in (r.ref.description, ("traits: " + ", ".join(r.ref.traits)) if r.ref.traits else "") if x)
                q = " ".join([r.ref.name, *r.ref.aliases])
                items.append({"id": iid, "title": r.ref.name, "reference": ref_txt, "prediction": r.pred.description,
                              "passages": index.search(q, 2)})
                item_rows[iid] = r
        judged = {}
        if items and gem:
            say("Judging character details with Gemini", 0.35)
            judged = gem_call("Character details", judge, gem, "character_details", items) or {}
        for r in cm.rows:
            name = r.ref.name if r.ref else (r.pred.name if r.pred else "")
            row = {"Character": name,
                   "Reference Details": (r.ref.description or "; ".join(r.ref.traits) or ("(declared, no description)" if r.ref and r.ref.declared else "")) if r.ref else "(not in reference)",
                   "LitMind Prediction": (r.pred.description or f"detected, {r.pred.mentions} mentions") if r.pred else "(not detected)",
                   "Name match": r.status + (f" ({r.how})" if r.how and r.status in ("Matched", "Duplicate") else ""),
                   "Evaluation": "", "Explanation": "", "Evidence": [], "_cands": list(r.candidates)}
            if r.status == "Matched":
                iid = next((k for k, v in item_rows.items() if v is r), None)
                if iid is not None and iid in judged:
                    j = judged[iid]
                    row.update(Evaluation=j["status"], Explanation=j["explanation"], Evidence=j["evidence"])
                elif iid is not None:
                    row.update(Evaluation="Correct (name); details: " + UNAVAILABLE, Explanation="Name matched; the description could not be compared.")
                else:
                    row.update(Evaluation="Correct", Explanation="Name matched. No descriptive details to compare, so details are Not Evaluated.")
            elif r.status == "Missing":
                row.update(Evaluation=M.INCORRECT, Explanation="In the reference but not detected by LitMind.")
            elif r.status == "Duplicate":
                row.update(Evaluation=M.INCORRECT, Explanation="Another LitMind entry already represents this reference character (duplicate).")
            elif r.status == "Needs confirmation":
                row.update(Evaluation=M.NOT_EVAL, Explanation="Could be: " + ", ".join(r.candidates) + ". Confirm which one to include it in the metrics.")
            else:
                row.update(Evaluation=M.INCORRECT if exhaustive else M.NOT_EVAL,
                           Explanation="Not in the reference." + ("" if exhaustive else " The reference is not exhaustive, so this is not counted as wrong."))
            cat.rows.append(row)
        cat.status = "Evaluated"
        cat.score, cat.score_label = (cat.metrics["f1"], "F1") if exhaustive else (cat.metrics["recall"], "recall")
        if cat.metrics["unresolved"]:
            cat.notes.append(f"{cat.metrics['unresolved']} detected name(s) need confirmation and are excluded from the metrics.")
        if not exhaustive:
            cat.notes.append(cat.metrics.get("precision_note", ""))

    # ------------------------------------------------------------------ relationships
    say("Matching relationships", 0.5)
    cat = rep.categories["Relationships"]
    if not av_ref["relationships"]:
        cat.reason = "The reference has no relationship annotations."
    elif not av_ref["characters"] and not reference.characters:
        cat.reason = "Relationship endpoints could not be resolved."
    else:
        rows = M.match_relationships(prediction.relationships, prediction.characters, reference.relationships,
                                     reference.characters, cm)
        pend = [r for r in rows if r.status == "Pending semantic check"]
        items = [{"id": i, "title": f"{r.ref.a} – {r.ref.b}", "reference": r.ref.type, "prediction": f"{r.pred.type}. {r.pred.reason}".strip(". "),
                  "passages": index.search(f"{r.ref.a} {r.ref.b}", 2)} for i, r in enumerate(pend)]
        judged = {}
        if items and gem:
            say("Judging relationship types with Gemini", 0.6)
            judged = gem_call("Relationship types", judge, gem, "relationship_types", items) or {}
        for i, r in enumerate(pend):
            j = judged.get(i)
            if j:
                r.status, r.note = j["status"], j["explanation"]
                r.evidence = j["evidence"]
            else:
                r.status, r.note = UNAVAILABLE, "The two relationship labels could not be compared mechanically."
        if not any(p.type for p in prediction.relationships) and prediction.relationships:
            cat.notes.append("LitMind has not assigned relationship types yet (run 'Explain relationships with Gemini' first); "
                             "matched pairs are therefore Partial.")
        if prediction.relationships:
            cat.notes.append(f"Only the {len(prediction.relationships)} relationship candidates that LitMind lists are evaluated; "
                             "reference relationships outside that list count as missed.")
        if not prediction.relationships:
            cat.notes.append("LitMind found no relationships. This is a valid result; all reference relationships count as missed.")
        resolved = [r for r in rows if not r.status.startswith("Not Evaluated")]
        cat.metrics = M.relationship_metrics(rows, reference.relationships_exhaustive)
        for r in rows:
            ev = getattr(r, "evidence", [])
            cat.rows.append({
                "Reference": f"{r.ref.a} – {r.ref.b}" if r.ref else "(not in reference)",
                "Reference type": (r.ref.type or "(none given)") if r.ref else "",
                "LitMind": f"{r.pred.a} – {r.pred.b}" if r.pred else "(not predicted)",
                "Predicted type": (r.pred.type or "(not determined)") if r.pred else "",
                "Evaluation": {"Missing": M.INCORRECT, "Duplicate": M.INCORRECT, "Not in reference": (M.INCORRECT if reference.relationships_exhaustive else M.NOT_EVAL)}.get(r.status, r.status),
                "Explanation": r.note or ("Reference relationship not predicted." if r.status == "Missing" else ""), "Evidence": ev})
        cat.status = "Evaluated"
        ex = reference.relationships_exhaustive
        cat.score, cat.score_label = (cat.metrics["strict_f1"], "strict F1") if ex else (cat.metrics["strict_recall"], "strict recall")
        if not ex:
            cat.notes.append(cat.metrics.get("precision_note", ""))

    # ------------------------------------------------------------------ protagonist
    say("Evaluating protagonist", 0.7)
    cat = rep.categories["Protagonist"]
    if not av_ref["protagonist"]:
        cat.reason = "The reference has no protagonist annotation."
    elif prediction.protagonist is None:
        cat.reason = "LitMind has not run protagonist analysis yet."
    else:
        target, prows = M.evaluate_protagonist_names(prediction.protagonist, reference.protagonists)
        pend = [r for r in prows if r.semantic]
        items = [{"id": i, "title": f"{target.name}: {r.label}", "reference": r.reference, "prediction": r.predicted,
                  "passages": index.search(f"{target.name} {r.label} {r.reference[:80]}", 2)} for i, r in enumerate(pend)]
        judged = {}
        if items and gem:
            say("Judging protagonist details with Gemini", 0.8)
            judged = gem_call("Protagonist fields", judge, gem, "protagonist_fields", items) or {}
        for i, r in enumerate(pend):
            j = judged.get(i)
            if j:
                r.status, r.explanation, r.evidence = j["status"], j["explanation"], j["evidence"]
            else:
                r.status, r.explanation = M.NOT_EVAL, UNAVAILABLE
        sc = M.protagonist_score(prows)
        cat.metrics = sc
        cat.rows = [{"Field": r.label, "Reference": r.reference or "(none)", "LitMind": r.predicted or "(none)",
                     "Evaluation": r.status, "Explanation": r.explanation, "Evidence": r.evidence} for r in prows]
        cat.score, cat.score_label = sc["score"], "mean field score"
        cat.status = "Evaluated" if sc["score"] is not None else "Not Evaluated"
        if sc["fields_not_evaluated"]:
            cat.notes.append(f"{sc['fields_not_evaluated']} field(s) were not evaluated (no reference value, or not produced by LitMind).")
        if not sc["name_correct"]:
            cat.notes.append("The predicted protagonist is not the reference protagonist, so descriptive fields are scored as incorrect.")

    # ------------------------------------------------------------------ genre
    say("Evaluating genre", 0.9)
    cat = rep.categories["Genre"]
    ge = M.genre_evaluation(reference.genres, prediction.genres, reference.is_multi_label)
    if not ge["evaluated"]:
        cat.reason = ge["reason"]
    else:
        expl = ""
        if gem and not ge["exact_match"]:
            expl = gem_call("Genre explanation", explain_genre, gem, reference.genres, prediction.genres, prediction.summary) or ""
        cat.metrics = {k: v for k, v in ge.items() if k not in ("reference", "predicted", "missing", "extra", "evaluated")}
        cat.rows = [{"Reference genre": ", ".join(reference.genres), "Predicted genre": ", ".join(prediction.genres),
                     "Exact match": "yes" if ge["exact_match"] else "no", "Missing labels": ", ".join(ge["missing"]) or "-",
                     "Extra labels": ", ".join(ge["extra"]) or "-",
                     "Explanation": expl or ("Labels agree." if ge["exact_match"] else "Labels differ. (Gemini explanation unavailable.)"),
                     "Evidence": []}]
        if not reference.is_multi_label and ge.get("in_secondary"):
            cat.notes.append("The reference label appears among LitMind's secondary genres but is not its primary genre, so it is not an exact match.")
        cat.score, cat.score_label = ((ge["f1"], "F1") if ge["multi_label"] else (ge["accuracy"], "exact-match accuracy"))
        cat.status = "Evaluated"

    say("Done", 1.0)
    rep.gemini_calls = (client.calls - start_calls) if client is not None else 0
    return rep
