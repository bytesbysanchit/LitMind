"""Gemini-based semantic judging, built on the existing GeminiClient (no second service).

Rules enforced here, not just requested in the prompt:
  * REFERENCE, PREDICTION and PASSAGES are sent as separate labelled blocks;
  * Gemini is asked to COMPARE with the reference, never to grade its own answer;
  * every reply is schema-validated; unknown ids are dropped, missing ids become "Not Evaluated";
  * evidence ids must be among the passages actually supplied, so citations cannot be invented;
  * Gemini can only produce a status and an explanation; it cannot change a reference value or a genre label.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..gemini_client import GeminiClient
from .matching import CORRECT, INCORRECT, NOT_EVAL, PARTIAL

_STATUS = {"correct": CORRECT, "match": CORRECT, "equivalent": CORRECT, "same": CORRECT,
           "partial": PARTIAL, "partially correct": PARTIAL, "partially_correct": PARTIAL,
           "incorrect": INCORRECT, "mismatch": INCORRECT, "wrong": INCORRECT, "contradicts": INCORRECT,
           "not_evaluated": NOT_EVAL, "not evaluated": NOT_EVAL, "insufficient": NOT_EVAL, "unknown": NOT_EVAL}


class Judgement(BaseModel):
    id: int
    status: str = NOT_EVAL
    explanation: str = Field(default="", description="At most two sentences.")
    evidence_ids: list[int] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def _norm(cls, v):
        return _STATUS.get(str(v).strip().casefold().replace("-", "_"), _STATUS.get(str(v).strip().casefold(), NOT_EVAL))


class JudgementBatch(BaseModel):
    judgements: list[Judgement] = Field(default_factory=list)


class GenreExplanation(BaseModel):
    explanation: str = ""


FRAME = (
    "You are an evaluation assistant. Compare each PREDICTION with the REFERENCE supplied by the user. "
    "The REFERENCE is the standard; do not substitute your own opinion and do not judge whether the "
    "prediction is plausible in general. Use PASSAGES only as supporting evidence and cite them by id. "
    "If the reference says nothing that lets you compare, answer not_evaluated. "
    "status is one of: correct (same meaning), partial (overlaps but misses or adds something), "
    "incorrect (contradicts or is unrelated), not_evaluated. explanation: at most two sentences. "
    "Return JSON {\"judgements\": [{\"id\", \"status\", \"explanation\", \"evidence_ids\"}]} with one entry per item id.\n\n")

TASKS = {
    "character_details": "TASK: compare the predicted description/role of each character with the reference description and traits.",
    "relationship_types": "TASK: decide whether the predicted relationship type means the same as the reference relationship type "
                          "(e.g. 'betrothed' and 'romance' are the same class; 'rivalry' and 'friendship' are not).",
    "protagonist_fields": "TASK: compare each predicted protagonist field with the reference field of the same name.",
}


def _item_block(it: dict) -> str:
    ev = "\n".join(f"    [{pid}] {txt}" for pid, txt in it.get("passages", [])) or "    (none)"
    return (f"ITEM {it['id']} — {it.get('title', '')}\n  REFERENCE: {it['reference']}\n"
            f"  PREDICTION: {it['prediction']}\n  PASSAGES:\n{ev}")


def judge(client: GeminiClient, task: str, items: list[dict], batch_size: int = 8) -> dict[int, dict]:
    """items: [{id, title, reference, prediction, passages: [(pid, text)]}] -> {id: {status, explanation, evidence}}."""
    results: dict[int, dict] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompt = FRAME + TASKS[task] + "\n\n" + "\n\n".join(_item_block(it) for it in batch)
        out = client._call(prompt, JudgementBatch)
        allowed = {it["id"]: {pid: txt for pid, txt in it.get("passages", [])} for it in batch}
        for j in out.judgements:
            if j.id not in allowed or j.id in results:
                continue                                   # unknown or repeated id: ignore
            results[j.id] = {"status": j.status, "explanation": j.explanation.strip(),
                             "evidence": [(pid, allowed[j.id][pid]) for pid in j.evidence_ids if pid in allowed[j.id]]}
        for it in batch:
            results.setdefault(it["id"], {"status": NOT_EVAL, "evidence": [],
                                          "explanation": "Gemini returned no judgement for this item."})
    return results


def explain_genre(client: GeminiClient, reference: list[str], predicted: list[str], excerpt: str = "") -> str:
    prompt = ("The REFERENCE genre labels are authoritative and must not be changed or replaced. In at most "
              "three sentences explain how the PREDICTED labels agree with or differ from them" +
              (", using the BOOK EXCERPT where helpful" if excerpt else "") + ". Do not say which label is 'better'.\n\n"
              f"REFERENCE: {', '.join(reference)}\nPREDICTED: {', '.join(predicted)}\n" +
              (f"BOOK EXCERPT:\n{excerpt[:3000]}" if excerpt else "") +
              "\nReturn JSON {\"explanation\": \"...\"}.")
    return client._call(prompt, GenreExplanation).explanation.strip()
