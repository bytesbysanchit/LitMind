"""Evaluation report: category results, the documented overall score, and CSV / Excel export."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

OVERALL_FORMULA = (
    "Overall accuracy = the plain average of the category scores that were evaluated (equal weights). "
    "Category scores: Characters = F1 if the reference is declared exhaustive, otherwise recall; "
    "Relationships = strict F1 (pair and type both correct) if the reference is exhaustive, otherwise strict recall; "
    "Protagonist = mean field score (Correct 1, Partial 0.5, Incorrect 0; fields without a reference value are excluded); "
    "Genre = exact-match accuracy (single label) or F1 (multi-label). "
    "Categories marked Not Evaluated are excluded from the average.")
CATEGORIES = ("Characters", "Relationships", "Protagonist", "Genre")


@dataclass
class CategoryResult:
    name: str
    status: str = "Not Evaluated"         # Evaluated | Partially evaluated | Not Evaluated
    reason: str = ""
    metrics: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    score: float | None = None
    score_label: str = ""


@dataclass
class EvaluationReport:
    book_title: str
    reference_origin: str
    reference_confirmed: bool
    categories: dict[str, CategoryResult]
    warnings: list[str] = field(default_factory=list)
    gemini_used: bool = False
    gemini_calls: int = 0
    created: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

    @property
    def overall(self) -> tuple[float | None, list[str]]:
        used = [c for c in self.categories.values() if c.score is not None and c.status != "Not Evaluated"]
        return (sum(c.score for c in used) / len(used) if used else None), [c.name for c in used]


def _evidence_text(evidence) -> str:
    return " | ".join(f"[{pid}] {txt}" for pid, txt in (evidence or []))


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def to_frames(rep: EvaluationReport) -> dict[str, pd.DataFrame]:
    overall, included = rep.overall
    summary = [("Book", rep.book_title), ("Created", rep.created),
               ("Reference origin", rep.reference_origin + ("" if rep.reference_confirmed else " (UNVERIFIED)")),
               ("Gemini used for semantic judgements", "yes" if rep.gemini_used else "no"),
               ("Gemini calls", rep.gemini_calls),
               ("Overall accuracy", _fmt(overall)), ("Categories included", ", ".join(included) or "none"),
               ("Overall formula", OVERALL_FORMULA)]
    for name, c in rep.categories.items():
        summary.append((f"{name}: status", c.status + (f" ({c.reason})" if c.reason else "")))
        if c.score is not None:
            summary.append((f"{name}: score ({c.score_label})", _fmt(c.score)))
        for k, v in c.metrics.items():
            if not isinstance(v, (list, dict)):
                summary.append((f"{name}: {k}", _fmt(v)))
    for w in rep.warnings:
        summary.append(("Warning", w))
    out = {"Summary": pd.DataFrame(summary, columns=["Item", "Value"])}
    for name, c in rep.categories.items():
        rows = [{**{k: v for k, v in r.items() if not k.startswith("_")}, "Evidence": _evidence_text(r.get("Evidence"))} for r in c.rows]
        out[name] = pd.DataFrame(rows) if rows else pd.DataFrame({"Note": [c.reason or "No rows."]})
    return out


def to_excel_bytes(rep: EvaluationReport) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in to_frames(rep).items():
            df.to_excel(w, sheet_name=name[:31], index=False)
            ws = w.sheets[name[:31]]
            for col in ws.columns:                                  # readable column widths
                width = min(60, max(len(str(c.value)) if c.value is not None else 0 for c in col) + 2)
                ws.column_dimensions[col[0].column_letter].width = max(10, width)
    return buf.getvalue()


def to_csv_bytes(rep: EvaluationReport) -> bytes:
    frames = []
    for name, df in to_frames(rep).items():
        d = df.copy()
        d.insert(0, "Category", name)
        frames.append(d)
    return pd.concat(frames, ignore_index=True).to_csv(index=False).encode("utf-8-sig")
