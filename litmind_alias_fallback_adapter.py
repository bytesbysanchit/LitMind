"""
LitMind — All-Character Alias Fallback Adapter
==============================================
A separate diagnostic adapter. Does NOT import or modify nlp_processor.py or prev.py.

Reads:
  Dattani_Tara_Corpus_Final.xlsx
  sheets CHARACTER_MASTER and UTTERANCE_CORPUS

Writes:
  reports/character_alias_fallback_comparison.xlsx
  reports/character_alias_fallback_candidates.csv

Important:
- spaCy PERSON output is preserved as baseline.
- Alias matches are candidate mentions, not automatically verified character mentions.
- This script does not claim accuracy/precision/recall/F1 without independent gold labels.
- Dialogue vs stage-direction classification is heuristic and flagged as such.
"""
from pathlib import Path
import argparse
import re
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher


def safe_text(value):
    return "" if pd.isna(value) else str(value)


def classify_context(text):
    """Heuristic only; never treat this as verified annotation."""
    t = text.strip()
    if not t:
        return "EMPTY"
    if re.search(r"[\(\[].{0,500}[\)\]]", t, flags=re.S):
        return "MIXED_OR_BRACKETED"
    stage_verbs = (
        "turns to", "slowly comes", "walks", "enters", "exits", "sits",
        "stands", "lies down", "looks at", "moves to", "crosses", "dances",
        "stage direction", "lights", "blackout", "pause", "silence"
    )
    low = t.lower()
    if any(v in low for v in stage_verbs) and not re.search(r'["“”]', t):
        return "POSSIBLE_STAGE_DIRECTION"
    return "DIALOGUE_OR_UNCLEAR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Dattani_Tara_Corpus_Final.xlsx")
    parser.add_argument("--output", default="reports/character_alias_fallback_comparison.xlsx")
    parser.add_argument("--model", default="en_core_web_sm")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Corpus workbook not found: {input_path.resolve()}")

    characters = pd.read_excel(input_path, sheet_name="CHARACTER_MASTER")
    utterances = pd.read_excel(input_path, sheet_name="UTTERANCE_CORPUS")
    req_c = {"Character_ID", "Play_ID", "Character_Name"}
    req_u = {"Speech_ID", "Play_ID", "Play", "Speech_OCR", "QA_Flag"}
    if not req_c.issubset(characters.columns):
        raise ValueError(f"CHARACTER_MASTER missing: {sorted(req_c-set(characters.columns))}")
    if not req_u.issubset(utterances.columns):
        raise ValueError(f"UTTERANCE_CORPUS missing: {sorted(req_u-set(utterances.columns))}")

    nlp = spacy.load(args.model)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    match_lookup = {}
    valid_chars = []
    for _, row in characters.iterrows():
        name = safe_text(row["Character_Name"]).strip()
        if not name or name.lower() == "nan":
            continue
        # Match whole token sequences, not arbitrary substrings (Dan won't match inside Chandan).
        key = f"LITMIND_CHAR_{row['Play_ID']}_{row['Character_ID']}"
        matcher.add(key, [nlp.make_doc(name)])
        match_lookup[nlp.vocab.strings[key]] = {
            "Character_ID": safe_text(row["Character_ID"]),
            "Play_ID": safe_text(row["Play_ID"]),
            "Character": name,
        }
        valid_chars.append({
            "Character_ID": safe_text(row["Character_ID"]),
            "Play_ID": safe_text(row["Play_ID"]),
            "Character": name,
        })

    utterances = utterances.copy()
    utterances["Play_ID_norm"] = utterances["Play_ID"].map(safe_text)
    detail_rows = []
    candidate_rows = []
    texts = [safe_text(x) for x in utterances["Speech_OCR"]]

    for (_, utt), doc in zip(utterances.iterrows(), nlp.pipe(texts, batch_size=32)):
        play_id = safe_text(utt["Play_ID"])
        ents = [(e.text, e.label_, e.start, e.end) for e in doc.ents]
        person_ents = [e for e in doc.ents if e.label_ == "PERSON"]
        alias_matches = []
        for match_id, start, end in matcher(doc):
            char = match_lookup[match_id]
            if char["Play_ID"] != play_id:
                continue
            alias_matches.append((char, start, end))

        # Keep baseline NER and alias candidates separate. A row represents one canonical alias hit.
        for char, start, end in alias_matches:
            exact = any(e.start == start and e.end == end for e in person_ents)
            overlap = any(e.start < end and e.end > start for e in person_ents)
            span = doc[start:end]
            row = {
                "Speech_ID": safe_text(utt["Speech_ID"]),
                "Play_ID": play_id,
                "Play": safe_text(utt["Play"]),
                "Character_ID": char["Character_ID"],
                "Character": char["Character"],
                "Matched_Surface": span.text,
                "Baseline_PERSON_Entities": "; ".join(f"{e.text}:{e.label_}" for e in doc.ents),
                "Baseline_Exact_PERSON": bool(exact),
                "Baseline_PERSON_Overlap": bool(overlap),
                "Alias_Fallback_Candidate": True,
                "Candidate_Recovery_vs_Exact_PERSON": not exact,
                "Context_Heuristic": classify_context(doc.text),
                "QA_Flag": safe_text(utt.get("QA_Flag", "")),
                "Text_Context": doc.text,
            }
            detail_rows.append(row)
            candidate_rows.append({
                "Speech_ID": row["Speech_ID"],
                "Play_ID": play_id,
                "Play": row["Play"],
                "Character_ID": char["Character_ID"],
                "Character_Candidate": char["Character"],
                "Evidence_Type": "CANONICAL_ALIAS_MATCH",
                "Baseline_NER_Label_For_Exact_Surface": next(
                    (e.label_ for e in doc.ents if e.start == start and e.end == end), ""
                ),
                "Baseline_Exact_PERSON": bool(exact),
                "Context_Heuristic": row["Context_Heuristic"],
                "Needs_Context_Review": True,
                "QA_Flag": row["QA_Flag"],
                "Text_Context": doc.text,
            })

    detail_cols = [
        "Speech_ID","Play_ID","Play","Character_ID","Character","Matched_Surface",
        "Baseline_PERSON_Entities","Baseline_Exact_PERSON","Baseline_PERSON_Overlap",
        "Alias_Fallback_Candidate","Candidate_Recovery_vs_Exact_PERSON",
        "Context_Heuristic","QA_Flag","Text_Context"
    ]
    detail = pd.DataFrame(detail_rows, columns=detail_cols)
    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        candidates = pd.DataFrame(columns=[
            "Speech_ID","Play_ID","Play","Character_ID","Character_Candidate","Evidence_Type",
            "Baseline_NER_Label_For_Exact_Surface","Baseline_Exact_PERSON","Context_Heuristic",
            "Needs_Context_Review","QA_Flag","Text_Context"
        ])

    summaries = []
    for char in valid_chars:
        sub = detail[
            (detail["Character_ID"] == char["Character_ID"]) &
            (detail["Play_ID"].astype(str) == char["Play_ID"])
        ] if len(detail) else detail
        summaries.append({
            "Play_ID": char["Play_ID"],
            "Character_ID": char["Character_ID"],
            "Character": char["Character"],
            "Canonical_Alias_Occurrences": int(len(sub)),
            "Unique_Utterances_With_Alias": int(sub["Speech_ID"].nunique()) if len(sub) else 0,
            "Baseline_Exact_PERSON_Matches": int(sub["Baseline_Exact_PERSON"].sum()) if len(sub) else 0,
            "Baseline_PERSON_Overlap_Matches": int(sub["Baseline_PERSON_Overlap"].sum()) if len(sub) else 0,
            "Alias_Candidates_Not_Exact_PERSON": int(sub["Candidate_Recovery_vs_Exact_PERSON"].sum()) if len(sub) else 0,
            "Alias_Candidates_Not_Exact_PERSON_Pct": round(100 * sub["Candidate_Recovery_vs_Exact_PERSON"].mean(), 2) if len(sub) else 0,
            "Interpretation": "Diagnostic only; alias candidate is not a verified character mention."
        })
    summary = pd.DataFrame(summaries)

    context_summary = (
        detail.groupby(["Play_ID","Context_Heuristic"], dropna=False)
        .agg(Alias_Candidate_Rows=("Speech_ID","size"),
             Unique_Utterances=("Speech_ID","nunique"),
             Not_Exact_PERSON=("Candidate_Recovery_vs_Exact_PERSON","sum"))
        .reset_index()
        if len(detail) else pd.DataFrame(columns=["Play_ID","Context_Heuristic","Alias_Candidate_Rows","Unique_Utterances","Not_Exact_PERSON"])
    )

    overview = pd.DataFrame([
        ["Corpus utterance rows", len(utterances)],
        ["Characters in CHARACTER_MASTER", len(valid_chars)],
        ["Canonical alias candidate rows", len(detail)],
        ["Unique utterances with at least one canonical alias", int(detail["Speech_ID"].nunique()) if len(detail) else 0],
        ["Baseline exact PERSON matches among alias hits", int(detail["Baseline_Exact_PERSON"].sum()) if len(detail) else 0],
        ["Alias candidates without exact PERSON match", int(detail["Candidate_Recovery_vs_Exact_PERSON"].sum()) if len(detail) else 0],
        ["Alias candidates overlapping any PERSON entity", int(detail["Baseline_PERSON_Overlap"].sum()) if len(detail) else 0],
        ["NER model", args.model],
        ["Evaluation status", "DIAGNOSTIC ONLY — no independent gold labels; do not report as accuracy/precision/recall/F1"],
        ["Fallback behavior", "Creates candidate mentions only; does not overwrite baseline NER or assert every match is a true character mention"],
        ["Context classification", "Heuristic only; all candidate rows are marked for contextual review"],
        ["Protected files", "nlp_processor.py and prev.py are not imported or modified"],
    ], columns=["Metric","Value"])

    methodology = pd.DataFrame([
        ["Baseline", "Fresh spaCy entities from the selected model; only PERSON entities are considered for comparison."],
        ["Fallback", "Case-insensitive whole-token-sequence matching of canonical names from CHARACTER_MASTER, scoped to the same Play_ID."],
        ["Substring safety", "PhraseMatcher matches token sequences, so Dan is not matched as a substring inside Chandan."],
        ["Interpretation", "Alias hits are candidates. Stage directions, mentions of other people, OCR noise, and context ambiguity may create false positives."],
        ["No gold labels", "No precision, recall, F1, or overall accuracy is computed."],
        ["Model dependency", f"The model '{args.model}' must be installed in the environment where this script is run."],
    ], columns=["Item","Detail"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_name("character_alias_fallback_candidates.csv")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="Summary", index=False)
        summary.to_excel(writer, sheet_name="Character_Summary", index=False)
        context_summary.to_excel(writer, sheet_name="Context_Breakdown", index=False)
        detail.to_excel(writer, sheet_name="Alias_Mention_Details", index=False)
        methodology.to_excel(writer, sheet_name="Methodology", index=False)
    candidates.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("LITMIND ALIAS FALLBACK ADAPTER COMPLETE")
    print(f"Utterances processed: {len(utterances)}")
    print(f"Characters: {len(valid_chars)}")
    print(f"Canonical alias candidate rows: {len(detail)}")
    print(f"Baseline exact PERSON matches: {int(detail['Baseline_Exact_PERSON'].sum()) if len(detail) else 0}")
    print(f"Alias candidates not exact PERSON: {int(detail['Candidate_Recovery_vs_Exact_PERSON'].sum()) if len(detail) else 0}")
    print(f"Excel report: {output_path.resolve()}")
    print(f"Candidate CSV: {csv_path.resolve()}")
    print("Diagnostic only: not an accuracy score.")

if __name__ == "__main__":
    main()
