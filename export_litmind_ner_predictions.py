#!/usr/bin/env python3
"""Export spaCy PERSON and canonical-name alias predictions for LitMind play utterances.

This is a prediction-export step, not an accuracy evaluator. Gold labels must be
annotated independently before computing precision/recall/F1.
Run from the LitMind project root:
  python export_litmind_ner_predictions.py --corpus Dattani_Tara_Corpus_Final.xlsx --out reports/ner_mention_predictions.csv
"""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path
from collections import defaultdict
from openpyxl import load_workbook

def clean(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()

def header_map(ws):
    vals = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
    return {clean(v): i for i, v in enumerate(vals) if v is not None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="Dattani_Tara_Corpus_Final.xlsx")
    ap.add_argument("--out", default="reports/ner_mention_predictions.csv")
    ap.add_argument("--model", default="en_core_web_sm")
    args = ap.parse_args()

    corpus = Path(args.corpus).expanduser().resolve()
    if not corpus.is_file():
        raise FileNotFoundError(f"Corpus workbook not found: {corpus}")

    try:
        import spacy
    except ImportError as exc:
        raise SystemExit("spaCy is not installed in this Python environment. Activate D:\\LitMind\\.venv and install project requirements first.") from exc
    try:
        nlp = spacy.load(args.model)
    except OSError as exc:
        raise SystemExit(f"spaCy model {args.model!r} is unavailable in this environment. Install/load it in the active .venv first.") from exc

    wb = load_workbook(corpus, read_only=True, data_only=True)
    required = {"CHARACTER_MASTER", "UTTERANCE_CORPUS"}
    if not required.issubset(wb.sheetnames):
        raise ValueError(f"Workbook must contain sheets {sorted(required)}")
    mw, uw = wb["CHARACTER_MASTER"], wb["UTTERANCE_CORPUS"]
    mh, uh = header_map(mw), header_map(uw)
    need_m = {"Play_ID", "Character_Name"}
    need_u = {"Speech_ID", "Play_ID", "Play", "Act", "Speaker", "Speech_OCR"}
    if not need_m.issubset(mh):
        raise ValueError(f"CHARACTER_MASTER missing: {sorted(need_m - set(mh))}")
    if not need_u.issubset(uh):
        raise ValueError(f"UTTERANCE_CORPUS missing: {sorted(need_u - set(uh))}")

    names_by_play = defaultdict(list)
    for row in mw.iter_rows(min_row=2, values_only=True):
        play_id = clean(row[mh["Play_ID"]]).upper()
        name = clean(row[mh["Character_Name"]])
        if play_id and name:
            names_by_play[play_id].append(name)

    rows_out = []
    n_utterances = 0
    for row in uw.iter_rows(min_row=2, values_only=True):
        speech_id = clean(row[uh["Speech_ID"]])
        play_id = clean(row[uh["Play_ID"]]).upper()
        play = clean(row[uh["Play"]])
        text = clean(row[uh["Speech_OCR"]])
        if not play_id or not text:
            continue
        n_utterances += 1
        doc = nlp(text)
        person_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        person_norm = {re.sub(r"[^a-z0-9]", "", x.casefold()) for x in person_entities}
        for name in names_by_play.get(play_id, []):
            pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(text):
                surface = match.group(0)
                norm_surface = re.sub(r"[^a-z0-9]", "", surface.casefold())
                exact_person = norm_surface in person_norm
                # Overlap means a PERSON entity includes the matched surface or is contained by it.
                start, end = match.span()
                overlap = any(
                    ent.start_char < end and ent.end_char > start and ent.label_ == "PERSON"
                    for ent in doc.ents
                )
                left = max(0, start - 100)
                right = min(len(text), end + 140)
                rows_out.append({
                    "Speech_ID": speech_id,
                    "Play_ID": play_id,
                    "Play": play,
                    "Act": clean(row[uh["Act"]]),
                    "Character_Candidate": name,
                    "Matched_Surface": surface,
                    "Start_Char": start,
                    "End_Char": end,
                    "Baseline_Exact_PERSON_Prediction": int(exact_person),
                    "Baseline_PERSON_Overlap_Prediction": int(overlap),
                    "All_PERSON_Entities": " ; ".join(person_entities),
                    "Local_Context": text[left:right],
                    "Gold_Mention": "",
                    "Gold_Referent": "",
                    "Gold_Status": "",
                    "Gold_Notes": "",
                })

    wb.close()
    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Speech_ID","Play_ID","Play","Act","Character_Candidate",
        "Matched_Surface","Start_Char","End_Char","Baseline_Exact_PERSON_Prediction",
        "Baseline_PERSON_Overlap_Prediction","All_PERSON_Entities","Local_Context",
        "Gold_Mention","Gold_Referent","Gold_Status","Gold_Notes"
    ]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    print("LITMIND NER PREDICTION EXPORT COMPLETE")
    print(f"Utterances processed: {n_utterances}")
    print(f"Canonical-name candidate occurrences exported: {len(rows_out)}")
    print(f"Exact PERSON predictions: {sum(r['Baseline_Exact_PERSON_Prediction'] for r in rows_out)}")
    print(f"PERSON overlap predictions: {sum(r['Baseline_PERSON_Overlap_Prediction'] for r in rows_out)}")
    print(f"Output: {out}")
    print("Gold columns are intentionally blank; annotate them independently before scoring metrics.")

if __name__ == "__main__":
    main()
