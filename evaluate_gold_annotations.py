#!/usr/bin/env python3
"""
LitMind gold-label evaluation harness.

Purpose:
- Audit an annotation workbook for usable gold labels.
- Optionally compare those labels against model predictions supplied as CSV.
- Never calculate metrics from uncertain labels or from the corpus's own source speaker column.

Prediction CSV format:
task,play,speech_id,prediction
speaker,Tara,TARA_S0005,Dan
relation,Tara,TARA_S0001,SIBLINGS

For relation predictions, use the predicted relation class in `prediction`.
Only relation rows whose Gold_Status is VERIFIED or NONE are eligible.
Speaker rows are eligible only when Gold_Speaker is a canonical label or UNKNOWN,
and Gold_Confidence is HIGH or MEDIUM. UNCERTAIN labels are excluded.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Alignment

SPEAKER_SHEET = "Speaker_Gold_Batch1"
RELATION_SHEET = "Relation_Gold_Batch1"

def norm(value):
    return "" if value is None else str(value).strip()

def load_vertical_sheet(ws):
    # This workbook stores field names vertically in column A and records horizontally from row 10.
    fields = [norm(ws.cell(row, 1).value) for row in range(1, 10)]
    records = []
    for row in range(10, ws.max_row + 1):
        vals = [ws.cell(row, col).value for col in range(1, len(fields) + 1)]
        record = {fields[i]: vals[i] for i in range(len(fields)) if fields[i]}
        # Ignore empty/template rows (including formatting-only rows).
        speech_id = norm(record.get("Speech_ID"))
        play_name = norm(record.get("Play"))
        if not speech_id or not play_name or speech_id.upper() in {"SPEECH_ID", "DRAFT STATUS"}:
            continue
        # Actual corpus IDs have a stable prefix plus underscore (e.g. TARA_S0001).
        if "_" not in speech_id:
            continue
        records.append(record)
    return records

def load_predictions(path):
    predictions = {}
    if not path:
        return predictions
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"task", "play", "speech_id", "prediction"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("Prediction CSV must have columns: task,play,speech_id,prediction")
        for row in reader:
            key = (norm(row["task"]).casefold(), norm(row["play"]).casefold(), norm(row["speech_id"]))
            if key in predictions:
                raise ValueError(f"Duplicate prediction key in CSV: {key}")
            predictions[key] = norm(row["prediction"])
    return predictions

def score(labels, preds):
    classes = sorted(set(labels) | set(preds))
    total = len(labels)
    correct = sum(a == b for a, b in zip(labels, preds))
    per_class = []
    macro_f1_values = []
    for cls in classes:
        tp = sum(a == cls and b == cls for a, b in zip(labels, preds))
        fp = sum(a != cls and b == cls for a, b in zip(labels, preds))
        fn = sum(a == cls and b != cls for a, b in zip(labels, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class.append({"class": cls, "support": sum(a == cls for a in labels),
                          "precision": precision, "recall": recall, "f1": f1})
        if sum(a == cls for a in labels) > 0:
            macro_f1_values.append(f1)
    return {
        "n": total,
        "accuracy": correct / total if total else None,
        "macro_f1": sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else None,
        "per_class": per_class,
        "confusion_pairs": dict(Counter(f"{a} -> {b}" for a, b in zip(labels, preds) if a != b)),
    }

def main():
    ap = argparse.ArgumentParser(description="Audit LitMind gold annotations and optionally score predictions.")
    ap.add_argument("--gold", required=True, help="Path to SOURCE_REVIEW or filled gold annotation XLSX")
    ap.add_argument("--predictions", help="Optional CSV with task,play,speech_id,prediction columns")
    ap.add_argument("--out", default="reports/gold_evaluation_report.xlsx", help="Output XLSX report path")
    args = ap.parse_args()

    gold_path = Path(args.gold).expanduser().resolve()
    if not gold_path.is_file():
        raise FileNotFoundError(f"Gold workbook not found: {gold_path}")
    wb = load_workbook(gold_path, data_only=True)
    for required_sheet in (SPEAKER_SHEET, RELATION_SHEET):
        if required_sheet not in wb.sheetnames:
            raise ValueError(f"Missing required sheet: {required_sheet}")

    speakers = load_vertical_sheet(wb[SPEAKER_SHEET])
    relations = load_vertical_sheet(wb[RELATION_SHEET])
    preds = load_predictions(args.predictions)

    speaker_eligible = []
    speaker_skipped = Counter()
    for row in speakers:
        gold = norm(row.get("Gold_Speaker"))
        confidence = norm(row.get("Gold_Confidence")).upper()
        if not gold:
            speaker_skipped["blank_gold"] += 1
        elif gold.upper() == "UNCERTAIN":
            speaker_skipped["uncertain_gold"] += 1
        elif confidence not in {"HIGH", "MEDIUM"}:
            speaker_skipped["low_or_missing_confidence"] += 1
        else:
            speaker_eligible.append(row)

    relation_eligible = []
    relation_skipped = Counter()
    for row in relations:
        status = norm(row.get("Gold_Status")).upper()
        label = norm(row.get("Gold_Relation")).upper()
        if status == "UNCERTAIN" or label == "UNCERTAIN":
            relation_skipped["uncertain_gold"] += 1
        elif status not in {"VERIFIED", "NONE"}:
            relation_skipped["not_verified_or_none"] += 1
        elif not label:
            relation_skipped["blank_relation_label"] += 1
        else:
            relation_eligible.append(row)

    def compare(task, rows, gold_field):
        y_true, y_pred, keys, missing = [], [], [], []
        for row in rows:
            key = (task, norm(row.get("Play")).casefold(), norm(row.get("Speech_ID")))
            if key not in preds:
                missing.append(key)
                continue
            y_true.append(norm(row.get(gold_field)).upper() if task == "relation" else norm(row.get(gold_field)))
            y_pred.append(preds[key].upper() if task == "relation" else preds[key])
            keys.append(key)
        return (score(y_true, y_pred) if y_true else None, len(missing), keys)

    speaker_metrics, speaker_missing, _ = compare("speaker", speaker_eligible, "Gold_Speaker")
    relation_metrics, relation_missing, _ = compare("relation", relation_eligible, "Gold_Relation")

    # Human-readable console summary.
    summary = {
        "gold_workbook": str(gold_path),
        "speaker_rows_total": len(speakers),
        "speaker_rows_eligible": len(speaker_eligible),
        "speaker_rows_skipped": dict(speaker_skipped),
        "relation_rows_total": len(relations),
        "relation_rows_eligible": len(relation_eligible),
        "relation_rows_skipped": dict(relation_skipped),
        "prediction_rows_loaded": len(preds),
        "speaker_metrics": speaker_metrics,
        "speaker_eligible_rows_missing_predictions": speaker_missing,
        "relation_metrics": relation_metrics,
        "relation_eligible_rows_missing_predictions": relation_missing,
        "metrics_status": "COMPUTED" if speaker_metrics or relation_metrics else "NOT_COMPUTED_NO_MATCHING_PREDICTIONS",
        "note": "Metrics are computed only for eligible gold rows with matching prediction keys. These metrics are only as valid as the independence and quality of the supplied gold labels."
    }

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = Workbook()
    ws = report.active
    ws.title = "Summary"
    ws.append(["Metric", "Value"])
    for k, v in summary.items():
        ws.append([k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v])
    ws.freeze_panes = "A2"
    ws.column_dimensions["A"].width = 42
    ws.column_dimensions["B"].width = 110
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="274060")
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    for name, metrics in (("Speaker_Metrics", speaker_metrics), ("Relation_Metrics", relation_metrics)):
        sh = report.create_sheet(name)
        sh.append(["Metric", "Value"])
        if metrics is None:
            sh.append(["Status", "NOT COMPUTED — no matching predictions for eligible gold rows"])
        else:
            sh.append(["n", metrics["n"]])
            sh.append(["accuracy", metrics["accuracy"]])
            sh.append(["macro_f1", metrics["macro_f1"]])
            sh.append([])
            sh.append(["Class", "Support", "Precision", "Recall", "F1"])
            for item in metrics["per_class"]:
                sh.append([item["class"], item["support"], item["precision"], item["recall"], item["f1"]])
        sh.column_dimensions["A"].width = 28
        for col in range(2, 6):
            sh.column_dimensions[chr(64 + col)].width = 16
        for cell in sh[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="274060")
        for row in sh.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)

    report.save(out_path)
    print("LITMIND GOLD EVALUATION REPORT")
    print(f"Speaker rows: {len(speakers)} total; {len(speaker_eligible)} eligible; skipped={dict(speaker_skipped)}")
    print(f"Relation rows: {len(relations)} total; {len(relation_eligible)} eligible; skipped={dict(relation_skipped)}")
    print(f"Prediction rows loaded: {len(preds)}")
    print(f"Metrics status: {summary['metrics_status']}")
    if speaker_metrics:
        print(f"Speaker: n={speaker_metrics['n']} accuracy={speaker_metrics['accuracy']:.3f} macro-F1={speaker_metrics['macro_f1']:.3f}")
    if relation_metrics:
        print(f"Relation: n={relation_metrics['n']} accuracy={relation_metrics['accuracy']:.3f} macro-F1={relation_metrics['macro_f1']:.3f}")
    print(f"Report: {out_path}")

if __name__ == "__main__":
    main()
