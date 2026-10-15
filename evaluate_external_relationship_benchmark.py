#!/usr/bin/env python3
"""LitMind E3 external relationship benchmark adapter.

Inputs:
  --gold-tsv path/to/character_relation_annotations.txt
  --predictions path/to/litmind_e3_pair_predictions.csv
Prediction CSV required columns:
  title, character_1, character_2, predicted_relation

The public gold file is a TSV with columns:
annotator, change, title, author, character_1, character_2, affinity,
coarse_category, fine_category, detail

Gold labels are crowd annotations, not adjudicated labels. This script aggregates
per pair by majority vote, excludes ties/unknown mappings, and reports coverage
and macro/weighted metrics. It does not claim that all E3 categories map perfectly.
"""
import argparse, csv, json, re
from collections import Counter, defaultdict
from pathlib import Path

MAP = {
    ("familial", "*"): "FAMILY",
    ("professional", "*"): "PROFESSIONAL",
    ("social", "friend"): "FRIEND / COMPANION",
    ("social", "lovers"): "ROMANTIC / LOVE",
    ("social", "unrequited love interest"): "ROMANTIC / LOVE",
    ("social", "enemy"): "CONFLICT / ENEMY",
    ("social", "rivals"): "CONFLICT / ENEMY",
    ("social", "acquaintance"): "INTERACTION",
}
UNKNOWN = {"", "nr", "unknown", "other", "none", "nan"}

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().casefold())

def canonical_pair(title, a, b):
    # Pair direction is irrelevant for this coarse relationship task.
    x, y = sorted([norm(a), norm(b)])
    return norm(title), x, y

def mapped_label(row):
    coarse = norm(row.get("coarse_category", ""))
    fine = norm(row.get("fine_category", ""))
    if coarse == "familial":
        return "FAMILY"
    if coarse == "professional":
        return "PROFESSIONAL"
    if coarse == "social":
        return MAP.get((coarse, fine))
    return None

def metric_block(golds, preds):
    labels = sorted(set(golds) | set(preds))
    if not golds:
        return {"n": 0, "accuracy": None, "macro_f1": None, "weighted_f1": None, "labels": {}}
    tp_fp_fn = {}
    f1s, weighted = [], []
    correct = sum(g == p for g,p in zip(golds,preds))
    for lab in labels:
        tp = sum(g == lab and p == lab for g,p in zip(golds,preds))
        fp = sum(g != lab and p == lab for g,p in zip(golds,preds))
        fn = sum(g == lab and p != lab for g,p in zip(golds,preds))
        support = sum(g == lab for g in golds)
        prec = tp/(tp+fp) if tp+fp else 0.0
        rec = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
        tp_fp_fn[lab] = {"precision":prec,"recall":rec,"f1":f1,"support":support}
        f1s.append(f1)
        weighted.append((f1,support))
    total = len(golds)
    return {
        "n": total, "accuracy": correct/total,
        "macro_f1": sum(f1s)/len(f1s) if f1s else None,
        "weighted_f1": sum(f*s for f,s in weighted)/total if total else None,
        "labels": tp_fp_fn
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold-tsv", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default="reports/external_relationship_evaluation.json")
    args=ap.parse_args()
    gp=Path(args.gold_tsv); pp=Path(args.predictions)
    if not gp.is_file(): raise FileNotFoundError(gp)
    if not pp.is_file(): raise FileNotFoundError(pp)
    with gp.open(encoding="utf-8-sig",newline="") as f:
        gold_rows=list(csv.DictReader(f,delimiter="\t"))
    required_gold={"title","character_1","character_2","coarse_category","fine_category"}
    if not gold_rows or not required_gold.issubset(gold_rows[0]):
        raise ValueError("Gold TSV headers do not match the published characterRelations schema.")
    with pp.open(encoding="utf-8-sig",newline="") as f:
        pred_rows=list(csv.DictReader(f))
    required_pred={"title","character_1","character_2","predicted_relation"}
    if not pred_rows or not required_pred.issubset(pred_rows[0]):
        raise ValueError(f"Prediction CSV must include {sorted(required_pred)}")
    votes=defaultdict(Counter)
    source_counts=Counter()
    for r in gold_rows:
        label=mapped_label(r)
        if not label:
            source_counts["unmapped_gold_annotations"]+=1
            continue
        pair=canonical_pair(r["title"],r["character_1"],r["character_2"])
        votes[pair][label]+=1
        source_counts["mapped_gold_annotations"]+=1
    gold={}
    tie_count=0
    for pair,counts in votes.items():
        maxv=max(counts.values())
        winners=[lab for lab,n in counts.items() if n==maxv]
        if len(winners)!=1:
            tie_count+=1
            continue
        gold[pair]=winners[0]
    pred={}
    for r in pred_rows:
        pair=canonical_pair(r["title"],r["character_1"],r["character_2"])
        label=(r["predicted_relation"] or "").strip().upper()
        if label in {"FRIEND","FRIEND/COMPANION","FRIEND / COMPANION","COMPANION"}: label="FRIEND / COMPANION"
        elif label in {"FAMILY","FAMILIAL"}: label="FAMILY"
        elif label in {"ROMANTIC","ROMANTIC/LOVE","ROMANTIC / LOVE","LOVE"}: label="ROMANTIC / LOVE"
        elif label in {"PROFESSIONAL","WORK"}: label="PROFESSIONAL"
        elif label in {"CONFLICT","ENEMY","CONFLICT/ENEMY","CONFLICT / ENEMY"}: label="CONFLICT / ENEMY"
        elif label in {"INTERACTION","ACQUAINTANCE"}: label="INTERACTION"
        else: label=None
        if label:
            pred[pair]=label
    matched=sorted(set(gold)&set(pred))
    y=[gold[k] for k in matched]; p=[pred[k] for k in matched]
    missing_pred=sorted(set(gold)-set(pred))
    extra_pred=sorted(set(pred)-set(gold))
    result={
        "benchmark":"Bamman et al. Character Relations in Literary Texts",
        "gold_annotation_rows":len(gold_rows),
        "gold_pairs_with_mapped_votes":len(votes),
        "gold_pairs_after_majority_vote_excluding_ties":len(gold),
        "majority_vote_tie_pairs_excluded":tie_count,
        "mapped_gold_annotations":source_counts["mapped_gold_annotations"],
        "unmapped_gold_annotations":source_counts["unmapped_gold_annotations"],
        "prediction_pairs":len(pred),
        "matched_pairs":len(matched),
        "gold_pairs_without_prediction":len(missing_pred),
        "prediction_pairs_without_gold":len(extra_pred),
        "metrics":metric_block(y,p),
        "limitations":[
            "Published annotations are crowdsourced and disagreements are not adjudicated.",
            "Majority vote is an evaluation protocol choice, not a gold label supplied by the dataset authors.",
            "The mapping from benchmark social/familial/professional categories to LitMind E3 labels is approximate.",
            "Only matching titles and character pairs are scored; unmatched pairs are reported as coverage, not silently treated as negatives.",
            "Metrics are valid only if predictions are exported from the frozen LitMind E3 baseline for the same work and character pairs."
        ]
    }
    out=Path(args.out)
    if not out.is_absolute(): out=Path.cwd()/out
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print("LITMIND EXTERNAL RELATIONSHIP EVALUATION")
    print(f"Gold annotation rows: {len(gold_rows)}")
    print(f"Majority-vote gold pairs: {len(gold)}; ties excluded: {tie_count}")
    print(f"Prediction pairs: {len(pred)}; matched pairs: {len(matched)}")
    print("Metrics:", json.dumps(result["metrics"],ensure_ascii=False))
    print("Report:",out)
if __name__=="__main__": main()
