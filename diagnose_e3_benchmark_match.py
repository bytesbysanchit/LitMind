#!/usr/bin/env python3
"""Diagnose why LitMind E3 predictions do not match the external benchmark."""
import argparse, csv, re
from pathlib import Path
from collections import Counter

def norm(s):
    s=(s or "").strip().casefold()
    s=re.sub(r"[\W_]+"," ",s,flags=re.UNICODE)
    return re.sub(r"\s+"," ",s).strip()

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold-tsv",default=r"data\external\character_relation_annotations.txt")
    ap.add_argument("--predictions",default=r"reports\e3_relationship_predictions.csv")
    ap.add_argument("--out",default=r"reports\e3_match_diagnostics.txt")
    args=ap.parse_args()
    gold_path=Path(args.gold_tsv); pred_path=Path(args.predictions)
    if not gold_path.exists(): raise FileNotFoundError(gold_path.resolve())
    if not pred_path.exists(): raise FileNotFoundError(pred_path.resolve())
    with gold_path.open(encoding="utf-8-sig",newline="") as f:
        gold=list(csv.DictReader(f,delimiter="\t"))
    pred=read_csv(pred_path)
    required={"title","character_1","character_2"}
    if not pred or not required.issubset(pred[0]):
        raise ValueError(f"Prediction CSV must include {sorted(required)}")
    gold_titles=Counter(r.get("title","").strip() for r in gold)
    pred_titles=Counter(r.get("title","").strip() for r in pred)
    gold_titles_norm={norm(t):t for t in gold_titles}
    lines=[]
    lines.append("LITMIND E3 MATCH DIAGNOSTICS")
    lines.append(f"Gold annotation rows: {len(gold)}")
    lines.append(f"Distinct gold titles: {len(gold_titles)}")
    lines.append(f"Prediction rows: {len(pred)}")
    lines.append("Prediction titles and exact normalized title availability:")
    for t,n in pred_titles.items():
        lines.append(f"  {t!r}: predictions={n}; gold title match={'YES: '+gold_titles_norm[norm(t)] if norm(t) in gold_titles_norm else 'NO'}")
    lines.append("")
    lines.append("Gold title candidates containing each prediction title:")
    for t in pred_titles:
        q=norm(t)
        candidates=[g for g in gold_titles if q in norm(g) or norm(g) in q]
        lines.append(f"  {t!r}: {candidates[:15] if candidates else 'none'}")
    lines.append("")
    lines.append("Prediction pairs:")
    for r in pred:
        title=r.get("title","")
        a=r.get("character_1","")
        b=r.get("character_2","")
        label=r.get("predicted_relation",r.get("relationship",r.get("relation","")))
        lines.append(f"  title={title!r} | {a!r} -- {b!r} | label={label!r}")
    lines.append("")
    lines.append("Suggested next step:")
    if not any(norm(t) in gold_titles_norm for t in pred_titles):
        lines.append("No prediction title exactly matches the benchmark. Run LitMind on a work actually present in the gold title list; do not force fuzzy matching.")
    else:
        lines.append("At least one title matches. Inspect canonical character names/aliases for pairs that fail to match. Keep title and pair matching exact unless an explicit documented alias map is added.")
    out=Path(args.out)
    if not out.is_absolute(): out=Path.cwd()/out
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))
    print(f"\nDiagnostic report: {out}")
if __name__=="__main__": main()
