#!/usr/bin/env python3
"""Conservative exact-pair relationship audit; abstentions are not silently dropped."""
import argparse,csv,re,json
from pathlib import Path
from collections import defaultdict,Counter

def norm(s):
    s=(s or "").strip().casefold()
    s=re.sub(r"[\W_]+"," ",s,flags=re.UNICODE)
    return re.sub(r"\s+"," ",s).strip()

def pair_key(title,a,b):
    return (norm(title),)+tuple(sorted((norm(a),norm(b))))

def gold_label(r):
    c=norm(r.get("coarse_category","")); f=norm(r.get("fine_category",""))
    if c=="familial": return "FAMILY"
    if c=="professional": return "PROFESSIONAL"
    if c=="social":
        if f=="friend": return "FRIEND / COMPANION"
        if f in {"lovers","unrequited love interest"}: return "ROMANTIC / LOVE"
        if f in {"enemy","rivals"}: return "CONFLICT / ENEMY"
        if f=="acquaintance": return "INTERACTION"
    return None

def pred_label(r):
    raw=r.get("predicted_relation",r.get("relationship",r.get("relation","")))
    x=norm(raw)
    if x in {"family","familial"}: return "FAMILY"
    if x in {"professional","work"}: return "PROFESSIONAL"
    if x in {"friend","companion","friend companion","friend / companion"}: return "FRIEND / COMPANION"
    if x in {"romantic","love","romantic love","romantic / love"}: return "ROMANTIC / LOVE"
    if x in {"conflict","enemy","conflict enemy","conflict / enemy"}: return "CONFLICT / ENEMY"
    if x in {"interaction","acquaintance"}: return "INTERACTION"
    if x in {"other unknown","other","unknown","other / unknown"}: return "OTHER / UNKNOWN"
    return "UNMAPPED: "+raw

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold-tsv",default=r"data\external\character_relation_annotations.txt")
    ap.add_argument("--predictions",default=r"reports\e3_relationship_predictions.csv")
    ap.add_argument("--out",default=r"reports\e3_exact_pair_metrics.json")
    args=ap.parse_args()
    with Path(args.gold_tsv).open(encoding="utf-8-sig",newline="") as f: gold=list(csv.DictReader(f,delimiter="\t"))
    with Path(args.predictions).open(encoding="utf-8-sig",newline="") as f: preds=list(csv.DictReader(f))
    votes=defaultdict(Counter)
    for r in gold:
        if norm(r.get("title"))!="dracula": continue
        lab=gold_label(r)
        if lab: votes[pair_key(r.get("title"),r.get("character_1"),r.get("character_2"))][lab]+=1
    consensus={}
    ties={}
    for k,v in votes.items():
        mx=max(v.values()); winners=[l for l,c in v.items() if c==mx]
        if len(winners)==1: consensus[k]=winners[0]
        else: ties[k]=dict(v)
    out_rows=[]; matched_clear=[]; matched_tie=0; unmatched=0
    for r in preds:
        if norm(r.get("title"))!="dracula": continue
        k=pair_key(r.get("title"),r.get("character_1"),r.get("character_2"))
        pl=pred_label(r)
        if k in consensus:
            gl=consensus[k]
            matched_clear.append((gl,pl))
            status="CORRECT" if pl==gl else "LABEL_DISAGREEMENT"
            votes_text="; ".join(f"{l}:{c}" for l,c in votes[k].most_common())
        elif k in ties:
            matched_tie+=1; gl="TIE/AMBIGUOUS"; status="EXCLUDED_GOLD_TIE"
            votes_text="; ".join(f"{l}:{c}" for l,c in ties[k].items())
        else:
            unmatched+=1; gl="NO_EXACT_GOLD_PAIR"; status="UNMATCHED_PAIR"; votes_text=""
        out_rows.append({"title":r.get("title",""),"character_1":r.get("character_1",""),"character_2":r.get("character_2",""),
                         "predicted_relation":r.get("predicted_relation",r.get("relationship",r.get("relation",""))),
                         "mapped_prediction":pl,"gold_label":gl,"gold_votes":votes_text,"status":status})
    n_clear=len(matched_clear); correct=sum(g==p for g,p in matched_clear)
    result={
      "title":"Dracula",
      "prediction_pairs":len(out_rows),
      "exact_pairs_with_gold":n_clear+matched_tie,
      "exact_pairs_with_clear_consensus":n_clear,
      "exact_pairs_with_gold_tie_excluded":matched_tie,
      "unmatched_prediction_pairs":unmatched,
      "exact_pair_coverage_of_predictions":(n_clear+matched_tie)/len(out_rows) if out_rows else None,
      "correct_clear_consensus_pairs":correct,
      "label_disagreements_on_clear_pairs":n_clear-correct,
      "accuracy_on_clear_exact_pairs":correct/n_clear if n_clear else None,
      "warning":"Dracula has only a small, crowd-sourced set of gold pairs. These are descriptive diagnostics, not a reliable estimate of general LitMind accuracy.",
      "rows":out_rows
    }
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print("LITMIND EXACT-PAIR METRICS (CORRECTED)")
    for k,v in result.items():
        if k!="rows": print(f"{k}: {v}")
    print("Report:",out.resolve())
if __name__=="__main__": main()
