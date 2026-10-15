#!/usr/bin/env python3
"""Produce a conservative exact-pair audit for LitMind's Dracula predictions."""
import argparse,csv,re
from pathlib import Path
from collections import defaultdict,Counter

def n(s):
    s=(s or "").strip().casefold()
    s=re.sub(r"[\W_]+"," ",s,flags=re.UNICODE)
    return re.sub(r"\s+"," ",s).strip()

def pair(title,a,b):
    return (n(title),)+tuple(sorted((n(a),n(b))))

def map_gold(r):
    c=n(r.get("coarse_category","")); f=n(r.get("fine_category",""))
    if c=="familial": return "FAMILY"
    if c=="professional": return "PROFESSIONAL"
    if c=="social":
        if f in {"friend"}: return "FRIEND / COMPANION"
        if f in {"lovers","unrequited love interest"}: return "ROMANTIC / LOVE"
        if f in {"enemy","rivals"}: return "CONFLICT / ENEMY"
        if f=="acquaintance": return "INTERACTION"
    return None

def map_pred(s):
    x=n(s)
    if x in {"family","familial"}: return "FAMILY"
    if x in {"professional","work"}: return "PROFESSIONAL"
    if x in {"friend companion","friend","companion","friend / companion"}: return "FRIEND / COMPANION"
    if x in {"romantic love","romantic","love","romantic / love"}: return "ROMANTIC / LOVE"
    if x in {"conflict enemy","conflict","enemy","conflict / enemy"}: return "CONFLICT / ENEMY"
    if x in {"interaction","acquaintance"}: return "INTERACTION"
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold-tsv",default=r"data\external\character_relation_annotations.txt")
    ap.add_argument("--predictions",default=r"reports\e3_relationship_predictions.csv")
    ap.add_argument("--out",default=r"reports\e3_dracula_exact_pair_audit.csv")
    args=ap.parse_args()
    with Path(args.gold_tsv).open(encoding="utf-8-sig",newline="") as f: gold=list(csv.DictReader(f,delimiter="\t"))
    with Path(args.predictions).open(encoding="utf-8-sig",newline="") as f: preds=list(csv.DictReader(f))
    votes=defaultdict(Counter); displays={}
    for r in gold:
        if n(r.get("title"))!="dracula": continue
        lab=map_gold(r)
        if not lab: continue
        k=pair(r.get("title"),r.get("character_1"),r.get("character_2"))
        votes[k][lab]+=1
        displays[k]=(r.get("character_1",""),r.get("character_2",""))
    consensus={}
    for k,v in votes.items():
        m=max(v.values()); winners=[lab for lab,count in v.items() if count==m]
        consensus[k]=winners[0] if len(winners)==1 else "TIE/AMBIGUOUS"
    rows=[]
    for r in preds:
        if n(r.get("title"))!="dracula": continue
        k=pair(r.get("title"),r.get("character_1"),r.get("character_2"))
        predlab=map_pred(r.get("predicted_relation",r.get("relationship",r.get("relation",""))))
        if k in consensus:
            goldlab=consensus[k]
            status="MATCHED_TIE" if goldlab=="TIE/AMBIGUOUS" else ("CORRECT" if predlab==goldlab else "LABEL_DISAGREEMENT")
            g_a,g_b=displays[k]
            vote_text="; ".join(f"{lab}:{cnt}" for lab,cnt in votes[k].most_common())
        else:
            goldlab="NO_EXACT_GOLD_PAIR"; status="UNMATCHED_PAIR"; g_a=g_b=""; vote_text=""
        rows.append({
            "Title":r.get("title",""),"Character_1":r.get("character_1",""),"Character_2":r.get("character_2",""),
            "Predicted_Relation":r.get("predicted_relation",r.get("relationship",r.get("relation",""))),
            "Mapped_Prediction":predlab or "UNMAPPED_LABEL","Exact_Gold_Character_1":g_a,"Exact_Gold_Character_2":g_b,
            "Gold_Majority_Label":goldlab,"Gold_Annotation_Votes":vote_text,"Audit_Status":status
        })
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]) if rows else ["Title","Character_1","Character_2","Audit_Status"]
    with out.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    counts=Counter(r["Audit_Status"] for r in rows)
    print("LITMIND EXACT-PAIR AUDIT")
    print(f"Predictions audited: {len(rows)}")
    for k,v in sorted(counts.items()): print(f"{k}: {v}")
    print("Only exact title + unordered character pair matches were counted.")
    print("This is a diagnostic audit, not a robust benchmark score.")
    print("Report:",out.resolve())
if __name__=="__main__": main()
