#!/usr/bin/env python3
"""Show Dracula benchmark character pairs and suggest name matches without changing labels."""
import argparse, csv, re
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter, defaultdict

def norm(s):
    s=(s or "").strip().casefold()
    s=re.sub(r"[\W_]+"," ",s,flags=re.UNICODE)
    return re.sub(r"\s+"," ",s).strip()

def sim(a,b):
    a,b=norm(a),norm(b)
    if not a or not b: return 0.0
    if a in b or b in a: return 0.92
    return SequenceMatcher(None,a,b).ratio()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold-tsv",default=r"data\external\character_relation_annotations.txt")
    ap.add_argument("--predictions",default=r"reports\e3_relationship_predictions.csv")
    ap.add_argument("--out",default=r"reports\e3_dracula_pair_diagnostics.txt")
    args=ap.parse_args()
    gp,pp=Path(args.gold_tsv),Path(args.predictions)
    if not gp.exists(): raise FileNotFoundError(gp.resolve())
    if not pp.exists(): raise FileNotFoundError(pp.resolve())
    with gp.open(encoding="utf-8-sig",newline="") as f:
        gold=list(csv.DictReader(f,delimiter="\t"))
    with pp.open(encoding="utf-8-sig",newline="") as f:
        preds=list(csv.DictReader(f))
    rows=[r for r in gold if norm(r.get("title",""))=="dracula"]
    pair_votes=defaultdict(Counter)
    display={}
    for r in rows:
        a=r.get("character_1","").strip(); b=r.get("character_2","").strip()
        key=tuple(sorted([norm(a),norm(b)]))
        display[key]=(a,b)
        label=(r.get("coarse_category","")+" / "+r.get("fine_category","")).strip(" /")
        pair_votes[key][label]+=1
    lines=["LITMIND DRACULA GOLD-PAIR DIAGNOSTICS",f"Raw Dracula annotation rows: {len(rows)}",
           f"Distinct unordered gold pairs: {len(pair_votes)}","",
           "PREDICTION -> BEST GOLD-PAIR CANDIDATES (name similarity only; suggestions are not automatic matches)"]
    for pred in preds:
        if norm(pred.get("title",""))!="dracula": continue
        a=pred.get("character_1",""); b=pred.get("character_2","")
        plabel=pred.get("predicted_relation",pred.get("relationship",pred.get("relation","")))
        candidates=[]
        for key,(ga,gb) in display.items():
            direct=sim(a,ga)+sim(b,gb)
            reverse=sim(a,gb)+sim(b,ga)
            score=max(direct,reverse)/2
            candidates.append((score,ga,gb,pair_votes[key].most_common(3)))
        candidates.sort(reverse=True)
        lines.append(f"\nPrediction: {a} -- {b} | LitMind label={plabel}")
        if candidates and candidates[0][0]>=0.60:
            for score,ga,gb,votes in candidates[:5]:
                lines.append(f"  similarity={score:.3f} | gold pair={ga} -- {gb} | annotations={votes}")
        else:
            lines.append("  No plausible name-similarity candidate. Check benchmark spelling/character identity manually.")
    lines.append("\nALL DISTINCT GOLD PAIRS FOR DRACULA")
    for key,votes in sorted(pair_votes.items(),key=lambda kv:(display[kv[0]][0].casefold(),display[kv[0]][1].casefold())):
        a,b=display[key]
        lines.append(f"  {a} -- {b} | votes={votes.most_common(5)}")
    lines.append("\nCAUTION: similarity suggestions are for human inspection only. Do not auto-merge characters or compute final metrics from fuzzy matches.")
    out=Path(args.out)
    if not out.is_absolute(): out=Path.cwd()/out
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport: {out}")
if __name__=="__main__": main()
