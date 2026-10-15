#!/usr/bin/env python3
"""Export the frozen LitMind E3 relationship predictions without editing frozen source files.

Run from the LitMind project root:
  python export_litmind_e3_predictions.py --text data/processed/Dracula.txt --out reports/e3_dracula_predictions.csv

This executes src/nlp_processor.py as-is using runpy, suppresses only the interactive plot window,
then exports the resulting in-memory relationship_results dictionary. It does not change
src/nlp_processor.py or src/prev.py.
"""
from __future__ import annotations
import argparse
import csv
import runpy
import sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",default="src/nlp_processor.py")
    parser.add_argument("--text",default="data/processed/Dracula.txt")
    parser.add_argument("--title",default="Dracula")
    parser.add_argument("--out",default="reports/e3_dracula_predictions.csv")
    args=parser.parse_args()
    source=Path(args.source).resolve()
    text=Path(args.text).resolve()
    out=Path(args.out).resolve()
    if not source.is_file(): raise FileNotFoundError(f"Frozen E3 source not found: {source}")
    if not text.is_file(): raise FileNotFoundError(f"Input text not found: {text}")
    out.parent.mkdir(parents=True,exist_ok=True)

    # Prevent matplotlib from opening a blocking GUI window; analysis remains unchanged.
    import matplotlib.pyplot as plt
    old_show=plt.show
    plt.show=lambda *a,**k: None
    old_argv=sys.argv[:]
    try:
        sys.argv=[str(source),str(text)]
        namespace=runpy.run_path(str(source),run_name="__main__")
    finally:
        sys.argv=old_argv
        plt.show=old_show

    results=namespace.get("relationship_results")
    if not isinstance(results,dict):
        raise RuntimeError("Frozen E3 did not expose relationship_results; no prediction CSV written.")
    fields=["title","character_1","character_2","predicted_relation","confidence","co_occurrence",
            "co_occurrence_strength","score","keywords","evidence_contexts"]
    with out.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for pair,result in sorted(results.items(),key=lambda item:(str(item[0][0]).casefold(),str(item[0][1]).casefold())):
            a,b=pair
            contexts=[]
            seen=set()
            for ev in result.get("evidence",[]) or []:
                context=" ".join(str(ev.get("context","")).split())
                if context and context not in seen:
                    seen.add(context); contexts.append(context)
            writer.writerow({
                "title":args.title,
                "character_1":str(a),
                "character_2":str(b),
                "predicted_relation":result.get("relationship","OTHER / UNKNOWN"),
                "confidence":result.get("confidence",""),
                "co_occurrence":result.get("co_occurrence",""),
                "co_occurrence_strength":result.get("strength",""),
                "score":result.get("score",""),
                "keywords":"; ".join(sorted({str(x) for x in result.get("keywords",[]) or []})),
                "evidence_contexts":" || ".join(contexts),
            })
    print("\nLITMIND E3 PREDICTION EXPORT COMPLETE")
    print(f"Input: {text}")
    print(f"Pair predictions exported: {len(results)}")
    print(f"Output: {out}")
    print("Frozen source files were not modified.")

if __name__=="__main__":
    main()
