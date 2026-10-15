#!/usr/bin/env python3
"""One-command LitMind E3 relationship benchmark runner.

Run from the LitMind project root:
  python run_litmind_e3_benchmark.py

Requires network access to download the public benchmark on first run.
Does not modify src/nlp_processor.py or src/prev.py.
"""
from pathlib import Path
import argparse, subprocess, sys, urllib.request

GOLD_URL="https://raw.githubusercontent.com/dbamman/characterRelations/master/character_relation_annotations.txt"

def run(cmd):
    print("\n>", " ".join(map(str,cmd)), flush=True)
    subprocess.run(list(map(str,cmd)),check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--text",default="data/processed/Dracula.txt")
    ap.add_argument("--title",default="Dracula")
    ap.add_argument("--source",default="src/nlp_processor.py")
    args=ap.parse_args()
    root=Path.cwd()
    for required in [Path(args.source),Path(args.text),Path("export_litmind_e3_predictions.py"),Path("evaluate_external_relationship_benchmark.py")]:
        if not (root/required).is_file():
            raise FileNotFoundError(f"Run this script from the LitMind project root. Missing: {required}")
    gold=Path("data/external/character_relation_annotations.txt")
    gold.parent.mkdir(parents=True,exist_ok=True)
    if not gold.is_file() or gold.stat().st_size<10000:
        print("Downloading public characterRelations benchmark...",flush=True)
        try:
            req=urllib.request.Request(GOLD_URL,headers={"User-Agent":"LitMind-Evaluation/1.0"})
            with urllib.request.urlopen(req,timeout=45) as response:
                content=response.read()
            if len(content)<10000:
                raise RuntimeError(f"Downloaded benchmark unexpectedly small ({len(content)} bytes)")
            gold.write_bytes(content)
        except Exception as exc:
            raise RuntimeError("Could not download benchmark. Check internet access, then rerun. " + str(exc))
    predictions=Path("reports/e3_relationship_predictions.csv")
    report=Path("reports/e3_external_benchmark_evaluation.json")
    run([sys.executable,"export_litmind_e3_predictions.py","--source",args.source,"--text",args.text,
         "--title",args.title,"--out",str(predictions)])
    run([sys.executable,"evaluate_external_relationship_benchmark.py","--gold-tsv",str(gold),
         "--predictions",str(predictions),"--out",str(report)])
    print("\nPIPELINE FINISHED")
    print(f"Predictions: {predictions.resolve()}")
    print(f"Evaluation report: {report.resolve()}")
    print("Interpret scores cautiously: the published gold is crowd-sourced and the category mapping is approximate.")

if __name__=="__main__":
    main()
