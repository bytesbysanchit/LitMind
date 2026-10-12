"""LitMind play-specific diagnostic. Does NOT modify or import nlp_processor.py.

Run from D:\\LitMind after copying this file into src or project root:
  python play_evaluation.py --corpus Dattani_Tara_Corpus_Final.xlsx --text-dir data/processed --out reports/play_evaluation
Requires: openpyxl, spacy, en_core_web_sm, networkx.
"""
from pathlib import Path
from collections import Counter
import argparse, re, csv
import openpyxl
import spacy
import networkx as nx

PLAY_ALIASES = {
  "TARA": ["TARA", "Tara"],
  "DANCE": ["DANCE", "Dance Like a Man", "Dance_Like_a_Man"],
}

def norm(s):
  s = str(s or "").replace("’", "'").strip()
  s = re.sub(r"\s+", " ", s)
  s = re.sub(r"[^A-Za-zÀ-ÿ .'’-]", "", s)
  return s.strip(" .'-").lower()

def tokens(s):
  return [x for x in re.findall(r"[a-z]+", norm(s)) if x not in {"mr", "mrs", "ms", "miss", "dr", "sir", "lady"}]

def play_key(play_id, play_name=""):
  pid = str(play_id or "").strip().upper()
  name = (str(play_id or "") + " " + str(play_name or "")).lower()

  # Use the corpus's actual Play_ID values first.
  if pid == "D01":
    return "Tara"
  if pid == "D02":
    return "Dance Like a Man"

  # Fallback for readable play names.
  if "tara" in name:
    return "Tara"
  if "dance" in name or "man" in name:
    return "Dance Like a Man"

  return None

def match_reference(name, refs):
  nt = tokens(name)
  if not nt: return None
  nset = set(nt)
  candidates = []
  for ref in refs:
    rt = tokens(ref); rset = set(rt)
    # Exact normalized name or a full one-token/last-name alias match.
    if nt == rt or (len(nt) == 1 and nt[0] in rset):
      candidates.append(ref)
  return candidates[0] if len(candidates) == 1 else None

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--corpus', required=True, help='Path to Dattani corpus Excel workbook')
  ap.add_argument('--text-dir', required=True, help='Folder containing the two clean input .txt files')
  ap.add_argument('--out', default='reports/play_evaluation', help='Output prefix (without extension)')
  args = ap.parse_args()
  corpus = Path(args.corpus); text_dir = Path(args.text_dir); out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  if not corpus.exists(): raise SystemExit(f'Corpus workbook not found: {corpus}')
  if not text_dir.exists(): raise SystemExit(f'Text directory not found: {text_dir}')
  nlp = spacy.load('en_core_web_sm')
  wb = openpyxl.load_workbook(corpus, read_only=True, data_only=True)
  master_ws = wb['CHARACTER_MASTER']; master_rows = list(master_ws.iter_rows(values_only=True))
  mh = [str(x or '').strip() for x in master_rows[0]]
  mi = {k: mh.index(k) for k in ['Play_ID','Character_Name']}
  refs = {'Tara': [], 'Dance Like a Man': []}
  for row in master_rows[1:]:
    if not row: continue
    pk = play_key(row[mi['Play_ID']])
    name = str(row[mi['Character_Name']] or '').strip()
    if pk and name and name not in refs[pk]: refs[pk].append(name)
  utter_ws = wb['UTTERANCE_CORPUS']; utter_rows = list(utter_ws.iter_rows(values_only=True))
  uh = [str(x or '').strip() for x in utter_rows[0]]
  ui = {k: uh.index(k) for k in ['Play_ID','Play','Speech_ID','Speaker','Speech_OCR']}
  utterances = {'Tara': [], 'Dance Like a Man': []}
  for row in utter_rows[1:]:
    if not row: continue
    pk = play_key(row[ui['Play_ID']], row[ui['Play']])
    if pk:
      utterances[pk].append((row[ui['Speech_ID']], str(row[ui['Speaker']] or '').strip(), str(row[ui['Speech_OCR']] or '')))
  workbook = openpyxl.Workbook(); summary = workbook.active; summary.title = 'Summary'
  summary.append(['Play','Reference speaker count','Utterance rows','Input text characters','PERSON mentions','Unique PERSON surfaces','Unique PERSON surfaces matched to reference','Reference names seen in NER','Reference names not seen in NER','False-positive/unmatched PERSON surfaces'])
  det_ws = workbook.create_sheet('NER_Details'); det_ws.append(['Play','PERSON surface','Mentions','Matched reference speaker','Example context'])
  miss_ws = workbook.create_sheet('Reference_Check'); miss_ws.append(['Play','Reference speaker','NER match found?','Matched surface(s)'])
  graph_ws = workbook.create_sheet('Interaction_Edges'); graph_ws.append(['Play','Speaker A','Speaker B','Adjacent-turn count','Evidence'])
  all_counts = {}
  for play, filename in [('Tara','Tara_clean_input.txt'),('Dance Like a Man','Dance_Like_a_Man_clean_input.txt')]:
    path = text_dir / filename
    if not path.exists(): raise SystemExit(f'Input text not found: {path}')
    text = path.read_text(encoding='utf-8')
    doc = nlp(text)
    surfaces = Counter(); contexts = {}
    for ent in doc.ents:
      if ent.label_ == 'PERSON':
        name = re.sub(r'\s+', ' ', ent.text.strip()).strip(" .,'’")
        if name:
          surfaces[name] += 1; contexts.setdefault(name, ent.sent.text[:240])
    matched = {name: match_reference(name, refs[play]) for name in surfaces}
    seen_refs = set(v for v in matched.values() if v)
    summary.append([play,len(refs[play]),len(utterances[play]),len(text),sum(surfaces.values()),len(surfaces),sum(1 for x in matched.values() if x),len(seen_refs),len(refs[play])-len(seen_refs),sum(1 for v in matched.values() if not v)])
    for name,count in surfaces.most_common():
      det_ws.append([play,name,count,matched[name] or '',contexts.get(name,'')])
    for ref in refs[play]:
      hits = [name for name,val in matched.items() if val == ref]
      miss_ws.append([play,ref,'YES' if hits else 'NO','; '.join(hits)])
    # Interaction graph: successive different speakers in the annotated utterance sequence.
    # This is a dialogue-turn interaction graph, NOT a semantic relationship graph.
    rows = utterances[play]
    edge_counts = Counter(); edge_evidence = {}
    for i in range(1, len(rows)):
      _, prev_speaker, prev_text = rows[i-1]
      speech_id, speaker, speech_text = rows[i]
      if not prev_speaker or not speaker or prev_speaker.lower() == speaker.lower(): continue
      a,b = sorted([prev_speaker, speaker], key=str.lower)
      edge_counts[(a,b)] += 1
      edge_evidence.setdefault((a,b), f'{rows[i-1][0]} → {speech_id}: {prev_speaker} / {speaker}')
    for (a,b),count in edge_counts.most_common():
      graph_ws.append([play,a,b,count,edge_evidence[(a,b)]])
    # Store speaker list in workbook for traceability.
    all_counts[play] = {'refs': refs[play], 'utterances': len(rows), 'ner_mentions': sum(surfaces.values()), 'unique_ner': len(surfaces), 'matched_refs': sorted(seen_refs)}
  wb.close()
  xlsx = out.with_suffix('.xlsx'); workbook.save(xlsx)
  # Human-readable edge CSV for easy inspection.
  csv_path = out.with_name(out.name + '_interaction_edges.csv')
  with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f); writer.writerow(['Play','Speaker A','Speaker B','Adjacent-turn count','Evidence'])
    for row in list(graph_ws.values)[1:]: writer.writerow(row)
  print('\nLITMIND PLAY DIAGNOSTIC COMPLETE')
  print('Original nlp_processor.py was not imported or modified.')
  print(f'Report: {xlsx.resolve()}')
  print(f'Interaction edges: {csv_path.resolve()}')
  for play, data in all_counts.items():
    print(f"{play}: reference speakers={len(data['refs'])}, utterance rows={data['utterances']}, PERSON mentions={data['ner_mentions']}, unique PERSON surfaces={data['unique_ner']}, reference names matched={len(data['matched_refs'])}")
  print('\nInterpretation: adjacent-turn edges show who spoke next to whom in the corpus sequence; they do NOT prove family/friend/enemy relationships.')

if __name__ == '__main__': main()
