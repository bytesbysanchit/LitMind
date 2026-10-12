"""
LitMind play-specific character adapter (separate from frozen E3).
Uses the annotated CHARACTER_MASTER and UTTERANCE_CORPUS as the reference for speaker identity.
Does not import or modify nlp_processor.py or prev.py.

Run from D:\\LitMind:
  python src/play_character_adapter.py --corpus Dattani_Tara_Corpus_Final.xlsx --out reports/play_character_adapter
Requires: openpyxl, spacy, en_core_web_sm
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import re
import openpyxl
import spacy

PLAY_NAMES = {"D01": "Tara", "D02": "Dance Like a Man"}
STAGE_ACTION_WORDS = {
  "listens", "laughs", "smiles", "mutters", "yawns", "grins", "sighs",
  "weeps", "cries", "pauses", "moves", "turns", "looks", "sits", "stands"
}
# Conservative aliases: only map variants supported by the corpus's character names/notes
# or previously inspected NER surfaces. These are candidate mappings for review, not truth labels.
ALIASES = {
  "dan": "Dan",
  "chandan": "Chandan",
  "chandu": "Chandan",
  "tara": "Tara",
  "bharati": "Bharati",
  "mrs bharati patel": "Bharati",
  "patel": "Patel",
  "roopa": "Roopa",
  "dr thakkar": "Dr Thakkar",
  "thakkar": "Dr Thakkar",
  "jairaj": "Jairaj",
  "jai": "Jairaj",
  "jairaj parekh": "Jairaj",
  "ratna": "Ratna",
  "ratna devi": "Ratna",
  "viswas": "Viswas",
  "vishy": "Viswas",
  "lata": "Lata",
  "lata parekh": "Lata",
  "amritlal": "Amritlal",
  "amritlal parekh": "Amritlal",
}

def clean(s):
  return re.sub(r"\s+", " ", str(s or "")).strip()

def norm(s):
  s = clean(s).replace("’", "'").casefold()
  s = re.sub(r"[^a-zà-ÿ.' -]", "", s)
  return re.sub(r"\s+", " ", s).strip(" .'-")

def play_key(play_id, play_name=""):
  pid = clean(play_id).upper()
  if pid in PLAY_NAMES:
    return PLAY_NAMES[pid]
  text = (pid + " " + clean(play_name)).casefold()
  if "tara" in text:
    return "Tara"
  if "dance" in text or "man" in text:
    return "Dance Like a Man"
  return None


# Explicit variants are candidate strings to inspect, not proof that a mention refers to a character.
ALIAS_VARIANTS = {
  "Dan": ["Dan"],
  "Chandan": ["Chandan", "Chandu"],
  "Tara": ["Tara"],
  "Bharati": ["Bharati", "Mrs Bharati Patel"],
  "Patel": ["Patel", "Mr Patel"],
  "Roopa": ["Roopa", "Rupa"],
  "Dr Thakkar": ["Dr Thakkar", "Thakkar", "Doctor Thakkar"],
  "Jairaj": ["Jairaj", "Jai", "Jairaj Parekh"],
  "Ratna": ["Ratna", "Ratna Devi"],
  "Viswas": ["Viswas", "Vishy"],
  "Lata": ["Lata", "Lata Parekh"],
  "Amritlal": ["Amritlal", "Amritlal Parekh"],
}

def literal_alias_hits(text, variants, limit=3):
  """Search actual dialogue text for literal variants; return counts and nearby sentence-like snippets."""
  total = 0
  examples = []
  for variant in variants:
    pattern = r"(?<![A-Za-z])" + r"\s+".join(re.escape(part) for part in variant.split()) + r"(?![A-Za-z])"
    hits = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    total += len(hits)
    for hit in hits:
      if len(examples) >= limit:
        break
      # Use punctuation-based local context instead of whole line, because one utterance can be long.
      left_candidates = [text.rfind(mark, 0, hit.start()) for mark in ".?!\n"]
      start = max(left_candidates) + 1
      right_candidates = [pos for mark in ".?!\n" if (pos := text.find(mark, hit.end())) >= 0]
      end = min(right_candidates) + 1 if right_candidates else min(len(text), hit.end()+180)
      snippet = re.sub(r"\s+", " ", text[start:end]).strip()
      examples.append((variant, snippet[:350]))
  return total, examples


def find_alias_mentions(play, rows, ref_names):
  """Find literal reference-name/alias candidates in utterance text for manual review."""
  found = []
  # Prefer longer aliases first so a full name is not double-counted as a surname/name fragment.
  candidates = []
  for canonical in ref_names:
    for variant in ALIAS_VARIANTS.get(canonical, [canonical]):
      candidates.append((variant, canonical))
  candidates.sort(key=lambda item: len(item[0]), reverse=True)

  for row in rows:
    text = row["text"]
    occupied = []
    for alias, canonical in candidates:
      pattern = r"(?<![A-Za-z])" + r"\s+".join(re.escape(part) for part in alias.split()) + r"(?![A-Za-z])"
      for hit in re.finditer(pattern, text, flags=re.IGNORECASE):
        if any(hit.start() < end and hit.end() > start for start, end in occupied):
          continue
        occupied.append((hit.start(), hit.end()))
        left = max(text.rfind(mark, 0, hit.start()) for mark in ".?!\n") + 1
        right_candidates = [pos for mark in ".?!\n" if (pos := text.find(mark, hit.end())) >= 0]
        end = min(right_candidates) + 1 if right_candidates else min(len(text), hit.end() + 180)
        context = re.sub(r"\s+", " ", text[left:end]).strip()
        found.append([
          play, canonical, alias, row["speech_id"], row["speaker_label"], row.get("qa_flag", ""),
          context[:350], "Literal text match; manually verify referent/context"
        ])
  return found



RELATION_CUES = {
  "FAMILY": [
    "father", "mother", "brother", "sister", "son", "daughter", "husband",
    "wife", "uncle", "aunt", "cousin", "parent", "child", "children",
    "stepfather", "step-father", "stepmother", "step-mother", "grandfather",
    "grandmother", "grandson", "granddaughter"
  ],
  "FRIEND_COMPANION": ["friend", "friends", "companion", "buddy"],
  "ROMANTIC_LOVE": ["love", "loves", "lover", "beloved", "sweetheart", "married", "marriage", "affair"],
  "CONFLICT": ["enemy", "enemies", "hate", "hates", "hated", "fight", "fighting", "betray", "betrayed", "blame", "kill"],
  "PROFESSIONAL": ["doctor", "patient", "boss", "colleague", "employee", "employer", "teacher", "student"]
}

def cue_categories(text):
  """Return lexical cue categories found in an utterance; candidate evidence only."""
  folded = norm(text)
  found = []
  for category, terms in RELATION_CUES.items():
    if any(re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", folded) for term in terms):
      found.append(category)
  return found


def target_mentions_in_turn(play, row, ref_names, label_to_character):
  """Create speaker-to-mentioned-character candidates from one annotated dialogue turn."""
  speaker = label_to_character.get((play, row["speaker_label"]))
  if not speaker:
    return []
  text = row["text"]
  candidates = []
  for canonical in ref_names:
    if canonical == speaker:
      continue
    for alias in ALIAS_VARIANTS.get(canonical, [canonical]):
      pattern = r"(?<![A-Za-z])" + r"\s+".join(re.escape(part) for part in alias.split()) + r"(?![A-Za-z])"
      for hit in re.finditer(pattern, text, flags=re.IGNORECASE):
        left = max(text.rfind(mark, 0, hit.start()) for mark in ".?!\n") + 1
        right_candidates = [pos for mark in ".?!\n" if (pos := text.find(mark, hit.end())) >= 0]
        end = min(right_candidates) + 1 if right_candidates else min(len(text), hit.end() + 180)
        context = re.sub(r"\s+", " ", text[left:end]).strip()
        candidates.append({
          "play": play, "source": speaker, "target": canonical, "alias": alias,
          "speech_id": row["speech_id"], "speaker_label": row["speaker_label"],
          "qa_flag": row.get("qa_flag", ""),
          "cue_categories": cue_categories(context), "context": context[:350]
        })
        break
  # One source-target pair per utterance; prefer the longest alias if several matched.
  unique = {}
  for item in candidates:
    key = (item["source"], item["target"])
    previous = unique.get(key)
    if previous is None or len(item["alias"]) > len(previous["alias"]):
      unique[key] = item
  return list(unique.values())

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--corpus", required=True)
  ap.add_argument("--out", default="reports/play_character_adapter")
  args = ap.parse_args()

  corpus = Path(args.corpus)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  if not corpus.exists():
    raise SystemExit(f"Corpus workbook not found: {corpus}")

  nlp = spacy.load("en_core_web_sm")
  source = openpyxl.load_workbook(corpus, read_only=True, data_only=True)
  if "CHARACTER_MASTER" not in source.sheetnames or "UTTERANCE_CORPUS" not in source.sheetnames:
    raise SystemExit("Workbook must contain CHARACTER_MASTER and UTTERANCE_CORPUS sheets.")

  # Read reference characters and their annotated speaker labels.
  master_ws = source["CHARACTER_MASTER"]
  master_rows = list(master_ws.iter_rows(values_only=True))
  master_headers = [clean(x) for x in master_rows[0]]
  required_master = ["Character_ID", "Play_ID", "Character_Name", "Source_Speaker_Label"]
  if any(x not in master_headers for x in required_master):
    raise SystemExit(f"Unexpected CHARACTER_MASTER headers: {master_headers}")
  mi = {name: master_headers.index(name) for name in required_master}
  characters = {}
  label_to_character = {}
  for row in master_rows[1:]:
    if not row:
      continue
    play = play_key(row[mi["Play_ID"]])
    name = clean(row[mi["Character_Name"]])
    label = clean(row[mi["Source_Speaker_Label"]]).upper()
    cid = clean(row[mi["Character_ID"]])
    if play and name:
      characters[(play, name)] = {"id": cid, "label": label}
      label_to_character[(play, label)] = name

  # Read annotated dialogue turns.
  utter_ws = source["UTTERANCE_CORPUS"]
  utter_rows = list(utter_ws.iter_rows(values_only=True))
  utter_headers = [clean(x) for x in utter_rows[0]]
  required_utter = ["Speech_ID", "Play_ID", "Play", "Act", "Speaker", "Speech_OCR", "QA_Flag"]
  if any(x not in utter_headers for x in required_utter):
    raise SystemExit(f"Unexpected UTTERANCE_CORPUS headers: {utter_headers}")
  ui = {name: utter_headers.index(name) for name in required_utter}

  turns = defaultdict(list)
  for row in utter_rows[1:]:
    if not row:
      continue
    play = play_key(row[ui["Play_ID"]], row[ui["Play"]])
    if play:
      turns[play].append({
        "speech_id": clean(row[ui["Speech_ID"]]),
        "act": clean(row[ui["Act"]]),
        "speaker_label": clean(row[ui["Speaker"]]).upper(),
        "text": clean(row[ui["Speech_OCR"]]),
        "qa_flag": clean(row[ui["QA_Flag"]]),
      })
  source.close()

  result = openpyxl.Workbook()
  summary = result.active
  summary.title = "Summary"
  summary.append(["Play", "Reference characters", "Utterance rows", "Rows with known speaker label",
                  "Unique annotated speakers", "PERSON mentions", "Unique PERSON surfaces",
                  "PERSON surfaces mapped to reference", "Distinct reference characters mapped",
                  "Action-like PERSON surfaces flagged"])
  speakers_ws = result.create_sheet("Speaker_Reference")
  speakers_ws.append(["Play", "Character_ID", "Reference character", "Source speaker label",
                      "Utterance rows with this speaker label"])
  ner_ws = result.create_sheet("NER_Candidates")
  ner_ws.append(["Play", "PERSON surface", "Mentions", "Mapped canonical character",
                 "Review flag", "Example context"])
  unmapped_ws = result.create_sheet("Unmapped_Speaker_Labels")
  unmapped_ws.append(["Play", "Speaker label in utterance corpus", "Utterance count", "Example Speech_ID"])
  alias_ws = result.create_sheet("Alias_Coverage")
  alias_ws.append(["Play", "Canonical reference character", "Candidate aliases searched",
                   "Literal occurrences in dialogue text", "Example dialogue context(s)",
                   "PERSON NER surfaces mapped", "Interpretation"])
  alias_mentions_ws = result.create_sheet("Alias_Mention_Candidates")
  alias_mentions_ws.append(["Play", "Candidate character", "Matched alias", "Speech_ID",
                            "Annotated speaker", "Source QA flag", "Context", "Evidence status"])
  nodes_ws = result.create_sheet("Interaction_Graph_Nodes")
  nodes_ws.append(["Play", "Character", "Annotated speaker label", "Utterance turns",
                   "Unique adjacent-speaker partners", "Weighted transition degree",
                   "Interpretation"])
  edges_ws = result.create_sheet("Turn_Transition_Edges")
  edges_ws.append(["Play", "Character A", "Character B", "Adjacent turn transitions",
                   "A then B", "B then A", "Example Speech_ID transitions",
                   "Interpretation"])
  target_evidence_ws = result.create_sheet("Speaker_Target_Evidence")
  target_evidence_ws.append(["Play", "Speaking character", "Mentioned character", "Matched alias",
                             "Speech_ID", "Source QA flag", "Lexical cue categories (candidate only)", "Context",
                             "Evidence status"])
  relation_summary_ws = result.create_sheet("Relationship_Candidate_Summary")
  relation_summary_ws.append(["Play", "Character A", "Character B", "Evidence utterances",
                              "Cue categories observed", "Example Speech_IDs", "Interpretation"])
  notes_ws = result.create_sheet("Methodology_Notes")
  notes_ws.append(["Topic", "Decision / limitation"])
  notes_ws.append(["Character reference", "CHARACTER_MASTER plus UTTERANCE_CORPUS annotated Speaker column are the reference for the two supplied plays; this is not an automatic speaker-diarization test."])
  notes_ws.append(["NER", "spaCy PERSON detections are candidate mentions. Missing or incorrect detections remain possible; literal alias hits are separately listed for review."])
  notes_ws.append(["Interaction graph", "Turn_Transition_Edges counts consecutive utterance rows with different known speakers within the same Act. This is a dialogue-turn transition graph, not a semantic relationship graph."])
  notes_ws.append(["Relationship candidates", "Speaker_Target_Evidence links an annotated speaker to another reference character named in that utterance. Cue categories are lexical hints only and require context review; they are not confirmed relation labels."])
  notes_ws.append(["Corpus quality", "Each evidence row retains the source QA_Flag. OCR and stage directions may still be mixed into Speech_OCR; flagged material should be reviewed before evaluation."])
  notes_ws.append(["Scope", "The adapter is currently tested against Tara and Dance Like a Man. Support for arbitrary plays is not established by these two cases alone."])
  notes_ws.append(["Record-count discrepancy", "The workbook's CORPUS_SUMMARY reports 993 Dance Like a Man turns, while UTTERANCE_CORPUS contains 996 rows for that play. This adapter analyzes the 996 UTTERANCE_CORPUS rows and does not silently reconcile the discrepancy."])
  notes_ws.append(["Accuracy claims", "Do not report overall precision/recall/F1 from these diagnostic outputs alone. Relationship metrics require compatible, independently verified pairwise labels."])

  for play in ["Tara", "Dance Like a Man"]:
    ref_names = [name for (p, name) in characters if p == play]
    rows = turns[play]
    play_text = "\n".join(row["text"] for row in rows)
    speaker_counts = Counter(row["speaker_label"] for row in rows if row["speaker_label"])
    known_rows = sum(1 for row in rows if (play, row["speaker_label"]) in label_to_character)

    # Build a conservative turn-transition graph from the annotated speaker sequence.
    # These edges represent adjacent dialogue turns only, never semantic relationships.
    transition_counts = Counter()
    transition_examples = defaultdict(list)
    for previous, current in zip(rows, rows[1:]):
      if previous["act"] != current["act"]:
        continue
      source_name = label_to_character.get((play, previous["speaker_label"]))
      target_name = label_to_character.get((play, current["speaker_label"]))
      if not source_name or not target_name or source_name == target_name:
        continue
      a, b = sorted([source_name, target_name], key=str.casefold)
      transition_counts[(a, b)] += 1
      direction = "A_then_B" if (source_name, target_name) == (a, b) else "B_then_A"
      transition_counts[(a, b, direction)] += 1
      if len(transition_examples[(a, b)]) < 5:
        transition_examples[(a, b)].append(f'{previous["speech_id"]} → {current["speech_id"]}')

    partners = defaultdict(set)
    weighted_degree = Counter()
    for (a, b), count in [(key, value) for key, value in transition_counts.items() if len(key) == 2]:
      partners[a].add(b)
      partners[b].add(a)
      weighted_degree[a] += count
      weighted_degree[b] += count
      a_then_b = transition_counts[(a, b, "A_then_B")]
      b_then_a = transition_counts[(a, b, "B_then_A")]
      edges_ws.append([play, a, b, count, a_then_b, b_then_a,
                       "; ".join(transition_examples[(a, b)]),
                       "Adjacent annotated speaker turns only; not proof of a relationship."])

    for name in ref_names:
      info = characters[(play, name)]
      nodes_ws.append([play, name, info["label"], speaker_counts.get(info["label"], 0),
                       len(partners[name]), weighted_degree[name],
                       "All reference speakers included; edge degree counts adjacent turn transitions, not relationships."])

    # Mention evidence: the annotated speaker says a known character's name/alias.
    # Cue categories are weak lexical candidates and must not be treated as gold labels.
    pair_evidence = defaultdict(list)
    for row in rows:
      for item in target_mentions_in_turn(play, row, ref_names, label_to_character):
        cues = item["cue_categories"]
        target_evidence_ws.append([
          play, item["source"], item["target"], item["alias"], item["speech_id"], row["qa_flag"],
          "; ".join(cues), item["context"],
          "Name mention by annotated speaker; cue tags are lexical candidates only, not confirmed relationships."
        ])
        key = tuple(sorted([item["source"], item["target"]], key=str.casefold))
        pair_evidence[key].append(item)
    for (a, b), items in sorted(pair_evidence.items()):
      cue_union = sorted({cue for item in items for cue in item["cue_categories"]})
      ids = list(dict.fromkeys(item["speech_id"] for item in items))[:8]
      relation_summary_ws.append([
        play, a, b, len(items), "; ".join(cue_union), "; ".join(ids),
        "Candidate evidence from name-mention utterances; review contexts before assigning any relationship label."
      ])

    for name in ref_names:
      info = characters[(play, name)]
      speakers_ws.append([play, info["id"], name, info["label"], speaker_counts.get(info["label"], 0)])

    for label, count in speaker_counts.items():
      if (play, label) not in label_to_character:
        example = next((r["speech_id"] for r in rows if r["speaker_label"] == label), "")
        unmapped_ws.append([play, label, count, example])

    surfaces = Counter()
    examples = {}
    # NER is used for name-mention candidates only; the annotated Speaker field is the
    # primary source for who is speaking in a given utterance.
    for row in rows:
      doc = nlp(row["text"])
      for ent in doc.ents:
        if ent.label_ != "PERSON":
          continue
        surface = clean(ent.text).strip(" .,'’")
        if not surface:
          continue
        surfaces[surface] += 1
        examples.setdefault(surface, ent.sent.text[:300])

    mapped_surfaces = {}
    for surface in surfaces:
      key = norm(surface)
      mapped = ALIASES.get(key)
      if not mapped:
        # Safe exact-name fallback, case-insensitive.
        for ref in ref_names:
          if key == norm(ref):
            mapped = ref
            break
      if mapped not in ref_names:
        mapped = None
      mapped_surfaces[surface] = mapped

    # Compare literal alias hits in Speech_OCR with NER mapping, keeping the two signals separate.
    for canonical in ref_names:
      variants = ALIAS_VARIANTS.get(canonical, [canonical])
      occurrence_count, occurrence_examples = literal_alias_hits(play_text, variants)
      ner_hits = [surface for surface, mapped_name in mapped_surfaces.items() if mapped_name == canonical]
      example_text = " || ".join(f"{variant}: {context}" for variant, context in occurrence_examples)
      if occurrence_count and not ner_hits:
        interpretation = "Name/alias occurs in Speech_OCR but no PERSON surface maps to it; inspect NER miss/context."
      elif not occurrence_count and not ner_hits:
        interpretation = "No literal alias hit and no PERSON mapping; verify text/alias coverage."
      else:
        interpretation = "Text hits and/or NER mappings found; inspect contexts before treating hits as character mentions."
      alias_ws.append([play, canonical, "; ".join(variants), occurrence_count, example_text,
                       "; ".join(ner_hits), interpretation])

    for mention_row in find_alias_mentions(play, rows, ref_names):
      alias_mentions_ws.append(mention_row)

    distinct_mapped = {name for name in mapped_surfaces.values() if name}
    action_flag_count = 0
    for surface, count in surfaces.most_common():
      action_like = norm(surface) in STAGE_ACTION_WORDS
      if action_like:
        action_flag_count += 1
      flag = "REVIEW: action-like PERSON surface" if action_like else (
        "MAPPED alias/canonical candidate" if mapped_surfaces[surface] else "REVIEW: unmapped PERSON surface"
      )
      ner_ws.append([play, surface, count, mapped_surfaces[surface] or "", flag, examples.get(surface, "")])

    summary.append([
      play, len(ref_names), len(rows), known_rows, len(speaker_counts),
      sum(surfaces.values()), len(surfaces),
      sum(1 for value in mapped_surfaces.values() if value),
      len(distinct_mapped), action_flag_count
    ])

  for ws in result.worksheets:
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for column_cells in ws.columns:
      letter = column_cells[0].column_letter
      max_len = max((len(str(cell.value or "")) for cell in column_cells[:200]), default=10)
      ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 55)

  xlsx = out.with_suffix(".xlsx")
  result.save(xlsx)
  print("\nLITMIND PLAY CHARACTER ADAPTER COMPLETE")
  print("Frozen nlp_processor.py and prev.py were not imported or modified.")
  print(f"Report: {xlsx.resolve()}")
  for row in list(summary.values)[1:]:
    print(f"{row[0]}: reference characters={row[1]}, utterances={row[2]}, known-speaker rows={row[3]}, "
          f"PERSON mentions={row[5]}, unique PERSON surfaces={row[6]}, "
          f"mapped PERSON surfaces={row[7]}, distinct mapped characters={row[8]}")
  for play in ["Tara", "Dance Like a Man"]:
    edge_count = sum(1 for row in edges_ws.iter_rows(min_row=2, values_only=True) if row[0] == play)
    candidate_count = sum(1 for row in relation_summary_ws.iter_rows(min_row=2, values_only=True) if row[0] == play)
    print(f"{play}: turn-transition edges={edge_count}, relationship candidate pairs for review={candidate_count}")
  print("\nAdded graph/evidence sheets:")
  print("  Interaction_Graph_Nodes + Turn_Transition_Edges: adjacent annotated speaker turns only.")
  print("  Speaker_Target_Evidence + Relationship_Candidate_Summary: name-mention and lexical-cue candidates only.")
  print("  Alias_Mention_Candidates: literal alias hits with Speech_ID and context.")
  print("\nImportant: Speaker_Reference uses the corpus's annotated Speaker column. "
        "NER_Candidates are mentions to review, not proof of relationships; no candidate sheet is a gold-standard relation graph.")

if __name__ == "__main__":
  main()
