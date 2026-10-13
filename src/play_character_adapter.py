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
import os
import re
import openpyxl
import spacy
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def classify_evidence_context(context):
  """Conservative quality screen; it does not assign a semantic relationship label."""
  folded = norm(context)
  if any(term in folded for term in [
    "copyright", "all rights reserved", "the title of the play",
    "a drama in two acts", "title of the play", "by chandan patel",
    "by mahesh dattani"
  ]):
    return "EXCLUDE_FROM_RELATION_CANDIDATES: title/byline/copyright metadata"
  stripped = clean(context)
  parenthetical_actions = re.findall(r"\(([^)]{1,120})\)", stripped)
  if (stripped.startswith("(") or parenthetical_actions) and any(
    re.search(r"\b" + re.escape(word) + r"\b", norm(part))
    for part in ([stripped] if stripped.startswith("(") else parenthetical_actions)
    for word in STAGE_ACTION_WORDS
  ):
    return "REVIEW_STAGE_DIRECTION: likely action/direction context"
  if not stripped:
    return "REVIEW: empty context"
  return "REVIEWABLE_NAME_MENTION: not a confirmed relationship"


def cue_categories_near_alias(context, alias):
  """Only report cue words close to the matched alias; still candidate evidence only."""
  folded = norm(context)
  alias_norm = norm(alias)
  match = re.search(r"(?<![a-z])" + r"\s+".join(re.escape(x) for x in alias_norm.split()) + r"(?![a-z])", folded)
  if not match:
    return []
  left = max(0, match.start() - 90)
  right = min(len(folded), match.end() + 90)
  local_context = folded[left:right]
  found = []
  for category, terms in RELATION_CUES.items():
    if any(re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", local_context) for term in terms):
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
          "cue_categories": cue_categories_near_alias(context, alias), "context": context[:350],
          "quality": classify_evidence_context(context)
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


def build_corpus_documented_relations(master_rows, master_headers):
  """Extract only explicit pairwise relations stated in CHARACTER_MASTER.Notes."""
  idx = {name: master_headers.index(name) for name in master_headers}
  out = []
  for row in master_rows[1:]:
    if not row:
      continue
    play = play_key(row[idx["Play_ID"]])
    source = clean(row[idx["Character_Name"]])
    note = clean(row[idx["Notes"]]) if "Notes" in idx else ""
    folded = norm(note)
    if play != "Dance Like a Man" or not source or not note:
      continue
    if source == "Ratna" and re.search(r"\bjairaj'?s wife\b", folded):
      out.append([play, "Jairaj", "Ratna", "SPOUSE", note, "CHARACTER_MASTER.Notes",
                  "Explicit relation stated in corpus character notes; direction normalized for display."])
    elif source == "Viswas" and re.search(r"\blata'?s fianc[eé]\b", folded):
      out.append([play, "Lata", "Viswas", "FIANCE", note, "CHARACTER_MASTER.Notes",
                  "Explicit relation stated in corpus character notes; direction normalized for display."])
    elif source == "Lata" and re.search(r"\bdaughter of jairaj and ratna\b", folded):
      out.append([play, "Jairaj", "Lata", "PARENT_CHILD", note, "CHARACTER_MASTER.Notes",
                  "Explicit relation stated in corpus character notes."])
      out.append([play, "Ratna", "Lata", "PARENT_CHILD", note, "CHARACTER_MASTER.Notes",
                  "Explicit relation stated in corpus character notes."])
    elif source == "Amritlal" and re.search(r"\bjairaj'?s father\b", folded):
      out.append([play, "Amritlal", "Jairaj", "PARENT_CHILD", note, "CHARACTER_MASTER.Notes",
                  "Explicit relation stated in corpus character notes."])
  return out


def add_tara_annotation_evidence(corpus_path, output_path):
  """Keep existing Tara utterance-level target/relation labels visible without miscalling them a graph."""
  src = openpyxl.load_workbook(corpus_path, read_only=True, data_only=True)
  out_rows = []
  if "TARA_FULL_ANNOTATION_V4" in src.sheetnames:
    data = list(src["TARA_FULL_ANNOTATION_V4"].iter_rows(values_only=True))
    headers = [clean(x) for x in data[0]]
    wanted = ["Speech_ID", "Speaker", "Primary_Text", "Masculinity_Target", "Secondary_Target",
              "Target_Relation", "Secondary_Target_Relation", "Target_Evidence_Type",
              "Source_QA_Flag", "Research_Memo"]
    ix = {name: headers.index(name) for name in wanted if name in headers}
    for row in data[1:]:
      rel = clean(row[ix["Target_Relation"]]) if "Target_Relation" in ix else ""
      rel2 = clean(row[ix["Secondary_Target_Relation"]]) if "Secondary_Target_Relation" in ix else ""
      if (rel and rel != "ND") or (rel2 and rel2 != "ND"):
        out_rows.append([
          clean(row[ix["Speech_ID"]]) if "Speech_ID" in ix else "",
          clean(row[ix["Speaker"]]) if "Speaker" in ix else "",
          clean(row[ix["Masculinity_Target"]]) if "Masculinity_Target" in ix else "",
          clean(row[ix["Secondary_Target"]]) if "Secondary_Target" in ix else "",
          rel, rel2,
          clean(row[ix["Target_Evidence_Type"]]) if "Target_Evidence_Type" in ix else "",
          clean(row[ix["Primary_Text"]])[:600] if "Primary_Text" in ix else "",
          clean(row[ix["Source_QA_Flag"]]) if "Source_QA_Flag" in ix else "",
          clean(row[ix["Research_Memo"]])[:600] if "Research_Memo" in ix else "",
          "Existing utterance-level target annotation; NOT a complete pairwise social-relation label."
        ])
  src.close()
  wb = openpyxl.load_workbook(output_path)
  if "Tara_Annotation_Evidence" in wb.sheetnames:
    del wb["Tara_Annotation_Evidence"]
  ws = wb.create_sheet("Tara_Annotation_Evidence")
  ws.append(["Speech_ID", "Speaker", "Masculinity_Target", "Secondary_Target", "Target_Relation",
             "Secondary_Target_Relation", "Target_Evidence_Type", "Annotated utterance",
             "Source QA flag", "Research memo", "Interpretation"])
  for row in out_rows:
    ws.append(row)
  ws.freeze_panes = "A2"
  ws.auto_filter.ref = ws.dimensions
  widths = [18, 18, 22, 22, 24, 28, 25, 70, 40, 70, 70]
  for i, width in enumerate(widths, 1):
    ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = width
  wb.save(output_path)



def build_semantic_relation_edges(corpus_path, output_path):
  """
  Add only corpus-documented Dance Like a Man relations and Tara's existing
  character-target annotations when both endpoints map to known characters.
  These remain evidence edges, not independently validated ground truth.
  """
  source_wb = openpyxl.load_workbook(corpus_path, read_only=True, data_only=True)
  master_data = list(source_wb["CHARACTER_MASTER"].iter_rows(values_only=True))
  master_headers = [clean(x) for x in master_data[0]]
  mi = {name: master_headers.index(name) for name in master_headers}
  references = defaultdict(dict)
  for row in master_data[1:]:
    if not row:
      continue
    play = play_key(row[mi["Play_ID"]])
    name = clean(row[mi["Character_Name"]])
    label = clean(row[mi["Source_Speaker_Label"]]).upper()
    if play and name:
      references[play][norm(name)] = name
      references[play][norm(label)] = name
      references[play][norm(name.replace("Dr ", ""))] = name

  rows_out = []
  for row in build_corpus_documented_relations(master_data, master_headers):
    play, a, b, relation, note, source_field, interpretation = row
    rows_out.append([play, a, b, relation, "CORPUS_DOCUMENTED_NOTE", 1, "",
                     note, "Documented in CHARACTER_MASTER.Notes; not independently gold-validated."])

  if "TARA_FULL_ANNOTATION_V4" in source_wb.sheetnames:
    data = list(source_wb["TARA_FULL_ANNOTATION_V4"].iter_rows(values_only=True))
    headers = [clean(x) for x in data[0]]
    ix = {name: headers.index(name) for name in headers}
    grouped = defaultdict(lambda: {"ids": [], "contexts": []})
    for row in data[1:]:
      play = play_key(row[ix["Play_ID"]], row[ix["Play"]]) if "Play_ID" in ix else "Tara"
      if play != "Tara":
        continue
      speaker = references[play].get(norm(row[ix["Speaker"]])) if "Speaker" in ix else None
      if not speaker:
        continue
      for target_col, rel_col in [
        ("Masculinity_Target", "Target_Relation"),
        ("Secondary_Target", "Secondary_Target_Relation")
      ]:
        target_raw = clean(row[ix[target_col]]) if target_col in ix else ""
        relation = clean(row[ix[rel_col]]).upper() if rel_col in ix else ""
        # Multiple targets are only accepted when each exact token maps to a
        # reference character; generic categories remain annotations, not nodes.
        if not target_raw or relation in {"", "ND", "SELF"}:
          continue
        target_parts = [x.strip() for x in target_raw.split(";") if x.strip()]
        targets = [references[play].get(norm(x)) for x in target_parts]
        if not targets or any(x is None for x in targets):
          continue
        for target in targets:
          if target == speaker:
            continue
          key = (play, speaker, target, relation, "TARA_UTTERANCE_ANNOTATION")
          grouped[key]["ids"].append(clean(row[ix["Speech_ID"]]) if "Speech_ID" in ix else "")
          grouped[key]["contexts"].append(clean(row[ix["Primary_Text"]])[:220] if "Primary_Text" in ix else "")
    for (play, source, target, relation, evidence_type), info in sorted(grouped.items()):
      ids = list(dict.fromkeys(x for x in info["ids"] if x))
      rows_out.append([play, source, target, relation, evidence_type, len(ids),
                       "; ".join(ids[:10]), " || ".join(info["contexts"][:2]),
                       "Existing utterance-level annotation mapped to known character endpoints; review source QA and context."])

  source_wb.close()
  wb = openpyxl.load_workbook(output_path)
  for sheet_name in ["Semantic_Relation_Edges", "Graph_Export_Index"]:
    if sheet_name in wb.sheetnames:
      del wb[sheet_name]
  ws = wb.create_sheet("Semantic_Relation_Edges")
  wb.create_sheet("Graph_Export_Index")
  ws.append(["Play", "Source character", "Target character", "Annotated/documented relation",
             "Evidence type", "Evidence count", "Example Speech_IDs", "Source evidence/context",
             "Interpretation"])
  for row in rows_out:
    ws.append(row)
  ws.freeze_panes = "A2"
  ws.auto_filter.ref = ws.dimensions
  for col, width in enumerate([22, 22, 22, 28, 34, 15, 40, 75, 75], 1):
    ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = width
  wb.save(output_path)
  return rows_out


def _draw_network(graph, output_path, title, edge_labels=None, directed=False):
  """Save a readable static graph; never infer missing semantic edges."""
  fig, ax = plt.subplots(figsize=(10, 7))
  ax.set_title(title, fontsize=13, pad=18)
  ax.axis("off")
  if graph.number_of_nodes() == 0:
    ax.text(0.5, 0.5, "No supported graph edges available", ha="center", va="center")
  else:
    pos = nx.spring_layout(graph, seed=19, k=1.25)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=1700, alpha=0.9)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    widths = []
    for _, _, data in graph.edges(data=True):
      widths.append(1.0 + min(float(data.get("weight", 1)), 8) * 0.35)
    edge_kwargs = {"width": widths, "alpha": 0.65}
    if directed:
      edge_kwargs.update({"arrows": True, "arrowsize": 16, "connectionstyle": "arc3,rad=0.06"})
    nx.draw_networkx_edges(graph, pos, ax=ax, **edge_kwargs)
    if edge_labels:
      nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=7)
  fig.tight_layout()
  fig.savefig(output_path, dpi=180, bbox_inches="tight")
  plt.close(fig)


def export_play_graphs(output_path, graph_dir, semantic_rows):
  """Export separate interaction and semantic-evidence graphs for both plays."""
  wb = openpyxl.load_workbook(output_path, read_only=True, data_only=True)
  nodes = list(wb["Interaction_Graph_Nodes"].iter_rows(min_row=2, values_only=True))
  interaction_edges = list(wb["Turn_Transition_Edges"].iter_rows(min_row=2, values_only=True))
  graph_dir.mkdir(parents=True, exist_ok=True)
  index_rows = []
  for play in ["Tara", "Dance Like a Man"]:
    ig = nx.Graph()
    for row in nodes:
      if row[0] == play:
        ig.add_node(clean(row[1]))
    for row in interaction_edges:
      if row[0] == play:
        a, b, weight = clean(row[1]), clean(row[2]), int(row[3] or 0)
        ig.add_edge(a, b, weight=weight)
    interaction_path = graph_dir / (("tara" if play == "Tara" else "dance_like_a_man") + "_interaction_graph.png")
    _draw_network(ig, interaction_path,
                  f"{play} — adjacent speaker-turn interaction\n(Not a semantic relationship graph)")
    index_rows.append([play, "Interaction graph", str(interaction_path.relative_to(output_path.parent)),
                       ig.number_of_nodes(), ig.number_of_edges(),
                       "Edges count consecutive annotated speaker turns within the same Act."])

    # Semantic edges are sourced only from explicit corpus notes or existing
    # Tara annotation labels; these two evidence types remain visibly distinct.
    sg = nx.Graph() if play == "Tara" else nx.DiGraph()
    relation_labels = {}
    relevant = [r for r in semantic_rows if r[0] == play]
    if play == "Tara":
      relevant = [r for r in relevant if r[4] == "TARA_UTTERANCE_ANNOTATION"]
    else:
      relevant = [r for r in relevant if r[4] == "CORPUS_DOCUMENTED_NOTE"]
    for row in relevant:
      _, a, b, relation, evidence_type, count, speech_ids, context, interpretation = row
      sg.add_edge(a, b, weight=int(count or 1))
      label = f"{relation} (n={count})"
      if (a, b) in relation_labels:
        relation_labels[(a, b)] += "\n" + label
      else:
        relation_labels[(a, b)] = label
    rel_path = graph_dir / (("tara" if play == "Tara" else "dance_like_a_man") + "_relationship_evidence_graph.png")
    subtitle = ("Existing utterance annotations; not independently verified" if play == "Tara"
                else "Explicit CHARACTER_MASTER notes; not a complete graph")
    _draw_network(sg, rel_path, f"{play} — semantic relationship evidence\n({subtitle})",
                  edge_labels=relation_labels, directed=(play != "Tara"))
    index_rows.append([play, "Relationship evidence graph", str(rel_path.relative_to(output_path.parent)),
                       sg.number_of_nodes(), sg.number_of_edges(), subtitle])

  wb.close()
  out_wb = openpyxl.load_workbook(output_path)
  idx = out_wb["Graph_Export_Index"]
  idx.append(["Play", "Graph type", "PNG file", "Nodes", "Edges", "Interpretation"])
  for row in index_rows:
    idx.append(row)
  idx.freeze_panes = "A2"
  idx.auto_filter.ref = idx.dimensions
  for col, width in enumerate([22, 28, 90, 12, 12, 70], 1):
    idx.column_dimensions[openpyxl.utils.get_column_letter(col)].width = width
  out_wb.save(output_path)
  return index_rows


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

  try:
    nlp = spacy.load(os.environ.get("LITMIND_SPACY_MODEL", "en_core_web_sm"))
    ner_available = True
  except OSError:
    # Keep the corpus-annotated speaker, alias, interaction, and documented-relation
    # outputs usable even if the optional spaCy model is not installed.
    nlp = None
    ner_available = False
    print("WARNING: spaCy model en_core_web_sm not found; NER surface counts will be skipped.")
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
                             "Speech_ID", "Source QA flag", "Lexical cue categories near alias (candidate only)", "Context",
                             "Evidence status / quality screen"])
  relation_summary_ws = result.create_sheet("Relationship_Candidate_Summary")
  relation_summary_ws.append(["Play", "Character A", "Character B", "Reviewable name-mention utterances",
                              "Cue categories near alias (candidate only)", "Example Speech_IDs",
                              "Excluded metadata/stage-direction mentions", "Interpretation"])
  excluded_ws = result.create_sheet("Excluded_Low_Quality_Evidence")
  excluded_ws.append(["Play", "Speaking character", "Mentioned character", "Matched alias",
                      "Speech_ID", "Quality flag", "Context"])
  notes_ws = result.create_sheet("Methodology_Notes")
  notes_ws.append(["Topic", "Decision / limitation"])
  notes_ws.append(["Character reference", "CHARACTER_MASTER plus UTTERANCE_CORPUS annotated Speaker column are the reference for the two supplied plays; this is not an automatic speaker-diarization test."])
  notes_ws.append(["NER", "spaCy PERSON detections are candidate mentions. Missing or incorrect detections remain possible; literal alias hits are separately listed for review. If en_core_web_sm is unavailable, NER counts are skipped while corpus-annotated speaker and evidence outputs are still generated."])
  notes_ws.append(["Interaction graph", "Turn_Transition_Edges counts consecutive utterance rows with different known speakers within the same Act. This is a dialogue-turn transition graph, not a semantic relationship graph."])
  notes_ws.append(["Relationship candidates", "Speaker_Target_Evidence links an annotated speaker to another reference character named in that utterance. Cue words are reported only near the matched alias; they remain lexical hints, not confirmed relation labels."])
  notes_ws.append(["Evidence quality screen", "Title/byline/copyright contexts are excluded from Relationship_Candidate_Summary. Likely stage-direction contexts are moved to Excluded_Low_Quality_Evidence for review. This conservative heuristic can still miss or misclassify contexts; it is not a human gold-standard annotation."])
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
    excluded_counts = Counter()
    for row in rows:
      for item in target_mentions_in_turn(play, row, ref_names, label_to_character):
        cues = item["cue_categories"]
        quality = item["quality"]
        target_evidence_ws.append([
          play, item["source"], item["target"], item["alias"], item["speech_id"], row["qa_flag"],
          "; ".join(cues), item["context"], quality
        ])
        key = tuple(sorted([item["source"], item["target"]], key=str.casefold))
        if quality.startswith("REVIEWABLE_NAME_MENTION"):
          pair_evidence[key].append(item)
        else:
          excluded_counts[key] += 1
          excluded_ws.append([play, item["source"], item["target"], item["alias"],
                              item["speech_id"], quality, item["context"]])
    for (a, b), items in sorted(pair_evidence.items()):
      cue_union = sorted({cue for item in items for cue in item["cue_categories"]})
      ids = list(dict.fromkeys(item["speech_id"] for item in items))[:8]
      relation_summary_ws.append([
        play, a, b, len(items), "; ".join(cue_union), "; ".join(ids), excluded_counts[(a, b)],
        "Reviewable name-mention evidence only; NOT a confirmed relation. Manually verify semantic meaning."
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
    if nlp is not None:
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

  documented_ws = result.create_sheet("Corpus_Documented_Relations")
  documented_ws.append(["Play", "Character A", "Character B", "Relation", "Source note", "Source field", "Interpretation"])
  for relation_row in build_corpus_documented_relations(master_rows, master_headers):
    documented_ws.append(relation_row)
  documented_ws.append(["Tara", "", "", "NOT A COMPLETE PAIRWISE GRAPH",
                       "Tara has utterance-level target/relation annotations, not a complete character-pair relation inventory.",
                       "TARA_FULL_ANNOTATION_V4",
                       "See Tara_Annotation_Evidence for existing annotations; do not treat them as a complete social graph."])
  source.close()

  for ws in result.worksheets:
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for column_cells in ws.columns:
      letter = column_cells[0].column_letter
      max_len = max((len(str(cell.value or "")) for cell in column_cells[:200]), default=10)
      ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 55)

  xlsx = out.with_suffix(".xlsx")
  result.save(xlsx)
  add_tara_annotation_evidence(corpus, xlsx)
  semantic_rows = build_semantic_relation_edges(corpus, xlsx)
  graph_dir = out.parent / (out.stem + "_graphs")
  graph_index = export_play_graphs(xlsx, graph_dir, semantic_rows)
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
  print("  Speaker_Target_Evidence + Relationship_Candidate_Summary: quality-screened name-mention candidates only.")
  print("  Excluded_Low_Quality_Evidence: title/byline/copyright or likely stage-direction contexts removed from candidate summaries.")
  print("  Alias_Mention_Candidates: literal alias hits with Speech_ID and context.")
  print("  Corpus_Documented_Relations: pairwise relations explicitly stated in CHARACTER_MASTER notes.")
  print("  Tara_Annotation_Evidence: existing utterance-level annotations, not a complete social-relation graph.")
  print("\nCHARACTERS")
  for play in ["Tara", "Dance Like a Man"]:
    names = [name for (p, name) in characters if p == play]
    print(f"  {play}: {', '.join(names)}")
  print("\nGRAPHS")
  for row in graph_index:
    print(f"  {row[0]} | {row[1]}: {row[3]} nodes, {row[4]} edges -> {row[2]}")
  print("\nRELATIONSHIP EVIDENCE")
  for play in ["Tara", "Dance Like a Man"]:
    subset = [r for r in semantic_rows if r[0] == play]
    print(f"  {play}: {len(subset)} evidence edges (annotation/documented evidence; not automatically confirmed truth)")
    for r in subset[:12]:
      print(f"    {r[1]} --[{r[3]}]--> {r[2]} (source={r[4]}, evidence_count={r[5]})")
    if len(subset) > 12:
      print(f"    ... {len(subset)-12} more evidence edges in Semantic_Relation_Edges sheet")
  print("\nImportant: Speaker_Reference uses the corpus's annotated Speaker column. "
        "Interaction edges are adjacent turns, not semantic relations. Semantic edges use existing annotation/documented evidence only.")

if __name__ == "__main__":
  main()
