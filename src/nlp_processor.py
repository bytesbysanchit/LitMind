from pathlib import Path
from collections import Counter
from itertools import combinations

import spacy
import networkx as nx
import matplotlib.pyplot as plt


# ============================================
# 1. LOAD MODEL AND TEXT
# ============================================

nlp = spacy.load("en_core_web_sm")

text_path = Path("data/processed/Sherlock_Holmes.txt")

with open(text_path, "r", encoding="utf-8") as file:
  text = file.read()

doc = nlp(text)


# ============================================
# 2. EXTRACT PERSON ENTITIES
# ============================================

character = Counter()

for ent in doc.ents:
  if ent.label_ == "PERSON":
    character[ent.text] += 1


# ============================================
# 3. STORE EXAMPLE CONTEXTS
# ============================================

character_context = {}

for ent in doc.ents:

  if ent.label_ != "PERSON":
    continue

  if ent.text not in character_context:
    character_context[ent.text] = []

  if len(character_context[ent.text]) < 3:
    character_context[ent.text].append(
      ent.sent.text
    )


# ============================================
# 4. DETECT CHARACTER-LIKE CONTEXTS
# ============================================

character_verbs = {
  "say",
  "ask",
  "reply",
  "answer",
  "remark",
  "cry",
  "exclaim",
}

context_scores = Counter()

for ent in doc.ents:

  if ent.label_ != "PERSON":
    continue

  for token in ent.sent:

    if token.lemma_.lower() in character_verbs:
      context_scores[ent.text] += 1
      break


# ============================================
# 5. NAME VARIANT DETECTION
# ============================================

titles = {
  "Mr",
  "Mrs",
  "Miss",
  "Ms",
  "Dr",
  "Lady",
  "Sir",
}


def has_title(name):
  first_word = name.split()[0]
  return first_word in titles


def is_name_variant(name1, name2):

  words1 = [
    word
    for word in name1.split()
    if word not in titles
  ]

  words2 = [
    word
    for word in name2.split()
    if word not in titles
  ]

  if len(words1) >= len(words2):
    full_name = words1
    short_name = words2
  else:
    full_name = words2
    short_name = words1

  return all(
    word in full_name
    for word in short_name
  )


name_variants = {}
names = list(character.keys())

for i in range(len(names)):

  for j in range(i + 1, len(names)):

    name1 = names[i]
    name2 = names[j]

    if not is_name_variant(name1, name2):
      continue

    if len(name1.split()) >= len(name2.split()):
      full_name = name1
      short_name = name2
    else:
      full_name = name2
      short_name = name1

    name_variants.setdefault(
      short_name,
      []
    ).append(full_name)


# ============================================
# 6. SELECT CANONICAL NAMES AND ALIASES
# ============================================

alias_map = {}

for short_name, full_names in name_variants.items():

  non_title_names = [
    name
    for name in full_names
    if not has_title(name)
  ]

  candidates = non_title_names or full_names

  best_full_name = max(
    candidates,
    key=lambda name: character[name]
  )

  total_frequency = sum(
    character[name]
    for name in full_names
  )

  if total_frequency >= 5:
    alias_map[short_name] = best_full_name


# Add title-based variants

for short_name, full_names in name_variants.items():

  if short_name not in alias_map:
    continue

  canonical = alias_map[short_name]

  for name in full_names:

    if has_title(name):
      alias_map[name] = canonical


# ============================================
# 7. CREATE CANONICAL NAME LOOKUP
# ============================================

canonical_lookup = {
  name: name
  for name in character
}

for alias, canonical in alias_map.items():
  canonical_lookup[alias] = canonical


# ============================================
# 8. MERGE CHARACTER DATA
# ============================================

merged_frequency = Counter()
merged_contexts = {}
merged_context_scores = Counter()

for ent in doc.ents:

  if ent.label_ != "PERSON":
    continue

  canonical = canonical_lookup.get(ent.text)

  if not canonical:
    continue

  # Frequency
  merged_frequency[canonical] += 1

  # Contexts
  if canonical not in merged_contexts:
    merged_contexts[canonical] = []

  if (
    len(merged_contexts[canonical]) < 5
    and ent.sent.text not in merged_contexts[canonical]
  ):
    merged_contexts[canonical].append(
      ent.sent.text
    )

  # Context score
  for token in ent.sent:

    if token.lemma_.lower() in character_verbs:
      merged_context_scores[canonical] += 1
      break


# ============================================
# 9. CREATE CHARACTER PROFILES
# ============================================

character_profiles = {}

for canonical, frequency in merged_frequency.most_common():

  # Frequency score
  if frequency >= 50:
    frequency_score = 3

  elif frequency >= 10:
    frequency_score = 2

  elif frequency >= 3:
    frequency_score = 1

  else:
    frequency_score = 0


  # Context score
  context_count = merged_context_scores.get(
    canonical,
    0
  )

  if context_count >= 5:
    context_score = 3

  elif context_count >= 3:
    context_score = 2

  elif context_count >= 1:
    context_score = 1

  else:
    context_score = 0


  aliases = [
    alias
    for alias, name in alias_map.items()
    if name == canonical
  ]

  total_score = (
    frequency_score + context_score
  )

  character_profiles[canonical] = {
    "frequency": frequency,
    "aliases": aliases,
    "contexts": merged_contexts.get(
      canonical,
      []
    ),
    "frequency_score": frequency_score,
    "context_score": context_score,
    "total_score": total_score,
  }


# ============================================
# 10. SELECT CHARACTER CANDIDATES
# ============================================

valid_characters = {
  name
  for name, profile in character_profiles.items()
  if profile["total_score"] >= 3
}


# ============================================
# 11. FIND CHARACTER CO-OCCURRENCE
# ============================================

sentence_characters = []

for sent in doc.sents:

  persons = set()

  for ent in sent.ents:

    if ent.label_ != "PERSON":
      continue

    canonical = canonical_lookup.get(ent.text)

    if (
      canonical
      and canonical in valid_characters
    ):
      persons.add(canonical)

  if len(persons) >= 2:

    sentence_characters.append(
      (
        sent.text.strip(),
        persons
      )
    )


# ============================================
# 12. COUNT CHARACTER PAIRS
# ============================================

pair_counts = Counter()

for sentence, persons in sentence_characters:

  for person1, person2 in combinations(
    sorted(persons),
    2
  ):

    pair_counts[
      (person1, person2)
    ] += 1


# ============================================
# 13. SELECT STRONG CHARACTER PAIRS
# ============================================

strong_pairs = Counter({
  pair: count
  for pair, count in pair_counts.items()
  if count >= 2
})


# ============================================
# 14. RELATIONSHIP CONTEXT EXTRACTION
# ============================================

pair_contexts = {
  pair: []
  for pair in strong_pairs
}

for sentence, persons in sentence_characters:

  for pair in strong_pairs:

    person1, person2 = pair

    if (
      person1 in persons
      and person2 in persons
    ):

      pair_contexts[pair].append(
        sentence
      )


# ============================================
# FAMILY RELATIONSHIP DETECTION
# ============================================

def check_married_relationship(
  parsed_context,
  pair
):
  person1, person2 = pair

  for token in parsed_context:

    if token.lemma_.lower() != "marry":
      continue

    subjects = [
      child.text
      for child in token.children
      if child.dep_ in {"nsubj", "nsubjpass"}
    ]

    objects = [
      child.text
      for child in token.children
      if child.dep_ in {"dobj", "obj"}
    ]

    explicit_characters = []

    for word in parsed_context:
      if word.text in pair:
        explicit_characters.append(
          word.text
        )

    print("\nMarriage Verb:", token.text)
    print("Subjects:", subjects)
    print("Objects:", objects)
    print(
      "Explicit characters:",
      explicit_characters
    )

family_keywords = {
  "married",
  "marry",
  "wife",
  "husband",
  "mother",
  "father",
  "brother",
  "sister",
  "son",
  "daughter",
  "parent",
  "parents",
  "family",
}

family_evidence = {}

for pair, contexts in pair_contexts.items():

  relation_evidence = []

  for context in contexts:

    parsed_context = nlp(context)

    check_married_relationship(
      parsed_context,
      pair
    )

    found_keywords = []

    for token in parsed_context:

      if token.lemma_.lower() in family_keywords:
        found_keywords.append(
          token.lemma_.lower()
        )

    if found_keywords:

      relation_evidence.append({
        "context": context,
        "keywords": found_keywords
      })

  if relation_evidence:

    family_evidence[pair] = relation_evidence


# ============================================
# DISPLAY FAMILY RELATIONSHIP EVIDENCE
# ============================================

print("\n" + "=" * 60)
print("FAMILY RELATIONSHIP EVIDENCE")
print("=" * 60)

for pair, evidences in family_evidence.items():

  print(
    f"\n{pair[0]} ↔ {pair[1]}"
  )

  for evidence in evidences:

    print(
      "Keywords:",
      ", ".join(evidence["keywords"])
    )

    print(
      "Context:",
      evidence["context"]
    )

# ============================================
# DISPLAY RELATIONSHIP CONTEXTS
# ============================================

print("\n" + "=" * 60)
print("RELATIONSHIP CONTEXTS")
print("=" * 60)

for pair, contexts in pair_contexts.items():

  print(
    f"\n{pair[0]} ↔ {pair[1]}"
  )

  for context in contexts:
    print(f"- {context}")


# ============================================
# TEST DEPENDENCY PARSING
# ============================================

def show_dependency_info(sentence):

  parsed_sentence = nlp(sentence)

  print("\nSentence:")
  print(sentence)

  print("\nDependency Information:")
  print("-" * 60)

  for token in parsed_sentence:

    print(
      token.text,
      "| POS:", token.pos_,
      "| DEP:", token.dep_,
      "| HEAD:", token.head.text
    )

test_context = pair_contexts[
  ("Sherlock Holmes", "Watson")
][0]

show_dependency_info(test_context)

# ============================================
# 15. CREATE CHARACTER GRAPH
# ============================================

graph = nx.Graph()

for (person1, person2), count in strong_pairs.items():

  graph.add_edge(
    person1,
    person2,
    weight=count
  )


# # ============================================
# # DISPLAY GRAPH INFORMATION
# # ============================================

# print("\n" + "-" * 60)
# print("CHARACTER CO-OCCURRENCE GRAPH")
# print("-" * 60)

# print(
#   "Graph Nodes:",
#   graph.number_of_nodes()
# )

# print(
#   "Graph Edges:",
#   graph.number_of_edges()
# )

# print("\nStrong Character Connections:")

# for (person1, person2), count in strong_pairs.most_common(10):

#   print(
#     f"{person1} <-> {person2}"
#     f" | Strength: {count}"
#   )


# ============================================
# 16. VISUALIZE CHARACTER GRAPH
# ============================================

plt.figure(figsize=(14, 10))

pos = nx.spring_layout(
  graph,
  k=1.2,
  iterations=100,
  seed=42
)

edge_widths = [
  graph[person1][person2]["weight"]
  for person1, person2 in graph.edges()
]

nx.draw(
  graph,
  pos,
  with_labels=True,
  width=edge_widths,
  node_size=1400,
  font_size=9
)

plt.title(
  "LitMind - Character Co-occurrence Graph",
  fontsize=16
)

plt.show()


# # ============================================
# # 17. CHARACTER DEGREE
# # ============================================

# print("\n" + "-" * 60)
# print("GRAPH ANALYSIS - DEGREE")
# print("-" * 60)

# degrees = dict(graph.degree())

# for name, degree in sorted(
#   degrees.items(),
#   key=lambda x: x[1],
#   reverse=True
# ):

#   print(
#     name,
#     "->",
#     degree
#   )


# # ============================================
# # 18. WEIGHTED CHARACTER DEGREE
# # ============================================

# print("\n" + "-" * 60)
# print("GRAPH ANALYSIS - WEIGHTED DEGREE")
# print("-" * 60)

# weighted_degrees = dict(
#   graph.degree(weight="weight")
# )

# for name, degree in sorted(
#   weighted_degrees.items(),
#   key=lambda x: x[1],
#   reverse=True
# ):

#   print(
#     name,
#     "->",
#     degree
#   )


# # ============================================
# # 19. DEGREE CENTRALITY
# # ============================================

# print("\n" + "-" * 60)
# print("GRAPH ANALYSIS - DEGREE CENTRALITY")
# print("-" * 60)

# degree_centrality = nx.degree_centrality(
#   graph
# )

# for name, score in sorted(
#   degree_centrality.items(),
#   key=lambda x: x[1],
#   reverse=True
# ):

#   print(
#     name,
#     "->",
#     round(score, 3)
#   )


# # ============================================
# # 20. BETWEENNESS CENTRALITY
# # ============================================

# print("\n" + "-" * 60)
# print("GRAPH ANALYSIS - BETWEENNESS CENTRALITY")
# print("-" * 60)

# betweenness_centrality = (
#   nx.betweenness_centrality(graph)
# )

# for name, score in sorted(
#   betweenness_centrality.items(),
#   key=lambda x: x[1],
#   reverse=True
# ):

#   print(
#     name,
#     "->",
#     round(score, 3)
#   )