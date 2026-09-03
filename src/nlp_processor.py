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
    character_context[ent.text].append(ent.sent.text)


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
    sentence_characters.append(persons)


# ============================================
# 12. COUNT CHARACTER PAIRS
# ============================================

pair_counts = Counter()

for persons in sentence_characters:

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
# 14. CREATE CHARACTER GRAPH
# ============================================

graph = nx.Graph()

for (person1, person2), count in strong_pairs.items():

  graph.add_edge(
    person1,
    person2,
    weight=count
  )


# ============================================
# 15. VISUALIZE CHARACTER GRAPH
# ============================================

plt.figure(figsize=(12, 8))

pos = nx.spring_layout(graph)

edge_widths = [
  graph[person1][person2]["weight"]
  for person1, person2 in graph.edges()
]

nx.draw(
  graph,
  pos,
  with_labels=True,
  width=edge_widths
)

# plt.show()


# ============================================
# 16. CHARACTER DEGREE
# ============================================

print("\nCharacter Degrees:\n")

degrees = dict(graph.degree())

for name, degree in sorted(
  degrees.items(),
  key=lambda x: x[1],
  reverse=True
):
  print(name, "->", degree)


# ============================================
# 17. WEIGHTED CHARACTER DEGREE
# ============================================

print("\nWeighted Character Degrees:\n")

weighted_degrees = dict(
  graph.degree(weight="weight")
)

for name, degree in sorted(
  weighted_degrees.items(),
  key=lambda x: x[1],
  reverse=True
):
  print(name, "->", degree)


# ============================================
# 18. DEGREE CENTRALITY
# ============================================

print("\nDegree Centrality:\n")

degree_centrality = nx.degree_centrality(graph)

for name, score in sorted(
  degree_centrality.items(),
  key=lambda x: x[1],
  reverse=True
):
  print(
    name,
    "->",
    round(score, 3)
  )


# ============================================
# 19. BETWEENNESS CENTRALITY
# ============================================

print("\nBetweenness Centrality:\n")

betweenness_centrality = (
  nx.betweenness_centrality(graph)
)

for name, score in sorted(
  betweenness_centrality.items(),
  key=lambda x: x[1],
  reverse=True
):
  print(
    name,
    "->",
    round(score, 3)
  )