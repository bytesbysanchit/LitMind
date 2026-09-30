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
# 14. RELATIONSHIP ANALYSIS
# ============================================


# --------------------------------------------
# FAMILY RELATIONSHIP KEYWORDS
# --------------------------------------------

family_keywords = {
  "married",
  "marry",
  "marriage",
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
}


# --------------------------------------------
# OTHER RELATIONSHIP KEYWORDS
# --------------------------------------------

professional_keywords = {
  "colleague",
  "employee",
  "employer",
  "assistant",
  "partner",
  "client",
  "detective",
  "inspector",
  "officer",
  "doctor",
  "lawyer",
}


friend_keywords = {
  "companion",
  "companions",
  "trusted",
  "trust",
  "faithful",
  "loyal",
  "dear",
}


conflict_keywords = {
  "enemy",
  "enemies",
  "rival",
  "rivals",
  "opponent",
  "opponents",
  "fight",
  "fought",
  "fighting",
  "attack",
  "attacked",
  "threat",
  "threatened",
  "arrest",
  "arrested",
  "revenge",
  "villain",
}


relation_keyword_map = {
  "PROFESSIONAL": professional_keywords,
  "FRIEND / COMPANION": friend_keywords,
  "CONFLICT / ENEMY": conflict_keywords,
}


# --------------------------------------------
# INTERACTION VERBS
# --------------------------------------------

interaction_verbs = {
  "ask",
  "answer",
  "reply",
  "tell",
  "help",
  "meet",
  "visit",
  "follow",
  "accompany",
  "join",
  "assist",
}


reporting_verbs = {
  "say",
  "remark",
  "ask",
  "reply",
  "answer",
  "tell",
  "cry",
  "exclaim",
}


# --------------------------------------------
# HELPER: CHARACTER NAME PARTS
# --------------------------------------------

def get_name_parts(name):

  parts = []

  for word in name.split():

    word = (
      word
      .replace(".", "")
      .replace(",", "")
      .strip()
      .lower()
    )

    if word not in {
      "mr",
      "mrs",
      "miss",
      "ms",
      "dr",
      "lady",
      "sir"
    }:
      parts.append(word)

  return parts


# --------------------------------------------
# HELPER: CHECK CHARACTER WORDS
# --------------------------------------------

def character_words_present(
  parsed_context,
  name
):

  name_parts = get_name_parts(name)

  context_words = {
    token.text.lower()
    for token in parsed_context
  }

  return any(
    part in context_words
    for part in name_parts
  )


# --------------------------------------------
# FAMILY RELATIONSHIP DETECTION
# --------------------------------------------

def detect_family_relation(
  context,
  pair
):

  person1, person2 = pair

  parsed_context = nlp(context)

  person1_parts = set(
    get_name_parts(person1)
  )

  person2_parts = set(
    get_name_parts(person2)
  )

  # ==========================================
  # 1. MARRIAGE VERB
  # ==========================================

  for token in parsed_context:

    if token.lemma_.lower() != "marry":
      continue

    subject_words = set()
    object_words = set()

    # ------------------------------------------
    # Find actual subject and object
    # ------------------------------------------

    for child in token.children:

      if child.dep_ in {
        "nsubj",
        "nsubjpass"
      }:

        for subtoken in child.subtree:
          subject_words.add(
            subtoken.text.lower()
          )

      elif child.dep_ in {
        "obj",
        "dobj"
      }:

        for subtoken in child.subtree:
          object_words.add(
            subtoken.text.lower()
          )

    # ------------------------------------------
    # CASE A:
    # Explicit character + explicit character
    #
    # Example:
    # "Holmes married Irene."
    # ------------------------------------------

    person1_subject = any(
      part in subject_words
      for part in person1_parts
    )

    person2_subject = any(
      part in subject_words
      for part in person2_parts
    )

    person1_object = any(
      part in object_words
      for part in person1_parts
    )

    person2_object = any(
      part in object_words
      for part in person2_parts
    )

    if (
      (person1_subject and person2_object)
      or
      (person2_subject and person1_object)
    ):
      return True

    # ------------------------------------------
    # CASE B:
    # Pronoun subject + explicit person object
    #
    # Example:
    # "He married Mrs. Stoner."
    # ------------------------------------------

    subject_is_pronoun = False

    for child in token.children:

      if child.dep_ in {
        "nsubj",
        "nsubjpass"
      } and child.pos_ == "PRON":

        subject_is_pronoun = True

    if not subject_is_pronoun:
      continue

    # ------------------------------------------
    # Find nearest PERSON before marriage verb
    # ------------------------------------------

    previous_person = None

    for ent in parsed_context.ents:

      if (
        ent.label_ == "PERSON"
        and ent.end <= token.i
      ):
        previous_person = ent

    if previous_person is None:
      continue

    previous_words = {
      word
      .replace(".", "")
      .replace(",", "")
      .lower()
      for word in previous_person.text.split()
    }

    previous_is_person1 = any(
      part in previous_words
      for part in person1_parts
    )

    previous_is_person2 = any(
      part in previous_words
      for part in person2_parts
    )

    # ------------------------------------------
    # Explicit object character
    # ------------------------------------------

    object_is_person1 = any(
      part in object_words
      for part in person1_parts
    )

    object_is_person2 = any(
      part in object_words
      for part in person2_parts
    )

    if (
      previous_is_person1
      and object_is_person2
    ):
      return True

    if (
      previous_is_person2
      and object_is_person1
    ):
      return True


  # ==========================================
  # 2. DIRECT FAMILY NOUN
  # ==========================================

  family_nouns = {
    "wife",
    "husband",
    "mother",
    "father",
    "brother",
    "sister",
    "son",
    "daughter",
    "parent"
  }

  for token in parsed_context:

    if token.lemma_.lower() not in family_nouns:
      continue

    connected_words = {
      subtoken.text.lower()
      for subtoken in token.subtree
    }

    person1_present = any(
      part in connected_words
      for part in person1_parts
    )

    person2_present = any(
      part in connected_words
      for part in person2_parts
    )

    if (
      person1_present
      and person2_present
    ):
      return True

  return False


# --------------------------------------------
# DIRECT INTERACTION DETECTION
# --------------------------------------------

def detect_direct_interaction(
  context,
  pair
):

  person1, person2 = pair

  parsed_context = nlp(context)

  person1_parts = get_name_parts(person1)
  person2_parts = get_name_parts(person2)


  # ------------------------------------------
  # 1. DIRECT SPEECH / ADDRESS
  # ------------------------------------------

  for token in parsed_context:

    if (
      token.lemma_.lower()
      not in reporting_verbs
    ):
      continue

    speaker = None

    # Find speaker of reporting verb.

    for child in token.children:

      if child.dep_ in {
        "nsubj",
        "nsubjpass"
      }:

        speaker_words = {
          subtoken.text.lower()
          for subtoken in child.subtree
        }

        if any(
          part in speaker_words
          for part in person1_parts
        ):
          speaker = person1

        elif any(
          part in speaker_words
          for part in person2_parts
        ):
          speaker = person2


    if speaker is None:
      continue


    # --------------------------------------
    # Find addressee near the end of speech
    # --------------------------------------

    other_person = (
      person2
      if speaker == person1
      else person1
    )

    other_parts = get_name_parts(
      other_person
    )

    for ent in parsed_context.ents:

      if ent.label_ != "PERSON":
        continue

      ent_words = {
        word.lower()
        for word in ent.text.split()
      }

      matches_other = any(
        part in ent_words
        for part in other_parts
      )

      if not matches_other:
        continue

      # Character should occur shortly before
      # the reporting verb.

      distance = token.i - ent.end

      if distance < 0 or distance > 6:
        continue

      # Avoid cases where the character is
      # merely an object/subject in the sentence.

      ent_tokens = list(ent)

      valid_address = False

      for ent_token in ent_tokens:

        if ent_token.dep_ in {
          "vocative",
          "npadvmod"
        }:
          valid_address = True

      if valid_address:

        return True


  # ------------------------------------------
  # 2. DIRECT ACTION BETWEEN CHARACTERS
  # ------------------------------------------

  for token in parsed_context:

    if (
      token.lemma_.lower()
      not in interaction_verbs
    ):
      continue

    subject_words = set()
    object_words = set()

    # ----------------------------------------
    # Find actual subject
    # ----------------------------------------

    for child in token.children:

      if child.dep_ in {
        "nsubj",
        "nsubjpass"
      }:

        for subtoken in child.subtree:

          subject_words.add(
            subtoken.text.lower()
          )

    # ----------------------------------------
    # Find actual object / recipient
    # ----------------------------------------

      elif child.dep_ in {
        "obj",
        "dobj",
        "iobj",
        "pobj"
      }:

        for subtoken in child.subtree:

          object_words.add(
            subtoken.text.lower()
          )

    # ----------------------------------------
    # Check whether person1/person2 are
    # actually on opposite sides of action
    # ----------------------------------------

    person1_subject = any(
      part in subject_words
      for part in person1_parts
    )

    person2_subject = any(
      part in subject_words
      for part in person2_parts
    )

    person1_object = any(
      part in object_words
      for part in person1_parts
    )

    person2_object = any(
      part in object_words
      for part in person2_parts
    )

    # ----------------------------------------
    # Genuine interaction:
    #
    # person1 -> action -> person2
    # OR
    # person2 -> action -> person1
    # ----------------------------------------

    if (
      person1_subject
      and person2_object
    ):
      return True

    if (
      person2_subject
      and person1_object
    ):
      return True


  return False


# --------------------------------------------
# RELATION KEYWORD DETECTION
# --------------------------------------------

def detect_keyword_relation(
  context,
  pair
):

  person1, person2 = pair

  parsed_context = nlp(context)

  person1_parts = get_name_parts(person1)
  person2_parts = get_name_parts(person2)

  found_relations = {
    "PROFESSIONAL": [],
    "FRIEND / COMPANION": [],
    "CONFLICT / ENEMY": [],
  }


  for token in parsed_context:

    word = token.lemma_.lower()

    matched_relation = None

    for relation, keywords in (
      relation_keyword_map.items()
    ):

      if word in keywords:

        matched_relation = relation
        break

    if matched_relation is None:
      continue


    # ----------------------------------------
    # Check nearby character evidence
    # ----------------------------------------

    start = max(
      0,
      token.i - 5
    )

    end = min(
      len(parsed_context),
      token.i + 6
    )

    nearby_words = {
      parsed_context[i].text.lower()
      for i in range(start, end)
    }

    person1_nearby = any(
      part in nearby_words
      for part in person1_parts
    )

    person2_nearby = any(
      part in nearby_words
      for part in person2_parts
    )

    if (
      person1_nearby
      and person2_nearby
    ):

      found_relations[
        matched_relation
      ].append(word)

  return found_relations


# --------------------------------------------
# BUILD RELATIONSHIP EVIDENCE
# --------------------------------------------

relationship_results = {}


for pair, contexts in (
  pair_contexts.items()
):

  family_evidence = []
  interaction_evidence = []
  keyword_evidence = []


  for context in contexts:

    # --------------------------------------
    # FAMILY
    # --------------------------------------

    if detect_family_relation(
      context,
      pair
    ):

      family_evidence.append({
        "context": context,
        "keywords": [
          token.lemma_.lower()
          for token in nlp(context)
          if (
            token.lemma_.lower()
            in family_keywords
          )
        ]
      })

      continue


    # --------------------------------------
    # DIRECT INTERACTION
    # --------------------------------------

    if detect_direct_interaction(
      context,
      pair
    ):

      interaction_evidence.append({
        "context": context,
        "keywords": []
      })


    # --------------------------------------
    # OTHER RELATIONSHIP KEYWORDS
    # --------------------------------------

    found_relations = (
      detect_keyword_relation(
        context,
        pair
      )
    )

    for relation, keywords in (
      found_relations.items()
    ):

      if keywords:

        keyword_evidence.append({
          "context": context,
          "keywords": keywords,
          "relation": relation
        })


  # ========================================
  # FINAL CLASSIFICATION
  # ========================================

  if family_evidence:

    relationship_results[pair] = {
      "relationship": "FAMILY",
      "keywords": [
        keyword
        for item in family_evidence
        for keyword in item["keywords"]
      ],
      "evidence": family_evidence,
      "score": len(family_evidence)
    }

    continue


  # ----------------------------------------
  # CONFLICT / PROFESSIONAL / FRIEND
  # ----------------------------------------

  relation_scores = {
    "PROFESSIONAL": 0,
    "FRIEND / COMPANION": 0,
    "CONFLICT / ENEMY": 0,
  }

  best_keyword_evidence = []


  for item in keyword_evidence:

    relation = item["relation"]

    relation_scores[relation] += (
      len(item["keywords"])
    )

    best_keyword_evidence.append(item)


  best_relation = max(
    relation_scores,
    key=relation_scores.get
  )

  best_score = relation_scores[
    best_relation
  ]


  # ----------------------------------------
  # Repeated direct interaction
  # ----------------------------------------

  if (
    len(interaction_evidence) >= 3
    and best_score == 0
  ):

    relationship_results[pair] = {
      "relationship": "FRIEND / COMPANION",
      "keywords": [],
      "evidence": interaction_evidence,
      "score": len(interaction_evidence)
    }

  elif best_score > 0:

    selected_evidence = [
      item
      for item in best_keyword_evidence
      if item["relation"] == best_relation
    ]

    relationship_results[pair] = {
      "relationship": best_relation,
      "keywords": [
        keyword
        for item in selected_evidence
        for keyword in item["keywords"]
      ],
      "evidence": selected_evidence,
      "score": best_score
    }

  elif interaction_evidence:

    relationship_results[pair] = {
      "relationship": "INTERACTION",
      "keywords": [],
      "evidence": interaction_evidence,
      "score": len(interaction_evidence)
    }

  else:

    relationship_results[pair] = {
      "relationship": "OTHER / UNKNOWN",
      "keywords": [],
      "evidence": [],
      "score": 0
    }

# ============================================
# RELATIONSHIP STRENGTH
# ============================================
# This is CO-OCCURRENCE strength, not semantic
# relationship strength.

for pair, result in relationship_results.items():

  co_occurrence = strong_pairs[pair]

  if co_occurrence >= 4:
    strength = "STRONG"

  elif co_occurrence >= 2:
    strength = "MODERATE"

  else:
    strength = "WEAK"

  result["co_occurrence"] = co_occurrence
  result["strength"] = strength

  # Relationship confidence
  if result["relationship"] == "FAMILY":
    result["confidence"] = "HIGH"

  elif result["score"] >= 3:
    result["confidence"] = "MEDIUM"

  elif result["score"] >= 1:
    result["confidence"] = "LOW"

  else:
    result["confidence"] = "LOW"


# ============================================
# FINAL RELATIONSHIP OUTPUT
# ============================================

print(
  "\n" + "=" * 60
)

print(
  "FINAL CHARACTER RELATIONSHIP ANALYSIS"
)

print(
  "=" * 60
)

for pair, result in relationship_results.items():

  person1, person2 = pair

  print(
    f"\n{person1} ↔ {person2}"
  )

  print(
    "Relationship:",
    result["relationship"]
  )

  print(
    "Confidence:",
    result["confidence"]
  )

  print(
    "Co-occurrence:",
    result["co_occurrence"]
  )

  print(
    "Co-occurrence Strength:",
    result["strength"]
  )

  if result["keywords"]:
    print(
      "Keywords:",
      ", ".join(
        sorted(
          set(result["keywords"])
        )
      )
    )
  else:
    print(
      "Keywords: None"
    )

  if result["evidence"]:

    print("Evidence:")

    shown_contexts = set()

    for evidence in result["evidence"]:

      context = evidence["context"]

      if context in shown_contexts:
        continue

      shown_contexts.add(context)

      print(
        " -",
        context
      )
  else:
    print(
      "Evidence: No reliable relationship evidence found."
    )


# ============================================
# RELATIONSHIP SUMMARY
# ============================================

print(
  "\n" + "=" * 60
)

print(
  "RELATIONSHIP SUMMARY"
)

print(
  "=" * 60
)

relationship_summary = Counter()

for result in relationship_results.values():
  relationship_summary[
    result["relationship"]
  ] += 1

for relation, count in relationship_summary.items():
  print(
    f"{relation}: {count}"
  )

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
