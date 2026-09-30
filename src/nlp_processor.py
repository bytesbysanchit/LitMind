from pathlib import Path
from collections import Counter
from itertools import combinations
import sys
import re
import math

import spacy
import networkx as nx
import matplotlib.pyplot as plt


# ============================================
# LITMIND - NOVEL ANALYZER
# ============================================

# Default novel used for demonstration.
# A different processed novel can be supplied from the command line:
# python src/nlp_processor.py path/to/novel.txt

DEFAULT_TEXT_PATH = Path(
  "data/processed/Sherlock_Holmes.txt"
)

text_path = (
  Path(sys.argv[1])
  if len(sys.argv) > 1
  else DEFAULT_TEXT_PATH
)


# ============================================
# 1. LOAD MODEL AND TEXT
# ============================================

nlp = spacy.load("en_core_web_sm")

with open(text_path, "r", encoding="utf-8") as file:
  text = file.read()

doc = nlp(text)

print("\n" + "=" * 60)
print("              LITMIND - NOVEL ANALYZER")
print("=" * 60)

print(f"\nInput Novel: {text_path.stem}")

print("\n" + "-" * 60)
print("1. TEXT PROCESSING")
print("-" * 60)
print("Characters in text:", len(text))
print("Sentences detected:", len(list(doc.sents)))
print("Tokens detected:", len(doc))


# ============================================
# 2. CHARACTER EXTRACTION
# ============================================

titles = {
  "mr", "mrs", "miss", "ms", "dr", "lady", "lord",
  "sir", "madam", "captain", "professor", "rev", "reverend",
}


def normalize_surface_name(name):
  name = (
    name
    .replace("’", "'")
    .replace("‘", "'")
    .strip()
  )

  name = re.sub(
    r"(?:'s|s')$",
    "",
    name,
    flags=re.IGNORECASE
  )

  name = re.sub(
    r"^[^A-Za-z]+|[^A-Za-z]+$",
    "",
    name
  )
  name = re.sub(r"\s+", " ", name).strip()
  name = name.replace(".", "")
  return name


document_indicators = {
  "journal", "diary", "volume", "chapter", "book", "letter",
  "account", "record", "notes", "newspaper", "article", "manuscript",
  "log", "entry", "telegram", "message", "report", "memoir",
  "license", "copyright", "permission", "terms", "ebook",
  "gutenberg",
}

place_indicators = {
  "street", "road", "avenue", "square", "station", "river", "lake",
  "hill", "mount", "bridge", "church", "castle", "manor", "hotel",
  "park", "garden", "house", "hall", "palace", "railway", "town",
  "city", "county", "island", "stoke",
}

organization_indicators = {
  "project", "society", "company", "university", "college",
  "publisher", "press", "edition",
}

role_only_words = {
  "man", "woman", "gentleman", "gentlemen", "lady", "ladies",
  "doctor", "nurse", "officer", "inspector", "detective", "professor",
  "servant", "master", "madam", "miss", "mrs", "mr", "dr", "sir", "lord",
  "majesty", "voivode",
}


def cleaned_words(name):
  return [
    re.sub(r"[^A-Za-z]", "", word).lower()
    for word in name.split()
    if re.sub(r"[^A-Za-z]", "", word)
  ]


def looks_like_non_character(name, non_person_counts):
  words = cleaned_words(name)

  if not words:
    return True

  # Literary NER sometimes absorbs punctuation or
  # surrounding prose into a PERSON span.
  if "," in name or "&" in name:
    return True

  # Ignore obvious OCR/copyright-style fragments such as
  # "M R C S L K..." or a single isolated letter.
  if len(words) == 1 and len(words[0]) <= 1:
    return True

  if (
    len(words) >= 4
    and sum(len(word) == 1 for word in words) >= 3
  ):
    return True

  # Reject OCR/heading artefacts such as "HOLMES Holmes".
  # A real character name should not repeat the same token
  # with different capitalization.
  lowered_words = [word.lower() for word in words]

  if len(lowered_words) != len(set(lowered_words)):
    return True

  # A multi-word PERSON span containing an all-caps word is
  # usually a header/OCR artefact rather than a character name.
  raw_words = name.split()

  if (
    len(raw_words) >= 2
    and any(
      len(re.sub(r"[^A-Za-z]", "", word)) >= 3
      and re.sub(r"[^A-Za-z]", "", word).isupper()
      for word in raw_words
    )
  ):
    return True

  if len(words) == 1 and words[0] in role_only_words:
    return True

  if any(word in document_indicators for word in words):
    return True

  if any(word in organization_indicators for word in words):
    return True

  if len(words) >= 2 and any(word in place_indicators for word in words):
    return True

  # Lowercase PERSON spans are usually descriptions,
  # objects, animals, or OCR noise rather than character names.
  # Keep title-cased literary names, but reject spans such as
  # "wolf" without maintaining a novel-specific animal list.
  if name == name.lower():
    return True

  return non_person_counts.get(name, 0) >= 2


# First collect non-PERSON labels so that a literary NER mistake such as
# a place/document being tagged PERSON can be rejected generically.
non_person_counts = Counter()

for ent in doc.ents:
  clean_name = normalize_surface_name(ent.text)

  if (
    clean_name
    and ent.label_ in {"GPE", "LOC", "FAC", "ORG"}
  ):
    non_person_counts[clean_name] += 1


character = Counter()

for ent in doc.ents:

  if ent.label_ != "PERSON":
    continue

  clean_name = normalize_surface_name(ent.text)

  if not looks_like_non_character(
    clean_name,
    non_person_counts
  ):
    character[clean_name] += 1


print("\n" + "-" * 60)
print("2. CHARACTER EXTRACTION")
print("-" * 60)
print("Total PERSON mentions:", sum(character.values()))
print("Unique PERSON names:", len(character))

print("\nTop Character Mentions:")

for name, count in character.most_common(10):
  print(f"  {name:<25} -> {count}")


# ============================================
# 3. STORE EXAMPLE CONTEXTS
# ============================================

character_context = {}

for ent in doc.ents:

  if ent.label_ != "PERSON":
    continue

  clean_name = normalize_surface_name(ent.text)

  if clean_name not in character:
    continue

  if clean_name not in character_context:
    character_context[clean_name] = []

  if len(character_context[clean_name]) < 3:
    character_context[clean_name].append(ent.sent.text)


# ============================================
# 4. DETECT CHARACTER-LIKE CONTEXTS
# ============================================

character_verbs = {
  "say", "ask", "reply", "answer", "remark", "cry", "exclaim",
}

context_scores = Counter()

for ent in doc.ents:

  if ent.label_ != "PERSON":
    continue

  clean_name = normalize_surface_name(ent.text)

  if clean_name not in character:
    continue

  for token in ent.sent:
    if token.lemma_.lower() in character_verbs:
      context_scores[clean_name] += 1
      break


# ============================================
# 3. CHARACTER NORMALIZATION
# ============================================

def name_tokens(name):
  return [
    word.lower()
    for word in cleaned_words(name)
    if word.lower() not in titles
  ]


def entity_text_matches_entity(entity, other_entity):
  return (
    entity.start < other_entity.end
    and other_entity.start < entity.end
  )


# Start with every extracted PERSON name mapping to itself.
alias_map = {}
canonical_lookup = {}

for name in character:
  alias_map[name] = name
  canonical_lookup[name] = name


# --------------------------------------------
# 3A. Build full-name candidates
# --------------------------------------------
#
# A canonical character name is preferably a
# multi-token PERSON name. This lets names such
# as:
#   Jonathan Harker
#   Quincey Morris
#   Abraham Van Helsing
# become canonical forms for shorter variants.
#
# No novel-specific names are hardcoded here.

full_name_candidates = [
  name
  for name in character
  if len(name_tokens(name)) >= 2
  and not looks_like_non_character(
    name,
    non_person_counts
  )
]


def candidate_key(name):
  return tuple(sorted(name_tokens(name)))


candidate_groups = {}

for name in full_name_candidates:
  candidate_groups.setdefault(
    candidate_key(name),
    []
  ).append(name)


canonical_names = set()

for variants in candidate_groups.values():
  canonical = max(
    variants,
    key=lambda name: (
      len(name_tokens(name)),
      character[name],
      len(name)
    )
  )
  canonical_names.add(canonical)


# --------------------------------------------
# 3B. Map exact / subset name variants
# --------------------------------------------

for name in character:
  if looks_like_non_character(
    name,
    non_person_counts
  ):
    continue

  tokens = set(name_tokens(name))

  if not tokens:
    continue

  # Exact token-set match.
  exact_matches = [
    candidate
    for candidate in canonical_names
    if set(name_tokens(candidate)) == tokens
  ]

  if exact_matches:
    canonical = max(
      exact_matches,
      key=lambda candidate: (
        character[candidate],
        len(candidate)
      )
    )
    canonical_lookup[name] = canonical
    if name != canonical:
      alias_map[name] = canonical
    continue

  # Subset match:
  # "Dracula" -> "Count Dracula"
  # "Harker"  -> "Jonathan Harker"
  # "Van Helsing" -> "Abraham Van Helsing"
  subset_matches = [
    candidate
    for candidate in canonical_names
    if tokens.issubset(
      set(name_tokens(candidate))
    )
  ]

  if len(subset_matches) == 1:
    canonical = subset_matches[0]
    canonical_lookup[name] = canonical
    if name != canonical:
      alias_map[name] = canonical
    continue

  if len(subset_matches) > 1:
    # Prefer the candidate with the strongest
    # corpus evidence, not a hardcoded name.
    canonical = max(
      subset_matches,
      key=lambda candidate: (
        character[candidate],
        len(name_tokens(candidate)),
        len(candidate)
      )
    )
    canonical_lookup[name] = canonical
    if name != canonical:
      alias_map[name] = canonical


# --------------------------------------------
# 3C. Generic nickname / abbreviation matching
# --------------------------------------------
#
# Only used when a short PERSON name does not
# already have a safer exact/subset mapping.
#
# Example:
#   Art -> Arthur Holmwood
#
# This is deliberately generic and only accepts
# a unique prefix candidate of length >= 3.

for name in character:
  if name in canonical_lookup and (
    canonical_lookup[name] != name
    or len(name_tokens(name)) >= 2
  ):
    continue

  tokens = name_tokens(name)

  if len(tokens) != 1:
    continue

  short = tokens[0]

  if len(short) < 3:
    continue

  prefix_matches = []

  for candidate in canonical_names:
    candidate_tokens = name_tokens(candidate)

    if len(candidate_tokens) >= 1:
      first = candidate_tokens[0]

      if (
        first.startswith(short)
        and first != short
      ):
        prefix_matches.append(candidate)

  if len(prefix_matches) == 1:
    canonical = prefix_matches[0]

    # A prefix alias is accepted only when the longer name
    # has actual corpus evidence. This avoids turning arbitrary
    # words into character aliases.
    if character.get(canonical, 0) >= 2:
      canonical_lookup[name] = canonical
      alias_map[name] = canonical


# --------------------------------------------
# 3C-FINAL. Re-check short prefix aliases
# --------------------------------------------
#
# This is intentionally a second pass over the already
# validated full-name candidates.  It protects against a
# normalization ordering issue where a short form such as
# "Art" is left unmapped even though "Arthur Holmwood" is
# a valid PERSON candidate.
#
# It is generic: no novel-specific character names are used.

for name in character:
  tokens = name_tokens(name)

  if len(tokens) != 1:
    continue

  short = tokens[0]

  if len(short) < 3:
    continue

  if canonical_lookup.get(name, name) != name:
    continue

  prefix_matches = []

  for candidate in full_name_candidates:
    candidate_tokens = name_tokens(candidate)

    if len(candidate_tokens) < 2:
      continue

    if looks_like_non_character(
      candidate,
      non_person_counts
    ):
      continue

    first = candidate_tokens[0]

    if (
      first.startswith(short)
      and first != short
      and character.get(candidate, 0) >= 2
    ):
      prefix_matches.append(candidate)

  if len(prefix_matches) == 1:
    canonical = prefix_matches[0]
    canonical_lookup[name] = canonical
    alias_map[name] = canonical


# --------------------------------------------
# 3D. Detect appositive aliases automatically
# --------------------------------------------
#
# This handles literary constructions such as:
#   "Arthur Holmwood, Lord Godalming, ..."
#
# without storing "Godalming" in the code.

for sent in doc.sents:
  person_entities = [
    ent
    for ent in sent.ents
    if ent.label_ == "PERSON"
  ]

  for entity in person_entities:

    clean_entity = normalize_surface_name(
      entity.text
    )

    if clean_entity not in character:
      continue

    for token in entity:

      if token.dep_ != "appos":
        continue

      head_entity = None

      for other in person_entities:

        if other == entity:
          continue

        if (
          other.start <= token.head.i < other.end
        ):
          head_entity = other
          break

      if head_entity is None:
        continue

      clean_head = normalize_surface_name(
        head_entity.text
      )

      if clean_head not in character:
        continue

      head_canonical = canonical_lookup.get(
        clean_head,
        clean_head
      )

      alias_map[clean_entity] = head_canonical
      canonical_lookup[clean_entity] = head_canonical


# --------------------------------------------
# 3E. Resolve alias chains
# --------------------------------------------

def resolve_canonical(name):
  current = name
  visited = set()

  while current in canonical_lookup:
    if current in visited:
      break

    visited.add(current)

    next_name = canonical_lookup[current]

    if next_name == current:
      break

    current = next_name

  return current


for name in list(canonical_lookup):
  canonical_lookup[name] = resolve_canonical(name)

for alias, canonical in list(alias_map.items()):
  alias_map[alias] = resolve_canonical(canonical)


# Remove non-character canonical targets after
# normalization. A canonical target must still be
# a valid PERSON-derived candidate.

for name in list(canonical_lookup):
  canonical = canonical_lookup[name]

  if looks_like_non_character(
    canonical,
    non_person_counts
  ):
    canonical_lookup[name] = name
    alias_map.pop(name, None)

# Remove aliases that point to a canonical name that is not
# present as a validated PERSON-derived candidate.
valid_canonical_targets = {
  candidate
  for candidate in character
  if not looks_like_non_character(
    candidate,
    non_person_counts
  )
}

for alias, canonical in list(alias_map.items()):
  if canonical not in valid_canonical_targets:
    alias_map.pop(alias, None)
    canonical_lookup[alias] = alias


print("\n" + "-" * 60)
print("3. CHARACTER NORMALIZATION")
print("-" * 60)

detected_aliases = sorted(
  [
    (alias, canonical)
    for alias, canonical in alias_map.items()
    if alias != canonical
  ],
  key=lambda item: item[0].lower()
)

print(
  "Aliases detected:",
  len(detected_aliases)
)

if detected_aliases:
  print("\nDetected Name Variants:")

  for alias, canonical in detected_aliases:
    print(
      f"  {alias} -> {canonical}"
    )


# ============================================
# 4. MERGE CHARACTER DATA
# ============================================


merged_frequency = Counter()
merged_contexts = {}
merged_context_scores = Counter()

for ent in doc.ents:

  if ent.label_ != "PERSON":
    continue

  clean_name = normalize_surface_name(ent.text)
  canonical = canonical_lookup.get(clean_name)

  if not canonical:
    continue

  merged_frequency[canonical] += 1

  if canonical not in merged_contexts:
    merged_contexts[canonical] = []

  if (
    len(merged_contexts[canonical]) < 5
    and ent.sent.text not in merged_contexts[canonical]
  ):
    merged_contexts[canonical].append(ent.sent.text)

  for token in ent.sent:
    if token.lemma_.lower() in character_verbs:
      merged_context_scores[canonical] += 1
      break


# ============================================
# 4. CHARACTER PROFILE / SELECTION
# ============================================

character_profiles = {}

for canonical, frequency in merged_frequency.most_common():

  if frequency >= 50:
    frequency_score = 3
  elif frequency >= 10:
    frequency_score = 2
  elif frequency >= 3:
    frequency_score = 1
  else:
    frequency_score = 0

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
    frequency_score
    + context_score
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


def looks_like_place(name):
  words = set(cleaned_words(name))

  return (
    len(words) >= 2
    and bool(words & place_indicators)
  )


def is_redundant_short_name(name, candidate_names):
  """
  Reject a short one-token PERSON span when a longer
  canonical-looking PERSON name starts with the same token.

  This is generic: it handles cases such as
  "Art" -> "Arthur Holmwood" and also protects
  "John"/"Jonathan"-style short NER fragments without
  hardcoding any novel character names.
  """

  parts = name_tokens(name)

  if len(parts) != 1:
    return False

  short = parts[0]

  if len(short) < 3:
    return False

  for candidate in candidate_names:
    candidate_parts = name_tokens(candidate)

    if len(candidate_parts) < 2:
      continue

    if (
      candidate != name
      and candidate_parts[0].startswith(short)
      and candidate_parts[0] != short
    ):
      return True

  return False


valid_character_names = set(character_profiles)

valid_characters = {
  name
  for name, profile in character_profiles.items()
  if (
    profile["total_score"] >= 3
    and not looks_like_non_character(
      name,
      non_person_counts
    )
    and not looks_like_place(name)
    and not is_redundant_short_name(
      name,
      valid_character_names
    )
  )
}

print("\n" + "-" * 60)
print("4. CHARACTER PROFILE / SELECTION")
print("-" * 60)
print(
  "Valid character candidates:",
  len(valid_characters)
)

print("\nTop Valid Characters:")

shown = 0

for name, profile in sorted(
  character_profiles.items(),
  key=lambda item: item[1]["frequency"],
  reverse=True
):

  if name not in valid_characters:
    continue

  print(
    f"  {name:<30}"
    f" Frequency: {profile['frequency']}"
    f" | Score: {profile['total_score']}"
  )

  shown += 1

  if shown >= 10:
    break


# ============================================
# 5. CHARACTER CO-OCCURRENCE
# ============================================

sentence_characters = []

for sent in doc.sents:

  persons = set()

  for ent in sent.ents:

    if ent.label_ != "PERSON":
      continue

    clean_name = normalize_surface_name(ent.text)
    canonical = canonical_lookup.get(clean_name)

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


pair_counts = Counter()

for sentence, persons in sentence_characters:

  for person1, person2 in combinations(
    sorted(persons),
    2
  ):
    pair_counts[(person1, person2)] += 1


strong_pairs = Counter({
  pair: count
  for pair, count in pair_counts.items()
  if count >= 2
})

print("\n" + "-" * 60)
print("5. CHARACTER CO-OCCURRENCE")
print("-" * 60)
print(
  "Character pairs detected:",
  len(pair_counts)
)
print(
  "Strong character pairs:",
  len(strong_pairs)
)

print("\nTop Character Connections:")

for (person1, person2), count in strong_pairs.most_common(10):
  print(
    f"  {person1} <-> {person2}"
    f" | Co-occurrence: {count}"
  )


# ============================================
# 6. RELATIONSHIP CONTEXT EXTRACTION
# ============================================

# A literary relationship is often described across several nearby
# sentences rather than in the exact sentence where both names occur.
# Keep the original same-sentence co-occurrence for graph construction,
# but give relationship extraction a small local context window.

sentence_records = []

for index, (sentence, persons) in enumerate(sentence_characters):
  sentence_records.append((index, sentence, persons))

RELATION_CONTEXT_RADIUS = 6

pair_contexts = {
  pair: []
  for pair in strong_pairs
}

for index, sentence, persons in sentence_records:

  for pair in strong_pairs:

    person1, person2 = pair

    if person1 not in persons or person2 not in persons:
      continue

    start_index = max(0, index - RELATION_CONTEXT_RADIUS)
    end_index = min(
      len(sentence_records),
      index + RELATION_CONTEXT_RADIUS + 1
    )

    window_sentences = [
      item[1]
      for item in sentence_records[start_index:end_index]
    ]

    context = " ".join(window_sentences).strip()

    if context and context not in pair_contexts[pair]:
      pair_contexts[pair].append(context)

print("\n" + "-" * 60)
print("6. RELATIONSHIP CONTEXT EXTRACTION")
print("-" * 60)
print(
  "Character pairs analyzed:",
  len(pair_contexts)
)
print(
  "Relationship contexts extracted:",
  sum(
    len(contexts)
    for contexts in pair_contexts.values()
  )
)


# ============================================
# 7. RELATIONSHIP ANALYSIS
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
  "coworker",
  "co-worker",
  "servant",
  "master",
  "solicitor",
  "lawyer",
  "doctor",
}


friend_keywords = {
  "friend",
  "friends",
  "friendship",
  "companion",
  "companions",
  "ally",
  "allies",
  "comrade",
  "comrades",
}


romantic_keywords = {
  "love",
  "lover",
  "lovers",
  "beloved",
  "affection",
  "affectionate",
  "adore",
  "engage",
  "engagement",
  "fiance",
  "fiancee",
  "fiancé",
  "fiancée",
  "betroth",
  "betrothal",
  "courtship",
  "court",
  "courting",
  "sweetheart",
  "suitor",
  "suitors",
  "betrothed",
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
  "hate",
  "hated",
  "hatred",
  "arrest",
  "arrested",
  "revenge",
  "villain",
}


relation_keyword_map = {
  "PROFESSIONAL": professional_keywords,
  "ROMANTIC / LOVE": romantic_keywords,
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
# HELPER: WORDS FROM DEPENDENCY SUBTREE
# --------------------------------------------

def words_from_subtree(token):
  return {
    subtoken.text.lower()
    for subtoken in token.subtree
  }


# --------------------------------------------
# FAMILY RELATIONSHIP DETECTION
# --------------------------------------------

def detect_family_relation(context, pair):
  """
  Detect family relations only when the family word
  has a grammatical connection to both characters.

  This prevents false positives such as:
  "Lucy's mother and Arthur's father..."
  "mother" and "father" are both present, but
  Lucy and Arthur are not directly related there.
  """
  person1, person2 = pair

  # Surface proximity is a recovery layer for literary prose.
  # Dependency parsing remains useful, but it is not the only evidence.
  found_relations = _window_proximity_relations(
    context,
    pair
  )

  parsed_context = nlp(context)

  person1_parts = set(get_name_parts(person1))
  person2_parts = set(get_name_parts(person2))

  def matches(parts, words):
    return any(part in words for part in parts)

  # ==========================================
  # 1. MARRIAGE VERB
  # ==========================================
  for token in parsed_context:

    if token.lemma_.lower() != "marry":
      continue

    subject_words = set()
    object_words = set()

    for child in token.children:

      if child.dep_ in {"nsubj", "nsubjpass"}:
        subject_words.update(words_from_subtree(child))

      elif child.dep_ in {"obj", "dobj"}:
        object_words.update(words_from_subtree(child))

    p1_subject = matches(person1_parts, subject_words)
    p2_subject = matches(person2_parts, subject_words)
    p1_object = matches(person1_parts, object_words)
    p2_object = matches(person2_parts, object_words)

    if (p1_subject and p2_object) or (p2_subject and p1_object):
      return True

    # Pronoun subject + explicit object.
    subject_is_pronoun = any(
      child.dep_ in {"nsubj", "nsubjpass"}
      and child.pos_ == "PRON"
      for child in token.children
    )

    if subject_is_pronoun:

      previous_person = None

      for ent in parsed_context.ents:
        if ent.label_ == "PERSON" and ent.end <= token.i:
          previous_person = ent

      if previous_person is not None:

        previous_name = normalize_surface_name(
          previous_person.text
        )
        previous_words = set(
          get_name_parts(previous_name)
        )

        if (
          previous_words & person1_parts
          and p2_object
        ):
          return True

        if (
          previous_words & person2_parts
          and p1_object
        ):
          return True

  # ==========================================
  # 2. DIRECT FAMILY-NOUN RELATIONS
  # ==========================================
  family_nouns = {
    "wife", "husband", "mother", "father",
    "brother", "sister", "son", "daughter",
    "parent", "parents",
  }

  for token in parsed_context:

    if token.lemma_.lower() not in family_nouns:
      continue

    possessors = [
      child
      for child in token.children
      if child.dep_ == "poss"
    ]

    possessor_words = set()

    for possessor in possessors:
      possessor_words.update(words_from_subtree(possessor))

    p1_possessor = matches(person1_parts, possessor_words)
    p2_possessor = matches(person2_parts, possessor_words)

    # Case A: John is Mary's brother.
    if token.dep_ in {"attr", "oprd"}:

      subject_words = set()

      for child in token.head.children:
        if child.dep_ in {"nsubj", "nsubjpass"}:
          subject_words.update(words_from_subtree(child))

      if p1_possessor and matches(person2_parts, subject_words):
        return True

      if p2_possessor and matches(person1_parts, subject_words):
        return True

      # John and Mary are brothers/sisters.
      if (
        matches(person1_parts, subject_words)
        and matches(person2_parts, subject_words)
      ):
        return True

    # Case B: Mary's brother is John.
    if token.dep_ in {"nsubj", "nsubjpass"}:

      predicate_words = set()

      for child in token.head.children:
        if child.dep_ in {"attr", "oprd"}:
          predicate_words.update(words_from_subtree(child))

      if p1_possessor and matches(person2_parts, predicate_words):
        return True

      if p2_possessor and matches(person1_parts, predicate_words):
        return True

    # Case C: John, Mary's brother, ...
    if token.dep_ == "appos":

      appos_words = words_from_subtree(token.head)

      if p1_possessor and matches(person2_parts, appos_words):
        return True

      if p2_possessor and matches(person1_parts, appos_words):
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
# RELATIONSHIP EVIDENCE DETECTION
# --------------------------------------------


def _pair_aliases(pair):
  """
  Return safe surface forms for the two already-selected characters.

  These forms are used only after co-occurrence has already selected the
  pair.  They are NOT used to create new character candidates.
  """
  aliases = {}

  for canonical in pair:
    values = {normalize_surface_name(canonical)}

    for alias, mapped in alias_map.items():
      if mapped == canonical:
        clean_alias = normalize_surface_name(alias)
        if clean_alias:
          values.add(clean_alias)

    # Literary prose often uses one part of a canonical name.
    # Keep only reasonably distinctive tokens.
    for value in list(values):
      for token in value.split():
        token = re.sub(r"[^A-Za-z]", "", token).lower()
        if len(token) >= 3:
          values.add(token)

    aliases[canonical] = sorted(
      values,
      key=len,
      reverse=True
    )

  return aliases


def _surface_pattern(name):
  """Create a word-boundary-safe pattern for a surface name."""
  words = [
    re.escape(word.lower())
    for word in normalize_surface_name(name).split()
    if word
  ]

  if not words:
    return None

  return r"\b" + r"\s+".join(words) + r"\b"


def _pair_name_patterns(pair):
  aliases = _pair_aliases(pair)
  patterns = {}

  for canonical, values in aliases.items():
    parts = []

    for value in values:
      pattern = _surface_pattern(value)
      if pattern:
        parts.append(pattern)

    patterns[canonical] = r"(?:" + "|".join(parts) + r")"

  return patterns


def _explicit_surface_relations(context, pair):
  """
  Detect relationship constructions from the surface form of the sentence.

  This is deliberately broader than the previous version, but it is NOT
  simple "keyword anywhere in the sentence" matching.  Every pattern below
  requires both members of the selected pair to participate in the same
  grammatical/relational frame.
  """
  patterns = _pair_name_patterns(pair)
  p1 = patterns[pair[0]]
  p2 = patterns[pair[1]]

  text = normalize_surface_name(context)
  text = re.sub(r"\s+", " ", text).strip()
  lower = text.lower()

  found = {
    "FAMILY": [],
    "ROMANTIC / LOVE": [],
    "FRIEND / COMPANION": [],
    "CONFLICT / ENEMY": [],
    "PROFESSIONAL": [],
  }

  pair_orders = [
    (p1, p2),
    (p2, p1),
  ]

  relation_patterns = {
    # ----------------------------------------
    # FAMILY
    # ----------------------------------------
    "FAMILY": [
      (r"{a}\s+(?:was|is|were|are)\s+"
       r"{b}'s\s+(?:wife|husband|mother|father|brother|sister|"
       r"son|daughter|parent|parents)", "family"),
      (r"{b}\s+(?:was|is|were|are)\s+"
       r"{a}'s\s+(?:wife|husband|mother|father|brother|sister|"
       r"son|daughter|parent|parents)", "family"),
      (r"{a}\s+(?:married|marries|marry)\s+{b}", "married"),
      (r"{a}\s+(?:was|is|were|are)\s+married\s+to\s+{b}", "married"),
      (r"{a}\s+and\s+{b}\s+(?:were|are|was|is)\s+married", "married"),
    ],

    # ----------------------------------------
    # ROMANTIC / LOVE
    # ----------------------------------------
    "ROMANTIC / LOVE": [
      (r"{a}\s+(?:was|is|were|are)\s+{b}'s\s+"
       r"(?:fianc[eé]e?|fiancé|fiancee|lover|sweetheart|beloved)", "romantic"),
      (r"{a}\s*,?\s+{b}'s\s+"
       r"(?:fianc[eé]e?|fiancé|fiancee|lover|sweetheart|beloved)", "romantic"),
      (r"{a}\s+(?:was|is|were|are)\s+(?:the|a|an)?\s*"
       r"(?:fianc[eé]e?|fiancé|fiancee|lover|sweetheart|beloved)\s+of\s+{b}", "romantic"),
      (r"{a}\s+(?:was|is|were|are)\s+{b}'s\s+(?:suitor|betrothed)", "suitor"),
      (r"{a}\s+(?:was|is|were|are)\s+(?:a|the|one\s+of\s+the)?\s*"
       r"(?:suitor|lover)s?\s+of\s+{b}", "suitor"),
      (r"{a}\s*,?\s+one\s+of\s+{b}'s\s+(?:suitors|lovers)", "suitor"),
      (r"{a}\s+(?:was|is|were|are)\s+(?:engaged|betrothed)\s+to\s+{b}", "engaged"),
      (r"{a}\s+and\s+{b}\s+(?:were|are|was|is)\s+(?:engaged|betrothed)", "engaged"),
      (r"{a}\s+(?:loved|loves|adored|courted|wooed)\s+{b}", "love"),
      (r"{a}\s+(?:was|is|were|are)\s+in\s+love\s+with\s+{b}", "love"),
      (r"{a}\s+and\s+{b}\s+(?:loved|love|adored)\s+each\s+other", "love"),
    ],

    # ----------------------------------------
    # FRIEND / COMPANION
    # ----------------------------------------
    "FRIEND / COMPANION": [
      (r"{a}\s+(?:was|is|were|are)\s+{b}'s\s+"
       r"(?:friend|companion|ally|comrade)", "friend"),
      (r"{a}\s*,?\s+{b}'s\s+"
       r"(?:friend|companion|ally|comrade)", "friend"),
      (r"{a}\s+(?:was|is|were|are)\s+(?:a|an|the)?\s*"
       r"(?:friend|companion|ally|comrade)\s+of\s+{b}", "friend"),
      (r"{a}\s+and\s+{b}\s+(?:were|are|was|is)\s+"
       r"(?:friends|companions|allies|comrades)", "friend"),
      (r"{a}\s+and\s+(?:my|our|his|her|their)?\s*"
       r"(?:friend|companion|ally|comrade)\s+{b}", "friend"),
      (r"{b}\s+and\s+(?:my|our|his|her|their)?\s*"
       r"(?:friend|companion|ally|comrade)\s+{a}", "friend"),
      (r"{a}\s*,?\s+(?:a|an|the)?\s*"
       r"(?:friend|companion|ally|comrade)\s+of\s+{b}", "friend"),
    ],

    # ----------------------------------------
    # PROFESSIONAL
    # ----------------------------------------
    "PROFESSIONAL": [
      (r"{a}\s+(?:was|is|were|are)\s+{b}'s\s+"
       r"(?:partner|client|colleague|assistant|employee|employer|"
       r"servant|master|doctor|lawyer|solicitor)", "professional"),
      (r"{a}\s*,?\s+{b}'s\s+"
       r"(?:partner|client|colleague|assistant|employee|employer|"
       r"servant|master|doctor|lawyer|solicitor)", "professional"),
      (r"{a}\s+(?:was|is|were|are|became)\s+(?:the|a|an)?\s*"
       r"(?:partner|client|colleague|assistant|employee|employer|"
       r"servant|master|doctor|lawyer|solicitor)\s+of\s+{b}", "professional"),
      (r"{a}\s+and\s+{b}\s+(?:were|are|was|is)\s+"
       r"(?:colleagues|partners|coworkers|co-workers)", "professional"),
      (r"{a}\s+(?:worked|works)\s+(?:with|for)\s+{b}", "professional"),
      (r"{a}\s+(?:was|is|were|are)\s+the\s+"
       r"(?:doctor|solicitor|lawyer|assistant)\s+of\s+{b}", "professional"),
    ],

    # ----------------------------------------
    # CONFLICT / ENEMY
    # ----------------------------------------
    "CONFLICT / ENEMY": [
      (r"{a}\s+(?:was|is|were|are)\s+{b}'s\s+"
       r"(?:enemy|rival|opponent|adversary|captor)", "enemy"),
      (r"{a}\s+(?:was|is|were|are)\s+(?:the|an|a)?\s*"
       r"(?:enemy|rival|opponent|adversary|captor)\s+of\s+{b}", "enemy"),
      (r"{a}\s+(?:attacked|attacks|threatened|threatens|hated|hates|"
       r"fought|fights|opposed|opposes|arrested|imprisoned|captured|abducted)\s+{b}", "conflict"),
      (r"{a}\s+(?:held|holds)\s+{b}\s+(?:captive|prisoner)", "captor"),
    ],
  }

  for relation, templates in relation_patterns.items():
    for a, b in pair_orders:
      for template, keyword in templates:
        pattern = template.format(a=a, b=b)
        if re.search(pattern, lower):
          if keyword not in found[relation]:
            found[relation].append(keyword)

  return found


def _surface_positions(text, forms):
  """Return token positions for any safe surface form."""
  tokens = re.findall(r"[A-Za-zÀ-ÿ]+(?:['’][A-Za-zÀ-ÿ]+)?", text.lower())
  positions = []

  form_tokens = []
  for form in forms:
    parts = re.findall(r"[A-Za-z]+", form.lower())
    if parts:
      form_tokens.append(parts)

  for form in form_tokens:
    size = len(form)
    for i in range(len(tokens) - size + 1):
      if tokens[i:i + size] == form:
        positions.append((i, i + size - 1))

  return positions


def _window_proximity_relations(context, pair):
  """
  Recover high-value literary relations when the exact dependency
  structure is missed by spaCy.

  A keyword must be close to BOTH selected characters.  This is
  intentionally different from broad keyword proximity, so a sentence
  such as "my friend Quincey ... Miss Lucy" does not become a Lucy-
  Quincey friendship merely because the word friend occurs nearby.
  """
  aliases = _pair_aliases(pair)
  text = context.lower()
  tokens = re.findall(r"[A-Za-zÀ-ÿ]+(?:['’][A-Za-zÀ-ÿ]+)?", text)

  p1_positions = _surface_positions(text, aliases[pair[0]])
  p2_positions = _surface_positions(text, aliases[pair[1]])

  if not p1_positions or not p2_positions:
    return {
      "FAMILY": [],
      "PROFESSIONAL": [],
      "ROMANTIC / LOVE": [],
      "FRIEND / COMPANION": [],
      "CONFLICT / ENEMY": [],
    }

  keyword_map = {
    # Family relations stay dependency-based; broad window proximity for
    # words such as "mother" or "father" creates false family links.
    "PROFESSIONAL": {
      "partner", "partners", "client", "clients", "colleague",
      "colleagues", "assistant", "employee", "employer", "servant",
      "master", "solicitor", "lawyer", "doctor", "worked", "work"
    },
    "ROMANTIC / LOVE": {
      # Do not treat a bare status word such as "married" or "engaged"
      # as pair evidence.  Those words can describe a third person:
      # "Irene Adler is married," remarked Holmes.
      # Explicit pair constructions are handled by _explicit_surface_relations
      # and the narrow literary bridge below.
      "love", "loved", "lover", "lovers", "beloved", "affection",
      "adore", "adored", "suitor", "suitors",
      "courted", "courtship",
      "sweetheart"
    },
    "FRIEND / COMPANION": {
      "friend", "friends", "friendship", "companion", "companions",
      "ally", "allies", "comrade", "comrades", "pal", "pals",
      "fellow", "fellows"
    },
    "CONFLICT / ENEMY": {
      "enemy", "enemies", "rival", "rivals", "opponent", "opponents",
      "attack", "attacked", "attacks", "threat", "threatened",
      "fight", "fought", "fighting", "hated", "hate", "captor",
      "captive", "prisoner", "imprisoned", "captured", "abducted"
    },
  }

  found = {relation: [] for relation in keyword_map}

  keyword_positions = {}
  for relation, keywords in keyword_map.items():
    for keyword in keywords:
      keyword_positions.setdefault(relation, []).extend(
        [i for i, token in enumerate(tokens) if token == keyword]
      )

  for relation, positions in keyword_positions.items():
    for keyword_index in positions:

      nearest_p1 = min(
        min(abs(keyword_index - start), abs(keyword_index - end))
        for start, end in p1_positions
      )
      nearest_p2 = min(
        min(abs(keyword_index - start), abs(keyword_index - end))
        for start, end in p2_positions
      )

      # Strong local evidence: the relation word is close to both
      # members of the already-selected pair. Romantic descriptions in
      # literary prose can span a few sentences, so allow a wider window
      # for explicit romantic vocabulary.
      proximity_limit = 14 if relation == "ROMANTIC / LOVE" else 10

      if nearest_p1 <= proximity_limit and nearest_p2 <= proximity_limit:
        keyword = tokens[keyword_index]

        # "my friend Quincey ... Lucy" is not enough. If the
        # relation word is immediately tied to only one person and
        # the other person is outside the local phrase, reject it.
        if keyword in {"friend", "friends", "companion", "pal", "fellow"}:
          local_left = max(0, keyword_index - 4)
          local_right = min(len(tokens), keyword_index + 5)
          local_tokens = set(tokens[local_left:local_right])

          p1_local = any(
            start <= keyword_index + 4 and end >= keyword_index - 4
            for start, end in p1_positions
          )
          p2_local = any(
            start <= keyword_index + 4 and end >= keyword_index - 4
            for start, end in p2_positions
          )

          if not (p1_local and p2_local):
            # Do not rescue the evidence from a broad context.  A
            # construction such as "friend Jonathan ... Count Dracula"
            # must not become a Dracula-Jonathan friendship unless BOTH
            # selected characters occur in the same local friend phrase.
            continue

        if keyword not in found[relation]:
          found[relation].append(keyword)

  return found


def detect_keyword_relation(context, pair):
  """
  Hybrid relationship detector.

  Order of evidence:
    1. Explicit surface constructions.
    2. Dependency evidence for relationship words.
    3. Subject/object evidence for direct romantic/conflict verbs.

  A bare keyword such as "friend" or "love" anywhere in a sentence is
  never enough.  The proximity layer first requires the keyword to be
  close to both selected characters; dependency/surface evidence can then
  strengthen the result.  This is the important protection against false
  positives.
  """
  person1, person2 = pair

  # Initialize all relationship buckets before adding evidence.
  # Start with the literary-proximity layer.  The previous build defined
  # this detector but never merged its result, which caused valid
  # relationship keywords to disappear before classification.
  found_relations = _window_proximity_relations(
    context,
    pair
  )

  parsed_context = nlp(context)

  person1_parts = {
    part
    for part in get_name_parts(person1)
    if len(part) >= 3
    and part not in {"van", "von", "der", "den"}
  }

  person2_parts = {
    part
    for part in get_name_parts(person2)
    if len(part) >= 3
    and part not in {"van", "von", "der", "den"}
  }

  # ----------------------------------------
  # 1. Explicit surface evidence
  # ----------------------------------------
  surface_relations = _explicit_surface_relations(
    context,
    pair
  )

  for relation, keywords in surface_relations.items():
    for keyword in keywords:
      if keyword not in found_relations[relation]:
        found_relations[relation].append(keyword)

  # ----------------------------------------
  # 2. Dependency evidence
  # ----------------------------------------
  for token in parsed_context:
    word = token.lemma_.lower()
    matched_relation = None

    for relation, keywords in relation_keyword_map.items():
      if word in keywords:
        matched_relation = relation
        break

    if matched_relation is None:
      continue

    # Descriptive modifiers/appositives are not relationship evidence by
    # themselves.  The surface detector above handles genuine constructions.
    if token.dep_ in {"amod", "compound", "nmod", "appos"}:
      continue

    keyword_subtree = words_from_subtree(token)

    p1_in_subtree = bool(keyword_subtree & person1_parts)
    p2_in_subtree = bool(keyword_subtree & person2_parts)

    if p1_in_subtree and p2_in_subtree:
      if word not in found_relations[matched_relation]:
        found_relations[matched_relation].append(word)
      continue

    # Copular relation: "Arthur and Lucy are friends."
    head = token.head
    if token.dep_ in {"attr", "acomp", "oprd"}:
      subject_words = set()
      for child in head.children:
        if child.dep_ in {"nsubj", "nsubjpass", "csubj"}:
          subject_words.update(words_from_subtree(child))

      if (
        bool(subject_words & person1_parts)
        and bool(subject_words & person2_parts)
      ):
        if word not in found_relations[matched_relation]:
          found_relations[matched_relation].append(word)
        continue

  # ----------------------------------------
  # 3. Direct relationship verbs
  # ----------------------------------------
  verbal_relations = {
    "ROMANTIC / LOVE": {
      "love", "adore", "court", "woo", "betroth", "engage",
      "propose"
    },
    "CONFLICT / ENEMY": {
      "hate", "attack", "fight", "threaten", "arrest", "oppose",
      "capture", "imprison", "abduct"
    },
  }

  for token in parsed_context:
    lemma = token.lemma_.lower()
    matched_relation = None

    for relation, verbs in verbal_relations.items():
      if lemma in verbs:
        matched_relation = relation
        break

    if matched_relation is None:
      continue

    subject_words = set()
    object_words = set()

    for child in token.children:
      if child.dep_ in {"nsubj", "nsubjpass", "csubj"}:
        subject_words.update(words_from_subtree(child))
      elif child.dep_ in {"obj", "dobj", "iobj", "obl", "pobj"}:
        object_words.update(words_from_subtree(child))

    p1_subject = bool(subject_words & person1_parts)
    p2_subject = bool(subject_words & person2_parts)
    p1_object = bool(object_words & person1_parts)
    p2_object = bool(object_words & person2_parts)

    if (p1_subject and p2_object) or (p2_subject and p1_object):
      if lemma not in found_relations[matched_relation]:
        found_relations[matched_relation].append(lemma)

  return found_relations


# --------------------------------------------
# RELATION-SAFE TEXT UNITS
# --------------------------------------------

def _relation_units(context):
  """Split relationship context into small, punctuation-aware units."""
  text = re.sub(r"\s+", " ", context).strip()

  if not text:
    return []

  # Preserve spaCy's good handling of abbreviations such as Mr./Dr.,
  # then split only the formatting artifacts that make a Gutenberg/OCR
  # block swallow multiple independent sentences.
  doc = nlp(text)
  raw_units = [
    sent.text.strip()
    for sent in doc.sents
    if sent.text.strip()
  ]

  cleaned = []
  abbreviations = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "vs", "etc", "fig", "no", "pg"
  }

  for raw in raw_units:
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = re.sub(r"\s*\*\s*(?:\*\s*){2,}", ". ", raw)

    if not raw:
      continue

    # Split terminal punctuation followed by a clear new sentence, while
    # protecting common title/abbreviation forms.
    pieces = []
    buffer = []
    chars = list(raw)

    for index, char in enumerate(chars):
      buffer.append(char)

      if char not in ".!?":
        continue

      remainder = "".join(chars[index + 1:])
      match = re.match(r"\s+([\"'“‘A-Z0-9])", remainder)

      if not match:
        continue

      before = "".join(buffer[:-1]).rstrip()
      previous_word_match = re.search(r"([A-Za-z]{1,12})$", before)
      previous_word = (
        previous_word_match.group(1).lower()
        if previous_word_match
        else ""
      )

      if char == "." and previous_word in abbreviations:
        continue

      piece = "".join(buffer).strip()
      if piece:
        pieces.append(piece)
      buffer = []

    tail = "".join(buffer).strip()
    if tail:
      pieces.append(tail)

    for piece in pieces:
      piece = re.sub(r"\s+", " ", piece).strip("-–— ")
      if piece:
        cleaned.append(piece)

  return cleaned


def _pair_appears(text, name, aliases=None):
  """Check whether a canonical character or alias occurs in text."""
  if aliases is None:
    aliases = _pair_aliases((name,))[name]

  lower = text.lower()

  for alias in aliases:
    pattern = _surface_pattern(alias)
    if pattern and re.search(pattern, lower):
      return True

  return False


def _family_evidence_units(context, pair):
  """Return small evidence units containing the relevant family wording."""
  units = _relation_units(context)
  aliases = _pair_aliases(pair)
  results = []

  for unit in units:
    lower = unit.lower()
    words = set(re.findall(r"[a-z]+", lower))

    if not (words & family_keywords):
      continue

    p1_present = _pair_appears(unit, pair[0], aliases[pair[0]])
    p2_present = _pair_appears(unit, pair[1], aliases[pair[1]])

    # Prefer units that explicitly mention both pair members.
    if p1_present and p2_present:
      results.append(unit)
      continue

    # Literary family statements often use a pronoun/possessive for the
    # second member, e.g. "Dr. Roylott ... married my mother".
    if p1_present or p2_present:
      results.append(unit)

  return results[:3]


def _bridged_relation_evidence(context, pair):
  """Recover narrow literary relations expressed with pronouns/nominals."""
  units = _relation_units(context)

  if len(units) < 2:
    return []

  aliases = _pair_aliases(pair)
  results = []

  romantic_patterns = [
    r"\b(?:you|he|she|they)\s+(?:are|were|is|was)\s+"
    r"(?:the|a|an)?\s*lover\s+of\s+(?:our|the|that|this)?\s*"
    r"(?:dear\s+)?(?:miss|girl|woman|lady)\b",
    r"\b(?:i|he|she|they)\s+"
    r"(?:loved|adored|courted)\s+(?:that|the|this)?\s*"
    r"(?:girl|woman|lady|miss)\b",
    r"\b(?:i|he|she|they)\s+"
    r"(?:wanted|wished)\s+to\s+marry\s+(?:her|him|them)\b",
    r"\bwe\s+(?:are|were)\s+to\s+be\s+married\b",
  ]

  # A bare construction such as "my friend Quincey" identifies the
  # speaker's relationship with Quincey, not necessarily Quincey's
  # relationship with every other character mentioned nearby.  Therefore
  # friendship is NOT inferred from a pronoun/nominal bridge.  Explicit
  # pair constructions are already handled by _explicit_surface_relations,
  # and repeated direct interaction is handled by the final classifier.

  for index, unit in enumerate(units):
    lower = unit.lower()

    is_romantic = any(
      re.search(pattern, lower)
      for pattern in romantic_patterns
    )

    if not is_romantic:
      continue

    # Literary romantic statements can use pronouns instead of repeating
    # both names. Keep the bridge local: only a few nearby units are
    # allowed to establish the pair.  The romantic wording itself is
    # highly specific, so this wider local span does not turn ordinary
    # mentions into romantic evidence.
    nearby = (
      units[max(0, index - 4):index] +
      units[index + 1:index + 5]
    )

    # The relation unit itself must contain at least one member of the
    # selected pair.  Otherwise a relation about a completely different
    # pair can leak into this pair merely because the evidence units happen
    # to be nearby.
    p1_in_unit = _pair_appears(
      unit, pair[0], aliases[pair[0]]
    )
    p2_in_unit = _pair_appears(
      unit, pair[1], aliases[pair[1]]
    )

    if not (p1_in_unit or p2_in_unit):
      continue

    p1_nearby = p1_in_unit or any(
      _pair_appears(item, pair[0], aliases[pair[0]])
      for item in nearby
    )
    p2_nearby = p2_in_unit or any(
      _pair_appears(item, pair[1], aliases[pair[1]])
      for item in nearby
    )

    if not (p1_nearby and p2_nearby):
      continue

    results.append({
      "context": unit,
      "keywords": ["love"],
      "relation": "ROMANTIC / LOVE"
    })

  # ----------------------------------------
  # Narrow friendship bridge
  # ----------------------------------------
  # Literary prose often says "my friend Quincey" or "friend Jonathan"
  # without repeating the other member of the relationship in the same
  # sentence.  Unlike the old broad keyword window, accept this only when
  # the named friend is one member of the selected pair AND the other
  # member occurs in the same or an immediately adjacent unit.
  friend_patterns = [
    r"\b(?:my|our|his|her|their)?\s*"
    r"(?:friend|companion|ally|comrade)\s+{name}\b",
    r"\b{name}\s*,?\s+(?:my|our|his|her|their)?\s*"
    r"(?:friend|companion|ally|comrade)\b",
  ]

  for index, unit in enumerate(units):
    lower = unit.lower()

    matched_friend = None
    for canonical in pair:
      for template in friend_patterns:
        pattern = template.format(
          name=_pair_name_patterns((canonical,))[canonical]
        )
        if re.search(pattern, lower):
          matched_friend = canonical
          break
      if matched_friend is not None:
        break

    if matched_friend is None:
      continue

    other = pair[1] if matched_friend == pair[0] else pair[0]
    nearby_friend_units = (
      units[max(0, index - 1):index] +
      units[index + 1:index + 2]
    )

    if (
      _pair_appears(unit, other, aliases[other])
      or any(
        _pair_appears(item, other, aliases[other])
        for item in nearby_friend_units
      )
    ):
      results.append({
        "context": unit,
        "keywords": ["friend"],
        "relation": "FRIEND / COMPANION"
      })

  # ----------------------------------------
  # Narrow adversarial/prisoner bridge
  # ----------------------------------------
  # Captivity is a strong relationship cue in literary narratives, but the
  # word "prisoner" can describe someone other than the selected pair.
  # Require one pair member to be explicitly tied to the captivity wording
  # and the other member to occur in the same or an immediately adjacent
  # unit.
  conflict_patterns = [
    r"\b(?:i|he|she|they|you)\s+(?:am|was|were|is|are)\s+"
    r"(?:a|the)?\s*(?:prisoner|captive)\b",
    r"\b(?:held|kept)\s+{name}\s+(?:captive|prisoner)\b",
    r"\b{name}\s+(?:was|is|were|are)\s+"
    r"(?:a|the)?\s*(?:prisoner|captive)\b",
  ]

  for index, unit in enumerate(units):
    lower = unit.lower()
    matched_member = None

    for canonical in pair:
      name_pattern = _pair_name_patterns((canonical,))[canonical]
      patterns = [
        template.format(name=name_pattern)
        for template in conflict_patterns
      ]

      if any(re.search(pattern, lower) for pattern in patterns):
        matched_member = canonical
        break

    if matched_member is None:
      continue

    other = pair[1] if matched_member == pair[0] else pair[0]
    nearby_conflict_units = (
      units[max(0, index - 1):index] +
      units[index + 1:index + 2]
    )

    if (
      _pair_appears(unit, other, aliases[other])
      or any(
        _pair_appears(item, other, aliases[other])
        for item in nearby_conflict_units
      )
    ):
      results.append({
        "context": unit,
        "keywords": ["prisoner"],
        "relation": "CONFLICT / ENEMY"
      })

  return results


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
    # Family relations may be described across nearby sentences, so
    # preserve the existing context-level family detector.

    if detect_family_relation(
      context,
      pair
    ):

      evidence_units = _family_evidence_units(
        context,
        pair
      )

      if evidence_units:
        for evidence_unit in evidence_units:
          family_evidence.append({
            "context": evidence_unit,
            "keywords": [
              token.lemma_.lower()
              for token in nlp(evidence_unit)
              if (
                token.lemma_.lower()
                in family_keywords
              )
            ]
          })
      else:
        # Keep a safe fallback if a literary construction spans units.
        family_evidence.append({
          "context": re.sub(r"\s+", " ", context).strip(),
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
    # NON-FAMILY RELATIONSHIP EVIDENCE
    # --------------------------------------
    # Do NOT run keyword detection over the complete 13-sentence
    # context. A keyword can otherwise belong to a third character.
    # Example: Jonathan + Lucy can share a context containing the word
    # "clients", even though "clients" describes Jonathan + Hawkins.
    # Evaluate non-family evidence at sentence level instead.

    context_sentences = _relation_units(context)

    for evidence_sentence in context_sentences:

      # ------------------------------------
      # DIRECT INTERACTION
      # ------------------------------------

      if detect_direct_interaction(
        evidence_sentence,
        pair
      ):

        interaction_evidence.append({
          "context": evidence_sentence,
          "keywords": []
        })


      # ------------------------------------
      # OTHER RELATIONSHIP KEYWORDS
      # ------------------------------------

      found_relations = detect_keyword_relation(
        evidence_sentence,
        pair
      )

      for relation, keywords in found_relations.items():

        if not keywords:
          continue

        if relation == "FAMILY":
          family_evidence.append({
            "context": evidence_sentence,
            "keywords": keywords
          })
        else:
          keyword_evidence.append({
            "context": evidence_sentence,
            "keywords": keywords,
            "relation": relation
          })

    # Narrow pronoun/nominal bridge.  This is intentionally limited to
    # explicit romantic constructions and nearby named characters.
    for bridge in _bridged_relation_evidence(context, pair):
      keyword_evidence.append({
        "context": bridge["context"],
        "keywords": bridge["keywords"],
        "relation": bridge["relation"]
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
    relation: 0
    for relation in relation_keyword_map
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
# 7. FINAL CHARACTER RELATIONSHIP ANALYSIS
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

      context = re.sub(r"\s+", " ", context).strip()

      if len(context) > 360:
        context = context[:357].rstrip() + "..."

      print(
        " -",
        context
      )
  else:
    print(
      "Evidence: No reliable relationship evidence found."
    )


# ============================================
# 8. RELATIONSHIP SUMMARY
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

# 9. CREATE CHARACTER NETWORK
# ============================================

graph = nx.Graph()

for (person1, person2), count in strong_pairs.items():

  graph.add_edge(
    person1,
    person2,
    weight=count
  )


print("\n" + "-" * 60)
print("9. CHARACTER NETWORK")
print("-" * 60)

print(
  "Graph Nodes:",
  graph.number_of_nodes()
)

print(
  "Graph Edges:",
  graph.number_of_edges()
)

print("\nStrong Character Connections:")

for (person1, person2), count in (
  strong_pairs.most_common(10)
):

  print(
    f"  {person1} <-> {person2}"
    f" | Weight: {count}"
  )


# ============================================
# 10. CHARACTER VISUALIZATION
# ============================================

print("\n" + "-" * 60)
print("10. CHARACTER VISUALIZATION")
print("-" * 60)

if graph.number_of_edges() > 0:

  # Keep the complete graph for analysis.
  # Show only the strongest edges in the figure so
  # dense novels remain readable.
  max_visual_edges = 25

  visual_edges = sorted(
    graph.edges(data=True),
    key=lambda edge: (
      edge[2]["weight"],
      edge[0],
      edge[1]
    ),
    reverse=True
  )[:max_visual_edges]

  visual_graph = nx.Graph()

  for person1, person2, data in visual_edges:
    visual_graph.add_edge(
      person1,
      person2,
      weight=data["weight"]
    )

  weighted_degree = dict(
    visual_graph.degree(weight="weight")
  )

  node_sizes = [
    1400 + min(
      weighted_degree.get(node, 0) * 90,
      2200
    )
    for node in visual_graph.nodes()
  ]

  edge_widths = [
    1.0 + 0.45 * visual_graph[person1][person2]["weight"]
    for person1, person2 in visual_graph.edges()
  ]

  plt.figure(figsize=(16, 12))

  pos = nx.spring_layout(
    visual_graph,
    k=2.0,
    iterations=300,
    seed=42
  )

  nx.draw_networkx_nodes(
    visual_graph,
    pos,
    node_size=node_sizes
  )

  nx.draw_networkx_edges(
    visual_graph,
    pos,
    width=edge_widths,
    alpha=0.65
  )

  nx.draw_networkx_labels(
    visual_graph,
    pos,
    font_size=9
  )

  plt.title(
    "LitMind - Character Co-occurrence Graph",
    fontsize=16
  )

  plt.axis("off")
  plt.tight_layout()
  plt.show()
  plt.close()

  print(
    "Visualization edges:",
    visual_graph.number_of_edges()
  )
  print(
    "Full analysis edges:",
    graph.number_of_edges()
  )

else:
  print("No character network could be visualized.")


# ============================================
# 11. NETWORK ANALYSIS
# ============================================

print("\n" + "-" * 60)
print("11. NETWORK ANALYSIS")
print("-" * 60)

if graph.number_of_nodes() > 0:

  degree_centrality = nx.degree_centrality(
    graph
  )

  print("\nTop Character Centrality:")

  for name, score in sorted(
    degree_centrality.items(),
    key=lambda item: item[1],
    reverse=True
  )[:5]:

    print(
      f"  {name:<25} -> {score:.3f}"
    )

  weighted_degrees = dict(
    graph.degree(weight="weight")
  )

  print("\nTop Weighted Connections:")

  for name, score in sorted(
    weighted_degrees.items(),
    key=lambda item: item[1],
    reverse=True
  )[:5]:

    print(
      f"  {name:<25} -> {score}"
    )

else:
  print("No character network could be generated.")


print("\n" + "=" * 60)
print("          LITMIND ANALYSIS COMPLETE")
print("=" * 60)


# ============================================
# OPTIONAL: ADDITIONAL NETWORK ANALYSIS
# ============================================

# The following advanced metrics can be enabled later
# for the final dashboard/PPT if required.
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
# # OPTIONAL: WEIGHTED CHARACTER DEGREE
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
# # OPTIONAL: DEGREE CENTRALITY
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
# # OPTIONAL: BETWEENNESS CENTRALITY
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
