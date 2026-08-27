from pathlib import Path
from collections import Counter
import spacy


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

titles = {
  "Mr",
  "Mrs",
  "Miss",
  "Ms",
  "Dr",
  "Lady",
  "Sir",
}


# ============================================
# 5. DETECT NAME VARIANTS
# ============================================

def is_name_variant(name1, name2):

  words1 = name1.split()
  words2 = name2.split()

  # Remove titles before comparison
  words1_without_title = [
    word for word in words1
    if word not in titles
  ]

  words2_without_title = [
    word for word in words2
    if word not in titles
  ]

  if len(words1_without_title) >= len(words2_without_title):
    full_name = words1_without_title
    short_name = words2_without_title
  else:
    full_name = words2_without_title
    short_name = words1_without_title

  return all(word in full_name for word in short_name)


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

    if short_name not in name_variants:
      name_variants[short_name] = []

    name_variants[short_name].append(full_name)


# ============================================
# 6. SELECT STRONG ALIAS CANDIDATES
# ============================================


def has_title(name):
  first_word = name.split()[0]
  return first_word in titles

alias_map = {}

for short_name, full_names in name_variants.items():

  # Prefer names without titles
  non_title_names = [
    name for name in full_names
    if not has_title(name)
  ]

  # If a non-title full name exists, prefer it
  if non_title_names:

    best_full_name = max(
      non_title_names,
      key=lambda name: character[name]
    )

  else:

    best_full_name = max(
      full_names,
      key=lambda name: character[name]
    )

  # Only create an alias when the selected
  # canonical name has enough evidence
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

canonical_lookup = {}

# Canonical names map to themselves
for name in character:
  canonical_lookup[name] = name

# Aliases map to canonical names
for alias, canonical in alias_map.items():
  canonical_lookup[alias] = canonical


# ============================================
# 8. MERGE CHARACTER FREQUENCIES
# ============================================

merged_frequency = Counter()

for ent in doc.ents:

  if ent.label_ != "PERSON":
    continue

  canonical = canonical_lookup.get(ent.text)

  if canonical:
    merged_frequency[canonical] += 1


# ============================================
# 9. MERGE CHARACTER CONTEXTS
# ============================================

merged_contexts = {}

for ent in doc.ents:

  if ent.label_ != "PERSON":
    continue

  canonical = canonical_lookup.get(ent.text)

  if not canonical:
    continue

  if canonical not in merged_contexts:
    merged_contexts[canonical] = []

  if len(merged_contexts[canonical]) < 5:
    if ent.sent.text not in merged_contexts[canonical]:
      merged_contexts[canonical].append(ent.sent.text)


# ============================================
# 10. MERGE CONTEXT SCORES
# ============================================

merged_context_scores = Counter()

for ent in doc.ents:

  if ent.label_ != "PERSON":
    continue

  canonical = canonical_lookup.get(ent.text)

  if not canonical:
    continue

  for token in ent.sent:

    if token.lemma_.lower() in character_verbs:
      merged_context_scores[canonical] += 1
      break


# ============================================
# 11. CREATE FINAL CHARACTER PROFILES
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
  context_count = merged_context_scores.get(canonical, 0)

  if context_count >= 5:
    context_score = 3
  elif context_count >= 3:
    context_score = 2
  elif context_count >= 1:
    context_score = 1
  else:
    context_score = 0

  # Find aliases
  aliases = [
    alias
    for alias, name in alias_map.items()
    if name == canonical
  ]

  total_score = frequency_score + context_score

  character_profiles[canonical] = {
    "frequency": frequency,
    "aliases": aliases,
    "contexts": merged_contexts.get(canonical, []),
    "frequency_score": frequency_score,
    "context_score": context_score,
    "total_score": total_score,
  }


# ============================================
# 12. RANK CHARACTERS
# ============================================

ranked_characters = sorted(
  character_profiles.items(),
  key=lambda x: x[1]["total_score"],
  reverse=True
)


# ============================================
# 13. FINAL OUTPUT
# ============================================

# print("\nTop Character Candidates:\n")

# for rank, (name, profile) in enumerate(
#   ranked_characters[:20],
#   start=1
# ):
#   print(
#     f"{rank}. {name}"
#     f" | Score: {profile['total_score']}"
#     f" | Frequency: {profile['frequency']}"
#     f" | Aliases: {profile['aliases']}"
#   )


# ============================================
# 14. SELECT VALID CHARACTER NODES
# ============================================

valid_characters = set()

# for name, profile in character_profiles.items():

  # Character should have either:
  # strong frequency evidence
  # OR strong contextual evidence


# print("\nValid Characters:")

# for name in sorted(valid_characters):
#   print(name)

#

sentence_characters = []

for sent in doc.sents:

  persons = set()

  for ent in sent.ents:
    if ent.label_ != "PERSON":
      continue

    canonical = canonical_lookup.get(ent.text)

    if canonical and canonical in valid_characters:
      persons.add(canonical)

  if len(persons) >= 2:
    sentence_characters.append(persons)

print("\nCharacter pairs by sentence:")

for persons in sentence_characters[:20]:
  print(persons)

from itertools import combinations

pair_counts = Counter()

for persons in sentence_characters:

  canonical_persons = set()

  for person in persons:
    canonical = canonical_lookup.get(person)

    if canonical:
      canonical_persons.add(canonical)

  for person1, person2 in combinations(
    sorted(canonical_persons), 2
  ):
    pair_counts[(person1, person2)] += 1


print("\nTop Character Pairs:")

for (person1, person2), count in pair_counts.most_common(20):
  print(
    person1,
    "<->",
    person2,
    "->",
    count
  )