from pathlib import Path
from collections import Counter
import spacy


# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


# Path of processed text
text_path = Path("data/processed/Sherlock_Holmes.txt")


# Read text file
with open(text_path, "r", encoding="utf-8") as file:
  text = file.read()


# Process the complete text
doc = nlp(text)


# --------------------------------
# 1. Count PERSON entities
# --------------------------------

character = Counter()

for ent in doc.ents:
  if ent.label_ == "PERSON":
    character[ent.text] += 1


# --------------------------------
# 2. Store 3 example contexts
#    for each PERSON
# --------------------------------

character_context = {}

for ent in doc.ents:
  if ent.label_ == "PERSON":

    if ent.text not in character_context:
      character_context[ent.text] = []

    if len(character_context[ent.text]) < 3:
      character_context[ent.text].append(ent.sent.text)


# --------------------------------
# 3. Count character-like contexts
# --------------------------------

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
  if ent.label_ == "PERSON":

    for token in ent.sent:
      if token.lemma_.lower() in character_verbs:
        context_scores[ent.text] += 1
        break


# --------------------------------
# 4. Create character profiles
# --------------------------------

character_profiles = {}

for name, count in character.most_common():

  # Frequency score
  if count >= 50:
    frequency_score = 3
  elif count >= 10:
    frequency_score = 2
  elif count >= 3:
    frequency_score = 1
  else:
    frequency_score = 0

  # Context score
  context_count = context_scores.get(name, 0)

  if context_count >= 50:
    context_score = 3
  elif context_count >= 10:
    context_score = 2
  elif context_count >= 3:
    context_score = 1
  else:
    context_score = 0

  # Total score
  total_score = frequency_score + context_score

  character_profiles[name] = {
    "count": count,
    "frequency_score": frequency_score,
    "context_score": context_score,
    "total_score": total_score,
    "contexts": character_context.get(name, [])
  }


# --------------------------------
# 5. Rank character candidates
# --------------------------------

ranked_characters = sorted(
  character_profiles.items(),
  key=lambda x: x[1]["total_score"],
  reverse=True
)


# --------------------------------
# 6. Display top candidates
# --------------------------------

print("\nTop Character Candidates:")

for name, profile in ranked_characters[:20]:
  print(name, "->", profile["total_score"])
