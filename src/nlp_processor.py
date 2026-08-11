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
# 3. Combine count + context
# --------------------------------

character_profiles = {}

for name, count in character.most_common():
    character_profiles[name] = {
        "count": count,
        "contexts": character_context.get(name, [])
    }


# --------------------------------
# 4. Test
# --------------------------------

print("Holmes Profile:")
print(character_profiles["Holmes"])