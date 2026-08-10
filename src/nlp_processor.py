from pathlib import Path
import spacy

nlp= spacy.load("en_core_web_sm")

text_path= Path("data/processed/Sherlock_Holmes.txt")

with open(text_path, "r", encoding= "utf-8") as file:
  text= file.read()

sample_text= text[:1000]

doc= nlp(sample_text)
character= set()

for ent in doc.ents:
  if ent.label_ == "PERSON":
    character.add(ent.text)

sorted_characters = sorted(character)

for character in sorted_characters:
  print(character)