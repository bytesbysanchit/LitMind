
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher.add("CHARACTER_ROOPA", [nlp.make_doc("Roopa")])

test_cases = {
    "TARA_S0057": "Oh, hi, Roopa. Come on in.",
    "TARA_S0178": "I'm Roopa. Hi, you're Tara.",
    "TARA_S0554": "Roopa, I think it will be better if you left.",
    "TARA_S0686": "Roopa slowly comes to him and lies down.",
}

for speech_id, text in test_cases.items():
    doc = nlp(text)

    ner_persons = [
        ent.text for ent in doc.ents
        if ent.label_ == "PERSON"
    ]

    alias_mentions = [
        doc[start:end].text
        for _, start, end in matcher(doc)
    ]

    print(f"\n{speech_id}: {text}")
    print("NER PERSON:", ner_persons)
    print("Alias matches:", alias_mentions)
