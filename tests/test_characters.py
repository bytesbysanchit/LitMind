import pytest
from litmind.characters import parse_surface, extract_characters, apply_review, CharacterResult, Character

def test_parse_strips_titles_possessive_and_headings():
    assert parse_surface("Dr. Seward's") == ("dr", ("Seward",))
    assert parse_surface("the Count") == ("count", ())
    assert parse_surface("DR. SEWARD'S DIARY")[1] == ("Seward",)
    assert parse_surface("Professor Van Helsing")[1] == ("Van", "Helsing")
    assert parse_surface("Neville St. Clair")[1] == ("Neville", "St", "Clair")

def _fake():
    mk = lambda n, al, m: Character(n, al, m, sentence_ids=[1, 2, 3])
    return CharacterResult([mk("Mina Harker", {"Mina Harker": 6}, 6), mk("Mina Murray", {"Mina Murray": 9, "Mina": 41}, 50),
                            mk("Danube", {"Danube": 9}, 9), mk("Lord Godalming", {"Godalming": 67}, 67)], [], ["s"] * 5, 100)

def test_apply_review_merges_drops_and_ignores_hallucinations():
    out = apply_review(_fake(), [["Mina Murray", "Mina Harker"], ["Mina Murray", "Nobody Real"]], ["Danube", "Ghost"])
    names = [c.name for c in out.characters]
    assert "Danube" not in names and len(out.characters) == 2
    mina = out.by_name("Mina Harker")
    assert mina.mentions == 56 and "Mina Harker" in mina.aliases

spacy = pytest.importorskip("spacy")
TEXT = ("Mr. Holmes looked at Watson. Holmes said that Sherlock Holmes never forgot a face. " * 6 +
        "Watson walked to Baker Street in London. Watson went in London again. They travelled in Whitby. " * 6 +
        "Mrs. Hudson smiled and she said hello to Holmes. " * 6)

def test_end_to_end_merges_aliases_and_rejects_places():
    try:
        res = extract_characters(TEXT, min_mentions=3)
    except RuntimeError:
        pytest.skip("spaCy model missing")
    names = [c.name for c in res.characters]
    assert sum("Holmes" in n for n in names) == 1          # Holmes / Mr. Holmes / Sherlock Holmes -> ONE character
    assert not any(n in ("London", "Whitby", "Baker Street") for n in names)
