import numpy as np, pandas as pd, pytest
from litmind.taste_model import TasteModel, load_books, clean_books, FormatError, COLUMNS
from litmind.relations import compute_relations
from litmind.characters import Character, CharacterResult
import io

def books(rows): 
    df = pd.DataFrame(rows, columns=["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"])
    df["ISBN"] = [str(i) for i in range(len(df))]
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return clean_books(df)

def synthetic():
    rng = np.random.default_rng(0)
    mystery = [(f"The Murder at Manor {i}", "Ann Sleuth", 1990 + i % 5, "Crime Press") for i in range(40)]
    other = [(f"Cooking with {w} {i}", f"Chef {i % 30}", 1995, "Food House") for i, w in enumerate(rng.choice(["rice", "beans", "soup"], 400))]
    return books(mystery[:15]), books(mystery[15:] + other)

def test_model_ranks_taste_above_random():
    hist, cat = synthetic()
    m = TasteModel().fit(hist, cat)
    good = m.predict_one("The Murder at Manor 99", "Ann Sleuth", 1992, "Crime Press")["percentile"]
    bad = m.predict_one("Cooking with rice 1", "Chef 3", 1995, "Food House")["percentile"]
    assert good > 90 and good > bad and m.report["cv_auc"] > 0.9

def test_history_must_not_leak_into_negatives_and_needs_5_books():
    hist, cat = synthetic()
    with pytest.raises(ValueError): TasteModel().fit(hist.head(3), cat)
    assert not set(m_ := TasteModel().fit(hist, cat).recommend(cat, hist, 50)["Book-Title"]) & set(hist["Book-Title"])

def test_format_error_and_junk_years():
    with pytest.raises(FormatError): load_books(io.StringIO("a,b\n1,2\n"))
    df = load_books(io.StringIO("ISBN,Book-Title,Book-Author,Year-Of-Publication,Publisher\n1,T,A,0,P\n2,T2,A2,2050,P\n3,T3,A3,1999,P\n"))
    assert df["Year"].isna().sum() == 2

def test_relations_rank_pairs_that_meet_more_than_chance():
    mk = lambda n, ids: Character(n, {n: len(ids)}, len(ids), sentence_ids=ids)
    together = list(range(0, 60, 3))
    res = CharacterResult([mk("A", together), mk("B", together), mk("C", list(range(0, 300, 3)))], [], [f"s{i}" for i in range(300)], 1)
    rels = compute_relations(res)
    assert (rels[0].a, rels[0].b) == ("A", "B") and rels[0].strength == 1.0 and rels[0].evidence
