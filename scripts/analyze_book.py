"""Command-line run of the full pipeline.

  python scripts/analyze_book.py data/samples/Dracula.txt
  python scripts/analyze_book.py book.txt --history my_read_books.csv --catalog data/Books.csv \
         --title "Dracula" --author "Bram Stoker" --year 1897 --gemini
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litmind import characters as C
from litmind.relations import compute_relations
from litmind.text_io import read_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("--min-mentions", type=int)
    ap.add_argument("--gemini", action="store_true", help="summary/genre + character review + relation explanations")
    ap.add_argument("--history"); ap.add_argument("--catalog")
    ap.add_argument("--title", default=""); ap.add_argument("--author", default="")
    ap.add_argument("--year", type=int); ap.add_argument("--publisher", default="")
    a = ap.parse_args()

    text = read_file(a.book)
    res = C.extract_characters(text, min_mentions=a.min_mentions)
    gc = None
    if a.gemini:
        from litmind.gemini_client import GeminiClient
        gc = GeminiClient()
        ctx = {c.name: [res.sentences[i][:220] for i in c.sentence_ids[:2]] for c in res.characters}
        rv = gc.review_characters(res, ctx)
        res = C.apply_review(res, rv.merge_groups, rv.drop)
    print(f"\n== {len(res.characters)} characters ==")
    for c in res.characters:
        print(f"{c.mentions:5d}  {c.name:24s} also: {', '.join(c.alias_list[:5])}")
    rels = compute_relations(res)
    if gc:
        gc.explain_relations(rels)
    print("\n== strongest relationships ==")
    for r in rels[:12]:
        print(f"{r.strength:.2f}  {r.a} & {r.b}  (together {r.together}x)  {r.relation_type} {r.reason}")
    if gc:
        rep = gc.analyze_book(text, a.title, a.author, progress=print)
        print("\n== summary ==\n", rep.analysis.primary_genre, "|", rep.analysis.one_line_pitch, "\n", rep.analysis.summary)
    if a.history and a.catalog:
        from litmind.taste_model import TasteModel, load_books
        hist, cat = load_books(a.history), load_books(a.catalog)
        m = TasteModel().fit(hist, cat)
        print("\n== taste ==", m.report)
        if a.title:
            print(m.predict_one(a.title, a.author, a.year, a.publisher))

if __name__ == "__main__":
    main()
