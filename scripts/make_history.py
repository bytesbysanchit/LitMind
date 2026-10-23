"""Build your reading-history CSV from a plain list of titles.

1. Make a text file, one book per line:  Title   or   Title | Author
       Dracula | Bram Stoker
       Murder on the Orient Express
2. Run:
       python3 scripts/make_history.py my_books.txt data/Books.csv my_read_books.csv

The output has the same columns as Books.csv, ready for the app's "Books I have read" box.
Books not found in Books.csv are listed so you can fix the spelling or skip them.
"""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litmind.taste_model import load_books

def norm(x): return re.sub(r"[^a-z0-9 ]", "", str(x).casefold()).strip()

def main(list_file, catalog_file, out_file):
    cat = load_books(catalog_file)
    cat["_t"] = cat["Book-Title"].map(norm)
    rows, missing = [], []
    for line in Path(list_file).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.strip() == "EOF" or "<<" in line or line.strip().startswith(("cat ", "$ ", "python")):
            print(f"SKIP {line!r}  (looks like a terminal command, not a book)")
            continue
        title, _, author = (p.strip() for p in line.partition("|"))
        t = norm(title)
        a = norm(author)
        exact = cat[cat["_t"] == t]
        # whole-word match only, and never for very short text (avoids 'eof' matching 'faceoff')
        contains = cat[cat["_t"].str.contains(r"\b" + re.escape(t) + r"\b", na=False)] if len(t) >= 5 else cat.iloc[0:0]
        hit, warn = exact, ""
        if a:
            # prefer a row whose author matches: exact title first, then titles containing it
            for pool in (exact, contains):
                by_author = pool[pool["_author"].str.contains(re.escape(a), na=False)]
                if not by_author.empty:
                    hit = by_author
                    break
            else:
                warn = "  (WARNING: author did not match - check this row)"
        if hit.empty:
            hit = contains
        if hit.empty:
            missing.append(line)
        else:
            rows.append(hit.iloc[0])
            print(f"OK   {line}  ->  {hit.iloc[0]['Book-Title']} / {hit.iloc[0]['Book-Author']}{warn}")
    import pandas as pd
    if not rows:
        sys.exit("\nNo books were matched, so nothing was written. Check that " + str(list_file) +
                 " has one 'Title | Author' per line, and that the spelling matches Books.csv.")
    out = pd.DataFrame(rows)[["ISBN","Book-Title","Book-Author","Year-Of-Publication","Publisher","Image-URL-S","Image-URL-M","Image-URL-L"]]
    out.drop_duplicates("ISBN").to_csv(out_file, index=False)
    print(f"\nWrote {len(out)} books to {out_file}")
    if missing:
        print("Not found in Books.csv:", *missing, sep="\n  ")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
