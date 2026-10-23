# LitMind 2

Rebuild of LitMind with a different approach to the three problems of the first version.

| Problem in v1 | Approach in v2 | Where |
|---|---|---|
| Places counted as characters; one person counted several times | spaCy NER **plus** label voting, dependency evidence, "the X" test, place/common-word filters, title stripping and alias clustering (short name → longer name, gender/dominance for shared surnames, nicknames, spellings, "the Count" → Dracula). Optional Gemini review merges what needs world knowledge. | `litmind/characters.py` |
| Relationship scorer unreliable | Strength is a transparent statistic (co-occurrence in 3-sentence windows, normalised, must beat chance). The *type and reason* come from Gemini, **only from quoted evidence**, and may be "unclear". | `litmind/relations.py`, `gemini_client.py` |
| Poor summary / genre | Gemini (`google-genai`): summary, genre, themes, setting, tone. Whole book in one call if it fits, otherwise map-reduce over chunks. | `litmind/gemini_client.py` |
| (new) "Will I like this book?" | Ridge linear regression trained on **your reading history CSV (same format as Books.csv)** vs. sampled catalogue books. | `litmind/taste_model.py` |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env            # then paste your Gemini key into .env
# put Books.csv in data/   (or set the path in the app's sidebar)
streamlit run app.py
```
CLI: `python scripts/analyze_book.py data/samples/Dracula.txt --gemini --history data/samples/my_read_books.csv --catalog data/Books.csv --title Dracula --author "Bram Stoker" --year 1897`
Tests: `pytest -q`

## Your reading-history CSV
Same columns as Books.csv: `ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-S, Image-URL-M, Image-URL-L`.
Easiest way to make it: copy the rows of the books you read out of Books.csv. Optional extra column `My-Rating` (1-5, 1-10 or 1-100).
`data/samples/my_read_books.csv` is a **demo** (30 gothic/mystery books chosen at random from Books.csv); replace it with your own.

## How to read the taste score
Books.csv has no ratings, so the regression target is: books you read = 1 (or rating/max), random catalogue books = 0.
Those zeros mean "not in my history", not "disliked". The result is therefore shown as a **percentile among catalogue books**
(not a probability), with the words/author/publisher that raised or lowered it. If none of a book's title words, author or
publisher were seen in training it says "Not enough information" instead of guessing. Features are title, author and
publisher only (that is all Books.csv contains); expect it to be strongest for authors/series you already read.
With a small or single-genre history the cross-validated AUC is optimistic (the demo shows ~0.99): trust it more with 50+ varied books.

## Known limits (character detection)
Measured on Dracula (13 characters, no places/rivers) and the Sherlock Holmes Adventures:
- Aliases that need world knowledge are not merged offline: *Lord Godalming* = *Arthur Holmwood*, *Mina Murray* = *Mina Harker*. The "Review with Gemini" button handles these.
- People sharing a surname with the same title type (e.g. Mrs. Westenra / Lucy Westenra) can merge; a shared surname with different gender titles (Jonathan vs Mina Harker) is separated.
- A few rare towns/historic names can survive (e.g. *Catherine* the Czarina in Dracula; the Gemini review drops these).
- Mention counts come from NER spans plus "the Count"-style titles, not from pronoun resolution, so they undercount.
- Gemini calls were not executed in development (no network access to Google); the logic was tested against a stub, so check the first real run and the model name (`GEMINI_MODEL`).
