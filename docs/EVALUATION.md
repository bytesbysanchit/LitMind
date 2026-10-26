# Corpus Accuracy Evaluation

Compares LitMind's results with a reference corpus you supply, for four categories: characters,
relationships, protagonist and genre. Open the **Corpus accuracy evaluation** tab.

## Workflow
1. Run the analyses you want evaluated: *Find characters*, optionally *Explain relationships with Gemini*,
   *Identify protagonist (Gemini)* and *Summarise with Gemini* (which also gives the genre).
2. Upload the reference corpus (`.xlsx`, `.csv`, `.json`, `.txt`, `.pdf`).
3. Check the detected sheets and columns; change them if needed. Preview the extracted annotations.
4. Tick the confirmation box ("I confirm this reference information is correct"). Nothing runs before that.
5. Run the evaluation. Review the tables, evidence and metrics. Export Excel or CSV.

## Accepted reference formats
| Format | How it is read |
|---|---|
| `.xlsx` | Every non-empty sheet is inspected. Sheet names and headers decide whether it is *characters*, *relationships*, *protagonist* or *genre*; you can override both. |
| `.csv` | One table; delimiter is detected (`, ; tab |`). |
| `.json` | Either the schema below, or a list of records (treated as one table). |
| `.txt` / `.pdf` | Raw book text. It supplies evidence passages. It contains no annotations, so you may ask Gemini to *propose* annotations; they are marked **UNVERIFIED** until you edit and confirm them. |

### Spreadsheet layout (recognised header names)
* **Characters**: `Name` (or *Character*), `Aliases` (*Also known as*, separated by `;`), `Description`, `Traits`.
* **Relationships**: `Character 1`, `Character 2`, `Relationship` (type), optional `Direction`, `Evidence`.
* **Protagonist**: either two columns `Field` / `Value` (Name, Narrative role, Traits, Motivations, Conflicts, Arc,
  Relationships, Passages) or one row per protagonist with those headers.
* **Genre**: a `Genre` column; several labels may be separated by `;`.

### JSON schema
```json
{
  "book_title": "Dracula",
  "characters": [{"name": "Mina Harker", "aliases": ["Mina Murray"], "description": "...", "traits": ["brave"]}],
  "relationships": [{"a": "Lucy Westenra", "b": "Arthur Holmwood", "type": "romance", "directed": false, "evidence": "..."}],
  "protagonists": [{"name": "Jonathan Harker", "narrative_role": "...", "traits": [], "motivations": "...",
                    "conflicts": "...", "arc": "...", "key_relationships": [], "passages": []}],
  "genres": ["Gothic", "Horror"],
  "multi_label_genre": true,
  "characters_exhaustive": false,
  "relationships_exhaustive": false,
  "text": "optional raw text for evidence passages"
}
```
Every field is optional. A category with no annotations is reported as **Not Evaluated**.
A sample workbook is in `data/samples/reference_dracula.xlsx` (one annotator's reading of the novel; illustrative).

## Rules the evaluation follows
* Reference and predictions are kept separate; Gemini never grades its own output and never supplies labels.
* Missing reference information is **Not Evaluated**, never an incorrect prediction.
* A prediction absent from the reference is **Not in reference**. It lowers precision only if you declare the
  reference exhaustive (tick the two "lists ALL" boxes); otherwise precision is shown as n/a.
* Names that could match several reference characters are listed under **Needs confirmation** and excluded
  from the metrics until you choose (no guessing).
* AI-proposed annotations stay **UNVERIFIED** until you confirm them, and the report says so.
* Gemini's judgements are model assessments against your reference, not independent human verification.

## Metrics
* **Characters**: recall = matched reference characters / reference characters. If exhaustive: precision =
  TP / (TP + not-in-reference + duplicates), F1 as usual. Descriptions are judged separately (Gemini).
* **Relationships**: pair-level recall (right two characters) and strict recall (pair **and** type correct); type
  accuracy among matched pairs; strict F1 if exhaustive. Types are compared mechanically through six classes
  (friendship, family, romance, conflict, mentorship, professional); labels that do not fit are judged by Gemini.
  LitMind does not guess types: until *Explain relationships* runs, matched pairs are Partial.
* **Protagonist**: name (exact, via aliases) plus descriptive fields judged against the reference. Field score:
  Correct 1, Partial 0.5, Incorrect 0; fields with no reference value are excluded. If the predicted protagonist
  is not the reference protagonist the score is 0.
* **Genre**: exact-match accuracy (single label) or precision/recall/F1 (multi-label). Gemini may *explain* a
  disagreement but cannot change the reference label.
* **Overall**: the plain average of the evaluated categories (formula shown in the app and in the export).

## Efficiency
The book is never sent whole. Passages are retrieved locally (TF-IDF), at most two per item, and items are batched
eight per request. Only items that need semantic comparison reach Gemini. Failed Gemini steps are reported and the
deterministic results are kept.
