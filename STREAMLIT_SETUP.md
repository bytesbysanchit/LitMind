# LitMind Streamlit setup

## What this stage adds

- `app.py`: Streamlit interface with separate Novel Analyzer and Play Analyzer modes.
- `litmind_backend.py`: shared backend facade; exposes novel mention counts and contexts to the UI.
- `requirements.txt`: project dependencies for the dashboard and existing analyzers.

The frozen files `src/nlp_processor.py` and `src/prev.py` are intentionally unchanged.

## Run on Windows (PowerShell)

Open PowerShell in the LitMind project root — the folder containing `app.py`, `litmind_backend.py`, `src/`, `data/`, and `Dattani_Tara_Corpus_Final.xlsx`.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

If PowerShell blocks activation, use the Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit should print a local URL, usually `http://localhost:8501`.

## Current UI scope

### Novel Analyzer
- Use one of the processed `.txt` samples in `data/processed/`, or upload a UTF-8 `.txt` file or text-based `.pdf`.
- Displays detected characters, mention counts, co-occurrence network, relationship candidates, supporting contexts and analyzer log.
- Scanned PDFs with no selectable text require OCR first.

### Play Analyzer
- Select Tara or Dance Like a Man from `Dattani_Tara_Corpus_Final.xlsx`.
- Displays corpus QA, annotated speaker turns, adjacent-turn interaction network, act-level sentiment when VADER is available, documented relationship evidence and review-required mention candidates.

## Interpretation notes

- Novel relationships are heuristic predictions, not validated facts.
- The Novel network represents co-occurrence/connection strength; it is not the same as a semantic relationship graph.
- Play character counts are annotated speaker turns, not NER mention counts.
- Play interaction edges represent adjacent annotated speaker turns, not semantic relationships.
- Documented play relationships are partial corpus-note evidence, not exhaustive gold labels.

## Known constraints

- This is the first Streamlit integration stage. It does not add Gemini summaries or deployment yet.
- Novel analysis runtime depends on input length and the installed spaCy model. Start with a shorter text to verify the environment before running a full-length book.
- Evaluation was paused by project decision; the UI does not present the diagnostic metrics as validated overall accuracy.
