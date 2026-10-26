"""Corpus accuracy evaluation: compare LitMind's predictions with a user-supplied reference corpus.

Pipeline (see pipeline.run_evaluation):
  1. corpus.parse_corpus / build_annotations   -> ReferenceAnnotations (user-confirmed)
  2. matching.*                                -> deterministic matching + precision/recall/F1
  3. gemini_eval.*                             -> semantic judgements, evidence-cited, schema-validated
  4. report.*                                  -> category metrics, overall score, CSV/Excel export
"""
