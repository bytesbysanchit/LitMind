"""Safe integration facade for LitMind's existing novel and play analyzers.

This module deliberately leaves src/nlp_processor.py and src/prev.py unchanged.
Run from the LitMind project root.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import io
import runpy
import sys

from src.play_analyzer import analyze_play, analyze_all_plays


PROJECT_ROOT = Path(__file__).resolve().parent
NOVEL_SCRIPT = PROJECT_ROOT / "src" / "nlp_processor.py"
DEFAULT_PLAY_CORPUS = PROJECT_ROOT / "Dattani_Tara_Corpus_Final.xlsx"


def analyze_novel(text_path: str | Path) -> dict[str, Any]:
    """Run the frozen novel script and expose its results as a structured dict.

    The script is executed in an isolated runpy namespace; its source file is
    not edited. Console output is captured and returned for debugging/UI logs.
    """
    input_path = Path(text_path).expanduser().resolve()
    if not NOVEL_SCRIPT.is_file():
        raise FileNotFoundError(f"Novel analyzer not found: {NOVEL_SCRIPT}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Novel text not found: {input_path}")

    old_argv = sys.argv[:]
    output = io.StringIO()
    # nlp_processor.py calls plt.show() at top level; suppress the blocking GUI.
    import matplotlib.pyplot as plt
    old_show = plt.show
    try:
        sys.argv = [str(NOVEL_SCRIPT), str(input_path)]
        plt.show = lambda *args, **kwargs: None
        with redirect_stdout(output):
            ns = runpy.run_path(str(NOVEL_SCRIPT), run_name="litmind_novel_run")
    finally:
        sys.argv = old_argv
        plt.show = old_show

    graph = ns.get("graph")
    relationship_results = ns.get("relationship_results", {})
    relationships = []
    for pair, result in relationship_results.items():
        relationships.append({
            "character_a": pair[0],
            "character_b": pair[1],
            "relationship": result.get("relationship", "OTHER / UNKNOWN"),
            "confidence": result.get("confidence"),
            "score": result.get("score", 0),
            "co_occurrence": result.get("co_occurrence", 0),
            "evidence": result.get("evidence", []),
        })

    characters = sorted(str(name) for name in ns.get("valid_characters", set()))

    # Expose the existing frozen analyzer's PERSON-entity counts to the UI.
    # Canonical aliases are merged only when the analyzer's own lookup maps them
    # to a selected valid character; we do not modify the NLP source file.
    raw_counts = ns.get("character", {})
    canonical_lookup = ns.get("canonical_lookup", {})
    valid_set = set(characters)
    mention_counts: dict[str, int] = {}
    for raw_name, raw_count in raw_counts.items():
        raw_name = str(raw_name)
        canonical = str(canonical_lookup.get(raw_name, raw_name))
        if canonical in valid_set:
            mention_counts[canonical] = mention_counts.get(canonical, 0) + int(raw_count)
    # If canonical mapping yields no selected-name matches, still show the actual
    # detected PERSON names instead of a misleading zero-count empty table.
    if not mention_counts:
        mention_counts = {str(name): int(count) for name, count in raw_counts.items()}

    raw_contexts = ns.get("character_context", {})
    character_context: dict[str, list[str]] = {}
    for raw_name, examples in raw_contexts.items():
        canonical = str(canonical_lookup.get(str(raw_name), str(raw_name)))
        target = canonical if canonical in valid_set else str(raw_name)
        character_context.setdefault(target, [])
        for example in examples if isinstance(examples, (list, tuple)) else [examples]:
            if example and len(character_context[target]) < 3 and str(example) not in character_context[target]:
                character_context[target].append(str(example))

    graph_edges = []
    if graph is not None:
        graph_edges = [
            {"character_a": str(a), "character_b": str(b), **dict(data)}
            for a, b, data in graph.edges(data=True)
        ]

    return {
        "mode": "novel",
        "input_path": str(input_path),
        "text_characters": len(ns.get("text", "")),
        "sentence_count": len(list(ns["doc"].sents)) if ns.get("doc") is not None else 0,
        "token_count": len(ns["doc"]) if ns.get("doc") is not None else 0,
        "characters": characters,
        "character_count": len(characters),
        "character_mention_counts": mention_counts,
        "character_context": character_context,
        "total_person_mentions": sum(mention_counts.values()),
        "interaction_graph_edges": graph_edges,
        "relationship_results": relationships,
        "console_output": output.getvalue(),
        "status": "PASS",
    }


def analyze_document(
    mode: str,
    *,
    text_path: str | Path | None = None,
    corpus_path: str | Path = DEFAULT_PLAY_CORPUS,
    play_id: str | None = None,
) -> dict[str, Any]:
    """Common entry point: mode='novel', 'play', or 'all_plays'."""
    selected = mode.strip().casefold()
    if selected == "novel":
        if text_path is None:
            raise ValueError("text_path is required when mode='novel'")
        return analyze_novel(text_path)
    if selected == "play":
        if play_id is None:
            raise ValueError("play_id is required when mode='play' (D01 or D02)")
        return analyze_play(corpus_path, play_id)
    if selected == "all_plays":
        return analyze_all_plays(corpus_path)
    raise ValueError("mode must be 'novel', 'play', or 'all_plays'")
