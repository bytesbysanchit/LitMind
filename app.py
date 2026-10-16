from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from litmind_backend import analyze_document

try:
    import pymupdf
except ImportError:
    pymupdf = None

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "Dattani_Tara_Corpus_Final.xlsx"
SAMPLE_DIR = ROOT / "data" / "processed"
SAMPLE_NOVELS = {
    "Dracula": SAMPLE_DIR / "Dracula.txt",
    "Sherlock Holmes": SAMPLE_DIR / "Sherlock_Holmes.txt",
}

st.set_page_config(
    page_title="LitMind | Literary Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2.2rem; max-width: 1500px;}
[data-testid="stMetric"] {background:rgba(127,139,160,.08);padding:1rem 1.1rem;border:1px solid rgba(127,139,160,.18);border-radius:14px;}
.lm-hero {padding:1.4rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,rgba(124,131,253,.20),rgba(39,174,160,.10));border:1px solid rgba(124,131,253,.25);margin-bottom:1.2rem;}
.lm-muted {color:var(--text-color);opacity:.72;font-size:.92rem;}
div[data-testid="stTabs"] button {font-weight:600;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_novel_analysis(path_text: str, modified: float) -> dict[str, Any]:
    # modified is included in the cache key so changed sample files are re-analysed.
    return analyze_document("novel", text_path=Path(path_text))


def _network(edges: list[dict[str, Any]], node_names: list[str] | None = None, kind: str = "cooccurrence") -> go.Figure:
    graph = nx.Graph()
    graph.add_nodes_from([str(x) for x in (node_names or []) if x])
    for edge in edges:
        a, b = edge.get("character_a"), edge.get("character_b")
        if not a or not b:
            continue
        value = edge.get("weight", edge.get("transition_count", edge.get("co_occurrence", 1)))
        try:
            value = max(1.0, float(value or 1))
        except (TypeError, ValueError):
            value = 1.0
        graph.add_edge(str(a), str(b), weight=value)
    fig = go.Figure()
    if not graph.nodes:
        fig.add_annotation(text="No connections were found for this input.", showarrow=False)
        fig.update_layout(height=430, template="plotly_dark")
        return fig
    pos = nx.spring_layout(graph, seed=19, weight="weight", k=0.95)
    ex, ey = [], []
    for a, b in graph.edges():
        x0, y0 = pos[a]; x1, y1 = pos[b]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1.8, color="rgba(203,213,225,.72)"), hoverinfo="skip", name="Connection"))
    palette = ["#8b5cf6", "#14b8a6", "#f59e0b", "#ec4899", "#60a5fa", "#a3e635", "#f97316"]
    nodes = list(graph.nodes)
    colors = [palette[i % len(palette)] for i, _ in enumerate(nodes)]
    hover = []
    sizes = []
    for node in nodes:
        strength = float(graph.degree(node, weight="weight"))
        hover.append(f"<b>{node}</b><br>Connection strength: {strength:.0f}")
        sizes.append(min(34, 17 + strength ** .35 * 2))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
        mode="markers+text", text=nodes, textposition="top center",
        hovertext=hover, hoverinfo="text",
        marker=dict(size=sizes, color=colors, line=dict(width=1.3, color="rgba(255,255,255,.8)")),
        textfont=dict(size=12, color="#FFFFFF", family="Arial Black, Arial, sans-serif"), name="Characters"
    ))
    fig.update_layout(
        height=540, template="plotly_dark", showlegend=False,
        margin=dict(l=15, r=15, t=20, b=15),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _render_novel(result: dict[str, Any], display_name: str) -> None:
    chars = result.get("characters", [])
    mentions = result.get("character_mention_counts", {})
    rels = result.get("relationship_results", [])
    edges = result.get("interaction_graph_edges", [])
    st.markdown(f'<div class="lm-hero"><h2 style="margin:0">Your story, explored</h2><p style="margin:.45rem 0 0;opacity:.8">Analysis results for <b>{display_name}</b></p></div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Characters found", f"{result.get('character_count', len(chars)):,}")
    m2.metric("Name mentions detected", f"{sum(mentions.values()):,}")
    m3.metric("Story sentences", f"{result.get('sentence_count', 0):,}")
    m4.metric("Connections", f"{len(edges):,}")
    st.caption("Character and relationship results are automated suggestions. Review the examples before treating a connection as a fact.")
    tabs = st.tabs(["Story overview", "Characters", "Character network", "Relationships"])
    with tabs[0]:
        left, right = st.columns([1.15, 1])
        with left:
            st.subheader("At a glance")
            st.write(f"LitMind selected **{len(chars)} character candidates** and detected **{sum(mentions.values()):,} named-person mentions** in the analysed text. Names can include false positives, so review the examples.")
            st.write(f"It also found **{len(rels)} relationship candidates** to inspect.")
        with right:
            if mentions:
                top = sorted(mentions.items(), key=lambda kv: kv[1], reverse=True)[:8]
                fig = go.Figure(go.Bar(x=[x[1] for x in top][::-1], y=[x[0] for x in top][::-1], orientation="h", marker_color="#14b8a6"))
                fig.update_layout(template="plotly_dark", height=330, margin=dict(l=10,r=10,t=20,b=10), xaxis_title="Mentions", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        st.subheader("Character names detected in this story")
        if mentions:
            df = pd.DataFrame([{"Character name": n, "Name mentions detected": count} for n, count in mentions.items()]).sort_values("Name mentions detected", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download character list", df.to_csv(index=False).encode("utf-8"), "litmind_characters.csv", "text/csv")
        else:
            st.info("The analyzer did not return person-name mentions for this text. Check the extracted text and language-model setup.")
        contexts = result.get("character_context", {})
        if contexts:
            with st.expander("See character examples in the text"):
                for name, context in list(contexts.items())[:25]:
                    st.markdown(f"**{name}**")
                    st.write(str(context)[:500])
    with tabs[2]:
        st.subheader("Who appears alongside whom?")
        st.caption("A line means the analyzer found a connection or co-occurrence in the text. It does not by itself prove friendship, family, or another relationship.")
        st.plotly_chart(_network(edges, chars), use_container_width=True)
        if edges:
            df = pd.DataFrame([{"Character": e.get("character_a"), "Also appears with": e.get("character_b"), "Shared appearances / strength": e.get("weight", e.get("co_occurrence", 1))} for e in edges])
            st.dataframe(df, use_container_width=True, hide_index=True)
    with tabs[3]:
        st.subheader("Possible relationships")
        st.caption("These are model-generated candidates, not confirmed facts.")
        if rels:
            for rel in rels:
                a, b = rel.get("character_a", "Character A"), rel.get("character_b", "Character B")
                label = str(rel.get("relationship", "Other / unclear")).replace("_", " ").title()
                with st.expander(f"{a}  ·  {label}  ·  {b}"):
                    score = rel.get("score", 0)
                    st.write(f"**Evidence score:** {score}")
                    evidence = rel.get("evidence", [])
                    if not evidence:
                        st.caption("No text excerpt was returned for this candidate.")
                    for item in evidence[:6]:
                        if isinstance(item, dict):
                            st.write(item.get("context", str(item)))
                        else:
                            st.write(str(item))
        else:
            st.info("No relationship candidates were returned.")


def _render_play(result: dict[str, Any]) -> None:
    summary = result.get("summary", {})
    chars = result.get("characters", [])
    edges = result.get("interaction_edge_rows", [])
    relations = result.get("semantic_relation_edges", [])
    title = result.get("title", result.get("play_id", "Selected play"))
    sections = result.get("sections")

    st.markdown(
        f'<div class="lm-hero"><h2 style="margin:0">{title}</h2>'
        '<p style="margin:.45rem 0 0;opacity:.8">Explore the characters and how dialogue moves between them.</p></div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dialogue entries", f"{summary.get('utterance_rows', 0):,}")
    m2.metric("Characters", f"{summary.get('reference_characters', len(chars)):,}")
    m3.metric("Conversation links", f"{len(edges):,}")
    section_count = len(sections) if isinstance(sections, (pd.DataFrame, list, tuple)) else 0
    m4.metric("Acts / sections", f"{section_count:,}")
    st.caption("Dialogue counts show who speaks in this edition. Conversation links show turn-taking, not family or friendship.")

    tab_names = ["Overview", "Characters", "Conversation map"]
    if relations:
        tab_names.append("Relationships")
    tabs = st.tabs(tab_names)
    with tabs[0]:
        st.subheader("At a glance")
        st.write(
            f"This edition contains **{summary.get('utterance_rows', 0):,} dialogue entries** "
            f"across **{summary.get('reference_characters', len(chars))} named speakers**."
        )
        mapped = summary.get("mapped_speaker_turns", 0)
        st.write(f"**{mapped:,} of {summary.get('utterance_rows', 0):,}** dialogue entries are linked to a known speaker.")
        unmapped = summary.get("unmapped_speaker_labels", {})
        blanks = summary.get("blank_speaker_rows", 0)
        if unmapped or blanks:
            st.warning("A few dialogue entries need a speaker-label check.")
        else:
            st.success("All dialogue entries are linked to a known speaker.")

    with tabs[1]:
        st.subheader("Characters in the play")
        if chars:
            df = pd.DataFrame([{
                "Character": c.get("name"),
                "Dialogue entries": c.get("speaker_turns", 0),
                "Example dialogue": c.get("first_context", ""),
            } for c in chars]).sort_values("Dialogue entries", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download character summary", df.to_csv(index=False).encode("utf-8"),
                "litmind_play_characters.csv", "text/csv",
            )
        st.caption("A dialogue entry is one recorded turn of speech. It is not the number of times a character's name appears.")

    with tabs[2]:
        st.subheader("Who speaks after whom?")
        st.caption("A connection means the two characters have consecutive dialogue turns somewhere in the play. It does not prove a personal relationship.")
        nodes = [c.get("name") for c in chars]
        st.plotly_chart(_network(edges, nodes, kind="turns"), use_container_width=True)
        if edges:
            df = pd.DataFrame([{
                "Characters": f"{e.get('character_a')} ↔ {e.get('character_b')}",
                "Dialogue switches": e.get("transition_count", 0),
                "A then B": e.get("a_then_b", 0),
                "B then A": e.get("b_then_a", 0),
            } for e in sorted(edges, key=lambda x: x.get("transition_count", 0), reverse=True)])
            st.dataframe(df, use_container_width=True, hide_index=True)

    if relations:
        with tabs[3]:
            st.subheader("Relationships described in the play notes")
            st.caption("These relationships come from the supplied character notes. They are a partial reference list, not relationships inferred from dialogue.")
            for e in relations:
                a, b = e.get("character_a"), e.get("character_b")
                label = str(e.get("relation_label", "Relationship")).replace("_", " ").title()
                with st.expander(f"{a} — {label} — {b}"):
                    evidence = e.get("evidence_text", "")
                    if evidence:
                        st.write(evidence)
                    else:
                        st.write("This connection is listed in the supplied character notes.")


st.markdown('<div class="lm-hero"><h1 style="margin:0">📚 LitMind</h1><p style="margin:.4rem 0 0;opacity:.82">Discover the characters, connections and emotional tone within a story.</p></div>', unsafe_allow_html=True)
with st.sidebar:
    st.header("Explore a story")
    mode = st.radio("What would you like to analyse?", ["Novel", "Play"], index=0, horizontal=False)
    st.divider()
    st.caption("Choose a built-in novel or upload your own story. Full-length books may take a few minutes to analyse.")

if mode == "Novel":
    st.subheader("Novel explorer")
    source_choice = st.selectbox(
        "Choose a novel or upload your own",
        ["Upload your own PDF / TXT", "Dracula — sample novel", "Sherlock Holmes — sample novel"],
        help="The two built-in examples are Dracula and Sherlock Holmes. You can also analyse your own text-based PDF or TXT."
    )
    input_path: Path | None = None
    display_name = ""
    temp_path: Path | None = None

    if source_choice == "Dracula — sample novel":
        sample_path = SAMPLE_NOVELS.get("Dracula")
        if sample_path and sample_path.is_file():
            input_path = sample_path
            display_name = "Dracula"
        else:
            st.error("Dracula sample file is missing from data/processed.")
    elif source_choice == "Sherlock Holmes — sample novel":
        sample_path = SAMPLE_NOVELS.get("Sherlock Holmes")
        if sample_path and sample_path.is_file():
            input_path = sample_path
            display_name = "Sherlock Holmes"
        else:
            st.error("Sherlock Holmes sample file is missing from data/processed.")
    else:
        uploaded = st.file_uploader(
            "Upload your novel",
            type=["pdf", "txt"],
            help="Text-based PDFs work best. Scanned PDFs need OCR before analysis."
        )
        if uploaded is not None:
            display_name = uploaded.name
            suffix = Path(uploaded.name).suffix.lower()
            try:
                if suffix == ".pdf":
                    if pymupdf is None:
                        st.error("PDF support is missing. Install it with `python -m pip install pymupdf`.")
                    else:
                        with pymupdf.open(stream=uploaded.getvalue(), filetype="pdf") as pdf:
                            extracted = "\n".join(page.get_text() for page in pdf)
                        if not extracted.strip():
                            st.error("No selectable text was found. This may be a scanned PDF; OCR is needed first.")
                        else:
                            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as f:
                                f.write(extracted)
                                temp_path = Path(f.name)
                            input_path = temp_path
                else:
                    decoded = uploaded.getvalue().decode("utf-8-sig")
                    if not decoded.strip():
                        st.error("This file is empty.")
                    else:
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as f:
                            f.write(decoded)
                            temp_path = Path(f.name)
                        input_path = temp_path
            except UnicodeDecodeError:
                st.error("Could not read this TXT as UTF-8. Save it as UTF-8 and upload it again.")
            except Exception as exc:
                st.error(f"Could not read the uploaded file: {exc}")
    if input_path is not None:
        st.caption(f"Ready to analyse: **{display_name or input_path.name}**")
        if st.button("Explore this novel", type="primary", use_container_width=True):
            try:
                with st.spinner("Exploring the story and identifying characters… Large books can take a few minutes on the first run."):
                    if temp_path is not None:
                        result = analyze_document("novel", text_path=input_path)
                    else:
                        result = _cached_novel_analysis(str(input_path.resolve()), input_path.stat().st_mtime)
                st.session_state["litmind_novel_result"] = result
                st.session_state["litmind_novel_name"] = display_name or input_path.name
            except Exception as exc:
                st.error(f"Could not analyse this novel: {exc}")
                st.info("Check that spaCy and the English language model are installed in this Python environment.")
            finally:
                if temp_path and temp_path.exists():
                    try: temp_path.unlink()
                    except OSError: pass
    if st.session_state.get("litmind_novel_result"):
        st.divider()
        _render_novel(st.session_state["litmind_novel_result"], st.session_state.get("litmind_novel_name", "Novel"))
else:
    st.subheader("Play explorer")
    if not DEFAULT_CORPUS.exists():
        st.error("The play corpus file `Dattani_Tara_Corpus_Final.xlsx` is missing from the project folder.")
        corpus_path = None
    else:
        corpus_path = DEFAULT_CORPUS
    play_choice = st.selectbox("Choose a play", ["Tara", "Dance Like a Man"])
    play_id = "D01" if play_choice == "Tara" else "D02"
    if corpus_path and st.button("Explore this play", type="primary", use_container_width=True):
        try:
            with st.spinner("Preparing the play analysis…"):
                result = analyze_document("play", corpus_path=corpus_path, play_id=play_id)
            st.session_state["litmind_play_result"] = result
            st.session_state["litmind_play_name"] = play_choice
        except Exception as exc:
            st.error(f"Could not analyse this play: {exc}")
    if st.session_state.get("litmind_play_result"):
        st.divider()
        _render_play(st.session_state["litmind_play_result"])

st.divider()
st.caption("LitMind · Explore characters, conversations and story connections.")
