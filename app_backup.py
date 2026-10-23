"""LitMind 2 - Streamlit app.   Run:  streamlit run app.py"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from litmind import characters as chars_mod
from litmind.gemini_client import GeminiClient, GeminiError, get_api_key
from litmind.relations import compute_relations
from litmind.taste_model import FormatError, TasteModel, load_books
from litmind.text_io import read_file, read_upload

ROOT = Path(__file__).resolve().parent
SAMPLES = {"Dracula": ROOT / "data/samples/Dracula.txt",
           "Sherlock Holmes (Adventures)": ROOT / "data/samples/Sherlock_Holmes.txt"}

st.set_page_config(page_title="LitMind 2", page_icon="📚", layout="wide")
st.title("📚 LitMind 2")
st.caption("Characters & aliases · relationships · Gemini summary & genre · taste match")


# ------------------------------------------------------------------ caching
@st.cache_resource(show_spinner="Loading spaCy model…")
def get_nlp(model: str):
    return chars_mod.load_nlp(model)


@st.cache_data(show_spinner="Reading book catalogue…")
def get_catalog(path: str):
    return load_books(path)


def gemini(key: str, model: str) -> GeminiClient:
    return GeminiClient(api_key=key or None, model=model or None)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    key = st.text_input("Gemini API key", value=get_api_key() or "", type="password",
                        help="Better: put GEMINI_API_KEY in a .env file.")
    model_name = st.text_input("Gemini model", value="", placeholder="default: gemini-3.5-flash")
    spacy_model = st.selectbox("spaCy model", ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"])
    st.divider()
    catalog_path = st.text_input("Books.csv path", value=str(ROOT / "data/Books.csv"))
    history_file = st.file_uploader("Books I have read (CSV, same columns as Books.csv)", type="csv")

ss = st.session_state

# ------------------------------------------------------------------ book input
st.subheader("1 · Choose a book")
c1, c2 = st.columns([1, 2])
sample = c1.selectbox("Sample", ["—"] + list(SAMPLES))
upload = c2.file_uploader("…or upload a .txt / text-based .pdf", type=["txt", "pdf"])
if upload is not None:
    name, text = upload.name, read_upload(upload.name, upload.getvalue())
elif sample != "—":
    name, text = sample, read_file(SAMPLES[sample])
else:
    name, text = None, None
if text:
    if ss.get("book_name") != name:
        for k in ("chars", "rels", "report", "review"):
            ss.pop(k, None)
        ss["book_name"] = name
    st.caption(f"{name}: {len(text):,} characters")
else:
    st.info("Pick a sample or upload a book to begin.")

tab_c, tab_s, tab_t = st.tabs(["Characters & relationships", "Summary & genre", "Taste match"])

# ------------------------------------------------------------------ tab 1
with tab_c:
    if text:
        min_m = st.slider("Minimum mentions", 3, 40, max(5, len(text) // 100_000))
        if st.button("Find characters", type="primary"):
            with st.spinner("Running NER and alias clustering (about 10–30 s for a novel)…"):
                ss["chars"] = chars_mod.extract_characters(text, get_nlp(spacy_model), min_mentions=min_m)
                ss["rels"] = compute_relations(ss["chars"])
                ss.pop("review", None)
        res = ss.get("chars")
        if res:
            if st.button("Review with Gemini (merge aliases, drop non-characters)"):
                try:
                    gc = gemini(key, model_name)
                    ctx = {c.name: [res.sentences[i][:220] for i in c.sentence_ids[:2]] for c in res.characters}
                    rv = gc.review_characters(res, ctx)
                    res = chars_mod.apply_review(res, rv.merge_groups, rv.drop)
                    ss["chars"], ss["review"] = res, rv
                    ss["rels"] = compute_relations(res)
                except GeminiError as e:
                    st.error(str(e))
            roles = {r.name.casefold(): r.role for r in getattr(ss.get("review"), "roles", [])}
            df = pd.DataFrame([{
                "Character": c.name,
                "Also called": ", ".join(c.alias_list[:6]),
                "Mentions": c.mentions,
                "Gender": c.gender or "",
                "Role (Gemini)": roles.get(c.name.casefold(), ""),
                "Note": c.note,
            } for c in res.characters])
            st.dataframe(df, width="stretch", hide_index=True)
            with st.expander(f"Rejected candidates ({len(res.rejected)}) – why they are not characters"):
                st.dataframe(pd.DataFrame(res.rejected).head(60), width="stretch", hide_index=True)

            rels = ss.get("rels") or []
            st.subheader("Relationships")
            st.caption("Strength = how often two characters share a 3-sentence window, normalised for how "
                       "often each appears (1.0 = always together). It measures connection, not the type of relationship.")
            if rels and st.button("Explain relationships with Gemini"):
                try:
                    with st.spinner("Asking Gemini…"):
                        gemini(key, model_name).explain_relations(rels)
                except GeminiError as e:
                    st.error(str(e))
            if rels:
                g = nx.Graph()
                for r in rels:
                    g.add_edge(r.a, r.b, weight=r.strength)
                pos = nx.spring_layout(g, seed=3, weight="weight")
                fig = go.Figure()
                for a, b, d in g.edges(data=True):
                    fig.add_trace(go.Scatter(x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines",
                                             line=dict(width=1 + 6 * d["weight"], color="rgba(150,150,170,.6)"),
                                             hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=[pos[n][0] for n in g], y=[pos[n][1] for n in g], mode="markers+text",
                                         text=list(g.nodes), textposition="top center",
                                         marker=dict(size=14, color="#7c83fd"), showlegend=False))
                fig.update_layout(height=480, xaxis=dict(visible=False), yaxis=dict(visible=False),
                                  margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, width="stretch")
                st.dataframe(pd.DataFrame([{
                    "A": r.a, "B": r.b, "Strength": r.strength, "Together": r.together,
                    "Relationship": r.relation_type, "Why": r.reason, "Tone": r.tone} for r in rels]),
                    width="stretch", hide_index=True)
                with st.expander("Evidence passages"):
                    for r in rels[:10]:
                        st.markdown(f"**{r.a} & {r.b}**")
                        for e in r.evidence:
                            st.write("› " + e)
            else:
                st.write("No pairs met the minimum co-occurrence.")

# ------------------------------------------------------------------ tab 2
with tab_s:
    if text:
        title_hint = st.text_input("Book title (optional, improves the answer)", value=name or "")
        author_hint = st.text_input("Author (optional)")
        if st.button("Summarise with Gemini", type="primary"):
            box = st.empty()
            try:
                ss["report"] = gemini(key, model_name).analyze_book(
                    text, title_hint, author_hint, progress=lambda m: box.info(m))
                box.empty()
            except GeminiError as e:
                box.error(str(e))
        rep = ss.get("report")
        if rep:
            a = rep.analysis
            st.markdown(f"### {a.primary_genre}")
            st.caption(" · ".join(a.sub_genres) + f"   |   {rep.calls} API call(s), {rep.mode}, model {rep.model}")
            st.markdown(f"> {a.one_line_pitch}")
            st.write(a.summary)
            st.markdown("**Themes:** " + ", ".join(a.themes))
            st.markdown(f"**Setting:** {a.setting}  \n**Tone:** {a.tone}")

# ------------------------------------------------------------------ tab 3
with tab_t:
    if history_file is None:
        st.info("Upload your reading history in the sidebar (same columns as Books.csv). "
                "A demo file is in data/samples/my_read_books.csv.")
    elif not Path(catalog_path).is_file():
        st.error(f"Books.csv not found at {catalog_path}. Put it in data/ or fix the path in the sidebar.")
    else:
        try:
            history = load_books(history_file)
            catalog = get_catalog(catalog_path)
        except FormatError as e:
            st.error(str(e))
            st.stop()
        st.write(f"{len(history)} books in your history · {len(catalog):,} in the catalogue.")
        if st.button("Train taste model", type="primary"):
            try:
                ss["taste"] = TasteModel().fit(history, catalog)
            except ValueError as e:
                st.error(str(e))
        model = ss.get("taste")
        if model:
            rp = model.report
            st.caption(f"Cross-validated AUC {rp.get('cv_auc', 'n/a')} · R² {rp.get('cv_r2', 'n/a')} · "
                       f"MAE {rp.get('cv_mae', 'n/a')} · {rp['n_negatives']} sampled unread books · "
                       f"ridge alpha {rp['best_alpha']}")
            st.markdown("##### Score a book")
            f1, f2, f3, f4 = st.columns(4)
            t = f1.text_input("Title", value=(name if name in SAMPLES else ""))
            au = f2.text_input("Author")
            yr = f3.number_input("Year", 0, 2026, 0)
            pu = f4.text_input("Publisher")
            if t and st.button("Predict match"):
                out = model.predict_one(t, au, int(yr) or None, pu)
                st.metric(out["verdict"], f"{out['percentile']:.0f}th percentile")
                st.caption("Percentile among typical catalogue books (not a probability).")
                st.write("**Raises the match:**", ", ".join(f"{n} ({v:+.3f})" for n, v in out["reasons"]["raises_match"]) or "–")
                st.write("**Lowers it:**", ", ".join(f"{n} ({v:+.3f})" for n, v in out["reasons"]["lowers_match"]) or "–")
            if st.button("Recommend 20 books from the catalogue"):
                st.dataframe(model.recommend(catalog, history, 20), width="stretch", hide_index=True)
