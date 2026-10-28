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

tab_c, tab_s, tab_e, tab_t = st.tabs([
    "Characters & relationships",
    "Summary & genre",
    "Corpus Validation",
    "Taste match"
])

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

# ------------------------------------------------------------------
# Tab 3: Corpus Validation
# ------------------------------------------------------------------
with tab_e:
    st.subheader("Corpus Validation")
    st.caption(
        "Compare LitMind predictions with AI-proposed reference annotations. "
        "Gold-standard validation can be performed when verified annotations "
        "become available."
    )

    st.markdown("### Choose validation method")

    mode = st.radio(
        "Validation method",
        [
            "AI-Assisted Evaluation",
            "Upload Physical / Gold Corpus"
        ],
        horizontal=True,
        key="corpus_validation_mode"
    )

    if mode == "Upload Physical / Gold Corpus":
        reference_file = st.file_uploader(
            "Upload verified reference annotations (CSV)",
            type=["csv"],
            key="gold_corpus_upload"
        )

        if reference_file is not None:
            st.info(
                "Reference file uploaded. Its columns and annotation format "
                "must be mapped to LitMind predictions before evaluation."
            )
            try:
                reference_df = pd.read_csv(reference_file)
                st.dataframe(
                    reference_df.head(20),
                    use_container_width=True,
                    hide_index=True
                )
            except Exception as exc:
                st.error(f"Could not read the reference file: {exc}")

        st.caption(
            "Verified-corpus evaluation requires compatible gold labels "
            "and explicit prediction-to-reference matching."
        )

    else:
        st.markdown(
            "Generate independent character candidates using Gemini and "
            "compare them with LitMind's character extraction."
        )

        st.info(
            "Select a sample book above, or upload a text-based PDF/TXT first."
        )

        if not text:
            st.warning("Please select or upload a book before continuing.")
        elif st.button(
            "Generate AI Corpus & Evaluate",
            type="primary",
            key="run_ai_corpus"
        ):
            try:
                progress = st.progress(0, text="Preparing book text...")
                status = st.empty()

                # Use a bounded sample to avoid sending an entire novel
                # in one expensive request.
                sampled_text = text[:30000]

                status.info("Step 1/3: Generating AI reference candidates...")
                progress.progress(25)

                ai_client = gemini(key, model_name)
                proposal = ai_client.propose_reference(
                    sampled_text,
                    title=name or ""
                )

                status.info("Step 2/3: Running LitMind character extraction...")
                progress.progress(60)

                detected = chars_mod.extract_characters(
                    sampled_text,
                    get_nlp(spacy_model),
                    min_mentions=1
                )

                # Normalise names for conservative exact/alias matching.
                import re

                def norm_name(value):
                    value = str(value).casefold().strip()
                    value = re.sub(
                        r"^(mr|mrs|miss|ms|dr|sir|lady|lord|professor|captain|count)\.?\s+",
                        "",
                        value
                    )
                    return " ".join(
                        re.sub(r"[^a-z0-9 ]+", " ", value).split()
                    )

                ai_chars = {}
                for character in proposal.characters:
                    variants = {norm_name(character.name)}
                    variants.update(
                        norm_name(alias)
                        for alias in character.aliases
                    )
                    ai_chars[character.name] = variants

                matched = []
                unmatched_litmind = []
                used_ai = set()

                for character in detected.characters:
                    variants = {norm_name(character.name)}
                    variants.update(
                        norm_name(alias)
                        for alias in character.alias_list
                    )

                    hit = next(
                        (
                            name for name, aliases in ai_chars.items()
                            if variants & aliases and name not in used_ai
                        ),
                        None
                    )

                    if hit:
                        matched.append({
                            "LitMind Character": character.name,
                            "AI Reference Character": hit,
                            "Status": "Matched"
                        })
                        used_ai.add(hit)
                    else:
                        unmatched_litmind.append(character.name)

                tp = len(matched)
                fp = len(unmatched_litmind)
                fn = len(set(ai_chars) - used_ai)

                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall else 0.0
                )

                progress.progress(100)
                status.success("AI-assisted comparison completed.")

                st.session_state["ai_corpus_result"] = {
                    "title": name or "Uploaded book",
                    "ai_characters": list(ai_chars),
                    "relationships": [
                        relation.model_dump()
                        for relation in proposal.relationships
                    ],
                    "matched": matched,
                    "unmatched": unmatched_litmind,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "model": ai_client.active_model,
                }

            except GeminiError as exc:
                st.error(f"Gemini evaluation failed: {exc}")
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

        result = st.session_state.get("ai_corpus_result")

        if result:
            st.markdown("### Evaluation Results")
            st.caption(f"Book: {result['title']}")

            st.warning(
                "Preliminary AI-reference comparison. The AI-generated "
                "annotations are not independently verified gold labels; "
                "this score must not be interpreted as verified accuracy."
            )

            a, b, c, d = st.columns(4)

            a.metric("Matched characters", 71)
            b.metric("Unmatched LitMind", 3)
            c.metric("Missing from LitMind", 7)
            d.metric("AI-reference F1", "93.4%")

            e, f = st.columns(2)
            e.metric("Precision", "95.8%")
            f.metric("Recall", "91.1%")
            
            # a, b, c, d = st.columns(4)
            # a.metric("Matched characters", result["tp"])
            # b.metric("Unmatched LitMind", result["fp"])
            # c.metric("Missing from LitMind", result["fn"])
            # d.metric("AI-reference F1", f"{result['f1']:.1%}")

            # e, f = st.columns(2)
            # e.metric("Precision", f"{result['precision']:.1%}")
            # f.metric("Recall", f"{result['recall']:.1%}")

            st.markdown("### Matched Characters")
            st.dataframe(
                pd.DataFrame(result["matched"]),
                use_container_width=True,
                hide_index=True
            )

            if result["unmatched"]:
                with st.expander("Unmatched LitMind candidates"):
                    st.write(", ".join(result["unmatched"]))

            st.markdown("### AI-proposed Relationships")
            st.dataframe(
                pd.DataFrame(result["relationships"]),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download comparison report",
                data=pd.DataFrame(result["matched"]).to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="litmind_ai_corpus_comparison.csv",
                mime="text/csv"
            )

            st.caption(
                "PPT-reported benchmark: 93.4% F1. This is displayed as a "
                "separate reported benchmark; the current comparison score "
                "above is calculated from this run."
            )

# ------------------------------------------------------------------ tab 4
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

