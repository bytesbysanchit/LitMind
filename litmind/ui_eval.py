"""Streamlit dashboard for Corpus Accuracy Evaluation (rendered as a tab of app.py)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .evaluation.corpus import (KINDS, CorpusError, TableMapping, build_annotations, parse_corpus,
                                suggest_mappings, validate_reference)
from .evaluation.passages import sample_text
from .evaluation.pipeline import EvalOptions, run_evaluation
from .evaluation.report import CATEGORIES, OVERALL_FORMULA, to_csv_bytes, to_excel_bytes
from .evaluation.schema import (RefCharacter, RefProtagonist, RefRelation, ReferenceAnnotations,
                                build_prediction)
from .gemini_client import GeminiError, gemini_configured

BADGE = {"Correct": "✅ Correct", "Partial": "🟡 Partial", "Incorrect": "❌ Incorrect", "Not Evaluated": "⚪ Not Evaluated"}


def _badge(v: str) -> str:
    for k, label in BADGE.items():
        if str(v).startswith(k):
            return label + str(v)[len(k):]
    return str(v)


def _pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _status_line(label: str, ok: bool, detail: str):
    st.markdown(f"{'✅' if ok else '⚪'} **{label}** — {detail}")


def render(ss, gemini_factory, book_text: str | None, book_name: str | None):
    st.subheader("Corpus Accuracy Evaluation")
    st.caption("Upload a reference corpus for this book. LitMind's results are compared with it for characters, "
               "relationships, protagonist and genre. Categories the reference does not cover are reported as "
               "Not Evaluated, never as wrong.")
    roles = {r.name: r.role for r in getattr(ss.get("review"), "roles", [])} if ss.get("review") else {}
    report_obj = ss.get("report")
    prediction = build_prediction(book_name or "", ss.get("chars"), ss.get("rels"), roles, ss.get("protagonist"),
                                  report_obj.analysis if report_obj else None)

    with st.expander("LitMind results available for evaluation", expanded=not prediction.characters):
        n_typed = sum(1 for r in prediction.relationships if r.type)
        _status_line("Characters", bool(prediction.characters), f"{len(prediction.characters)} detected" if prediction.characters else "run 'Find characters' first")
        _status_line("Relationships", ss.get("rels") is not None,
                     f"{len(prediction.relationships)} pairs, {n_typed} with a type" if ss.get("rels") is not None else "run 'Find characters' first")
        if prediction.relationships and not n_typed:
            st.caption("Tip: run 'Explain relationships with Gemini' so relationship types can be compared.")
        _status_line("Protagonist", prediction.protagonist is not None, "analysed" if prediction.protagonist else "run 'Identify protagonist' on the Characters tab")
        _status_line("Genre", bool(prediction.genres), ", ".join(prediction.genres) if prediction.genres else "run 'Summarise with Gemini' first")

    # ------------------------------------------------------------ 1. upload
    up = st.file_uploader("Reference corpus", type=["xlsx", "csv", "json", "txt", "pdf"], key="ref_upload",
                          help="Excel/CSV tables, a JSON file in the annotation schema, or plain text / PDF of the book.")
    if up is None:
        st.info("Upload a reference corpus to begin.")
        return
    sig = (up.name, up.size)
    if ss.get("ref_sig") != sig:
        for k in list(ss.keys()):
            if k in ("ref_parsed", "ref_report", "proposal", "ref_overrides") or k.startswith(("map_", "kind_", "col_", "amb_")):
                ss.pop(k, None)
        ss["ref_sig"] = sig
        try:
            ss["ref_parsed"] = parse_corpus(up.name, up.getvalue())
        except CorpusError as e:
            ss["ref_parsed"] = None
            ss["ref_error"] = str(e)
        else:
            ss.pop("ref_error", None)
    if ss.get("ref_error"):
        st.error(ss["ref_error"])
        return
    parsed = ss["ref_parsed"]

    c1, c2, c3 = st.columns(3)
    c1.metric("File", parsed.filename)
    c2.metric("Type", parsed.kind.upper())
    c3.metric("Sheets / tables" if parsed.tables else "Text length", len(parsed.tables) if parsed.tables else f"{len(parsed.text):,} chars")
    for w in parsed.warnings:
        st.warning(w)

    # ------------------------------------------------------------ 2. mapping / detection
    title = st.text_input("Book name", value=book_name or parsed.filename.rsplit(".", 1)[0], key="ref_title")
    mappings: list[TableMapping] = []
    if parsed.tables:
        st.markdown("##### Detected sheets and columns")
        st.caption("Check the automatic detection and change it if needed.")
        for sheet, df in parsed.tables.items():
            auto = ss.setdefault(f"map_{sheet}", suggest_mappings(parsed)[list(parsed.tables).index(sheet)])
            with st.expander(f"{sheet}  ({len(df)} rows, {len(df.columns)} columns)", expanded=auto.kind != "ignore"):
                kind = st.selectbox("This sheet contains", KINDS, index=KINDS.index(auto.kind) if auto.kind in KINDS else 4, key=f"kind_{sheet}")
                cols = {}
                if kind in ("characters", "relationships", "protagonist", "genre"):
                    from .evaluation.corpus import SYNONYMS
                    fields = list(SYNONYMS[kind]) if not ("field" in auto.columns and kind == "protagonist") else ["field", "value"]
                    cc = st.columns(min(4, len(fields)))
                    for i, f in enumerate(fields):
                        default = auto.columns.get(f) if kind == auto.kind else None
                        opts = ["(none)"] + list(df.columns)
                        pick = cc[i % len(cc)].selectbox(f, opts, index=opts.index(default) if default in opts else 0, key=f"col_{sheet}_{kind}_{f}")
                        if pick != "(none)":
                            cols[f] = pick
                st.dataframe(df.head(10), width="stretch", hide_index=True)
                mappings.append(TableMapping(sheet, kind, cols))
    exhaustive_c = st.checkbox("The reference lists ALL characters in the book (enables precision)", key="ex_c")
    exhaustive_r = st.checkbox("The reference lists ALL relationships in the book (enables precision)", key="ex_r")
    multi = st.selectbox("Genre labelling", ["Detect from the reference", "Single label", "Multi-label"], key="ref_multi")
    multi_val = {"Detect from the reference": None, "Single label": False, "Multi-label": True}[multi]

    try:
        ref = build_annotations(parsed, mappings, title=title, characters_exhaustive=exhaustive_c,
                                relationships_exhaustive=exhaustive_r, multi_label=multi_val)
    except CorpusError as e:
        st.error(str(e))
        return

    # ------------------------------------------------------------ 3. raw text only: AI proposal (unverified)
    no_annotations = not any(ref.available().values())
    if no_annotations and parsed.text:
        st.info("This corpus contains text but no annotations. LitMind cannot evaluate against raw text alone. "
                "You can ask Gemini to PROPOSE reference annotations from the text; they stay marked UNVERIFIED until "
                "you review, edit and confirm them. (The proposal never sees LitMind's predictions.)")
        if st.button("Propose reference annotations with Gemini"):
            try:
                with st.spinner("Reading sampled passages…"):
                    ss["proposal"] = gemini_factory().propose_reference(sample_text(parsed.text), title)
            except GeminiError as e:
                st.error(str(e))
        prop = ss.get("proposal")
        if prop is not None:
            st.warning("AI-proposed annotations: UNVERIFIED. Edit them, then confirm below. The proposal is based on a sample of the text and may be incomplete.")
            ch = st.data_editor(pd.DataFrame([{"name": c.name, "aliases": "; ".join(c.aliases), "description": c.description} for c in prop.characters]
                                             or [{"name": "", "aliases": "", "description": ""}]), num_rows="dynamic", key="prop_chars", width="stretch")
            rl = st.data_editor(pd.DataFrame([{"a": r.a, "b": r.b, "type": r.type, "evidence": r.evidence} for r in prop.relationships]
                                             or [{"a": "", "b": "", "type": "", "evidence": ""}]), num_rows="dynamic", key="prop_rels", width="stretch")
            pr = prop.protagonist
            pname = st.text_input("Protagonist name", pr.name, key="prop_pname")
            prole = st.text_input("Narrative role", pr.narrative_role, key="prop_prole")
            ptraits = st.text_input("Traits (; separated)", "; ".join(pr.personality_traits), key="prop_ptraits")
            pmot = st.text_input("Motivations", pr.motivations, key="prop_pmot")
            pconf = st.text_input("Major conflicts", pr.major_conflicts, key="prop_pconf")
            parc = st.text_input("Character arc", pr.character_arc, key="prop_parc")
            genres = st.text_input("Genres (; separated)", "; ".join(prop.genres), key="prop_genres")
            if st.checkbox("I have reviewed these annotations and confirm them as the reference", key="prop_confirm"):
                ref = ReferenceAnnotations(
                    book_title=title, source_text=parsed.text, origin="ai_proposed_user_confirmed", confirmed=True,
                    characters=[RefCharacter(**{k: r[k] for k in ("name", "aliases", "description")}) for _, r in ch.iterrows() if str(r["name"]).strip()],
                    relationships=[RefRelation(**{k: r[k] for k in ("a", "b", "type", "evidence")}) for _, r in rl.iterrows() if str(r["a"]).strip() and str(r["b"]).strip()],
                    protagonists=[RefProtagonist(name=pname, narrative_role=prole, traits=ptraits, motivations=pmot, conflicts=pconf, arc=parc)] if pname.strip() else [],
                    genres=genres, characters_exhaustive=False, multi_label_genre=multi_val)
            else:
                st.info("Tick the confirmation box to use these annotations.")
                return

    # ------------------------------------------------------------ 4. preview + confirm
    st.markdown("##### Reference preview")
    av = ref.available()
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Characters", sum(1 for c in ref.characters if c.declared))
    p2.metric("Relationships", len(ref.relationships))
    p3.metric("Protagonists", len(ref.protagonists))
    p4.metric("Genre labels", len(ref.genres))
    for note in validate_reference(ref):
        (st.warning if "UNVERIFIED" in note or "No usable" in note else st.caption)(note)
    if not any(av.values()):
        return
    with st.expander("Show annotation details"):
        if av["characters"]:
            st.dataframe(pd.DataFrame([c.model_dump() for c in ref.characters if c.declared]), width="stretch", hide_index=True)
        if av["relationships"]:
            st.dataframe(pd.DataFrame([r.model_dump() for r in ref.relationships]), width="stretch", hide_index=True)
        if av["protagonist"]:
            st.dataframe(pd.DataFrame([p.model_dump() for p in ref.protagonists]), width="stretch", hide_index=True)
        if av["genre"]:
            st.write(", ".join(ref.genres))
    confirmed = st.checkbox("I confirm this reference information is correct and may be used as the standard.", key="ref_confirmed")
    use_gem = st.checkbox("Use Gemini for semantic comparisons (descriptions, relationship meanings, genre explanation)",
                          value=gemini_configured(), disabled=not gemini_configured(), key="use_gem")
    if not gemini_configured():
        st.caption("Gemini is not configured on this server, so only the deterministic checks will run.")

    # ------------------------------------------------------------ 5. run
    if st.button("Run evaluation", type="primary", disabled=not confirmed):
        bar = st.progress(0.0, text="Starting")
        try:
            client = gemini_factory() if use_gem else None
        except GeminiError as e:
            st.error(str(e))
            client = None
        ss["ref_report"] = run_evaluation(prediction, ref, client=client, book_text=book_text or "",
                                          options=EvalOptions(use_gemini=use_gem, overrides=ss.get("ref_overrides", {})),
                                          progress=lambda msg, f: bar.progress(min(1.0, f), text=msg))
        bar.empty()
    rep = ss.get("ref_report")
    if rep is None:
        return
    _show_report(rep, ss, ref, prediction, book_text, gemini_factory, use_gem, confirmed)


def _show_report(rep, ss, ref, prediction, book_text, gemini_factory, use_gem, confirmed):
    st.divider()
    if not rep.reference_confirmed:
        st.error("The reference was AI-proposed and not confirmed. These results are UNVERIFIED.")
    if rep.gemini_used:
        st.caption("Semantic judgements were made by Gemini against your reference. They are model assessments, "
                   "not independent human verification. Deterministic metrics (matching, precision, recall, F1) involve no AI.")
    for w in rep.warnings:
        st.warning(w)
    overall, included = rep.overall
    m = st.columns(5)
    m[0].metric("Overall", _pct(overall), help=OVERALL_FORMULA)
    for i, name in enumerate(CATEGORIES, 1):
        c = rep.categories[name]
        m[i].metric(name, _pct(c.score) if c.status != "Not Evaluated" else "Not Evaluated", help=c.score_label or c.reason)
    with st.expander("How the overall score is calculated"):
        st.write(OVERALL_FORMULA)
        st.write("Included in this run: " + (", ".join(included) or "none"))

    # Plain-language conclusion based only on categories evaluated in this run.
    evaluated = [
        (name, rep.categories[name].score)
        for name in CATEGORIES
        if rep.categories[name].status != "Not Evaluated"
        and rep.categories[name].score is not None
    ]
    if not evaluated:
        st.info(
            "Conclusion: no accuracy conclusion can be drawn yet. The reference file does not "
            "contain usable annotations for any evaluated category."
        )
    elif not rep.reference_confirmed:
        st.warning(
            "Conclusion: preliminary comparison only. The reference annotations are not confirmed, "
            "so the displayed scores must not be reported as verified accuracy."
        )
    else:
        best_name, best_score = max(evaluated, key=lambda item: item[1])
        worst_name, worst_score = min(evaluated, key=lambda item: item[1])
        st.markdown("### Evaluation conclusion")
        st.write(
            f"LitMind's overall evaluation score is **{_pct(overall)}** across "
            f"{len(included)} evaluated categor{'y' if len(included) == 1 else 'ies'}. "
            f"The strongest category is **{best_name}** ({_pct(best_score)}); "
            f"the category needing the most improvement is **{worst_name}** ({_pct(worst_score)})."
        )
        st.caption(
            "Interpret this as accuracy against the uploaded reference annotations, not a universal "
            "measure of performance. The result is only as reliable as the completeness and correctness "
            "of the confirmed reference data."
        )

    tabs = st.tabs(list(CATEGORIES))
    for tab, name in zip(tabs, CATEGORIES):
        c = rep.categories[name]
        with tab:
            if c.status == "Not Evaluated":
                st.info(f"Not Evaluated — {c.reason}")
                continue
            for n in c.notes:
                if n:
                    st.caption(n)
            if name == "Characters":
                mm = c.metrics
                a, b, d, e = st.columns(4)
                a.metric("Recall", _pct(mm["recall"]))
                b.metric("Precision", _pct(mm["precision"]) if mm["precision"] is not None else "n/a")
                d.metric("F1", _pct(mm["f1"]) if mm["f1"] is not None else "n/a")
                e.metric("Duplicates", mm["dup"])
                _resolve_ambiguous(ss, c, rep)
            elif name == "Relationships":
                mm = c.metrics
                a, b, d, e = st.columns(4)
                a.metric("Pair recall", _pct(mm["pair_recall"]))
                b.metric("Strict recall (pair + type)", _pct(mm["strict_recall"]))
                d.metric("Type accuracy", _pct(mm["type_accuracy"]))
                e.metric("Strict F1", _pct(mm["strict_f1"]) if mm["strict_f1"] is not None else "n/a")
            elif name == "Protagonist":
                st.metric("Protagonist score", _pct(c.score), help="Mean of field scores: Correct 1, Partial 0.5, Incorrect 0")
            else:
                mm = c.metrics
                st.metric(c.score_label.capitalize(), _pct(c.score))
            df = pd.DataFrame([{k: v for k, v in r.items() if k != "Evidence" and not k.startswith("_")} for r in c.rows])
            if "Evaluation" in df:
                df["Evaluation"] = df["Evaluation"].map(_badge)
            st.dataframe(df, width="stretch", hide_index=True)
            ev = [(r, r.get("Evidence")) for r in c.rows if r.get("Evidence")]
            if ev:
                with st.expander(f"Supporting passages ({len(ev)})"):
                    for r, e in ev:
                        st.markdown("**" + str(next(iter(r.values()))) + "** — " + str(r.get("Explanation", "")))
                        for pid, txt in e:
                            st.caption(f"[passage {pid}] {txt}")
            bad = [r for r in c.rows if str(r.get("Evaluation", "")).startswith(("Incorrect", "Partial"))]
            if bad:
                with st.expander(f"Missing, partial and incorrect predictions ({len(bad)})"):
                    st.dataframe(pd.DataFrame([{k: v for k, v in r.items() if k != "Evidence" and not k.startswith("_")} for r in bad]), width="stretch", hide_index=True)

    x1, x2, _ = st.columns([1, 1, 3])
    x1.download_button("Download Excel report", to_excel_bytes(rep), file_name="litmind_evaluation.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    x2.download_button("Download CSV report", to_csv_bytes(rep), file_name="litmind_evaluation.csv", mime="text/csv")


def _resolve_ambiguous(ss, cat, rep):
    amb = [r for r in cat.rows if str(r.get("Name match", "")).startswith("Needs confirmation")]
    if not amb:
        return
    st.warning(f"{len(amb)} detected name(s) could refer to more than one reference character. Choose which one each is "
               "(or 'none'), then run the evaluation again. Until then they are excluded from the metrics.")
    choices = ss.setdefault("ref_overrides", {})
    for r in amb:
        cands = list(r.get("_cands", []))
        pick = st.selectbox(f"'{r['Character']}' is…", ["(undecided)"] + cands + ["none of these"], key=f"amb_{r['Character']}")
        if pick == "none of these":
            choices[r["Character"]] = None
        elif pick != "(undecided)":
            choices[r["Character"]] = pick
        else:
            choices.pop(r["Character"], None)
