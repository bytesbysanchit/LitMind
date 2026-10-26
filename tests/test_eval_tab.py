"""Drives the real evaluation tab. Streamlit's AppTest cannot upload files, so file_uploader is replaced by a
fake that returns the reference workbook; everything after that is the production code path."""
import tempfile
from pathlib import Path
from streamlit.testing.v1 import AppTest
from tests.helpers import REF_SHEETS, workbook

TMP = Path(tempfile.gettempdir()) / "litmind_ref_test.xlsx"


def script():
    import os
    from pathlib import Path
    import streamlit as st
    from litmind import ui_eval
    from litmind.characters import Character
    from litmind.relations import Relation
    from litmind.gemini_client import ProtagonistAnalysis
    data = Path(os.environ["REF_TEST_FILE"]).read_bytes()

    class Up:
        name, size = "ref.xlsx", len(data)
        def getvalue(self): return data
    orig_uploader = st.file_uploader
    st.file_uploader = lambda *a, **k: Up()
    ss = st.session_state
    if "chars" not in ss:
        class R:   # minimal CharacterResult-compatible object
            characters = [Character("Lucy Westenra", {"Lucy": 3}, 300, sentence_ids=[0]), Character("Dracula", {"the Count": 9}, 200, sentence_ids=[1]),
                          Character("Mina Harker", {"Mina": 5}, 100, sentence_ids=[2]), Character("Mary", {"Mary": 5}, 10, sentence_ids=[3])]
        ss["chars"] = R()
        ss["rels"] = [Relation("Lucy Westenra", "Mina Harker", 5, 0.4, relation_type="friendship"),
                      Relation("Dracula", "Lucy Westenra", 4, 0.3, relation_type="enemy")]
        ss["protagonist"] = ProtagonistAnalysis(name="Jonathan Harker", narrative_role="narrator and hero")
    try:
        ui_eval.render(ss, lambda: None, "Lucy and Dracula met at night.\n\nMina wrote in her journal about Lucy.", "Dracula")
    finally:
        st.file_uploader = orig_uploader          # AppTest runs in-process: never leak the fake into other tests


def run_app():
    import os
    TMP.write_bytes(workbook(**REF_SHEETS))
    os.environ["REF_TEST_FILE"] = str(TMP)
    return AppTest.from_function(script, default_timeout=60).run()


def test_tab_detects_sheets_requires_confirmation_then_runs():
    at = run_app()
    assert not at.exception, at.exception
    assert {s.label for s in at.selectbox if s.label == "This sheet contains"}                         # one per sheet
    kinds = [s.value for s in at.selectbox if s.label == "This sheet contains"]
    assert sorted(kinds) == ["characters", "genre", "protagonist", "relationships"]
    run_btn = [b for b in at.button if b.label == "Run evaluation"][0]
    assert run_btn.disabled                                                                             # not before confirmation
    [c for c in at.checkbox if c.key == "ref_confirmed"][0].check().run()
    [b for b in at.button if b.label == "Run evaluation"][0].click().run()
    assert not at.exception, at.exception
    rep = at.session_state["ref_report"]
    assert rep.categories["Characters"].status == "Evaluated"
    assert rep.categories["Genre"].status == "Not Evaluated"                                            # no genre predicted: not wrong
    assert len(at.tabs) >= 4 and any("Not Evaluated" in m.value for m in at.info)
    labels = [m.label for m in at.metric]
    assert "Overall" in labels and "Characters" in labels


def test_ambiguous_name_is_surfaced_for_confirmation():
    at = run_app()
    [c for c in at.checkbox if c.key == "ref_confirmed"][0].check().run()
    [b for b in at.button if b.label == "Run evaluation"][0].click().run()
    assert not at.exception, at.exception
    rows = at.session_state["ref_report"].categories["Characters"].rows
    assert any(r["Name match"].startswith("Not in reference") and r["Character"] == "Mary" for r in rows)


def test_bad_upload_shows_friendly_error():
    import os
    TMP.write_bytes(b"this is not an excel file")
    os.environ["REF_TEST_FILE"] = str(TMP)
    at = AppTest.from_function(script, default_timeout=60).run()
    assert not at.exception and any("Excel" in e.value for e in at.error)


# ------------------------------------------------------------ raw-text corpus -> AI proposal -> confirm -> run
TXT = Path(tempfile.gettempdir()) / "litmind_ref_test.txt"


def script_txt():
    import os
    from pathlib import Path
    import streamlit as st
    from litmind import ui_eval
    from litmind.characters import Character
    from litmind.gemini_client import GeminiClient, ProtagonistAnalysis
    from tests.helpers import SmartStub
    data = Path(os.environ["REF_TEST_FILE"]).read_bytes()

    class Up:
        name, size = "book.txt", len(data)
        def getvalue(self): return data
    orig_uploader = st.file_uploader
    st.file_uploader = lambda *a, **k: Up()
    ss = st.session_state
    if "chars" not in ss:
        class R:
            characters = [Character("Jonathan Harker", {"Jonathan": 3}, 205, sentence_ids=[0])]
        ss["chars"], ss["rels"] = R(), []
        ss["protagonist"] = ProtagonistAnalysis(name="Jonathan Harker", narrative_role="narrator and hero", motivations="to survive")
    try:
        ui_eval.render(ss, lambda: GeminiClient(backend=SmartStub()), None, "Dracula")
    finally:
        st.file_uploader = orig_uploader


def run_txt():
    import os
    TXT.write_text(("Jonathan Harker travelled to the castle of the Count. " * 8 + "\n\n") * 6)
    os.environ["REF_TEST_FILE"] = str(TXT)
    return AppTest.from_function(script_txt, default_timeout=60).run()


def test_raw_text_cannot_be_evaluated_until_proposal_is_confirmed():
    at = run_txt()
    assert not at.exception, at.exception
    assert any("no annotations" in i.value for i in at.info)
    assert not [b for b in at.button if b.label == "Run evaluation"]                                    # nothing to run yet
    [b for b in at.button if b.label.startswith("Propose reference")][0].click().run()
    assert not at.exception, at.exception
    assert any("UNVERIFIED" in w.value for w in at.warning)
    assert not [b for b in at.button if b.label == "Run evaluation"]                                    # still blocked: not confirmed
    [c for c in at.checkbox if c.key == "prop_confirm"][0].check().run()
    assert not at.exception, at.exception
    [c for c in at.checkbox if c.key == "ref_confirmed"][0].check().run()
    [b for b in at.button if b.label == "Run evaluation"][0].click().run()
    assert not at.exception, at.exception
    rep = at.session_state["ref_report"]
    assert rep.reference_origin == "ai_proposed_user_confirmed" and rep.reference_confirmed
    assert rep.categories["Protagonist"].status == "Evaluated" and rep.categories["Protagonist"].metrics["name_correct"]
