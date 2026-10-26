import json, re
from pathlib import Path
import pytest
from litmind import gemini_client as g
from litmind.characters import CharacterResult, extract_characters
from litmind.relations import compute_relations
from litmind.render import clean_label_list, clean_summary, format_genres, summary_paragraphs
from tests.helpers import SmartStub

ROOT = Path(__file__).resolve().parents[1]
SRC = [p for p in list(ROOT.glob("*.py")) + list((ROOT / "litmind").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))]


# ------------------------------------------------------------ summary rendering
def test_summary_json_object_is_unwrapped():
    raw = '{"summary": "Harker travels to Transylvania.\\n\\nHe escapes.", "primary_genre": "Gothic"}'
    assert clean_summary(raw) == "Harker travels to Transylvania.\n\nHe escapes."


def test_summary_fenced_json_escaped_newlines_and_metadata_removed():
    assert clean_summary('```json\n{"summary": "One. Two."}\n```') == "One. Two."
    assert clean_summary("First part.\\n\\nSecond part.") == "First part.\n\nSecond part."
    assert clean_summary('Real text.\n"primary_genre": "Gothic",\n"themes": ["x"]') == "Real text."


def test_summary_duplicates_removed_and_long_text_paragraphed():
    assert clean_summary("Same para.\n\nSame para.\n\nOther.") == "Same para.\n\nOther."
    wall = " ".join(f"Sentence number {i} goes here with some words." for i in range(30))
    assert len(summary_paragraphs(wall)) > 2
    assert clean_summary(None) == "" and clean_summary("") == ""


def test_broken_json_summary_is_recovered_not_shown_raw():
    out = clean_summary('{"summary": "A tale of dread.", "themes": ["fear"')        # truncated JSON
    assert out == "A tale of dread." and "{" not in out


def test_genre_formatting():
    assert format_genres('["Gothic", "Horror"]', None) == ("Gothic", ["Horror"])
    assert format_genres("Gothic, Horror", ["Horror", "Mystery"]) == ("Gothic", ["Horror", "Mystery"])
    assert format_genres("Mystery", ["mystery", "Crime"]) == ("Mystery", ["Crime"])
    assert format_genres(None, None) == ("", []) and clean_label_list({"a": "X"}) == ["X"]


def test_analyze_book_sanitises_json_in_summary_and_genre():
    class S:
        def generate(self, *, model, prompt, schema, system):
            return json.dumps({"one_line_pitch": "p", "summary": '{"summary": "Clean prose.\\n\\nMore."}',
                               "primary_genre": '["gothic"]', "sub_genres": ["Gothic", "Horror"]})
    rep = g.GeminiClient(backend=S()).analyze_book("short text")
    assert rep.analysis.summary == "Clean prose.\n\nMore." and rep.analysis.primary_genre == "Gothic"
    assert rep.analysis.sub_genres == ["Horror"]                                   # primary not repeated


def test_empty_summary_is_an_error_not_blank_output():
    class S:
        def generate(self, *, model, prompt, schema, system):
            return json.dumps({"one_line_pitch": "p", "summary": "", "primary_genre": "Gothic"})
    with pytest.raises(g.GeminiError, match="no usable summary"):
        g.GeminiClient(backend=S()).analyze_book("x")


def test_protagonist_analysis_through_client():
    c = g.GeminiClient(backend=SmartStub())
    p = c.identify_protagonist([{"name": "Jonathan Harker", "mentions": 205}], "summary", {"Jonathan Harker": ["He wrote."]})
    assert p.name == "Jonathan Harker" and p.determinable


# ------------------------------------------------------------ empty states, no minimum counts
def test_zero_characters_and_relationships_are_valid_results():
    res = extract_characters("Nothing but the wind and rain was heard that long evening. " * 6, min_mentions=1)
    assert isinstance(res, CharacterResult) and compute_relations(res) == []


def test_single_character_book_displays_normally():
    from litmind.characters import Character
    one = CharacterResult([Character("Solo", {"Solo": 3}, 3, sentence_ids=[0, 1, 2])], [], ["a", "b", "c"], 10)
    assert compute_relations(one) == []


def test_no_count_floor_hides_a_character():
    text = ("Mr. Holmes met Mrs. Hudson. " * 2 + "Mr. Holmes spoke to Watson. Watson nodded. " * 3 +
            "The Danube was wide.\n\n") * 3
    res = extract_characters(text, min_mentions=1)
    assert res.characters, "characters found with no minimum"
    assert not any("fewer than" in r["reason"] for r in res.rejected)


def test_pair_seen_once_can_be_listed():
    from litmind.characters import Character
    sents = [f"s{i}" for i in range(30)]
    res = CharacterResult([Character("A", {"A": 4}, 4, sentence_ids=[0, 10, 20, 25]), Character("B", {"B": 4}, 4, sentence_ids=[0, 11, 21, 29]),
                           Character("C", {"C": 1}, 1, sentence_ids=[28])], [], sents, 1)
    pairs = {(r.a, r.b): r.together for r in compute_relations(res)}
    assert ("A", "B") in pairs and pairs[("A", "B")] >= 1


# ------------------------------------------------------------ API key handling
SECRET = "AIzaSyFAKEKEY_1234567890abcdefghijk"


def test_error_messages_never_contain_the_key_or_chain_to_it():
    class Leaky:
        def generate(self, *, model, prompt, schema, system):
            raise RuntimeError(f"400 Bad request to https://x/y?key={SECRET} using {SECRET}")
    c = g.GeminiClient(api_key=SECRET, backend=Leaky(), sleep=lambda s: None)
    with pytest.raises(g.GeminiError) as ei:
        c.analyze_book("x")
    assert SECRET not in str(ei.value)
    assert "[hidden]" in str(ei.value)
    assert ei.value.__cause__ is None                                           # original exception not chained


def test_missing_key_message_is_safe_and_configured_flag(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(g, "_load_dotenv", lambda: None)
    assert g.gemini_configured() is False
    with pytest.raises(g.GeminiError, match="not configured"):
        g.GeminiClient()
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    assert g.gemini_configured() is True


def test_no_key_field_or_hardcoded_secret_in_source():
    for p in SRC:
        src = p.read_text(encoding="utf-8")
        assert 'type="password"' not in src, p
        assert not re.search(r"AIza[0-9A-Za-z_\-]{20,}", src), p
        assert not re.search(r'text_input\(\s*["\']Gemini API key', src), p
    ui = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "get_api_key" not in ui                                              # the UI never even reads the key
    assert "GEMINI_API_KEY=your-key-here" in (ROOT / ".env.example").read_text()
    assert ".env" in (ROOT / ".gitignore").read_text().split()


def test_key_never_reaches_the_rendered_page(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert SECRET not in str(at.main) and SECRET not in str(at.sidebar) and SECRET not in str(at.session_state)
    assert any("configured" in s.value for s in at.success)
    assert not [w for w in at.text_input if "api" in w.label.lower() and "key" in w.label.lower()]


# ------------------------------------------------------------ removed feature, UI
def test_cross_book_genre_comparison_does_not_exist():
    pat = re.compile(r"compare_?genre|genre_?compar|cross_?book|book_?to_?book|genre_?similar", re.I)
    for p in SRC + [ROOT / "README.md"]:
        assert not pat.search(p.read_text(encoding="utf-8")), p
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not any("compar" in t.label.lower() and "genre" in t.label.lower() for t in at.tabs)


def test_ui_has_no_minimum_count_controls_and_has_evaluation_tab():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
    assert not at.slider and "Corpus accuracy evaluation" in [t.label for t in at.tabs]
    at.selectbox[0].select("Sherlock Holmes (Adventures)").run()
    [b for b in at.button if b.label == "Find characters"][0].click().run()
    assert not at.exception
    shown = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption) + " ".join(i.value for i in at.info)
    assert "minimum" not in shown.lower()
    assert len(at.dataframe[0].value) <= 20 and any("Show all" in c.label for c in at.checkbox)
    assert any("Upload a reference corpus" in i.value for i in at.info)


def test_app_applies_no_mention_floor_to_characters():
    """The app itself must not drop characters for having few mentions (checked via what it stores)."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
    at.selectbox[0].select("Sherlock Holmes (Adventures)").run()
    [b for b in at.button if b.label == "Find characters"][0].click().run()
    chars = at.session_state["chars"].characters
    assert min(c.mentions for c in chars) == 1 and len(chars) > 60
    assert not any("fewer than" in r["reason"] for r in at.session_state["chars"].rejected)
    assert any(c.label == f"Show all {len(chars)} characters" for c in at.checkbox)


def test_ui_applies_no_mention_floor_to_the_character_list():
    """Regression guard: the app must list every detected character (no hidden minimum-mentions cut-off)."""
    from streamlit.testing.v1 import AppTest
    from litmind.characters import extract_characters, load_nlp
    from litmind.text_io import read_file
    path = ROOT / "data/samples/Dracula.txt"
    expected = len(extract_characters(read_file(path), load_nlp(), min_mentions=1, max_characters=500).characters)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
    at.selectbox[0].select("Dracula").run()
    [b for b in at.button if b.label == "Find characters"][0].click().run()
    assert not at.exception
    assert any(f"Show all {expected} characters" == c.label for c in at.checkbox), [c.label for c in at.checkbox]
    assert expected > 30                                                       # with a floor of 8 there would be only 13


# ------------------------------------------------------------ evaluation tab, driven like a user
def _app_with_dracula_characters():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    at.selectbox[0].select("Dracula").run()
    [b for b in at.button if b.label == "Find characters"][0].click().run()
    return at


def _uploader(at):
    return [f for f in at.file_uploader if f.key == "ref_upload"][0]


def test_evaluation_tab_full_flow_with_excel_reference(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from litmind import gemini_client as gc
    monkeypatch.setattr(gc, "_load_dotenv", lambda: None)
    at = _app_with_dracula_characters()
    ref = (ROOT / "data/samples/dracula_reference_demo.xlsx").read_bytes()
    _uploader(at).upload("dracula_reference_demo.xlsx", ref).run()
    assert not at.exception
    assert any("Detected sheets and columns" in m.value for m in at.markdown)           # sheets + mapping shown
    run_btn = [b for b in at.button if b.label == "Run evaluation"][0]
    assert run_btn.proto.disabled                                                        # cannot run before confirming
    [c for c in at.checkbox if c.key == "ref_confirmed"][0].check().run()
    [b for b in at.button if b.label == "Run evaluation"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Overall" in labels and "Characters" in labels and "Genre" in labels
    genre_metric = next(m for m in at.metric if m.label == "Genre")
    assert genre_metric.value == "Not Evaluated"                                         # LitMind genre not run: not wrong
    assert any("Pair recall" == m.label for m in at.metric)


def test_evaluation_tab_rejects_bad_files_with_a_friendly_message():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    _uploader(at).upload("broken.json", b"{not json").run()
    assert not at.exception
    assert any("malformed" in e.value for e in at.error)
    _uploader(at).upload("empty.csv", b"").run()
    assert any("empty" in e.value.lower() for e in at.error)


def test_evaluation_tab_text_only_corpus_requires_confirmation_of_ai_annotations(monkeypatch):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    _uploader(at).upload("book.txt", ("A long enough passage of plain text without any annotations. " * 20).encode()).run()
    assert not at.exception
    assert any("no annotations" in i.value for i in at.info)
    assert not [b for b in at.button if b.label == "Run evaluation"]                     # nothing to run yet
