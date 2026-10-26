import io, json
import pandas as pd
import pytest
from litmind import gemini_client as g
from litmind.evaluation import corpus as C, matching as M
from litmind.evaluation.gemini_eval import judge, Judgement
from litmind.evaluation.pipeline import EvalOptions, run_evaluation
from litmind.evaluation.report import to_csv_bytes, to_excel_bytes, to_frames
from litmind.evaluation.schema import *
from litmind.gemini_client import ProtagonistAnalysis
from tests.helpers import REF_SHEETS, SmartStub, workbook


def client(stub):
    return g.GeminiClient(backend=stub, sleep=lambda s: None, max_retries=0)


def ref_from_workbook(**kw):
    p = C.parse_corpus("ref.xlsx", workbook(**REF_SHEETS))
    return C.build_annotations(p, C.suggest_mappings(p), title="Dracula", **kw)


def pred(**over):
    base = Prediction("Dracula",
        [PredCharacter("Van Helsing", [], 451, "Dutch professor"), PredCharacter("Lucy Westenra", ["Lucy"], 317, "a young woman who is preyed upon by the vampire"),
         PredCharacter("Dracula", ["the Count"], 208, "ancient vampire"), PredCharacter("Mina Harker", [], 111, "resourceful heroine"),
         PredCharacter("Mina Murray", ["Mina"], 52), PredCharacter("Arthur Holmwood", [], 165), PredCharacter("Jonathan Harker", [], 205)],
        [PredRelation("Lucy Westenra", "Arthur Holmwood", "romance", "engaged", 0.4, 30), PredRelation("Mina Harker", "Jonathan Harker", "friendship", "", 0.3, 20)],
        ProtagonistAnalysis(name="Jonathan Harker", narrative_role="narrator and hero", motivations="to escape the castle"),
        ["Gothic", "Mystery"], "summary")
    for k, v in over.items():
        setattr(base, k, v)
    return base


# ------------------------------------------------------------ corpus upload and parsing
def test_xlsx_sheets_and_columns_detected():
    p = C.parse_corpus("ref.xlsx", workbook(**REF_SHEETS))
    kinds = {m.sheet: m.kind for m in C.suggest_mappings(p)}
    assert kinds == {"Characters": "characters", "Relations": "relationships", "Protagonist": "protagonist", "Genre": "genre"}
    ref = C.build_annotations(p, C.suggest_mappings(p))
    assert ref.available() == {"characters": True, "relationships": True, "protagonist": True, "genre": True}
    assert ref.characters[0].aliases == ["Mina Murray", "Mina"] and ref.genres == ["Gothic", "Horror"]


def test_user_can_override_sheet_and_column_mapping():
    df = pd.DataFrame({"Who": ["A Person", "B Person"], "Other names": ["ap", ""]})
    p = C.parse_corpus("x.xlsx", workbook(Sheet1=df))
    assert C.suggest_mappings(p)[0].kind == "ignore"                      # not guessable
    ref = C.build_annotations(p, [C.TableMapping("Sheet1", "characters", {"name": "Who", "aliases": "Other names"})])
    assert [c.name for c in ref.characters] == ["A Person", "B Person"] and ref.characters[0].aliases == ["ap"]


def test_csv_json_txt_and_semicolon_csv():
    p = C.parse_corpus("c.csv", b"name;aliases\nMina Harker;Mina\nRenfield;\n")
    assert list(p.tables.values())[0].shape == (2, 2)
    j = json.dumps({"characters": [{"name": "Mina"}, "Lucy"], "genres": ["Gothic"], "protagonists": {"name": "Jonathan"}}).encode()
    ref = C.build_annotations(C.parse_corpus("r.json", j))
    assert [c.name for c in ref.characters] == ["Mina", "Lucy"] and ref.protagonists[0].name == "Jonathan"
    t = C.parse_corpus("book.txt", ("It was a dark night. " * 30).encode())
    assert t.text and not C.build_annotations(t).available()["characters"]


@pytest.mark.parametrize("name,data,msg", [
    ("a.exe", b"x", "Unsupported"), ("a.csv", b"", "empty"), ("a.json", b"{bad", "malformed"), ("a.xlsx", b"not excel", "Excel"),
    ("a.txt", b"too short", "too short"), ("a.csv", b"name,aliases\n", "no rows"), ("a.json", b"[]", "object")])
def test_invalid_empty_and_unsupported_files(name, data, msg):
    with pytest.raises(C.CorpusError, match=msg):
        C.parse_corpus(name, data)


def test_incomplete_reference_reports_missing_categories():
    p = C.parse_corpus("g.csv", b"Genre\nGothic\n")
    ref = C.build_annotations(p, C.suggest_mappings(p))
    notes = " ".join(C.validate_reference(ref))
    assert "No characters annotations" in notes and "No protagonist" in notes


# ------------------------------------------------------------ characters
def test_character_matching_aliases_duplicates_and_misses():
    ref = ref_from_workbook()
    m = M.match_characters(pred().characters, ref.characters)
    st = {r.pred.name if r.pred else "-": r.status for r in m.rows}
    assert st["Lucy Westenra"] == "Matched" and st["Mina Harker"] == "Matched" and st["Mina Murray"] == "Duplicate"
    assert st["Van Helsing"] == "Not in reference" and st["-"] == "Missing"       # Renfield not detected


def test_ambiguous_name_needs_confirmation_then_override():
    ref = [RefCharacter(name="Mary Sutherland"), RefCharacter(name="Mary Holder")]
    pr = [PredCharacter("Mary", [], 10)]
    m = M.match_characters(pr, ref)
    assert m.rows[0].status == "Needs confirmation" and set(m.rows[0].candidates) == {"Mary Sutherland", "Mary Holder"}
    assert m.counts["tp"] == 0 and m.counts["unresolved"] == 1                    # excluded, not guessed
    m2 = M.match_characters(pr, ref, {"Mary": "Mary Holder"})
    assert m2.rows[0].status == "Matched" and m2.rows[0].ref.name == "Mary Holder"


def test_precision_only_when_reference_declared_exhaustive():
    ref = ref_from_workbook()
    m = M.match_characters(pred().characters, ref.characters)
    assert M.character_metrics(m, False)["precision"] is None
    ex = M.character_metrics(m, True)
    assert ex["recall"] == pytest.approx(3 / 4) and ex["precision"] == pytest.approx(3 / (3 + 1 + 1)) and ex["f1"] is not None


def test_missing_reference_info_is_not_a_wrong_prediction():
    ref = ref_from_workbook()
    rep = run_evaluation(pred(), ref)                                   # no Gemini
    row = next(r for r in rep.categories["Characters"].rows if r.get("Character") == "Van Helsing")
    assert row["Evaluation"] == "Not Evaluated"                         # reference non-exhaustive


# ------------------------------------------------------------ relationships
def test_relationship_matching_and_type_mismatch():
    ref = ref_from_workbook()
    p = pred()
    p.characters.append(PredCharacter("Renfield", [], 37))
    p.relationships.append(PredRelation("Dracula", "Renfield", "friendship", "", 0.2, 5))   # reference says master/servant
    cm = M.match_characters(p.characters, ref.characters)
    rows = {(r.pred.a, r.pred.b) if r.pred else ("-", r.ref.a): r for r in M.match_relationships(p.relationships, p.characters, ref.relationships, ref.characters, cm)}
    assert rows[("Lucy Westenra", "Arthur Holmwood")].status == "Correct"           # 'engaged' == 'romance'
    assert rows[("Mina Harker", "Jonathan Harker")].status == "Incorrect"          # 'married' vs 'friendship'
    assert rows[("Dracula", "Renfield")].status == "Pending semantic check"        # 'master and servant': not mechanical


def test_relationship_without_predicted_type_is_partial_and_empty_is_valid():
    ref = ref_from_workbook()
    p = pred(relationships=[PredRelation("Lucy Westenra", "Arthur Holmwood", "", "", 0.4, 30)])
    rep = run_evaluation(p, ref)
    assert rep.categories["Relationships"].rows[0]["Evaluation"] == "Partial"
    rep0 = run_evaluation(pred(relationships=[]), ref)                          # zero relationships: valid
    assert rep0.categories["Relationships"].status == "Evaluated" and rep0.categories["Relationships"].metrics["pair_recall"] == 0.0


def test_relation_types_canonicalised():
    assert M.canon_type("betrothed") == "romance" and M.canon_type("sworn enemies") == "conflict"
    assert M.canon_type("complicated") is None


# ------------------------------------------------------------ protagonist
def test_protagonist_name_and_field_scoring():
    ref = ref_from_workbook().protagonists
    _, rows = M.evaluate_protagonist_names(pred().protagonist, ref)
    by = {r.field: r for r in rows}
    assert by["name"].status == "Correct" and by["narrative_role"].semantic and by["traits"].status == "Not Evaluated"
    for r in rows:
        if r.semantic:
            r.status = {"narrative_role": "Correct", "motivations": "Partial"}[r.field]
    sc = M.protagonist_score(rows)
    assert sc["score"] == pytest.approx((1 + 1 + 0.5) / 3) and sc["fields_not_evaluated"] >= 1


def test_wrong_or_undetermined_protagonist_scores_zero():
    ref = ref_from_workbook().protagonists
    wrong = M.protagonist_score(M.evaluate_protagonist_names(ProtagonistAnalysis(name="Dracula"), ref)[1])
    none = M.protagonist_score(M.evaluate_protagonist_names(ProtagonistAnalysis(determinable=False), ref)[1])
    assert wrong["score"] == 0.0 and none["score"] == 0.0 and not wrong["name_correct"]


def test_protagonist_not_run_is_not_evaluated_not_wrong():
    rep = run_evaluation(pred(protagonist=None), ref_from_workbook())
    assert rep.categories["Protagonist"].status == "Not Evaluated" and "not run" in rep.categories["Protagonist"].reason


# ------------------------------------------------------------ genre
def test_single_label_genre_metrics():
    assert M.genre_evaluation(["Science Fiction"], ["sci-fi"], False)["exact_match"]
    r = M.genre_evaluation(["Horror"], ["Gothic", "Horror"], False)
    assert r["accuracy"] == 0.0 and r["in_secondary"] and r["missing"] == ["horror"]


def test_multi_label_genre_metrics():
    r = M.genre_evaluation(["Gothic", "Horror"], ["Gothic", "Mystery"], True)
    assert (r["precision"], r["recall"], r["f1"]) == (0.5, 0.5, 0.5) and r["missing"] == ["horror"] and r["extra"] == ["mystery"]


def test_genre_not_evaluated_without_reference_or_prediction():
    assert not M.genre_evaluation([], ["Gothic"], False)["evaluated"] and not M.genre_evaluation(["Gothic"], [], False)["evaluated"]


# ------------------------------------------------------------ gemini parsing and validation
def test_judge_drops_unknown_ids_invalid_citations_and_fills_missing():
    items = [{"id": 0, "title": "a", "reference": "r", "prediction": "p", "passages": [(5, "text five")]},
             {"id": 1, "title": "b", "reference": "r", "prediction": "p", "passages": []}]
    out = judge(client(SmartStub(mode="junk")), "character_details", items)
    assert set(out) == {0, 1}                                         # id 9999 ignored
    assert all(pid in (5,) for v in out.values() for pid, _ in v["evidence"])   # invented passage ids removed
    assert out[1]["status"] in ("Correct", "Not Evaluated")


def test_status_values_are_normalised():
    assert Judgement(id=1, status="Partially correct").status == "Partial" and Judgement(id=1, status="???").status == "Not Evaluated"


def test_reference_and_prediction_are_sent_as_separate_blocks_and_no_book_dump():
    stub = SmartStub()
    ref = ref_from_workbook()
    ref.source_text = ("Lucy Westenra and Arthur Holmwood were engaged. " * 3 + "\n\n") * 400      # long corpus
    run_evaluation(pred(), ref, client=client(stub))
    prompts = [p for n, p in stub.log if n == "JudgementBatch"]
    assert prompts and all("REFERENCE:" in p and "PREDICTION:" in p and "PASSAGES:" in p for p in prompts)
    assert max(len(p) for _, p in stub.log) < 12_000                    # never the whole corpus


# ------------------------------------------------------------ pipeline end to end, failures, unconfirmed
def test_pipeline_with_gemini_end_to_end_and_overall_formula():
    ref = ref_from_workbook()
    rep = run_evaluation(pred(), ref, client=client(SmartStub("correct")))
    assert rep.gemini_used and rep.gemini_calls >= 2
    chars = rep.categories["Characters"]
    assert next(r for r in chars.rows if r["Character"] == "Dracula")["Evaluation"] == "Correct"
    prot = rep.categories["Protagonist"]
    assert prot.score is not None and prot.score > 0.5
    overall, used = rep.overall
    scores = [rep.categories[n].score for n in used]
    assert overall == pytest.approx(sum(scores) / len(scores)) and set(used) <= {"Characters", "Relationships", "Protagonist", "Genre"}


def test_not_evaluated_categories_excluded_from_overall():
    ref = ref_from_workbook()
    ref.protagonists, ref.genres = [], []
    rep = run_evaluation(pred(), ref)
    assert rep.categories["Genre"].status == "Not Evaluated" and rep.categories["Protagonist"].status == "Not Evaluated"
    assert set(rep.overall[1]) <= {"Characters", "Relationships"}


def test_gemini_failure_keeps_deterministic_results_and_warns():
    rep = run_evaluation(pred(), ref_from_workbook(), client=client(SmartStub(mode="raise")))
    assert rep.categories["Characters"].metrics["recall"] == pytest.approx(0.75)
    assert any("Gemini step failed" in w for w in rep.warnings)
    assert all("RuntimeError" not in w for w in rep.warnings)


def test_unconfirmed_ai_reference_is_flagged_unverified():
    ref = ref_from_workbook()
    ref.origin, ref.confirmed = "ai_proposed", False
    rep = run_evaluation(pred(), ref)
    assert any("NOT confirmed" in w for w in rep.warnings)
    assert "UNVERIFIED" in to_frames(rep)["Summary"].to_string()


def test_genre_explanation_does_not_change_reference_labels():
    ref = ref_from_workbook()
    rep = run_evaluation(pred(), ref, client=client(SmartStub()))
    row = rep.categories["Genre"].rows[0]
    assert row["Reference genre"] == "Gothic, Horror" and "differ" in row["Explanation"]


def test_export_excel_and_csv_roundtrip():
    rep = run_evaluation(pred(), ref_from_workbook(), client=client(SmartStub()))
    xl = pd.read_excel(io.BytesIO(to_excel_bytes(rep)), sheet_name=None)
    assert {"Summary", "Characters", "Relationships", "Protagonist", "Genre"} <= set(xl)
    csv = pd.read_csv(io.BytesIO(to_csv_bytes(rep)))
    assert "Category" in csv.columns and set(csv["Category"]) >= {"Summary", "Characters"}
