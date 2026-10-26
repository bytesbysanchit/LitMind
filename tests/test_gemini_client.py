import json, pytest
from litmind import gemini_client as g
from litmind.relations import Relation

class Stub:
    def __init__(self, replies): self.replies, self.log = list(replies), []
    def generate(self, *, model, prompt, schema, system):
        self.log.append((model, prompt)); r = self.replies.pop(0)
        if isinstance(r, Exception): raise r
        return r if isinstance(r, str) else json.dumps(r)

ANALYSIS = {"one_line_pitch": "p", "summary": "s", "primary_genre": "gothic horror", "sub_genres": [], "themes": ["fear"]}
class Err(Exception):
    def __init__(self, code): super().__init__(f"{code} error"); self.code = code

def client(replies, **kw): 
    b = Stub(replies); return g.GeminiClient(backend=b, sleep=lambda s: None, **kw), b

def test_single_call_and_genre_normalised():
    c, b = client([ANALYSIS]); rep = c.analyze_book("short text")
    assert rep.mode == "single" and rep.calls == 1 and rep.analysis.primary_genre in ("Gothic", "Horror")

def test_map_reduce_for_long_books(monkeypatch):
    monkeypatch.setattr(g, "SINGLE_CALL_CHARS", 100); monkeypatch.setattr(g, "CHUNK_CHARS", 300)
    text = "\n\n".join(["para " * 20] * 8)
    c, b = client([{"notes": "n"}] * 20 + [ANALYSIS])
    n_chunks = len(g._split_chunks(text, 300)); c.backend.replies = [{"notes": "n"}] * n_chunks + [ANALYSIS]
    rep = c.analyze_book(text)
    assert rep.mode == "map-reduce" and rep.calls == n_chunks + 1

def test_retry_on_429_then_success():
    c, b = client([Err(429), Err(503), ANALYSIS]); assert c.analyze_book("x").calls == 3

def test_falls_back_to_next_model_on_404():
    c, b = client([Err(404), ANALYSIS]); c.analyze_book("x")
    assert b.log[0][0] != b.log[1][0]

def test_bad_key_gives_friendly_error():
    c, _ = client([Err(403)])
    with pytest.raises(g.GeminiError, match="API key"): c.analyze_book("x")

def test_malformed_json_retried_once():
    c, _ = client(["not json", ANALYSIS]); assert c.analyze_book("x").analysis.summary == "s"

def test_explain_relations_matches_pairs_in_either_order():
    r = Relation("A", "B", 5, .5, 1.0, ["ev"])
    c, _ = client([{"relations": [{"a": "b", "b": "a", "relation_type": "rivals", "reason": "r", "tone": "negative"}]}])
    c.explain_relations([r]); assert r.relation_type == "rivals"

def test_missing_key_message(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False); monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(g, "_load_dotenv", lambda: None)
    with pytest.raises(g.GeminiError, match="GEMINI_API_KEY"): g.GeminiClient()

def test_persistent_503_falls_back_to_next_model():
    c, b = client([Err(503)] * 3 + [ANALYSIS])          # 1 try + 2 retries on model 1, then model 2 works
    rep = c.analyze_book("x")
    assert rep.model == c.models[1] and rep.calls == 4

def test_connection_reset_is_retried_not_fatal():
    c, b = client([ConnectionResetError(54, "Connection reset by peer"), ANALYSIS])
    assert c.analyze_book("x").calls == 2

def test_persistent_network_failure_gives_helpful_message():
    c, _ = client([ConnectionResetError(54, "Connection reset by peer")] * 40)
    with pytest.raises(g.GeminiError, match="VPN"): c.analyze_book("x")
