import io, json, re
import pandas as pd


def workbook(**sheets) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    return buf.getvalue()


REF_SHEETS = dict(
    Characters=pd.DataFrame({"Character Name": ["Mina Harker", "Dracula", "Lucy Westenra", "Renfield"],
                             "Also known as": ["Mina Murray; Mina", "Count Dracula", "Lucy", ""],
                             "Description": ["resourceful heroine", "ancient vampire", "young woman who is preyed upon", ""]}),
    Relations=pd.DataFrame({"Character 1": ["Lucy Westenra", "Mina Harker", "Dracula"], "Character 2": ["Arthur Holmwood", "Jonathan Harker", "Renfield"],
                            "Relationship": ["engaged", "married", "master and servant"]}),
    Protagonist=pd.DataFrame({"Field": ["Name", "Role", "Motivations"], "Value": ["Jonathan Harker", "narrator and hero", "survive and protect Mina"]}),
    Genre=pd.DataFrame({"Genre": ["Gothic; Horror"]}),
)


class SmartStub:
    """Backend that answers each evaluation prompt with schema-shaped JSON. `mode` can inject faults."""
    def __init__(self, status="correct", mode="ok"):
        self.status, self.mode, self.log = status, mode, []

    def generate(self, *, model, prompt, schema, system):
        self.log.append((schema.__name__, prompt))
        n = schema.__name__
        if self.mode == "raise":
            raise RuntimeError("503 UNAVAILABLE")
        if n == "JudgementBatch":
            ids = [int(i) for i in re.findall(r"ITEM (\d+)", prompt)]
            pas = re.findall(r"^\s+\[(\d+)\] ", prompt, re.M)
            js = [{"id": i, "status": self.status, "explanation": "Judged against the reference.", "evidence_ids": [int(pas[0])] if pas else []} for i in ids]
            if self.mode == "junk":
                js = js[:-1] + [{"id": 9999, "status": "correct", "explanation": "x", "evidence_ids": [123456]}] + [{**js[0], "evidence_ids": [987654]}]
            return json.dumps({"judgements": js})
        if n == "GenreExplanation":
            return json.dumps({"explanation": "The labels differ in emphasis."})
        if n == "ReferenceProposal":
            return json.dumps({"characters": [{"name": "Jonathan Harker", "aliases": ["Jonathan"], "description": "solicitor"}], "relationships": [], "protagonist": {"name": "Jonathan Harker"}, "genres": ["Gothic"]})
        if n == "ProtagonistAnalysis":
            return json.dumps({"name": "Jonathan Harker", "aliases": ["Jonathan"], "narrative_role": "narrator and hero", "motivations": "to survive and protect Mina"})
        raise AssertionError(n)
