"""Create data/samples/dracula_reference_demo.xlsx, a small hand-written reference corpus for trying the
Corpus Accuracy Evaluation feature. The annotations are the developer's own reading of the novel and are
deliberately NOT exhaustive. Replace them with your own verified annotations for real evaluations."""
import sys
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data/samples/dracula_reference_demo.xlsx"
chars = pd.DataFrame([
    ("Dracula", "Count Dracula; the Count; Voivode", "Transylvanian vampire count who travels to England"),
    ("Jonathan Harker", "Jonathan", "young English solicitor who visits Castle Dracula"),
    ("Mina Harker", "Mina Murray; Mina", "Jonathan's fiancee and later wife, a schoolmistress who helps organise the hunt"),
    ("Lucy Westenra", "Lucy", "Mina's friend, courted by three men, who becomes Dracula's victim"),
    ("Abraham Van Helsing", "Van Helsing; the Professor", "Dutch doctor and vampire expert who leads the hunt"),
    ("John Seward", "Dr. Seward; Jack Seward; Jack", "doctor who runs the asylum and was a suitor of Lucy"),
    ("Arthur Holmwood", "Lord Godalming; Godalming; Art", "Lucy's fiance, later Lord Godalming"),
    ("Quincey Morris", "Quincey; Morris", "wealthy Texan, suitor of Lucy, who dies in the final pursuit"),
    ("Renfield", "", "asylum patient who eats insects and serves Dracula"),
    ("Peter Hawkins", "Hawkins", "solicitor and Jonathan's employer"),
], columns=["Character Name", "Also known as", "Description"])
rels = pd.DataFrame([
    ("Lucy Westenra", "Arthur Holmwood", "engaged", ""),
    ("Jonathan Harker", "Mina Harker", "engaged, then married", ""),
    ("Jonathan Harker", "Peter Hawkins", "employer and employee", ""),
    ("John Seward", "Renfield", "doctor and patient", ""),
    ("Abraham Van Helsing", "John Seward", "former teacher and student", ""),
    ("Dracula", "Jonathan Harker", "captor and prisoner", ""),
    ("Abraham Van Helsing", "Dracula", "adversaries", ""),
], columns=["Character 1", "Character 2", "Relationship", "Evidence"])
prot = pd.DataFrame([("Name", "Jonathan Harker"), ("Role", "narrator of the opening chapters and one of the group that hunts Dracula"),
                     ("Motivations", "to survive Castle Dracula, then protect Mina and destroy Dracula")], columns=["Field", "Value"])
genre = pd.DataFrame({"Genre": ["Gothic; Horror"]})
note = pd.DataFrame({"Note": ["Demo reference written by the developer. Not exhaustive. Verify before relying on it."]})
OUT.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    for n, d in [("README", note), ("Characters", chars), ("Relationships", rels), ("Protagonist", prot), ("Genre", genre)]:
        d.to_excel(w, sheet_name=n, index=False)
print("wrote", OUT)
