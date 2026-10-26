"""Writes data/samples/reference_dracula.xlsx: an ILLUSTRATIVE reference corpus (one annotator, from reading
the novel) showing the expected workbook layout. Replace it with your own annotations for real evaluation."""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "samples" / "reference_dracula.xlsx"
characters = pd.DataFrame([
    ("Count Dracula", "Dracula; the Count; Voivode Dracula", "Transylvanian vampire count who travels to England", ""),
    ("Jonathan Harker", "Jonathan; Harker", "English solicitor who visits the castle and becomes Dracula's prisoner", "resourceful, anxious"),
    ("Mina Harker", "Mina Murray; Mina; Wilhelmina Murray", "Jonathan's fiancee then wife, schoolmistress, who helps organise the pursuit", "intelligent, brave"),
    ("Lucy Westenra", "Lucy; Westenra", "Mina's friend, courted by three men, becomes Dracula's victim", ""),
    ("Abraham Van Helsing", "Van Helsing; the Professor; Helsing", "Dutch doctor and expert who leads the fight against Dracula", "learned, determined"),
    ("John Seward", "Seward; Jack Seward; Dr. Seward; Jack", "Physician who runs an asylum and loves Lucy", ""),
    ("Arthur Holmwood", "Arthur; Lord Godalming; Godalming; Art; Holmwood", "Lucy's fiance, later Lord Godalming", ""),
    ("Quincey Morris", "Quincey; Morris", "Texan adventurer and Lucy's suitor", "brave, loyal"),
    ("R. M. Renfield", "Renfield", "Patient in Seward's asylum who serves Dracula", ""),
    ("Peter Hawkins", "Hawkins; Mr. Hawkins", "Solicitor who sends Jonathan to Transylvania", ""),
], columns=["Character Name", "Also known as", "Description", "Traits"])
relations = pd.DataFrame([
    ("Jonathan Harker", "Mina Harker", "married"), ("Lucy Westenra", "Arthur Holmwood", "engaged"),
    ("Abraham Van Helsing", "John Seward", "mentor and former student"), ("John Seward", "R. M. Renfield", "doctor and patient"),
    ("Count Dracula", "Jonathan Harker", "enemy"), ("Count Dracula", "Mina Harker", "enemy"),
    ("Lucy Westenra", "Mina Harker", "friends"), ("Jonathan Harker", "Peter Hawkins", "employer and employee"),
    ("Lucy Westenra", "Quincey Morris", "suitor"), ("Abraham Van Helsing", "Lucy Westenra", "doctor and patient"),
], columns=["Character 1", "Character 2", "Relationship"])
protagonist = pd.DataFrame([
    ("Name", "Jonathan Harker"), ("Aliases", "Jonathan"), ("Narrative role", "narrator of the opening chapters and a member of the group that hunts Dracula"),
    ("Motivations", "to escape Dracula's castle and then to protect Mina"), ("Major conflicts", "imprisonment in the castle and the fight against Dracula"),
], columns=["Field", "Value"])
genre = pd.DataFrame({"Genre": ["Gothic; Horror"]})
if __name__ == "__main__":
    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        for n, d in [("Characters", characters), ("Relationships", relations), ("Protagonist", protagonist), ("Genre", genre)]:
            d.to_excel(w, sheet_name=n, index=False)
    print("wrote", OUT)
