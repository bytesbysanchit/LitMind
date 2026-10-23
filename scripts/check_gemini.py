"""Tiny connectivity test: one very small request. Run from the project folder:

    python3 scripts/check_gemini.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litmind.gemini_client import ChunkNotes, GeminiClient, GeminiError, get_api_key

key = get_api_key()
print("API key found:", bool(key), "(length %d)" % len(key) if key else "")
try:
    c = GeminiClient()
    out = c._call("Put the word OK in the notes field.", ChunkNotes)
    print(f"SUCCESS with model {c.active_model}: {out.notes!r}  ({c.calls} call(s))")
except GeminiError as e:
    print("FAILED:", e)
