import pymupdf
from pathlib import Path

pdf_path= Path("data/raw/Sherlock_Holmes.pdf")
processed_path= Path("data/processed/Sherlock_Holmes.txt")

# Extract text from a PDF file
def extract_text(pdf_path):
  with pymupdf.open(pdf_path) as doc:
    text= ""

    for page in doc:
      text+= page.get_text()

  return text

extracted_text = extract_text(pdf_path)
# print(type(extracted_text))
# print(len(extracted_text))

with open(processed_path, "w", encoding="utf-8") as file:
  file.write(extracted_text)