import pymupdf
from pathlib import Path

# Extract text from a PDF file
def extract_text(pdf_path):
  with pymupdf.open(pdf_path) as doc:
    text= ""

    for page in doc:
      text+= page.get_text()

  return text

def clean_text(text):
    words = text.strip().split()
    cleaned_text = " ".join(words)
    return cleaned_text

processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
  pdf_files = Path("data/raw").glob("*.pdf")

  for pdf_path in pdf_files:
    extracted_text = extract_text(pdf_path)
    cleaned_text = clean_text(extracted_text)

    processed_path = processed_dir / f"{pdf_path.stem}.txt"

    with open(processed_path, "w", encoding="utf-8") as file:
        file.write(cleaned_text)