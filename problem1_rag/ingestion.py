from pathlib import Path
from typing import List, Dict
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pypdf import PdfReader


CORPUS_DIR = Path("data/corpus")


def load_markdown(path: Path) -> str:
    """Read a Markdown file as plain text."""
    return path.read_text(encoding="utf-8")


def load_html(path: Path) -> str:
    """Extract visible text from an HTML file."""
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that don't contain useful document content.
    for element in soup(["script", "style"]):
        element.decompose()

    return soup.get_text(separator="\n", strip=True)


def load_pdf(path: Path) -> str:
    """Extract text from all pages of a PDF."""
    reader = PdfReader(str(path))

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def load_document(path: Path) -> str:
    """Load a PDF, HTML, or Markdown document."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)

    if suffix in {".html", ".htm"}:
        return load_html(path)

    if suffix in {".md", ".markdown"}:
        return load_markdown(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def load_corpus(corpus_dir: Path) -> List[Dict]:
    """Load every supported document in the corpus."""
    documents = []

    supported_extensions = {".pdf", ".html", ".htm", ".md", ".markdown"}

    for path in sorted(corpus_dir.iterdir()):

        if not path.is_file():
            continue

        if path.suffix.lower() not in supported_extensions:
            continue

        text = load_document(path)

        documents.append(
            {
                "document_id": path.stem,
                "source": path.name,
                "file_type": path.suffix.lower().lstrip("."),
                "text": text,
            }
        )

    return documents

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

os.getenv("CHUNK_SIZE", "500")

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks without breaking words."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        target_end = min(start + chunk_size, len(text))

        # End the chunk at a word boundary.
        if target_end < len(text):
            end = text.rfind(" ", start, target_end)

            if end <= start:
                end = target_end
        else:
            end = target_end

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Keep approximately `overlap` characters from the previous chunk.
        overlap_start = max(start, end - overlap)

        # Start at the beginning of the word containing overlap_start.
        previous_space = text.rfind(" ", start, overlap_start)

        if previous_space == -1:
            start = overlap_start
        else:
            start = previous_space + 1

    return chunks

def create_chunks(document: Dict) -> List[Dict]:
    """Create chunk records with metadata for a single document."""

    chunks = chunk_text(
        document["text"],
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    chunk_records = []

    for index, chunk in enumerate(chunks):
        chunk_records.append(
            {
                "chunk_id": f"{document['document_id']}_{index}",
                "document_id": document["document_id"],
                "source": document["source"],
                "file_type": document["file_type"],
                "chunk_index": index,
                "text": chunk,
            }
        )

    return chunk_records

if __name__ == "__main__":
    documents = load_corpus(CORPUS_DIR)

    print(f"Loaded {len(documents)} documents.")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}\n")

    all_chunks = []

    for document in documents:
        chunks = create_chunks(document)
        all_chunks.extend(chunks)

        print("=" * 70)
        print(f"Document: {document['source']}")
        print(f"Chunks: {len(chunks)}")

    print("\n" + "=" * 70)
    print(f"Total chunks: {len(all_chunks)}")

    print("\nExample chunk record:")
    print(all_chunks[0])