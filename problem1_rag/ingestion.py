from pathlib import Path
from typing import List, Dict

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


if __name__ == "__main__":
    documents = load_corpus(CORPUS_DIR)

    print(f"Loaded {len(documents)} documents.\n")

    for document in documents:
        print("=" * 70)
        print(f"Document: {document['source']}")
        print(f"Type: {document['file_type']}")
        print(f"Characters: {len(document['text'])}")
        print()
        print(document["text"][:500])
        print()