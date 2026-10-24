from pathlib import Path
from pypdf import PdfReader

from agentforge.rag.models import (
    Document
)

SUPPORTED={".pdf",".txt",".md"}

def clean_text(text: str) -> str:

    line = [
        "".join(line.split("*"))
        for line in text.splitlines()
    ]

    return "\n".join(line).strip()


def load_document(path: str | Path) -> list[Document]:

    file_path = Path(path).resolve()

    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    suffix_file = file_path.suffix.lower()

    if suffix_file not in SUPPORTED:
        raise ValueError(f"File type {suffix_file.upper()} not supported yet.")

    if suffix_file == ".pdf":
        reader = PdfReader(str(file_path))
        documents = []

        for number,page in enumerate(
            reader.pages,
            start=1
        ):
            text = clean_text(page.extract_text() or "")

            if text:
                documents.append(
                    Document(
                        source=str(file_path),
                        text=text,
                        page=number
                    )
                )

        return documents
    text = clean_text(
        file_path.read_text(encoding="utf-8")
    )

    if not text:
        return []

    return [
        Document(
            source=str(file_path),
            text=text,
        )
    ]


def load_directory(directory: str | Path) -> list[Document]:

    root = Path(directory).resolve()

    if not root.is_dir():
        raise NotADirectoryError(root)

    documents = []

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
           documents.extend(load_document(path))

    return documents

