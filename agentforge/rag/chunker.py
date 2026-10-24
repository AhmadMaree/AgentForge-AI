
from hashlib import sha256
from agentforge.rag.models import Document, Chunk


class TextChunker:
    def __init__(self, chunk_size: int = 800, overlap:int = 120):
        if chunk_size <= 0:
            raise ValueError("chunk_size should be positive")

        if not 0 <= overlap <= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def spilt(
        self,
        documents: list[Document]
    ) -> list[Chunk]:
        chunks = []

        for doc in documents:

            document_id = sha256(
                (
                    f"{doc.source}|"
                    f"{doc.page}|"
                    f"{doc.text}"
                ).encode("utf-8")
            ).hexdigest()
            
            start = 0
            index = 0

            while start < len(doc.text):
                end = min(
                    start + self.chunk_size,
                    len(doc.text)
                )

                text = doc.text[start:end].strip()

                if text:
                    raw_id = (
                        f"{document_id}|{index}|"
                        f"{self.chunk_size}|{self.overlap}"
                    )

                    chunk_id = sha256(
                        raw_id.encode("utf-8")
                    ).hexdigest()

                    chunks.append(
                        Chunk(
                            id= chunk_id,
                            document_id=document_id,
                            source=doc.source,
                            page=doc.page,
                            text=text,
                            chunk_index=index,
                        )
                    )

                if end == len(doc.text):
                    break

                start = end - self.overlap
                index +=1

        return chunks


        