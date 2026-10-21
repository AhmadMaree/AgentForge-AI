from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
class BaseEmbedding(ABC):

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...


class localTransformerEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    ):
        self.model = SentenceTransformer(model_name)
        self._dimention = int(self.model.get_sentence_embedding_dimension())


    @property

    def dimension(self) -> int:
       return self._dimention

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError(
                "Cannot embed empty text"
            )

        vector = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return vector.tolist()
        